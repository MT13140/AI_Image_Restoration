"""AI 图像修复（Inpainting）引擎。

默认模型：**LaMa**（Large-Mask inpainting，big-lama，FFC 架构）。
* 论文：Resolution-robust Large Mask Inpainting with Fourier Convolutions (WACV 2022)
* 权重：big-lama（TorchScript / PyTorch state dict），约 200MB
* 获取方式：通过 ``simple-lama-inpainting`` 包在首次使用时下载到本地缓存，
  也可以把 ``big-lama.pt`` 手动放到 ``models/`` 目录（离线可用）。

备用后端：``opencv``（Telea 算法）——传统扩散式修复，不是 AI 模型，
只在 LaMa 不可用时作为兜底，并且在 UI 中明确标注。

本模块不包含任何伪造实现：模型不可用时会抛出明确错误或降级并说明原因。
"""

from __future__ import annotations

import os
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from . import image_processor as ip
from . import mask_processor as mp
from config import DEFAULT_FEATHER, DEFAULT_MASK_DILATE, MODELS_DIR, clear_cuda_cache, detect_device, get_logger

logger = get_logger("ai_restore.inpainting")

#: 允许的本地 LaMa 权重文件名
LAMA_LOCAL_FILES = ("big-lama.pt", "big_lama.pt", "big_lama_fp32.pt", "lama_fp32.jit", "big-lama.jit")
#: 官方权重地址（simple-lama-inpainting v0.1.0 release，LaMa big-lama TorchScript）
LAMA_MODEL_URL = os.environ.get(
    "LAMA_MODEL_URL",
    "https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt",
)


def download_weights(dest: Path, url: str = LAMA_MODEL_URL, retries: int = 20) -> Path:
    """下载 LaMa 权重到指定路径（支持断点续传与重试，纯 Python 实现）。

    说明：之所以自己实现下载，是因为 ``simple-lama-inpainting`` 会把权重放到
    torch hub 缓存目录后再用 ``torch.jit.load(路径)`` 加载，而 PyTorch 的 JIT
    在 Windows 上无法打开包含非 ASCII 字符的路径（例如用户名是中文）。
    本项目用"文件对象"方式加载，因此不受中文路径影响。
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    attempt = 0
    while True:
        have = part.stat().st_size if part.exists() else 0
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "AI-Image-Restoration/1.0"})
            if have:
                req.add_header("Range", f"bytes={have}-")
            with urllib.request.urlopen(req, timeout=60) as resp:
                total = int(resp.headers.get("Content-Length", 0) or 0) + have
                mode = "ab" if have else "wb"
                with part.open(mode) as fh:
                    last_log = time.time()
                    while True:
                        block = resp.read(1 << 20)
                        if not block:
                            break
                        fh.write(block)
                        have += len(block)
                        if time.time() - last_log > 10:
                            last_log = time.time()
                            pct = f"{have / total * 100:.0f}%" if total else "?"
                            print(f"  正在下载 LaMa 权重: {have / 1048576:.0f} MB ({pct})", flush=True)
            if total and part.stat().st_size < total:
                raise IOError(f"下载不完整: {part.stat().st_size}/{total}")
            if dest.exists():
                dest.unlink()
            part.replace(dest)
            logger.info("LaMa 权重已保存: %s (%.1f MB)", dest, dest.stat().st_size / 1048576)
            return dest
        except Exception as exc:  # 断流 → 续传
            attempt += 1
            if attempt > retries:
                raise RuntimeError(f"LaMa 权重下载失败（已重试 {retries} 次）: {exc}") from exc
            print(f"  下载中断（第 {attempt} 次），2s 后续传…", flush=True)
            time.sleep(min(15.0, 2.0 * attempt))


@dataclass
class InpaintOutcome:
    """一次修复的完整输出。"""

    image: np.ndarray                      # 修复结果（RGB）
    mask_used: np.ndarray                  # 实际使用的（扩张后）Mask
    backend: str = "lama"
    backend_label: str = ""
    seconds: float = 0.0
    info: Dict[str, object] = field(default_factory=dict)
    raw_image: Optional[np.ndarray] = None  # 模型原始输出（融合前，便于质量分析）


class LaMaBackend:
    """LaMa 模型的加载与推理封装。"""

    label = "LaMa (big-lama) 深度学习修复"

    def __init__(self, device: Optional[str] = None, model_path: Optional[Path] = None, use_jit: bool = True):
        self.device_name = device or detect_device().device
        self.model_path = Path(model_path) if model_path else self._find_local_model()
        self.use_jit = bool(use_jit)
        self._model = None
        self._mode = ""          # "jit" 或 "package"
        self._lock = threading.Lock()
        self.load_seconds = 0.0
        self.load_error: Optional[str] = None

    # ---------------- 加载 ----------------
    @staticmethod
    def _find_local_model() -> Optional[Path]:
        for name in LAMA_LOCAL_FILES:
            p = MODELS_DIR / name
            if p.exists() and p.stat().st_size > 1024 * 1024:
                return p
        return None

    @property
    def available(self) -> bool:
        """是否已加载成功（不会触发加载）。"""
        return self._model is not None

    @property
    def mode(self) -> str:
        return self._mode

    def ensure_loaded(self) -> bool:
        if self._model is not None:
            return True
        with self._lock:
            if self._model is not None:
                return True
            t0 = time.time()
            try:
                import torch  # noqa: F401

                if self.model_path is None:
                    # 没有本地权重 → 下载到 models/ 后再用文件对象加载
                    target = MODELS_DIR / "big-lama.pt"
                    self.model_path = download_weights(target)
                self._load_from_path(self.model_path)
                self.load_error = None
            except Exception as exc:
                logger.warning("LaMa 优先路径失败: %s: %s", exc.__class__.__name__, exc)
                # 兜底：尝试 simple-lama-inpainting 自带封装
                try:
                    self._load_from_package()
                    self.load_error = None
                except Exception as exc2:
                    self.load_error = f"{exc.__class__.__name__}: {exc} / 兜底失败: {exc2.__class__.__name__}: {exc2}"
                    self._model = None
                    logger.warning("LaMa 加载失败: %s", self.load_error)
            self.load_seconds = time.time() - t0
        return self._model is not None

    def _load_from_path(self, path: Path) -> None:
        import torch

        # 关键：用文件对象加载，规避 PyTorch JIT 在 Windows 上不支持非 ASCII 路径的问题
        with open(path, "rb") as fh:
            model = torch.jit.load(fh, map_location=self.device_name)
        model.eval()
        model.to(self.device_name)
        self._model = model
        self._mode = "jit"
        logger.info("已加载本地 LaMa 模型: %s (device=%s)", path, self.device_name)

    def _load_from_package(self) -> None:
        """通过 simple-lama-inpainting 加载（首次使用会下载权重）。"""
        from simple_lama_inpainting import SimpleLama

        cached = getattr(self, "model_path", None)
        if cached is not None and Path(cached).exists():
            # 让包使用我们已经下载好的权重（其内部使用 torch.jit.load(路径)，
            # 若路径含中文可能失败，因此这一步只是兜底）
            os.environ["LAMA_MODEL"] = str(cached)
        model = SimpleLama(device=self.device_name)
        self._model = model
        self._mode = "package"
        cache = os.environ.get("LAMA_MODEL", "包内自动缓存")
        logger.info("已通过 simple-lama-inpainting 加载 LaMa (device=%s, weights=%s)", self.device_name, cache)

    # ---------------- 推理 ----------------
    @staticmethod
    def _pad_to_multiple(image: np.ndarray, mask: np.ndarray, multiple: int = 8):
        """LaMa 的 FFC 结构要求尺寸可被 8 整除，这里做反射填充。"""
        h, w = image.shape[:2]
        ph = (multiple - h % multiple) % multiple
        pw = (multiple - w % multiple) % multiple
        if ph == 0 and pw == 0:
            return image, mask, (0, 0)
        img = cv2.copyMakeBorder(image, 0, ph, 0, pw, cv2.BORDER_REFLECT_101)
        m = cv2.copyMakeBorder(mask, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=0)
        return img, m, (ph, pw)

    def infer(self, image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """执行一次修复，返回与输入同尺寸的 RGB 结果。"""
        if not self.ensure_loaded():
            raise RuntimeError(f"LaMa 模型不可用：{self.load_error}")

        img = mp.to_uint8(image_rgb)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        mask_l = mp.ensure_binary(mask, img.shape[:2])
        if mp.is_empty(mask_l):
            return img.copy()

        if self._mode == "package":
            return self._infer_package(img, mask_l)
        return self._infer_jit(img, mask_l)

    def _infer_package(self, img: np.ndarray, mask_l: np.ndarray) -> np.ndarray:
        from PIL import Image

        padded_img, padded_mask, _ = self._pad_to_multiple(img, mask_l)
        pil_img = Image.fromarray(padded_img)
        pil_mask = Image.fromarray(padded_mask)
        result = self._model(pil_img, pil_mask)
        out = np.asarray(result.convert("RGB") if hasattr(result, "convert") else result)
        if out.shape[:2] != img.shape[:2]:
            out = out[: img.shape[0], : img.shape[1]]
        return np.ascontiguousarray(out)

    def _infer_jit(self, img: np.ndarray, mask_l: np.ndarray) -> np.ndarray:
        import torch

        padded_img, padded_mask, _ = self._pad_to_multiple(img, mask_l)
        img_f = padded_img.astype(np.float32) / 255.0
        mask_f = (padded_mask > 0).astype(np.float32)

        image_t = torch.from_numpy(img_f).permute(2, 0, 1).unsqueeze(0).to(self.device_name)
        mask_t = torch.from_numpy(mask_f).unsqueeze(0).unsqueeze(0).to(self.device_name)

        with torch.inference_mode():
            out = self._model(image_t, mask_t)
        if isinstance(out, (list, tuple)):
            out = out[0]
        out = out.clamp(0, 1)
        # 掩膜内使用模型输出，掩膜外保留原图
        composed = out * mask_t + image_t * (1 - mask_t)
        arr = composed.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(arr[: img.shape[0], : img.shape[1]])


class OpenCVBackend:
    """传统扩散式修复（Telea），仅作为 LaMa 不可用时的兜底方案。"""

    label = "OpenCV Telea（传统算法 · 备用）"

    def __init__(self, radius: int = 6, method: str = "telea"):
        self.radius = int(radius)
        self.method = method
        self.load_error: Optional[str] = None

    @property
    def available(self) -> bool:
        return True

    def ensure_loaded(self) -> bool:
        return True

    def infer(self, image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        img = mp.to_uint8(image_rgb)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        m = mp.ensure_binary(mask, img.shape[:2])
        if mp.is_empty(m):
            return img.copy()
        flag = cv2.INPAINT_TELEA if self.method == "telea" else cv2.INPAINT_NS
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        out = cv2.inpaint(bgr, m, self.radius, flag)
        return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


class InpaintEngine:
    """统一修复引擎，负责后端选择、Mask 预处理和后处理流水线。"""

    BACKENDS = ("lama", "opencv")

    def __init__(self, backend: str = "lama", device: Optional[str] = None, model_path: Optional[Path] = None):
        self.backend_name = (backend or "lama").lower()
        self.device_name = device or detect_device().device
        self.model_path = model_path
        self._lama: Optional[LaMaBackend] = None
        self._opencv: Optional[OpenCVBackend] = None
        self.disabled_reason = ""

    # ---------------- 后端 ----------------
    def _get_backend(self):
        if self.backend_name == "opencv":
            if self._opencv is None:
                self._opencv = OpenCVBackend()
            return self._opencv
        if self._lama is None:
            self._lama = LaMaBackend(device=self.device_name, model_path=self.model_path)
        return self._lama

    @property
    def backend_label(self) -> str:
        return self._get_backend().label

    def ensure_loaded(self) -> Tuple[bool, str]:
        """确保后端可用；LaMa 失败时给出明确原因。"""
        backend = self._get_backend()
        ok = backend.ensure_loaded()
        reason = ""
        if not ok:
            reason = getattr(backend, "load_error", "") or "模型加载失败"
        return ok, reason

    def load_ms(self) -> float:
        backend = self._get_backend()
        return float(getattr(backend, "load_seconds", 0.0)) * 1000.0

    # ---------------- 主流程 ----------------
    def inpaint(
        self,
        image_rgb: np.ndarray,
        mask: np.ndarray,
        dilate: int = DEFAULT_MASK_DILATE,
        feather: int = DEFAULT_FEATHER,
        color_match: float = 0.5,
        sharpen: float = 0.0,
        denoise: int = 0,
        seamless: bool = False,
        seamless_mode: str = "mixed",
        max_side: int = 0,
        fallback: bool = True,
        grain: bool = True,
        legacy_blend: bool = False,
        return_raw: bool = False,
        roi: Optional[Tuple[int, int, int, int]] = None,
        grain_exclude: Optional[np.ndarray] = None,
        tile_components: bool = False,
        max_tiles: int = 160,
        tile_group_px: int = 6,
        repair_mask: Optional[np.ndarray] = None,
        grain_strength: float = 0.8,
        hole_grow: int = 0,
    ) -> InpaintOutcome:
        """执行完整修复流程：Detect -> Mask -> Inpaint -> PostProcess。

        ``roi``（x, y, w, h）不为空时，只在局部区域做**原生分辨率**重建，
        再把结果贴回去 —— 用于人脸保护：避免整图缩放导致五官变形，
        同时保证 ROI 之外的像素与原图逐字节一致。

        ``repair_mask``（第五轮）：**必须 100% 用模型输出替换**的水印本体区域；
        ``mask`` 则是包含过渡带的融合外边界。分离后，半透明水印的边缘不会再
        落在"保留原图"的过渡带里，从而消除边缘残留。

        ``hole_grow``（第六轮）：**只把"送进模型的孔洞"扩大** ``hole_grow`` 像素，
        写回范围（mask / repair_mask）保持不变。
        为什么需要：文字笔画掩膜通常很窄，LaMa 在"零余量孔洞"里会顺着孔洞两侧的
        像素把原文字抄回来（实测残留 21%，红字会变成金黄色块）；把模型孔洞扩开
        几像素后残留降到 0.2%，而且**不会**扩大实际修改的区域。
        """
        img = mp.to_uint8(image_rgb)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        mask_l = mp.clean_mask(mp.ensure_binary(mask, img.shape[:2]), min_area=6, close_px=1)
        if mp.is_empty(mask_l):
            raise ValueError("Mask 为空：请先用画笔/矩形/自动检测标出需要修复的区域")

        core_l: Optional[np.ndarray] = None
        if repair_mask is not None and not mp.is_empty(repair_mask):
            core_l = mp.clean_mask(mp.ensure_binary(repair_mask, img.shape[:2]), min_area=4, close_px=0)
            core_l = cv2.bitwise_and(core_l, mask_l)
            if mp.is_empty(core_l):
                core_l = None

        # ---------------- 分块局部重建（密集/平铺水印 + 人脸保护的关键） ----------------
        if tile_components and roi is None:
            tiled = self._inpaint_tiles(
                img, mask_l, core_l, dilate=dilate, feather=feather, color_match=color_match,
                sharpen=sharpen, denoise=denoise, seamless=seamless, seamless_mode=seamless_mode,
                max_side=max_side, fallback=fallback, grain=grain,
                legacy_blend=legacy_blend, return_raw=return_raw, grain_exclude=grain_exclude,
                max_tiles=max_tiles, group_px=tile_group_px, grain_strength=grain_strength,
                hole_grow=hole_grow,
            )
            if tiled is not None:
                return tiled

        if roi is None:
            return self._inpaint_core(
                img, mask_l, dilate=dilate, feather=feather, color_match=color_match,
                sharpen=sharpen, denoise=denoise, seamless=seamless, seamless_mode=seamless_mode,
                max_side=max_side, fallback=fallback, grain=grain,
                legacy_blend=legacy_blend, return_raw=return_raw, grain_exclude=grain_exclude,
                core_mask=core_l, grain_strength=grain_strength, hole_grow=hole_grow,
            )

        # ---------------- ROI 局部高保真重建 ----------------
        h, w = img.shape[:2]
        x, y, rw, rh = [int(v) for v in roi]
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(w, x + rw), min(h, y + rh)
        if x1 - x0 < 16 or y1 - y0 < 16:
            return self._inpaint_core(
                img, mask_l, dilate=dilate, feather=feather, color_match=color_match,
                sharpen=sharpen, denoise=denoise, seamless=seamless, seamless_mode=seamless_mode,
                max_side=max_side, fallback=fallback, grain=grain,
                legacy_blend=legacy_blend, return_raw=return_raw, grain_exclude=grain_exclude,
                core_mask=core_l, grain_strength=grain_strength, hole_grow=hole_grow,
            )

        crop = np.ascontiguousarray(img[y0:y1, x0:x1])
        crop_mask = np.ascontiguousarray(mask_l[y0:y1, x0:x1])
        if mp.is_empty(crop_mask):
            return self._inpaint_core(
                img, mask_l, dilate=dilate, feather=feather, color_match=color_match,
                sharpen=sharpen, denoise=denoise, seamless=seamless, seamless_mode=seamless_mode,
                max_side=max_side, fallback=fallback, grain=grain,
                legacy_blend=legacy_blend, return_raw=return_raw, grain_exclude=grain_exclude,
                core_mask=core_l, grain_strength=grain_strength, hole_grow=hole_grow,
            )

        crop_excl = None
        if grain_exclude is not None:
            crop_excl = mp.ensure_binary(grain_exclude, (h, w))[y0:y1, x0:x1]
        crop_core = core_l[y0:y1, x0:x1] if core_l is not None else None

        # ROI 内按原生分辨率推理（LaMa 对局部区域更敏感，能保住五官细节）
        inner = self._inpaint_core(
            crop, crop_mask, dilate=dilate, feather=feather, color_match=color_match,
            sharpen=sharpen, denoise=denoise, seamless=seamless, seamless_mode=seamless_mode,
            max_side=0, fallback=fallback, grain=grain,
            legacy_blend=legacy_blend, return_raw=return_raw, grain_exclude=crop_excl,
            core_mask=crop_core, grain_strength=grain_strength, hole_grow=hole_grow,
        )

        # 贴回去：只在"掩膜 + 少量余量"范围内替换，ROI 之外保持原样
        paste = mp.dilate_mask(inner.mask_used, max(2, int(dilate) + int(feather) + 2))
        full_result = img.copy()
        region = full_result[y0:y1, x0:x1]
        sel = paste > 0
        region[sel] = inner.image[sel]

        full_raw = None
        if return_raw and inner.raw_image is not None:
            full_raw = img.copy()
            raw_region = full_raw[y0:y1, x0:x1]
            raw_region[sel] = inner.raw_image[sel]

        full_mask = np.zeros((h, w), np.uint8)
        full_mask[y0:y1, x0:x1] = inner.mask_used
        info = dict(inner.info)
        info["roi"] = [x0, y0, x1 - x0, y1 - y0]
        info["roi_mode"] = "局部原生分辨率重建（人脸保护）"
        info["resolution"] = f"{w}×{h}"
        return InpaintOutcome(full_result, full_mask, inner.backend, inner.backend_label,
                              inner.seconds, info, full_raw)

    def _inpaint_tiles(
        self, image: np.ndarray, mask: np.ndarray, core: Optional[np.ndarray], *, dilate: int, feather: int,
        color_match: float, sharpen: float, denoise: int, seamless: bool, seamless_mode: str,
        max_side: int, fallback: bool, grain: bool, legacy_blend: bool, return_raw: bool,
        grain_exclude: Optional[np.ndarray], max_tiles: int, group_px: int, grain_strength: float,
        hole_grow: int = 0,
    ) -> Optional[InpaintOutcome]:
        """把 Mask 拆成若干小块，**只生成原始修复内容**，最后统一后处理一次。

        为什么需要它：
        * 平铺/密集水印会让整块 Mask 覆盖图片的很大比例，LaMa 在这种"大面积重画"下
          会改写本来正常的内容（人脸会被重画成另一个样子）；
        * 拆成小块后，每块只让 LaMa 看局部上下文（皮肤/布料纹理），
          既保住原有结构，又真正做到"最小修改"。

        第五轮改进：tile **不再各自做 tone/grain/融合**（那会产生块状色差），
        而是只把每一块的 LaMa 原始输出拼回原图坐标系，最后**全局做一次**
        颜色匹配 / 高频匹配 / 边缘融合。
        """
        m = mp.ensure_binary(mask)
        # 先把距离很近的碎片合并成组，减少推理次数
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(group_px) + 1,) * 2)
        grouped = cv2.dilate(m, kernel)
        n, labels, stats, _ = cv2.connectedComponentsWithStats((grouped > 0).astype(np.uint8), 8)
        comps = [i for i in range(1, n) if int(stats[i, cv2.CC_STAT_AREA]) > 0]
        if not comps or len(comps) > int(max_tiles):
            return None            # 碎片太多 → 交回整块流程，避免推理次数失控

        h, w = image.shape[:2]
        full_mask = np.zeros((h, w), np.uint8)
        gen_full = image.copy()                     # 未覆盖区域 = 原图
        # 统一准备后端（只做原始推理，不做后处理）
        backend = self._get_backend()
        notes: List[str] = []
        if not backend.ensure_loaded():
            reason = getattr(backend, "load_error", "") or "未知原因"
            if self.backend_name == "lama" and fallback:
                notes.append(f"LaMa 不可用（{reason}），已自动降级为 OpenCV Telea 传统算法")
                backend = self._get_backend_opencv()
            else:
                raise RuntimeError(f"{backend.label} 不可用：{reason}")

        total_seconds = 0.0

        for i in comps:
            comp = (labels == i) & (m > 0)
            if not np.any(comp):
                continue
            comp_mask = np.zeros((h, w), np.uint8)
            comp_mask[comp] = 255
            x = int(stats[i, cv2.CC_STAT_LEFT]); y = int(stats[i, cv2.CC_STAT_TOP])
            bw = int(stats[i, cv2.CC_STAT_WIDTH]); bh = int(stats[i, cv2.CC_STAT_HEIGHT])
            pad = int(min(max(24, max(bw, bh) * 0.9), 120))     # 控制 ROI 大小，避免显存/耗时失控
            x0, y0 = max(0, x - pad), max(0, y - pad)
            x1, y1 = min(w, x + bw + pad), min(h, y + bh + pad)
            if x1 - x0 < 12 or y1 - y0 < 12:
                continue
            # 单块 ROI 过大时仍然按上限缩放处理（否则 4K 级别会显存溢出）
            tile_max_side = 0 if max(x1 - x0, y1 - y0) <= 1400 else (int(max_side) if max_side else 1600)
            crop = np.ascontiguousarray(image[y0:y1, x0:x1])
            crop_mask = mp.dilate_mask(np.ascontiguousarray(comp_mask[y0:y1, x0:x1]), dilate) if dilate else \
                np.ascontiguousarray(comp_mask[y0:y1, x0:x1])
            t0 = time.time()
            work, scale = ip.resize_max_side(crop, tile_max_side) if tile_max_side else (crop, 1.0)
            work_mask = cv2.resize(crop_mask, (work.shape[1], work.shape[0]),
                                   interpolation=cv2.INTER_NEAREST) if scale != 1.0 else crop_mask
            model_mask = (mp.dilate_mask(work_mask, int(hole_grow))
                          if int(hole_grow) > 0 else work_mask)
            gen = backend.infer(work, model_mask)
            if scale != 1.0:
                gen = cv2.resize(gen, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_LANCZOS4)
            total_seconds += time.time() - t0
            sel = crop_mask > 0
            region = gen_full[y0:y1, x0:x1]
            region[sel] = gen[sel]
            full_mask[y0:y1, x0:x1] = np.maximum(full_mask[y0:y1, x0:x1], crop_mask)

        if not np.any(full_mask):
            return None

        # ---- 全局统一后处理（只做一次：颜色匹配 → 高频匹配 → 边缘融合）----
        t_post = time.time()
        result, post_info = ip.finalize_result(
            image, gen_full, full_mask,
            feather=feather, color_match=color_match, sharpen=sharpen, denoise=denoise,
            seamless=seamless, seamless_mode=seamless_mode, grain=grain,
            legacy=bool(legacy_blend), grain_exclude=grain_exclude,
            core_mask=core, grain_strength=grain_strength,
        )
        post_seconds = time.time() - t_post
        info: Dict[str, object] = {
            "backend": self.backend_name if backend is self._get_backend() else "opencv",
            "backend_label": backend.label,
            "device": self.device_name,
            "mask_pixels": int(np.count_nonzero(full_mask)),
            "mask_ratio": round(mp.area_ratio(full_mask), 4),
            "dilate": int(dilate),
            "tiled_repair": len(comps),
            "resolution": f"{w}×{h}",
            "post": post_info,
            "notes": notes + [f"分块局部重建：{len(comps)} 块只生成原始内容，随后**全局统一**做一次颜色/高频/融合处理"],
            "timings": {"infer_ms": round(total_seconds * 1000.0, 1),
                        "post_ms": round(post_seconds * 1000.0, 1)},
        }
        raw_full = gen_full if return_raw else None
        return InpaintOutcome(result, full_mask, "tile-" + info["backend"], backend.label, total_seconds, info, raw_full)

    def _inpaint_core(
        self,
        image_rgb: np.ndarray,
        mask: np.ndarray,
        dilate: int = DEFAULT_MASK_DILATE,
        feather: int = DEFAULT_FEATHER,
        color_match: float = 0.5,
        sharpen: float = 0.0,
        denoise: int = 0,
        seamless: bool = False,
        seamless_mode: str = "mixed",
        max_side: int = 0,
        fallback: bool = True,
        grain: bool = True,
        legacy_blend: bool = False,
        return_raw: bool = False,
        grain_exclude: Optional[np.ndarray] = None,
        core_mask: Optional[np.ndarray] = None,
        grain_strength: float = 0.8,
        hole_grow: int = 0,
    ) -> InpaintOutcome:
        """在给定图像上执行 Detect -> Inpaint -> PostProcess（不做 ROI 处理）。"""
        t0 = time.time()
        img = mp.to_uint8(image_rgb)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        original = img.copy()

        mask_l = mp.clean_mask(mp.ensure_binary(mask, original.shape[:2]), min_area=6, close_px=1)
        if mp.is_empty(mask_l):
            raise ValueError("Mask 为空：请先用画笔/矩形/自动检测标出需要修复的区域")

        work_img, scale = ip.resize_max_side(original, max_side) if max_side else (original, 1.0)
        if scale != 1.0:
            work_mask = cv2.resize(mask_l, (work_img.shape[1], work_img.shape[0]), interpolation=cv2.INTER_NEAREST)
        else:
            work_mask = mask_l

        if dilate:
            work_mask = mp.dilate_mask(work_mask, dilate)

        core_l = None
        if core_mask is not None and not mp.is_empty(core_mask):
            core_l = cv2.bitwise_and(mp.ensure_binary(core_mask, original.shape[:2]), mask_l)
            if mp.is_empty(core_l):
                core_l = None

        backend = self._get_backend()
        ok = backend.ensure_loaded()
        used_backend = self.backend_name
        used_label = backend.label
        notes: List[str] = []

        if not ok:
            reason = getattr(backend, "load_error", "") or "未知原因"
            if self.backend_name == "lama" and fallback:
                notes.append(f"LaMa 不可用（{reason}），已自动降级为 OpenCV Telea 传统算法")
                backend = self._get_backend_opencv()
                used_backend, used_label = "opencv", backend.label
            else:
                raise RuntimeError(f"{backend.label} 不可用：{reason}")

        t_infer = time.time()
        # 送给模型的孔洞可以再扩几像素（写回范围不变）：避免 LaMa 在"零余量孔洞"里抄回文字
        model_mask = mp.dilate_mask(work_mask, int(hole_grow)) if int(hole_grow) > 0 else work_mask
        gen = backend.infer(work_img, model_mask)
        infer_seconds = time.time() - t_infer

        if scale != 1.0:
            gen = cv2.resize(gen, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_LANCZOS4)
            full_mask = mask_l if not dilate else mp.dilate_mask(mask_l, dilate)
        else:
            full_mask = work_mask
        full_core = core_l
        if core_l is not None and scale != 1.0:
            full_core = cv2.resize(core_l, (work_img.shape[1], work_img.shape[0]), interpolation=cv2.INTER_NEAREST)
        if full_core is not None and dilate:
            full_core = mp.dilate_mask(full_core, dilate)

        t_post = time.time()
        result, post_info = ip.finalize_result(
            original,
            gen,
            full_mask,
            feather=feather,
            color_match=color_match,
            sharpen=sharpen,
            denoise=denoise,
            seamless=seamless,
            seamless_mode=seamless_mode,
            grain=bool(grain),
            legacy=bool(legacy_blend),
            grain_exclude=grain_exclude,
            core_mask=full_core,
            grain_strength=grain_strength,
        )
        post_seconds = time.time() - t_post

        clear_cuda_cache()
        elapsed = time.time() - t0
        info = {
            "backend": used_backend,
            "backend_label": used_label,
            "device": self.device_name,
            "mask_pixels": int(np.count_nonzero(full_mask)),
            "mask_ratio": round(mp.area_ratio(full_mask), 4),
            "dilate": int(dilate),
            "scaled": scale != 1.0,
            "scale": round(scale, 4),
            "resolution": f"{original.shape[1]}×{original.shape[0]}",
            "post": post_info,
            "notes": notes,
            # 分阶段耗时（第六轮性能统计：LaMa 推理 / 后处理）
            "timings": {"infer_ms": round(infer_seconds * 1000.0, 1),
                        "post_ms": round(post_seconds * 1000.0, 1)},
        }
        raw = gen if return_raw else None
        return InpaintOutcome(result, full_mask, used_backend, used_label, elapsed, info, raw)

    def _get_backend_opencv(self):
        if self._opencv is None:
            self._opencv = OpenCVBackend()
        return self._opencv


_ENGINE_CACHE: Dict[Tuple[str, str, str], InpaintEngine] = {}


def get_engine(backend: str = "lama", device: Optional[str] = None, model_path: Optional[Path] = None) -> InpaintEngine:
    """按 (后端, 设备, 权重路径) 缓存引擎实例，避免重复加载模型。"""
    dev = device or detect_device().device
    key = (backend.lower(), dev, str(model_path or ""))
    if key not in _ENGINE_CACHE:
        _ENGINE_CACHE[key] = InpaintEngine(backend=backend, device=dev, model_path=model_path)
    return _ENGINE_CACHE[key]
