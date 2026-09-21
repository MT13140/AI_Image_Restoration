"""图像读写与后处理模块。

包含：图片读取（含 EXIF 方向纠正）、分辨率控制、结果融合（羽化 /
SeamlessClone）、局部色彩匹配、锐化与降噪、对比图生成、结果保存。
"""

from __future__ import annotations

import io
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from . import mask_processor as mp
from .error_handler import AppError, check_image_size, log_exception
from config import FONT_CANDIDATES, OUTPUTS_DIR, TEMP_DIR, ensure_dirs

#: 支持的图片扩展名（用于提示语与上传校验）
SUPPORTED_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")

#: 默认允许的最大像素数（约 6000 万，超过则提示过大）
MAX_PIXELS = 60_000_000


# --------------------------------------------------------------------------
# 读写
# --------------------------------------------------------------------------
def _array_from_numpy(arr: np.ndarray) -> np.ndarray:
    """numpy 数组 -> RGB uint8（自动识别灰度 / RGBA / float / bool）。"""
    a = np.asarray(arr)
    if a.dtype == bool:
        a = a.astype(np.uint8) * 255
    elif a.dtype == np.uint16:
        a = (a / 257.0).round().astype(np.uint8)
    elif a.dtype in (np.float32, np.float64, np.float16):
        a = a.astype(np.float32)
        if a.size and float(np.nanmax(a)) <= 1.0001:
            a = a * 255.0
        a = np.nan_to_num(a, nan=0.0, posinf=255.0, neginf=0.0)
        a = np.clip(a, 0, 255).astype(np.uint8)
    elif a.dtype != np.uint8:
        a = np.clip(a, 0, 255).astype(np.uint8)

    if a.ndim == 2:                      # 灰度
        return cv2.cvtColor(a, cv2.COLOR_GRAY2RGB)
    if a.ndim == 3 and a.shape[2] == 4:   # RGBA
        rgb = a[..., :3].astype(np.float32)
        alpha = (a[..., 3:4].astype(np.float32)) / 255.0
        # 透明区域合成到白底（避免出现黑边）
        rgb = rgb * alpha + 255.0 * (1.0 - alpha)
        return np.clip(rgb, 0, 255).astype(np.uint8)
    if a.ndim == 3 and a.shape[2] == 3:   # RGB / BGR 无法区分，统一按 RGB 处理
        return np.ascontiguousarray(a)
    if a.ndim == 3 and a.shape[2] == 1:
        return cv2.cvtColor(a[..., 0], cv2.COLOR_GRAY2RGB)
    raise AppError("图片数据格式异常，请重新上传图片。", detail=f"numpy shape={getattr(a, 'shape', None)}")


def _pil_to_rgb(im: Image.Image) -> np.ndarray:
    """PIL.Image -> RGB uint8（处理 EXIF 方向、调色板/CMYK/透明通道）。"""
    try:
        im = ImageOps.exif_transpose(im)
    except Exception:
        pass
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        rgba = im.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, rgba)
    if im.mode != "RGB":
        im = im.convert("RGB")
    return np.asarray(im).copy()


def _read_from_path(path: Any) -> Tuple[np.ndarray, str, str]:
    """从文件读取图片 -> (RGB array, 扩展名, 原始模式)。"""
    p = Path(str(path))
    if not p.exists():
        raise AppError("文件不存在或已被移动，请重新上传图片。", detail=str(p))
    if p.is_dir():
        raise AppError("这是一个文件夹，请选择图片文件。", detail=str(p))
    try:
        if p.stat().st_size <= 0:
            raise AppError("图片文件是空的（0 字节），请重新上传。", detail=str(p))
    except OSError as exc:
        raise AppError("文件无法读取（权限不足或已被占用），请换一张图片。", detail=str(exc)) from exc

    try:
        with Image.open(p) as im:
            raw_mode = im.mode
            arr = _pil_to_rgb(im)
    except AppError:
        raise
    except Exception as exc:  # PIL 的各种读取异常
        raise AppError(
            "图片读取失败，请重新上传 JPG、PNG 或 WEBP 格式的图片。",
            detail=f"{exc.__class__.__name__}: {exc}",
        ) from exc
    return arr, p.suffix.lower(), raw_mode


def _extract_source(value: Any, depth: int = 0) -> Tuple[str, Any]:
    """把各种输入类型归一为 ("array"|"pil"|"path"|"bytes", 值)。"""
    if depth > 4:
        raise AppError("图片数据格式异常，请重新上传图片。", detail="嵌套层级过深")
    if value is None:
        raise AppError("请先上传图片。")
    if isinstance(value, np.ndarray):
        return "array", value
    if isinstance(value, Image.Image):
        return "pil", value
    if isinstance(value, (str, Path)):
        return "path", value
    if isinstance(value, (bytes, bytearray)):
        return "bytes", bytes(value)
    if isinstance(value, dict):
        # Gradio FileData / ImageEditor 值 / 自定义结构
        for key in ("path", "name", "url"):
            if value.get(key):
                return "path", value[key]
        for key in ("background", "composite", "image", "value", "data"):
            if value.get(key) is not None:
                return _extract_source(value[key], depth + 1)
        raise AppError("图片数据格式异常，请重新上传图片。", detail=f"dict keys={list(value)[:8]}")
    if isinstance(value, (tuple, list)):
        if len(value) == 0:
            raise AppError("图片数据为空，请重新上传图片。")
        return _extract_source(value[0], depth + 1)
    raise AppError("图片数据格式异常，请重新上传图片。", detail=f"type={type(value).__name__}")


def normalize_image(
    source: Any,
    max_side: Optional[int] = None,
    allow_resize: bool = True,
    keep_original_size: bool = True,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """**统一图片入口**：任何来源的图片都先经过这里，输出 RGB uint8 numpy。

    支持：numpy 数组（灰度/RGB/RGBA/float/bool）、PIL.Image、文件路径、
    文件 bytes、Gradio FileData(dict)、list/tuple 包装，以及 None 的安全报错。
    自动处理 EXIF 方向、透明通道、调色板、CMYK；可选按长边缩放（保持比例）。

    返回 ``(rgb_array, info)``，info 内含尺寸、格式、是否缩放等信息。
    """
    kind, raw = _extract_source(source)
    origin_format = "未知"

    try:
        if kind == "array":
            arr = _array_from_numpy(raw)
            origin_format = "上传图像"
        elif kind == "pil":
            origin_format = getattr(raw, "format", None) or "上传图像"
            arr = _pil_to_rgb(raw)
        elif kind == "path":
            arr, ext, raw_mode = _read_from_path(raw)
            origin_format = (ext.lstrip(".") or raw_mode).upper()
        else:  # bytes
            with Image.open(io.BytesIO(raw)) as im:
                origin_format = im.format or "未知"
                arr = _pil_to_rgb(im)
    except AppError:
        raise
    except Exception as exc:  # noqa: BLE001
        log_exception(exc, context="normalize_image")
        raise AppError(
            "图片读取失败，请重新上传 JPG、PNG 或 WEBP 格式的图片。",
            detail=f"{exc.__class__.__name__}: {exc}",
        ) from exc

    if arr is None or not isinstance(arr, np.ndarray) or arr.size == 0:
        raise AppError("图片数据为空，请重新上传图片。")

    h, w = int(arr.shape[0]), int(arr.shape[1])
    check_image_size(w, h, max_pixels=MAX_PIXELS)

    info: Dict[str, object] = {
        "width": w,
        "height": h,
        "orig_width": w,
        "orig_height": h,
        "format": str(origin_format).upper(),
        "mode": "RGB",
        "source": kind,
        "resized": False,
        "scale": 1.0,
    }

    if allow_resize and max_side and max(h, w) > int(max_side):
        scale = float(max_side) / float(max(h, w))
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        arr = cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        info.update({"width": new_w, "height": new_h, "resized": True, "scale": round(scale, 5)})
    if not keep_original_size:
        info.pop("orig_width", None)
        info.pop("orig_height", None)

    return np.ascontiguousarray(arr), info


def image_info_text(info: Dict[str, object]) -> str:
    """把 normalize_image 的 info 渲染成一句便于展示的中文。"""
    if not info:
        return ""
    w = info.get("width")
    h = info.get("height")
    fmt = info.get("format", "未知")
    txt = f"{w} × {h} ｜ 格式：{fmt}"
    if info.get("resized"):
        txt += f"（原图 {info.get('orig_width')} × {info.get('orig_height')}，已按上限缩放）"
    return txt


def load_image(path) -> np.ndarray:
    """兼容旧接口：读取文件 -> RGB uint8。"""
    arr, _info = normalize_image(path)
    return arr


def save_image(path, image: np.ndarray) -> str:
    """保存图片（自动识别通道），返回绝对路径字符串。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    arr = mp.to_uint8(image)
    if arr.ndim == 3 and arr.shape[2] == 3:
        Image.fromarray(arr).save(p)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        Image.fromarray(arr).save(p)
    else:
        Image.fromarray(arr).save(p)
    return str(p.resolve())


def resize_max_side(image: np.ndarray, max_side: int) -> Tuple[np.ndarray, float]:
    """长边限制缩放，返回 (图像, 缩放系数)。"""
    if not max_side or max_side <= 0:
        return image, 1.0
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return image, 1.0
    scale = max_side / float(longest)
    new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    return cv2.resize(image, new_size, interpolation=interp), scale


def to_pil(image: np.ndarray) -> Image.Image:
    arr = mp.to_uint8(image)
    return Image.fromarray(arr)


def to_bytes(image: np.ndarray, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    to_pil(image).save(buf, format=fmt)
    return buf.getvalue()


# --------------------------------------------------------------------------
# 文本标注（中文字体）
# --------------------------------------------------------------------------
_FONT_CACHE: Dict[int, ImageFont.FreeTypeFont] = {}


def _get_font(size: int = 22) -> ImageFont.ImageFont:
    size = max(10, int(size))
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    font: ImageFont.ImageFont
    for cand in FONT_CANDIDATES:
        try:
            font = ImageFont.truetype(cand, size)
            _FONT_CACHE[size] = font
            return font
        except Exception:
            continue
    font = ImageFont.load_default()
    _FONT_CACHE[size] = font
    return font


def draw_label(image: np.ndarray, text: str, position: Tuple[int, int] = (12, 12), size: Optional[int] = None) -> np.ndarray:
    """在图片左上/指定位置绘制带底色的文字标注（支持中文）。"""
    img = to_pil(image).convert("RGB")
    draw = ImageDraw.Draw(img)
    font_size = size or max(14, int(min(img.size) * 0.035))
    font = _get_font(font_size)
    x, y = position
    bbox = draw.textbbox((x, y), text, font=font)
    pad = max(4, font_size // 5)
    draw.rectangle([bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad], fill=(0, 0, 0))
    draw.text((x, y), text, font=font, fill=(255, 255, 255))
    return np.asarray(img)


# --------------------------------------------------------------------------
# 融合与后处理
# --------------------------------------------------------------------------
def blend_with_mask(
    original: np.ndarray,
    generated: np.ndarray,
    mask: np.ndarray,
    feather: int = 6,
) -> np.ndarray:
    """用羽化权重把修复结果融合回原图（核心融合步骤）。"""
    orig = mp.to_uint8(original).astype(np.float32)
    gen = mp.to_uint8(generated).astype(np.float32)
    if gen.shape[:2] != orig.shape[:2]:
        gen = cv2.resize(gen, (orig.shape[1], orig.shape[0]), interpolation=cv2.INTER_LANCZOS4)
    w = mp.feather_mask(mask, feather)[..., None]
    out = gen * w + orig * (1.0 - w)
    return np.clip(out, 0, 255).astype(np.uint8)


def seamless_blend(
    original: np.ndarray,
    generated: np.ndarray,
    mask: np.ndarray,
    mode: str = "mixed",
) -> Optional[np.ndarray]:
    """OpenCV SeamlessClone 融合，缓解修复区域与周围的接缝。失败返回 None。"""
    orig = mp.to_uint8(original)
    gen = mp.to_uint8(generated)
    if gen.shape[:2] != orig.shape[:2]:
        gen = cv2.resize(gen, (orig.shape[1], orig.shape[0]), interpolation=cv2.INTER_LANCZOS4)
    m = mp.ensure_binary(mask, orig.shape[:2])
    if mp.is_empty(m):
        return None
    ys, xs = np.nonzero(m)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    center = ((x0 + x1) // 2, (y0 + y1) // 2)
    flag = cv2.MIXED_CLONE if mode == "mixed" else cv2.NORMAL_CLONE
    try:
        return cv2.seamlessClone(gen, orig, m, center, flag)
    except Exception:
        return None


def match_region_color(
    reference: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    strength: float = 0.6,
) -> np.ndarray:
    """把修复区域的均值/标准差向周围参考区靠拢，减少明暗与色偏。

    reference: 融合前的原图；target: 模型输出；mask: 修复区域。
    """
    ref = mp.to_uint8(reference).astype(np.float32)
    tgt = mp.to_uint8(target).astype(np.float32)
    m = mp.ensure_binary(mask, ref.shape[:2])
    if mp.is_empty(m):
        return mp.to_uint8(target)
    k = max(9, int(0.05 * min(ref.shape[:2])) | 1)
    ring = mp.dilate_mask(m, max(6, k // 2)) > 0
    ring &= ~(m > 0)  # 只取掩膜外的邻环作为参考
    if ring.sum() < 50:
        return mp.to_uint8(target)

    out = tgt.copy()
    strength = float(np.clip(strength, 0.0, 1.0))
    for c in range(3):
        tgt_pixels = tgt[m > 0, c]
        ref_pixels = ref[ring, c]
        if tgt_pixels.size == 0 or ref_pixels.size == 0:
            continue
        t_mu, t_sd = float(tgt_pixels.mean()), float(tgt_pixels.std() + 1e-6)
        r_mu, r_sd = float(ref_pixels.mean()), float(ref_pixels.std() + 1e-6)
        scale = float(np.clip(r_sd / t_sd, 0.6, 1.6))
        adjusted = (out[..., c] - t_mu) * scale + t_mu
        adjusted = adjusted + (r_mu - float((adjusted[m > 0]).mean()))
        ch = out[..., c]
        ch[m > 0] = ch[m > 0] * (1 - strength) + adjusted[m > 0] * strength
        out[..., c] = ch
    return np.clip(out, 0, 255).astype(np.uint8)


def unsharp_in_region(
    image: np.ndarray,
    mask: np.ndarray,
    amount: float = 0.6,
    radius: int = 5,
) -> np.ndarray:
    """在修复区域做非锐化掩膜，恢复细节观感。"""
    if amount <= 0:
        return mp.to_uint8(image)
    img = mp.to_uint8(image)
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=max(1, radius / 3.0))
    sharp = cv2.addWeighted(img, 1 + amount, blur, -amount, 0)
    m = mp.ensure_binary(mask, img.shape[:2])
    out = img.copy()
    sel = m > 0
    out[sel] = sharp[sel]
    return out


def denoise_in_region(image: np.ndarray, mask: np.ndarray, strength: int = 6) -> np.ndarray:
    """在修复区域做双边滤波降噪（保边）。"""
    if strength <= 0:
        return mp.to_uint8(image)
    img = mp.to_uint8(image)
    filtered = cv2.bilateralFilter(img, d=5, sigmaColor=strength * 4, sigmaSpace=strength)
    m = mp.ensure_binary(mask, img.shape[:2])
    out = img.copy()
    sel = m > 0
    out[sel] = filtered[sel]
    return out


def _mask_scale(mask: np.ndarray, fallback: int = 20) -> float:
    """用 Mask 尺寸估计低频校正的空间尺度（取最大连通域的短边一半）。"""
    boxes = mp.mask_to_boxes(mask, min_area=16)
    if not boxes:
        return float(fallback)
    _x, _y, w, h = max(boxes, key=lambda b: b[2] * b[3])
    return float(max(6, min(w, h) // 2))


def match_local_tone(
    original: np.ndarray,
    generated: np.ndarray,
    mask: np.ndarray,
    strength: float = 0.8,
    sigma: Optional[float] = None,
    decay_width: Optional[float] = None,
) -> np.ndarray:
    """局部光照/色彩校正：把 Mask 外围的真实光照（低频）延续进 Mask 内部。

    做法（避免"补丁感"的关键）：
      1. 对原图做低通，得到"光照/色彩场"；
      2. **只用 Mask 外部像素**做归一化卷积（``blur(x*w)/blur(w)``），
         向 Mask 内部插值出一条平滑的"期望光照场"——因此不会把水印本身的亮度带进来；
      3. 计算模型输出自身的低频场，两者之差就是需要补的"光照偏移"；
      4. 只把这个**低频偏移**加到模型输出上，模型生成的纹理/细节完全保留。

    这样修复区就会自然延续周围的明暗渐变，不再出现整块偏亮/偏暗的阴影补丁。
    """
    orig = mp.to_uint8(original).astype(np.float32)
    gen = mp.to_uint8(generated).astype(np.float32)
    if gen.shape[:2] != orig.shape[:2]:
        gen = cv2.resize(gen, (orig.shape[1], orig.shape[0]), interpolation=cv2.INTER_LANCZOS4)
    m = mp.ensure_binary(mask, orig.shape[:2])
    if mp.is_empty(m):
        return mp.to_uint8(generated)

    sigma = float(sigma) if sigma else max(3.0, _mask_scale(m) * 0.45)
    valid = (m == 0).astype(np.float32)[..., None]

    low_orig = cv2.GaussianBlur(orig, (0, 0), sigma)
    num_o = cv2.GaussianBlur(low_orig * valid, (0, 0), sigma)
    den_o = cv2.GaussianBlur(valid[..., 0], (0, 0), sigma)[..., None]
    target_low = num_o / np.maximum(den_o, 1e-3)      # 掩膜内的"期望光照场"

    low_gen = cv2.GaussianBlur(gen, (0, 0), sigma)
    delta = (target_low - low_gen) * float(np.clip(strength, 0.0, 1.0))

    # 关键：只在"接缝附近"施加校正，越往 Mask 内部越弱（到中心为 0），
    # 这样只修正边缘处的明暗/色彩不连续，而不会把整块修复区改色（避免新的色块）。
    if decay_width and decay_width > 0:
        dist = cv2.distanceTransform((m > 0).astype(np.uint8), cv2.DIST_L2, 3)
        t = np.clip(dist / float(decay_width), 0.0, 1.0)
        weight = 1.0 - (t * t * (3.0 - 2.0 * t))       # 边界 1 → 内部 0
        delta = delta * weight[..., None]

    out = gen + delta
    return np.clip(out, 0, 255).astype(np.uint8)


def match_local_grain(
    original: np.ndarray,
    generated: np.ndarray,
    mask: np.ndarray,
    strength: float = 0.8,
    sigma: float = 1.6,
    seed: int = 1234,
    exclude_mask: Optional[np.ndarray] = None,
    mode: str = "transfer",
) -> np.ndarray:
    """局部高频/颗粒**统计匹配**：让修复区的细节能量与周围真实区域一致。

    第五轮改进（对应"皮肤塑料感"）：

    * ``mode="transfer"``（默认）：先从原图提取高频层（纹理 + 传感器颗粒），
      用 ``cv2.inpaint`` 把**周围真实高频**平滑延续进掩膜内部（不生成新结构、
      不制造明显噪点），再按能量比例补到修复区上；
      如果修复区比周围更锐，则按比例**轻微衰减**，避免"补丁过锐/磨皮"。
    * ``mode="noise"``：旧行为（叠加与周围等强度的随机噪声），保留兼容。
    """
    orig = mp.to_uint8(original).astype(np.float32)
    gen = mp.to_uint8(generated).astype(np.float32)
    if gen.shape[:2] != orig.shape[:2]:
        gen = cv2.resize(gen, (orig.shape[1], orig.shape[0]), interpolation=cv2.INTER_LANCZOS4)
    m = mp.ensure_binary(mask, orig.shape[:2])
    if mp.is_empty(m):
        return mp.to_uint8(generated)

    ring = cv2.dilate(m, np.ones((9, 9), np.uint8)) - m
    ring_bool = ring > 0
    if int(ring_bool.sum()) < 50:
        ring_bool = (cv2.dilate(m, np.ones((25, 25), np.uint8)) > 0) & (m == 0)
    if int(ring_bool.sum()) < 20:
        return mp.to_uint8(generated)

    def highpass(img: np.ndarray) -> np.ndarray:
        return img - cv2.GaussianBlur(img, (0, 0), sigma)

    hf_ring = highpass(orig)
    hf_in = highpass(gen)
    sel = m > 0
    sd_ring = float(np.mean([hf_ring[..., c][ring_bool].std() for c in range(3)]))
    sd_in = float(np.mean([hf_in[..., c][sel].std() for c in range(3)]))
    if not np.isfinite(sd_ring) or not np.isfinite(sd_in):
        return mp.to_uint8(generated)

    if mode == "noise":
        if sd_ring <= sd_in * 0.95:
            return mp.to_uint8(generated)
        need = float(np.sqrt(max(0.0, sd_ring ** 2 - sd_in ** 2)) * np.clip(strength, 0.0, 2.0))
        rng = np.random.default_rng(seed)
        noise = rng.normal(0.0, need, size=(m.shape[0], m.shape[1], 3)).astype(np.float32)
        if exclude_mask is not None and not mp.is_empty(exclude_mask):
            noise[mp.ensure_binary(exclude_mask, m.shape) > 0] = 0.0
        out = gen.copy()
        out[sel] = np.clip(out[sel] + noise[sel], 0, 255)
        return out.astype(np.uint8)

    # ---------- transfer：把周围真实高频延续进掩膜，再按能量比例匹配 ----------
    donor = np.zeros_like(hf_ring)
    for c in range(3):
        layer = hf_ring[..., c]
        enc = np.clip(layer * 8.0 + 128.0, 0, 255).astype(np.uint8)      # 放大后编码，避免量化损失
        filled = cv2.inpaint(enc, m, 3, cv2.INPAINT_TELEA).astype(np.float32)
        d = (filled - 128.0) / 8.0
        # 关键：inpaint 在掩膜内会形成"斜坡"式低频分量，若直接叠加会变成色块/阴影，
        # 因此只保留其中的高频（微纹理/颗粒），并把掩膜内的均值归零。
        d = d - cv2.GaussianBlur(d, (0, 0), max(2.0, sigma * 6.0))
        if np.any(sel):
            d = d - float(d[sel].mean())
        donor[..., c] = d
    sd_donor = float(np.mean([donor[..., c][sel].std() for c in range(3)]))
    if sd_donor <= 1e-6:
        return mp.to_uint8(generated)

    out = gen.copy()
    target = sd_ring
    if sd_in < target:                       # 偏平滑 → 补上真实高频
        k = float(np.sqrt(max(0.0, target ** 2 - sd_in ** 2)) / sd_donor)
        k = float(np.clip(k, 0.0, 1.2)) * float(np.clip(strength, 0.0, 1.5))
        out[sel] = np.clip(out[sel] + (donor * k)[sel], 0, 255)
    elif sd_in > target * 1.15:              # 偏锐/有噪 → 轻微衰减到接近周围
        k = float(np.clip((sd_in - target) / max(1e-6, sd_in), 0.0, 0.4)) * float(np.clip(strength, 0.0, 1.5))
        out[sel] = np.clip(out[sel] - (hf_in * k)[sel], 0, 255)

    if exclude_mask is not None and not mp.is_empty(exclude_mask):
        # 面孔/皮肤区域：只保留"较弱"的纹理恢复（不叠加到塑料感方向）
        keep_out = mp.ensure_binary(exclude_mask, m.shape) > 0
        out[keep_out] = np.clip(gen[keep_out] + (out[keep_out] - gen[keep_out]) * 0.5, 0, 255)
    return out.astype(np.uint8)


def soft_blend(
    original: np.ndarray,
    generated: np.ndarray,
    mask: np.ndarray,
    feather: int = 6,
    core_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """平滑羽化融合：alpha 由距离变换得到 smoothstep，**只作用于 Mask 内部**。

    * 掩膜外像素完全保持原样（不会把修复结果糊到正常区域上）；
    * 掩膜中心 100% 使用模型输出（保证水印被真正去掉，不会被羽化"混回来"）；
    * 过渡宽度会按 Mask 实际厚度自动收缩，避免细笔画被整体羽化。

    **第五轮改进（修复 Mask 与融合 Mask 分离）**

    ``core_mask``（repair_mask）表示"必须 100% 用模型输出替换"的水印本体区域。
    传入后：alpha 在 core_mask 内恒为 1，只在 core_mask 与 mask(blend) 边界之间的
    过渡带里从 1 平滑降到 0。

    这样一来，半透明水印的边缘一定落在"完全替换"区里，不会再出现
    "水印中心被抹掉、边缘还残留"的情况；而过渡带位于水印之外，是完全安全的区域。
    """
    orig = mp.to_uint8(original).astype(np.float32)
    gen = mp.to_uint8(generated).astype(np.float32)
    if gen.shape[:2] != orig.shape[:2]:
        gen = cv2.resize(gen, (orig.shape[1], orig.shape[0]), interpolation=cv2.INTER_LANCZOS4)
    m = mp.ensure_binary(mask, orig.shape[:2])
    if mp.is_empty(m):
        return mp.to_uint8(original)

    core = None
    if core_mask is not None and not mp.is_empty(core_mask):
        core = mp.ensure_binary(core_mask, orig.shape[:2])
        core = cv2.bitwise_and(core, m)

    dist = cv2.distanceTransform((m > 0).astype(np.uint8), cv2.DIST_L2, 3)
    thickness = float(dist.max()) if dist.size else 0.0
    if feather and feather > 0 and thickness > 0:
        f = min(float(feather), max(1.0, thickness * 0.8))
        alpha = np.clip(dist / f, 0.0, 1.0)
        alpha = alpha * alpha * (3.0 - 2.0 * alpha)      # smoothstep
    else:
        alpha = (m > 0).astype(np.float32)

    if core is not None:
        alpha = np.maximum(alpha, (core > 0).astype(np.float32))   # 水印本体 100% 替换

    a = alpha[..., None]
    out = gen * a + orig * (1.0 - a)
    return np.clip(out, 0, 255).astype(np.uint8)


def finalize_result(
    original: np.ndarray,
    generated: np.ndarray,
    mask: np.ndarray,
    feather: int = 6,
    color_match: float = 0.5,
    sharpen: float = 0.0,
    denoise: int = 0,
    seamless: bool = False,
    seamless_mode: str = "mixed",
    grain: bool = True,
    legacy: bool = False,
    grain_exclude: Optional[np.ndarray] = None,
    core_mask: Optional[np.ndarray] = None,
    grain_strength: float = 0.8,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """完整后处理流水线（v2：局部光照场校正 + 颗粒匹配 + 平滑羽化融合）。

    旧版（``legacy=True``）会走原来的"全局均值/方差匹配 + 高斯羽化"，
    保留它是为了做效果对比，默认不再使用。

    v2 的顺序：
        1. 用 **掩膜外** 像素估计"原图 − 模型输出"的低频差场，并向内插值
           （归一化卷积）→ 修正光照/亮度/色偏，但不引入水印本身的亮度；
        2. 颗粒匹配：把修复区的细节能量补到与周围一致，避免"过于干净"的补丁感；
        3. 平滑羽化融合：alpha 由距离变换得到 smoothstep 过渡，**只影响 Mask 内部**，
           Mask 外像素保持原样；
        4. 可选降噪 / 锐化（锐化限定在收缩后的 Mask 内，避免在接缝处制造光晕）。
    """
    t0 = time.time()
    steps: list = []
    gen = generated

    if legacy:
        if color_match > 0:
            gen = match_region_color(original, gen, mask, strength=color_match)
            steps.append(f"旧版色彩匹配({color_match:.2f})")
        blended = blend_with_mask(original, gen, mask, feather)
        steps.append(f"旧版羽化融合({feather}px)")
        if denoise > 0:
            blended = denoise_in_region(blended, mask, strength=denoise)
            steps.append(f"局部降噪({denoise})")
        if sharpen > 0:
            blended = unsharp_in_region(blended, mask, amount=sharpen)
            steps.append(f"局部锐化({sharpen:.2f})")
        return blended, {"steps": steps, "seconds": round(time.time() - t0, 3), "mode": "legacy"}

    m = mp.ensure_binary(mask, original.shape[:2])

    if color_match > 0:
        # 只在接缝附近做低频光照校正（越往里越弱），避免整块改色
        gen = match_local_tone(
            original, gen, m,
            strength=float(color_match),
            decay_width=max(10.0, float(feather) * 2.5),
        )
        steps.append(f"接缝处光照校正({color_match:.2f})")

    if grain:
        gen = match_local_grain(original, gen, m, strength=float(grain_strength),
                                exclude_mask=grain_exclude)
        steps.append(f"高频/颗粒匹配({grain_strength:.2f})")

    if denoise > 0:
        gen = denoise_in_region(gen, m, strength=denoise)
        steps.append(f"修复区降噪({denoise})")

    core = None
    if core_mask is not None and not mp.is_empty(core_mask):
        core = mp.ensure_binary(core_mask, original.shape[:2])
    blended = soft_blend(original, gen, m, feather=int(feather), core_mask=core)
    steps.append(f"平滑羽化融合({feather}px)" + ("（水印本体 100% 替换）" if core is not None else ""))

    if seamless:
        refined = seamless_blend(original, gen, m, mode=seamless_mode)
        if refined is not None:
            blended = refined
            steps.append(f"SeamlessClone({seamless_mode})")

    if sharpen > 0:
        sharp_zone = core if core is not None else m
        sharp_zone = mp.erode_mask(sharp_zone, max(1, int(feather) + 1)) if feather else sharp_zone
        if not mp.is_empty(sharp_zone):
            blended = unsharp_in_region(blended, sharp_zone, amount=sharpen)
        steps.append(f"局部锐化({sharpen:.2f})")

    return blended, {"steps": steps, "seconds": round(time.time() - t0, 3), "mode": "v2"}


# --------------------------------------------------------------------------
# 对比图
# --------------------------------------------------------------------------
def side_by_side(
    before: np.ndarray,
    after: np.ndarray,
    labels: Sequence[str] = ("原图", "修复结果"),
    gap: int = 16,
    background: Tuple[int, int, int] = (24, 24, 28),
) -> np.ndarray:
    """生成带中文标签的左右对比图。"""
    a = mp.to_uint8(before)
    b = mp.to_uint8(after)
    if a.ndim == 2:
        a = cv2.cvtColor(a, cv2.COLOR_GRAY2RGB)
    if b.ndim == 2:
        b = cv2.cvtColor(b, cv2.COLOR_GRAY2RGB)
    if b.shape[:2] != a.shape[:2]:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_LANCZOS4)

    a = draw_label(a, labels[0] if len(labels) > 0 else "")
    b = draw_label(b, labels[1] if len(labels) > 1 else "")

    h = max(a.shape[0], b.shape[0])
    w = a.shape[1] + gap + b.shape[1]
    canvas = np.zeros((h, w, 3), np.uint8)
    canvas[:] = background
    canvas[: a.shape[0], : a.shape[1]] = a
    canvas[: b.shape[0], a.shape[1] + gap : a.shape[1] + gap + b.shape[1]] = b
    return canvas


def difference_map(before: np.ndarray, after: np.ndarray, gain: float = 3.0) -> np.ndarray:
    """差异热力图，便于检查修复区域。"""
    a = mp.to_uint8(before)
    b = mp.to_uint8(after)
    if a.shape[:2] != b.shape[:2]:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_LANCZOS4)
    diff = cv2.absdiff(a, b)
    if diff.ndim == 3:
        diff = diff.max(axis=2)
    diff = np.clip(diff.astype(np.float32) * gain, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(diff, cv2.COLORMAP_JET)[..., ::-1]


# --------------------------------------------------------------------------
# 修复后自动质检（第五轮新增）
# --------------------------------------------------------------------------
def quality_check(
    original: np.ndarray,
    result: np.ndarray,
    blend_mask: np.ndarray,
    core_mask: Optional[np.ndarray] = None,
    face_mask: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """修复结果自动检查：Mask 外是否被改动 + 修复区与周围是否一致。

    返回可直接展示的指标与警告文案。判定阈值偏保守（宁可提示，不放过明显伪影）。
    """
    orig = mp.to_uint8(original)
    res = mp.to_uint8(result)
    h, w = orig.shape[:2]
    m = mp.ensure_binary(blend_mask, (h, w))
    core = mp.ensure_binary(core_mask, (h, w)) if core_mask is not None else m
    sel = (core > 0)
    ring = ((cv2.dilate(m, np.ones((17, 17), np.uint8)) > 0) & (m == 0))
    if not np.any(sel) or int(ring.sum()) < 30:
        return {"warnings": [], "available": False}

    diff = np.abs(res.astype(np.int16) - orig.astype(np.int16)).max(axis=2)
    outside = (m == 0)
    outside_ratio = float(np.mean(diff[outside] > 2)) if np.any(outside) else 0.0

    lab_o = cv2.cvtColor(orig, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab_r = cv2.cvtColor(res, cv2.COLOR_RGB2LAB).astype(np.float32)

    # 绝对判据：修复区应当"像周围的局部内容"（而不是像"带水印的原图"）
    dl = float(lab_r[..., 0][sel].mean() - lab_r[..., 0][ring].mean())
    da = float(lab_r[..., 1][sel].mean() - lab_r[..., 1][ring].mean())
    db = float(lab_r[..., 2][sel].mean() - lab_r[..., 2][ring].mean())
    dl_before = float(lab_o[..., 0][sel].mean() - lab_o[..., 0][ring].mean())
    da_before = float(lab_o[..., 1][sel].mean() - lab_o[..., 1][ring].mean())
    db_before = float(lab_o[..., 2][sel].mean() - lab_o[..., 2][ring].mean())

    def _hf(img: np.ndarray, where: np.ndarray) -> float:
        g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
        return float((g - cv2.GaussianBlur(g, (0, 0), 1.6))[where].std())

    def _lap(img: np.ndarray, where: np.ndarray) -> float:
        g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return float(cv2.Laplacian(g, cv2.CV_32F)[where].var())

    hf_in, hf_ring = _hf(res, sel), _hf(res, ring)
    hf_ratio = hf_in / max(1e-6, hf_ring)
    sharp_ratio = _lap(res, sel) / max(1e-6, _lap(res, ring))
    hf_before_ratio = _hf(orig, sel) / max(1e-6, _hf(orig, ring))

    # 水印残留检测：修复区内的"局部亮度偏移"能量（半透明水印的主要特征）
    def _overlay_energy(img: np.ndarray) -> float:
        g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        bg = cv2.medianBlur(g, 15)          # 局部背景（OpenCV 大核限制内）
        return float(np.abs(g.astype(np.float32) - bg.astype(np.float32))[sel].mean())

    resid_before, resid_after = _overlay_energy(orig), _overlay_energy(res)
    residue_cut = 1.0 - resid_after / max(1e-6, resid_before)

    face_change = None
    if face_mask is not None and not mp.is_empty(face_mask):
        fm = mp.ensure_binary(face_mask, (h, w))
        keep = (fm > 0) & (mp.dilate_mask(m, 2) == 0)
        if int(keep.sum()) > 30:
            face_change = float(diff[keep].max())

    warnings: List[str] = []
    if outside_ratio > 0.001:
        warnings.append(f"Mask 之外有 {outside_ratio * 100:.3f}% 像素被改动（正常应为 0）")
    # 亮度/色差：只有当"比原图明显更不一致"时才告警（避免把眼睛/头发等本来就不同的内容误判）
    if abs(dl) > 14.0 and abs(dl) > abs(dl_before) * 1.5 + 3.0:
        warnings.append(f"修复区与周围邻域的亮度关系明显变差（ΔL={dl:+.1f}，原图该处 {dl_before:+.1f}），"
                        f"可能出现阴影/亮块")
    if (abs(da) > 8.0 and abs(da) > abs(da_before) * 1.5 + 2.0) or \
       (abs(db) > 8.0 and abs(db) > abs(db_before) * 1.5 + 2.0):
        warnings.append(f"修复区与周围邻域出现新的色差（Δa={da:+.1f}, Δb={db:+.1f}）")
    # 注意：五官/睫毛等真实内容本身就会产生较大的局部亮度偏移，
    # 因此这里只在"几乎没被消除"时才告警（避免误报）。
    if resid_before > 5.0 and residue_cut < 0.25:
        warnings.append(f"水印可能仍有残留：修复区内亮度偏移只消除了 {residue_cut * 100:.0f}%"
                        f"（建议把「Mask 扩张」调大 1~2 像素后重试）")
    if hf_ratio < 0.55:
        warnings.append(f"修复区纹理明显少于周围（高频比 {hf_ratio:.2f}，原图该处 {hf_before_ratio:.2f}），可能有涂抹感")
    if hf_ratio > 1.9:
        warnings.append(f"修复区比周围更多噪（高频比 {hf_ratio:.2f}），可能出现颗粒/补丁感")
    if sharp_ratio < 0.50:
        warnings.append(f"修复区比周围明显更糊（清晰度比 {sharp_ratio:.2f}）")
    if sharp_ratio > 2.0:
        warnings.append(f"修复区比周围明显更锐（清晰度比 {sharp_ratio:.2f}），可能有补丁感")
    if face_change is not None and face_change > 2:
        warnings.append(f"人脸区域（Mask 之外）被改动了（最大差 {face_change:.0f}）")

    return {
        "available": True,
        "outside_change_ratio": round(outside_ratio, 6),
        "edge_luma_delta": round(dl, 2),
        "edge_luma_delta_before": round(dl_before, 2),
        "edge_chroma_delta": [round(da, 2), round(db, 2)],
        "edge_chroma_delta_before": [round(da_before, 2), round(db_before, 2)],
        "hf_ratio": round(hf_ratio, 3),
        "hf_ratio_before": round(hf_before_ratio, 3),
        "sharp_ratio": round(sharp_ratio, 3),
        "residue_cut": round(residue_cut, 3),
        "residue_before": round(resid_before, 2),
        "residue_after": round(resid_after, 2),
        "face_outside_change": face_change,
        "warnings": warnings,
    }


# --------------------------------------------------------------------------
# 输出
# --------------------------------------------------------------------------
def make_output_path(prefix: str = "restored", ext: str = ".png") -> Path:
    ensure_dirs()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return OUTPUTS_DIR / f"{prefix}_{stamp}{ext}"


def make_temp_path(prefix: str = "tmp", ext: str = ".png") -> Path:
    ensure_dirs()
    # 注意：``%f``（微秒）只有 datetime.strftime 支持，time.strftime 会抛
    # "Invalid format string"（早期版本这里踩过坑，导致"生成下载副本失败"）。
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    return TEMP_DIR / f"{prefix}_{stamp}{ext}"


def image_stats(image: np.ndarray) -> Dict[str, object]:
    """基础统计信息，用于 UI 展示。"""
    arr = mp.to_uint8(image)
    h, w = arr.shape[:2]
    mean = float(arr.mean()) if arr.size else 0.0
    return {"width": int(w), "height": int(h), "channels": int(arr.shape[2]) if arr.ndim == 3 else 1, "mean": round(mean, 2)}
