"""人脸保护（Face-aware Inpainting）模块。

目标：**去除水印不能以破坏人脸为代价**。

策略：
1. 检测人脸（优先 OpenCV 官方 YuNet，带 5 点关键点；不可用时退回 OpenCV 自带 Haar 级联，
   Haar 完全离线、无额外依赖）；
2. 由关键点构造"五官核心区"（双眼/眉毛/鼻子/嘴巴）与"面部区"掩膜；
3. 判断用户涂抹区域与人脸的接触情况，分三种模式：
   * ``normal``        —— 水印不碰脸：按常规流程处理；
   * ``face_edge``     —— 只碰到脸部边缘：脸部范围内的扩张收紧到最小；
   * ``face_protect``  —— 压到五官：进入保护模式，只修真正被水印覆盖的像素，
                          并用高分辨率 ROI 重建（避免整张缩放到 2560 导致五官变形）。
4. 面部区域禁用"颗粒匹配"、收紧羽化宽度，**Mask 之外的人脸像素保证不变**。

本模块只读取图像、产出掩膜与"修复计划"，不修改任何既有检测能力。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from config import MODELS_DIR, get_logger
from . import mask_processor as mp

logger = get_logger("ai_restore.face")

#: 官方 YuNet 权重（约 227KB）文件名（放在 models/ 下，缺失时自动退回 Haar）
YUNET_FILENAMES = (
    "face_detection_yunet_2023mar.onnx",
    "face_detection_yunet.onnx",
)

#: 人脸关键点在 YuNet 输出中的顺序
LM_RIGHT_EYE, LM_LEFT_EYE, LM_NOSE, LM_MOUTH_R, LM_MOUTH_L = 0, 1, 2, 3, 4


@dataclass
class FaceRegion:
    """一张人脸。"""

    box: Tuple[int, int, int, int]          # x, y, w, h（原图坐标）
    score: float = 0.0
    landmarks: Optional[np.ndarray] = None  # (5, 2) 原图坐标；None 表示只有框
    source: str = "yunet"

    @property
    def center(self) -> Tuple[float, float]:
        x, y, w, h = self.box
        return (x + w / 2.0, y + h / 2.0)


@dataclass
class FaceReport:
    """用户涂抹区域与所有人脸的关系。"""

    faces: List[FaceRegion] = field(default_factory=list)
    mode: str = "normal"                  # normal / face_edge / face_protect
    overlap_ratio: float = 0.0            # 涂抹区域落在人脸内的比例
    critical_ratio: float = 0.0           # 涂抹区域落在五官核心区的比例
    face_mask: Optional[np.ndarray] = None
    critical_mask: Optional[np.ndarray] = None
    message: str = ""


# --------------------------------------------------------------------------
# 检测
# --------------------------------------------------------------------------
def _find_yunet() -> Optional[Path]:
    for name in YUNET_FILENAMES:
        p = MODELS_DIR / name
        if p.exists() and p.stat().st_size > 100_000:   # 排除误下载的错误的文件
            return p
    return None


def face_detector_info() -> Dict[str, object]:
    """当前可用的人脸检测能力（用于界面/日志展示）。"""
    yunet = _find_yunet()
    return {
        "yunet_model": str(yunet) if yunet else "",
        "yunet_available": bool(yunet) and hasattr(cv2, "FaceDetectorYN"),
        "haar_available": os.path.exists(cv2.data.haarcascades + "haarcascade_frontalface_default.xml"),
        "landmarks": bool(yunet) and hasattr(cv2, "FaceDetectorYN"),
    }


def _detect_yunet(image_bgr: np.ndarray, model_path: Path, score_thr: float = 0.6) -> List[FaceRegion]:
    h, w = image_bgr.shape[:2]
    det = cv2.FaceDetectorYN.create(str(model_path), "", (w, h), score_thr, 0.3, 5000)
    det.setInputSize((w, h))
    _, faces = det.detect(image_bgr)
    out: List[FaceRegion] = []
    if faces is None:
        return out
    for f in faces:
        x, y, bw, bh = [float(v) for v in f[:4]]
        score = float(f[-1])
        lms = np.array([[float(f[4 + 2 * i]), float(f[5 + 2 * i])] for i in range(5)], np.float32)
        out.append(FaceRegion(
            box=(int(round(x)), int(round(y)), int(round(bw)), int(round(bh))),
            score=score, landmarks=lms, source="yunet",
        ))
    return out


def _detect_haar(image_bgr: np.ndarray) -> List[FaceRegion]:
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    if not os.path.exists(cascade_path):
        return []
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    cascade = cv2.CascadeClassifier(cascade_path)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
    return [FaceRegion(box=(int(x), int(y), int(w), int(h)), score=0.5, source="haar") for (x, y, w, h) in faces]


def _detect_yunet_scaled(
    image_rgb: np.ndarray, model_path: Path, max_side: Optional[int], score_thr: float
) -> List[FaceRegion]:
    """在指定缩放尺度上用 YuNet 检测，坐标自动映射回原图。"""
    h, w = image_rgb.shape[:2]
    scale = 1.0 if not max_side else min(1.0, float(max_side) / float(max(h, w)))
    small = (
        cv2.resize(image_rgb, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
        if scale < 1.0 else image_rgb
    )
    faces = _detect_yunet(cv2.cvtColor(small, cv2.COLOR_RGB2BGR), model_path, score_thr)
    if scale < 1.0 and faces:
        inv = 1.0 / scale
        for f in faces:
            x, y, bw, bh = f.box
            f.box = (int(round(x * inv)), int(round(y * inv)), int(round(bw * inv)), int(round(bh * inv)))
            if f.landmarks is not None:
                f.landmarks = f.landmarks.astype(np.float32) * inv
    return faces


def detect_faces(
    image_rgb: np.ndarray, max_side: int = 1280, score_thr: float = 0.6, robust: bool = True
) -> List[FaceRegion]:
    """**鲁棒**人脸检测（第五轮改进）。

    依次尝试，找到即停（不会为了检测而修改原图）：

    1. 常规：长边缩到 1280 + 阈值 0.6（快）
    2. 原始分辨率 + 同阈值（小脸/远处人脸更易检出）
    3. 低阈值 0.35（侧脸、低头、遮挡、动作模糊）
    4. 2× 上采样尺度 + 阈值 0.45（极小脸）
    5. Haar 级联兜底（YuNet 不可用或全部失败时）

    目标：尽可能让"脸上有水印"这种情况进入保护流程，而不是因为漏检而放任 AI 重绘人脸。
    """
    img = mp.to_uint8(image_rgb)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    h, w = img.shape[:2]

    yunet = _find_yunet()
    faces: List[FaceRegion] = []
    if yunet is not None and hasattr(cv2, "FaceDetectorYN"):
        attempts: List[Tuple[Optional[int], float]] = [(max_side, score_thr)]
        if robust:
            attempts += [
                (None, score_thr),
                (max_side, 0.35),
                (None, 0.35),
                (int(max(h, w) * 2), 0.45),      # 上采样再试（小脸）
            ]
        for side, thr in attempts:
            try:
                found = _detect_yunet_scaled(img, yunet, side, thr)
            except Exception as exc:  # noqa: BLE001
                logger.warning("YuNet 检测失败（side=%s, thr=%.2f）：%s", side, thr, exc)
                found = []
            found = [f for f in found if f.box[2] >= 20 and f.box[3] >= 20]
            if found:
                faces = found
                break

    if not faces:
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        faces = _detect_haar(bgr)
        if not faces and robust and max(h, w) > 900:
            scale = 900.0 / float(max(h, w))
            small = cv2.resize(bgr, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
            inv = 1.0 / scale
            faces = _detect_haar(small)
            for f in faces:
                x, y, bw, bh = f.box
                f.box = (int(round(x * inv)), int(round(y * inv)), int(round(bw * inv)), int(round(bh * inv)))

    faces = [f for f in faces if f.box[2] >= 20 and f.box[3] >= 20]
    faces.sort(key=lambda f: -f.score)
    return faces


# --------------------------------------------------------------------------
# 掩膜构造
# --------------------------------------------------------------------------
def _ellipse(mask: np.ndarray, cx: float, cy: float, rx: float, ry: float, angle: float = 0.0) -> None:
    cv2.ellipse(
        mask,
        (int(round(cx)), int(round(cy))),
        (max(2, int(round(rx))), max(2, int(round(ry)))),
        angle, 0, 360, 255, -1,
    )


def build_face_masks(shape: Tuple[int, int], faces: Sequence[FaceRegion]) -> Tuple[np.ndarray, np.ndarray]:
    """返回 (面部掩膜, 五官核心区掩膜)。"""
    h, w = int(shape[0]), int(shape[1])
    face_mask = np.zeros((h, w), np.uint8)
    critical = np.zeros((h, w), np.uint8)
    for f in faces:
        x, y, bw, bh = f.box
        # 面部区域：椭圆，略微内缩，避免把头发/背景算成脸
        cx, cy = x + bw / 2.0, y + bh / 2.0
        _ellipse(face_mask, cx, cy + bh * 0.02, bw * 0.47, bh * 0.52)

        if f.landmarks is not None and len(f.landmarks) >= 5:
            lm = f.landmarks
            eye_r, eye_l = lm[LM_RIGHT_EYE], lm[LM_LEFT_EYE]
            nose, mouth_r, mouth_l = lm[LM_NOSE], lm[LM_MOUTH_R], lm[LM_MOUTH_L]
            inter_eye = float(np.linalg.norm(eye_r - eye_l))
            r_eye = max(4.0, inter_eye * 0.30)
            # 眼睛 + 眉毛
            for pt in (eye_r, eye_l):
                _ellipse(critical, pt[0], pt[1], r_eye, r_eye * 0.80)
                _ellipse(critical, pt[0], pt[1] - inter_eye * 0.22, r_eye * 1.05, r_eye * 0.55)
            # 鼻子
            _ellipse(critical, nose[0], nose[1], inter_eye * 0.30, inter_eye * 0.34)
            # 嘴巴
            mcx, mcy = (mouth_r[0] + mouth_l[0]) / 2.0, (mouth_r[1] + mouth_l[1]) / 2.0
            mw = max(8.0, float(np.linalg.norm(mouth_r - mouth_l)))
            _ellipse(critical, mcx, mcy, mw * 0.68, mw * 0.42)
        else:
            # 只有人脸框（Haar）：按比例估计五官区
            _ellipse(critical, x + bw * 0.32, y + bh * 0.40, bw * 0.15, bh * 0.09)
            _ellipse(critical, x + bw * 0.68, y + bh * 0.40, bw * 0.15, bh * 0.09)
            _ellipse(critical, x + bw * 0.50, y + bh * 0.60, bw * 0.14, bh * 0.11)
            _ellipse(critical, x + bw * 0.50, y + bh * 0.78, bw * 0.22, bh * 0.10)
        # 五官核心区在面部范围内
        critical = cv2.bitwise_and(critical, face_mask)
    return face_mask, critical


# --------------------------------------------------------------------------
# 修复计划
# --------------------------------------------------------------------------
@dataclass
class RepairPlan:
    """交给修复引擎执行的具体参数（人脸保护策略的落地结果）。"""

    mode: str = "normal"
    dilate: int = 4
    feather: int = 6
    grain: bool = True
    grain_strength: float = 0.8
    tone_strength: float = 0.35
    use_roi: bool = False
    roi: Optional[Tuple[int, int, int, int]] = None   # x, y, w, h
    tile_mode: bool = False                            # 分块局部重建（密集/平铺水印）
    face_mask: Optional[np.ndarray] = None
    critical_mask: Optional[np.ndarray] = None
    notes: List[str] = field(default_factory=list)

    def summary(self) -> str:
        tag = {"normal": "常规", "face_edge": "面部边缘保护", "face_protect": "面部五官保护"}.get(self.mode, self.mode)
        extra = "｜高分辨率局部重建" if self.use_roi else ""
        if self.tile_mode:
            extra += "｜分块局部修复"
        grain_txt = f"高频恢复 {self.grain_strength:.2f}" if self.grain else "高频恢复 关"
        return f"{tag}（扩张 {self.dilate}px，羽化 {self.feather}px，{grain_txt}）{extra}"


def enforce_protection(
    mask: np.ndarray,
    faces: Sequence[FaceRegion],
    max_feature_cover: float = 0.45,
    max_component_ratio: float = 0.03,
) -> Tuple[np.ndarray, List[str], Dict[str, float]]:
    """五官保护**硬约束**：某个五官区域被水印覆盖过多时，宁可不修，也不让 AI 改写五官。

    （对应需求："不允许为了消除一个小水印而重建整个脸颊、鼻子或嘴部"）

    判定方式（第四轮细化）：
    * 覆盖不严重（≤ ``max_feature_cover``）→ 正常修复；
    * 覆盖严重 → **只保留细小的水印碎片**做局部修复（配合分块局部重建，
      小碎片只会用到周围皮肤/毛发纹理，不会改写五官），
      而**丢弃过大的遮挡块**（这类块一旦重建就会把眼睛/鼻子/嘴"重画"）。
    """
    m = mp.ensure_binary(mask)
    notes: List[str] = []
    stats: Dict[str, float] = {}
    if not faces:
        return m, notes, stats
    face_mask, critical = build_face_masks(m.shape, faces)
    n, labels, stat, _ = cv2.connectedComponentsWithStats((critical > 0).astype(np.uint8), 8)
    dropped_big = 0
    kept_small = 0
    for i in range(1, n):
        feature_area = int(stat[i, cv2.CC_STAT_AREA])
        if feature_area < 40:
            continue
        comp = labels == i
        cover = float(np.count_nonzero(m[comp])) / float(feature_area)
        if cover <= max_feature_cover:
            continue
        # 覆盖过密：逐个碎片判断，过大的遮挡块跳过，细小碎片保留
        sub = (m > 0) & comp
        n2, lab2, st2, _ = cv2.connectedComponentsWithStats(sub.astype(np.uint8), 8)
        for j in range(1, n2):
            area = int(st2[j, cv2.CC_STAT_AREA])
            if area <= 0:
                continue
            if area > max_component_ratio * feature_area:
                m[lab2 == j] = 0
                dropped_big += 1
            else:
                kept_small += 1
    if dropped_big:
        notes.append(
            f"五官保护：跳过 {dropped_big} 个过大遮挡块（重建它们会改写五官）；"
            f"{kept_small} 个细小水印碎片仍在五官附近做了局部修复"
        )
    elif kept_small:
        notes.append(f"五官保护：{kept_small} 个细小水印碎片做了局部修复（未大面积重建五官）")
    face_area = int(np.count_nonzero(face_mask))
    if face_area > 0:
        stats["face_cover"] = float(np.count_nonzero(m[face_mask > 0])) / face_area
    stats["dropped_features"] = float(dropped_big)
    stats["kept_fragments"] = float(kept_small)
    return m, notes, stats


def analyze(
    mask: np.ndarray,
    faces: Sequence[FaceRegion],
    shape: Optional[Tuple[int, int]] = None,
) -> FaceReport:
    """判断涂抹区域与人脸/五官的重叠程度。"""
    m = mp.ensure_binary(mask, shape)
    report = FaceReport(faces=list(faces))
    if not faces:
        report.message = "未检测到人脸"
        return report
    face_mask, critical = build_face_masks(m.shape, faces)
    report.face_mask, report.critical_mask = face_mask, critical
    total = float(np.count_nonzero(m)) or 1.0
    overlap = float(np.count_nonzero(cv2.bitwise_and(m, face_mask))) / total
    crit = float(np.count_nonzero(cv2.bitwise_and(m, critical))) / total
    report.overlap_ratio, report.critical_ratio = overlap, crit
    if crit > 0.02:
        report.mode = "face_protect"
        report.message = f"检测到 {len(faces)} 张人脸，且涂抹区域压到五官（{crit * 100:.1f}%），已启用面部保护模式"
    elif overlap > 0.05:
        report.mode = "face_edge"
        report.message = f"检测到 {len(faces)} 张人脸，涂抹区域接触面部边缘（{overlap * 100:.1f}%），已收紧面部重建范围"
    else:
        report.mode = "normal"
        report.message = f"检测到 {len(faces)} 张人脸，但涂抹区域不在人脸范围内，按常规流程处理"
    return report


def make_plan(
    report: FaceReport,
    image_shape: Tuple[int, int],
    repair_mask: np.ndarray,
    base_dilate: int = 4,
    base_feather: int = 6,
    base_tone: float = 0.35,
    base_grain: bool = True,
    protect_faces: bool = True,
    prefer_local: bool = True,
) -> RepairPlan:
    """根据人脸分析结果生成修复参数（人脸优先保护、最小修改、局部优先）。

    第五轮策略调整：
    * **局部优先**：只要 Mask 的包围盒（含 padding）不是几乎覆盖整张图，就只用**一个 ROI**
      做原生分辨率重建（一次 LaMa、一次融合），避免全图缩放导致的模糊；
    * **tile 只在真正超大时启用**：`ratio > 0.35` 或 `碎片 ≥ 25 且 ratio > 0.20`；
    * 人脸模式：扩张 1~2px、羽化 3~4px、光照校正 ≤0.25、**高频恢复以低强度启用**
      （不再直接禁用，避免皮肤塑料感）；水印在脸上占比过高时进一步收紧并提示重涂。
    """
    plan = RepairPlan(
        dilate=int(base_dilate), feather=int(base_feather),
        grain=bool(base_grain), tone_strength=float(base_tone), grain_strength=0.8,
        face_mask=report.face_mask, critical_mask=report.critical_mask,
    )
    m = mp.ensure_binary(repair_mask)
    n_comp, _, stats_c, _ = cv2.connectedComponentsWithStats((m > 0).astype(np.uint8), 8)
    comps = sum(1 for i in range(1, n_comp) if int(stats_c[i, cv2.CC_STAT_AREA]) >= 6)
    ratio = mp.area_ratio(m)
    h, w = int(image_shape[0]), int(image_shape[1])

    # ---------- 局部优先：一个 ROI，一次推理，一次融合 ----------
    boxes = mp.mask_to_boxes(m, min_area=20)
    if boxes and prefer_local:
        x0 = min(b[0] for b in boxes)
        y0 = min(b[1] for b in boxes)
        x1 = max(b[0] + b[2] for b in boxes)
        y1 = max(b[1] + b[3] for b in boxes)
        pad = int(min(max(24, max(x1 - x0, y1 - y0) * 0.6), 160))
        rx0, ry0 = max(0, x0 - pad), max(0, y0 - pad)
        rx1, ry1 = min(w, x1 + pad), min(h, y1 + pad)
        roi_ratio = ((rx1 - rx0) * (ry1 - ry0)) / float(w * h)
        if (rx1 - rx0) >= 24 and (ry1 - ry0) >= 24 and roi_ratio <= 0.90:
            plan.use_roi = True
            plan.roi = (rx0, ry0, rx1 - rx0, ry1 - ry0)

    # ---------- 分块重建：仅在超大 Mask 时作为兜底 ----------
    plan.tile_mode = bool(ratio > 0.35 or (comps >= 25 and ratio > 0.20))
    if plan.tile_mode and plan.use_roi:
        plan.use_roi = False          # 两者互斥：分块本身就是在做局部
    if plan.tile_mode:
        plan.notes.append(
            f"水印覆盖面积很大（{comps} 块碎片，{ratio * 100:.1f}%），已启用分块重建："
            f"每块只生成原始内容，最后由全局统一做一次颜色/纹理/融合处理"
        )
    if not protect_faces or report.mode == "normal" or report.face_mask is None:
        if report.mode == "normal" and report.faces:
            plan.notes.append(report.message)
        return plan

    # ---------- 面部保护 ----------
    plan.mode = report.mode
    plan.dilate = 1 if report.mode == "face_protect" else 2
    plan.feather = 3 if report.mode == "face_protect" else 4
    plan.grain = bool(base_grain)
    plan.grain_strength = 0.45          # 皮肤：低强度真实高频恢复（不叠加明显噪声）
    plan.tone_strength = min(float(base_tone), 0.25)   # 面部只做很轻的接缝校正
    plan.notes.append(report.message)

    # 水印在面部占比过高 → 进一步收紧（宁小不大），并提示重新精确涂抹
    face_area = int(np.count_nonzero(report.face_mask))
    if face_area > 0:
        face_cover = float(np.count_nonzero(cv2.bitwise_and(m, report.face_mask))) / face_area
        if face_cover > 0.25:
            plan.dilate = 0
            plan.feather = 2
            plan.notes.append(
                f"水印覆盖了面部约 {face_cover * 100:.0f}% 的面积：已把修复范围收紧到最小"
                f"（扩张 0px / 羽化 2px）。建议重新精确涂抹一小块后再修，效果会明显更自然"
            )
        elif face_cover > 0.12:
            plan.notes.append(f"水印覆盖面部约 {face_cover * 100:.0f}%：已使用最小扩张与低强度纹理恢复")
    return plan


def protect_mask_for(report: FaceReport, shape: Tuple[int, int]) -> Optional[np.ndarray]:
    """返回"不要在脸上加颗粒/强化"的区域（用于后处理）。"""
    if report.face_mask is None:
        return None
    return mp.ensure_binary(report.face_mask, shape)


def describe_capability() -> str:
    info = face_detector_info()
    if info["yunet_available"]:
        return "人脸检测：YuNet（含眼/鼻/嘴关键点）"
    if info["haar_available"]:
        return "人脸检测：Haar 级联（离线，无关键点）"
    return "人脸检测：不可用"
