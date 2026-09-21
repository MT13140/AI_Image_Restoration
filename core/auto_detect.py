"""自动水印检测引擎 v2（第六轮专项重写）。

任务书要求（第二 ~ 十一章）总结成四句话：

1. **提高召回**：小字、旋转、半透明、复杂背景上的水印都不能轻易漏；
2. **识别完整**：同一水印的多行文字 / 账号 / 旁边的小图标要合成**一个**水印块；
3. **识别重复**：同模板的重复 / 平铺水印要尽量把**全部实例**都找出来；
4. **只给候选**：输出"区域 + 置信度分层"，配合 Mask 预览由用户确认，
   绝不自动修复、也绝不自动删除。

本模块的检测链路（全部是真实、可解释的图像处理 / OCR，无虚构模型）：

    多尺度 OCR（全局）
        ↓
    局部对比度"文字笔画图" → 笔画聚类 → **逐块去倾斜放大 OCR**
        （这一步负责小字、倾斜 30~45°、竖排文字）
        ↓
    文字行合并 → 水印块合并（多行 + 账号 + 相邻小图标）
        ↓
    水印块 → 模板聚类 → 模板匹配 + 点阵展开（找齐重复/平铺水印）
        ↓
    半透明响应 / 重复检测（无 OCR 证据时的兜底证据）
        ↓
    Watermark Score（类型 + 位置 + 尺寸 + 重复性 + 透明性 + 人脸冲突）
        ↓
    高 / 中 / 低置信度分层 → 候选 Mask（贴合笔画，而不是大矩形）

**这里没有任何修复算法**：修复仍走第五轮验证过的
``core/pipeline.py`` → ``core/inpainting.py``（LaMa + repair/blend 双 Mask +
人脸保护 + ROI 优先 + 高频纹理匹配）。
"""

from __future__ import annotations

import difflib
import math
import re
import time
from dataclasses import dataclass, field
from collections import OrderedDict
import hashlib
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from config import get_logger
from . import face_protector as fp
from . import mask_processor as mp
from . import watermark_detector as wd
from .ocr_detector import classify_text

logger = get_logger("ai_restore.autodetect")

#: 置信度分层阈值（"低"的下限由 sensetivity 决定）
TIER_HIGH_MIN_EXTRA = 0.0
TIER_MEDIUM_MIN = 0.40

#: 敏感度预设：积极 = 减少漏检优先（本项目场景就是用户明确要删水印）
SENSITIVITY_PRESETS: Dict[str, Dict[str, Any]] = {
    "conservative": {
        "label": "保守",
        "min_confidence": 0.46,
        "select_threshold": 0.70,
        "select_tiers": ("high",),
        "max_area_ratio": 0.20,
        "max_region_ocr": 8,
        "max_candidates": 48,
        "sweep_angles": (),
    },
    "balanced": {
        "label": "平衡",
        "min_confidence": 0.33,
        "select_threshold": 0.60,
        "select_tiers": ("high",),
        "max_area_ratio": 0.28,
        "max_region_ocr": 14,
        "max_candidates": 72,
        "sweep_angles": (-25.0, 25.0),
    },
    "aggressive": {
        "label": "积极",
        "min_confidence": 0.22,
        "select_threshold": 0.50,
        "select_tiers": ("high", "medium"),
        "max_area_ratio": 0.35,
        "max_region_ocr": 20,
        "max_candidates": 96,
        "sweep_angles": (-25.0, 25.0, -45.0, 45.0),
    },
}

DEFAULT_SENSITIVITY = "aggressive"


# --------------------------------------------------------------------------
# 基础几何工具
# --------------------------------------------------------------------------
def _norm_angle(deg: float) -> float:
    """把角度归一化到 (-90, 90]。"""
    a = float(deg)
    while a <= -90.0:
        a += 180.0
    while a > 90.0:
        a -= 180.0
    return a


def _poly_of_rect(rect: Sequence[float]) -> np.ndarray:
    """``(cx, cy, w, h, theta)`` → 4 个角点（(4,2) float32）。"""
    cx, cy, w, h, theta = [float(v) for v in rect]
    t = math.radians(theta)
    u = np.array([math.cos(t), math.sin(t)], np.float32)
    v = np.array([-u[1], u[0]], np.float32)
    c = np.array([cx, cy], np.float32)
    return np.array([
        c - u * (w / 2.0) - v * (h / 2.0),
        c + u * (w / 2.0) - v * (h / 2.0),
        c + u * (w / 2.0) + v * (h / 2.0),
        c - u * (w / 2.0) + v * (h / 2.0),
    ], np.float32)


def _rect_from_points(points: np.ndarray) -> Tuple[float, float, float, float, float]:
    """任意点集 → 最小外接旋转矩形 ``(cx, cy, w, h, theta)``（w ≥ h）。

    ``theta`` 是**长边方向**与 x 轴的夹角（图像坐标，y 向下），范围 (-90, 90]。
    "文字行方向"就是长边方向，因此 ``theta`` 可直接当作去倾斜角度使用。

    实现要点：``cv2.boxPoints`` 返回的角点顺序固定，但"哪条边是长边"取决于
    ``minAreaRect`` 的返回角度，**不能靠下标奇偶判断**（早期版本就栽在这里：
    水平文字被算成 +89°，去倾斜时反而把文字转成竖条）。
    """
    pts = np.asarray(points, np.float32).reshape(-1, 2)
    if pts.shape[0] < 2:
        return (0.0, 0.0, 1.0, 1.0, 0.0)
    (cx, cy), (w, h), ang = cv2.minAreaRect(pts)
    box = cv2.boxPoints(((cx, cy), (w, h), ang)).astype(np.float32)
    edges = np.roll(box, -1, axis=0) - box
    lens = np.linalg.norm(edges, axis=1)
    idx = int(np.argmax(lens))                      # 最长边 = 文字行方向
    vec = edges[idx]
    theta = _norm_angle(math.degrees(math.atan2(float(vec[1]), float(vec[0]))))
    long_side = float(lens[idx])
    short_side = float(lens[(idx + 1) % 4])          # 相邻边与最长边垂直
    if short_side > long_side:                       # 数值兜底（几乎不会发生）
        long_side, short_side = short_side, long_side
        theta = _norm_angle(theta + 90.0)
    return (float(cx), float(cy), float(max(1.0, long_side)), float(max(1.0, short_side)),
            float(theta))


def _scale_evidence(items: Sequence["TextEvidence"], factor: float) -> List["TextEvidence"]:
    """把证据坐标整体缩放，用于统一到"检测坐标系"（所有证据必须同一坐标系）。"""
    if not items or abs(float(factor) - 1.0) < 1e-9:
        return list(items)
    out: List[TextEvidence] = []
    for it in items:
        poly = (np.asarray(it.poly, np.float32) * float(factor)).astype(np.float32)
        out.append(TextEvidence(
            poly=poly, text=it.text, score=it.score, source=it.source,
            kind=it.kind, prior=it.prior, reasons=list(it.reasons),
        ))
    return out


def _envelope(poly_or_rect) -> Tuple[int, int, int, int]:
    """旋转矩形 / 角点 / 轴对齐框 → 轴对齐包围盒 ``(x, y, w, h)``（整型，允许越界）。

    支持三种输入（这个函数被多处复用，必须兼容）：

    * ``(cx, cy, w, h, theta)`` —— 旋转矩形；
    * ``(x, y, w, h)`` —— 已经是轴对齐框（**候选的 bbox 就是这种**）；
    * ``(N, 2)`` 角点数组。

    注意：早期版本把 ``(x, y, w, h)`` 当成"两个点"来做 reshape，
    结果包围盒全错（例如 (33,666,296,83) 被算成 (33,83,263,583)），
    导致候选去重时把真正的整块水印挤掉 —— A 模式曾经因此漏检。
    """
    arr = np.asarray(poly_or_rect)
    if arr.ndim == 1 and arr.size == 5:
        pts = _poly_of_rect(arr)
    elif arr.ndim == 1 and arr.size == 4:
        x, y, w, h = [float(v) for v in arr]
        return (int(math.floor(x)), int(math.floor(y)),
                max(1, int(math.ceil(w))), max(1, int(math.ceil(h))))
    else:
        pts = arr.astype(np.float32).reshape(-1, 2)
    x0 = int(np.floor(pts[:, 0].min()))
    y0 = int(np.floor(pts[:, 1].min()))
    x1 = int(np.ceil(pts[:, 0].max()))
    y1 = int(np.ceil(pts[:, 1].max()))
    return x0, y0, max(1, x1 - x0), max(1, y1 - y0)


def _clip_rect(rect: Sequence[float], shape: Tuple[int, int]) -> Tuple[float, float, float, float, float]:
    """把旋转矩形的中心夹回画面内。"""
    H, W = int(shape[0]), int(shape[1])
    cx, cy, w, h, theta = [float(v) for v in rect]
    return (float(np.clip(cx, 0, W - 1)), float(np.clip(cy, 0, H - 1)),
            max(1.0, w), max(1.0, h), theta)


def _iou_rect(a: Sequence[float], b: Sequence[float]) -> float:
    ax0, ay0, aw, ah = _envelope(a)
    bx0, by0, bw, bh = _envelope(b)
    ax1, ay1, bx1, by1 = ax0 + aw, ay0 + ah, bx0 + bw, by0 + bh
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = float((ix1 - ix0) * (iy1 - iy0))
    union = float(aw * ah + bw * bh) - inter
    return inter / max(1e-6, union)


def _intersection_area(a: Sequence[float], b: Sequence[float]) -> float:
    """两个轴对齐框的交集面积（用于"小框被大框包含"的合并判断）。"""
    ax0, ay0, aw, ah = _envelope(a)
    bx0, by0, bw, bh = _envelope(b)
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax0 + aw, bx0 + bw), min(ay0 + ah, by0 + bh)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    return float((ix1 - ix0) * (iy1 - iy0))


def _local_frame(cx: float, cy: float, theta: float, px: float, py: float) -> Tuple[float, float]:
    """把点 ``(px, py)`` 表示在以 ``(cx, cy)`` 为原点、方向 ``theta`` 的坐标系里。"""
    t = math.radians(theta)
    dx, dy = float(px - cx), float(py - cy)
    return (dx * math.cos(t) + dy * math.sin(t), -dx * math.sin(t) + dy * math.cos(t))


# --------------------------------------------------------------------------
# 文字笔画图（局部对比度）
# --------------------------------------------------------------------------
def _odd(v: int, lo: int = 3) -> int:
    v = int(max(lo, v))
    return v if v % 2 == 1 else v + 1


def local_contrast(gray: np.ndarray, kernel: Optional[int] = None) -> np.ndarray:
    """局部对比度图：``|gray - median(gray)|``，水印/文字的笔画会明显抬升。"""
    h, w = gray.shape[:2]
    if kernel is None:
        kernel = _odd(int(max(7, min(h, w) * 0.05)), 5)
    kernel = _odd(min(kernel, max(5, (min(h, w) // 2) * 2 - 1)), 5)
    if kernel >= min(h, w):
        kernel = _odd(max(3, min(h, w) // 3), 3)
    bg = cv2.medianBlur(gray, kernel)
    diff = cv2.absdiff(gray, bg).astype(np.float32)
    return cv2.GaussianBlur(diff, (0, 0), 1.0)


def textness_map(img_rgb: np.ndarray, max_side: int = 1500) -> Tuple[np.ndarray, float, np.ndarray]:
    """生成"文字笔画强度图"（0~1）。

    返回 ``(文字图, 缩放比例, 检测用的缩小图)``。缩放比例 = 检测图 / 原图。
    """
    img = mp.to_uint8(img_rgb)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    h, w = img.shape[:2]
    scale = min(1.0, float(max_side) / float(max(1, max(h, w))))
    if scale < 1.0:
        small = cv2.resize(img, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                           interpolation=cv2.INTER_AREA)
    else:
        small = img
        scale = 1.0
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    d = local_contrast(gray)
    p = float(np.percentile(d, 99.0)) if d.size else 0.0
    norm = np.clip(d / max(8.0, p), 0.0, 1.0).astype(np.float32)
    return norm, scale, small


# --------------------------------------------------------------------------
# OCR 证据
# --------------------------------------------------------------------------
@dataclass
class TextEvidence:
    """一条 OCR 证据（坐标一律是"原图分辨率"下的坐标）。"""

    poly: np.ndarray                     # (4, 2) float32
    text: str
    score: float
    source: str = "OCR"
    kind: str = "text"
    prior: float = 0.45
    reasons: List[str] = field(default_factory=list)
    rect: Tuple[float, float, float, float, float] = (0, 0, 1, 1, 0)

    def __post_init__(self) -> None:
        if self.rect[2] <= 1.0 and self.poly is not None:
            self.rect = _rect_from_points(self.poly)

    @property
    def center(self) -> Tuple[float, float]:
        return (self.rect[0], self.rect[1])

    @property
    def theta(self) -> float:
        return self.rect[4]

    @property
    def height(self) -> float:
        return self.rect[3]

    @property
    def width(self) -> float:
        return self.rect[2]


def _ocr_patch(
    detector: wd.WatermarkDetector,
    src: np.ndarray,
    cx: float, cy: float, w: float, h: float, theta: float,
    min_score: float,
    source: str,
    target_h: float = 96.0,
    min_scale: float = 0.30,
    max_scale: float = 8.0,
    force_scale: Optional[float] = None,
    pad_ratio: float = 0.20,
) -> List[TextEvidence]:
    """在 ``src`` 上按"中心 + 尺寸 + 角度"取一块区域做 OCR，并把框映射回 ``src`` 坐标。

    这一招同时解决三件事：
    * 小字 → 放大后再识别；
    * 旋转文字 → 先转正再识别（30~45° 也能读）；
    * 文字 + 账号 → 各自成为一条证据，后续再合并成同一个水印块。
    """
    w = max(8.0, float(w))
    h = max(8.0, float(h))
    s = float(force_scale) if force_scale else float(np.clip(target_h / h, min_scale, max_scale))
    long_side = max(w, h) * s
    if long_side > 2048:
        s *= 2048.0 / long_side
    s = max(0.05, s)
    Wp = int(max(24, round(w * s * (1.0 + 2 * pad_ratio))))
    Hp = int(max(24, round(h * s * (1.0 + 2 * pad_ratio))))
    M = cv2.getRotationMatrix2D((float(cx), float(cy)), float(theta), s)
    c = M @ np.array([cx, cy, 1.0], np.float64)
    M[0, 2] += Wp / 2.0 - c[0]
    M[1, 2] += Hp / 2.0 - c[1]
    try:
        patch = cv2.warpAffine(src, M, (Wp, Hp), flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REPLICATE)
    except Exception as exc:  # pragma: no cover - 极端尺寸
        logger.debug("区域裁剪失败：%s", exc)
        return []
    try:
        # 注意：WatermarkDetector.ocr 是"文字水印检测器"包装类，
        # 真正的 OCR 引擎在它的 .detector 上（OCRDetector.detect 返回 TextItem 列表）。
        items = detector.ocr.detector.detect(patch, min_score=float(min_score))
    except Exception as exc:
        logger.warning("区域 OCR 失败（%s）：%s", source, exc)
        return []
    if not items:
        return []
    Minv = cv2.invertAffineTransform(M)
    out: List[TextEvidence] = []
    for it in items:
        try:
            pts = np.asarray(it.box, np.float32).reshape(-1, 2)
            mapped = cv2.transform(pts.reshape(1, -1, 2), Minv).reshape(-1, 2)
        except Exception:
            continue
        out.append(TextEvidence(
            poly=mapped.astype(np.float32),
            text=str(it.text).strip(),
            score=float(it.score),
            source=source,
            kind=getattr(it, "kind", "text"),
            prior=float(getattr(it, "watermark_score", 0.45)),
            reasons=list(getattr(it, "reasons", []) or []),
        ))
    return out


def _text_shape(text: str) -> str:
    """去掉空白/数字/标点后的"文字形状"，用于判断是否同一水印模板。"""
    return re.sub(r"[\W_]+", "", (text or "").lower(), flags=re.UNICODE)


_RE_CJK = re.compile(r"[\u3400-\u9fff\uf900-\ufaff]")
_RE_ALNUM = re.compile(r"[0-9A-Za-z]")


def _is_cjk_dominant(text: str) -> bool:
    """是否以中日韩字符为主（用于区分"英文水印"和"画面正文/字幕"）。"""
    s = text or ""
    cjk = len(_RE_CJK.findall(s))
    alnum = len(_RE_ALNUM.findall(s))
    return cjk >= 3 and cjk >= alnum


def _alnum_ratio(text: str) -> float:
    """字母+数字占比（英文/数字水印的特征：很高）。"""
    s = re.sub(r"\s", "", text or "")
    if not s:
        return 0.0
    return len(_RE_ALNUM.findall(s)) / float(len(s))


def _similar_text(a: str, b: str, ratio_thr: float = 0.80) -> bool:
    sa, sb = _text_shape(a), _text_shape(b)
    if not sa or not sb:
        return False
    if sa == sb:
        return True
    if len(sa) >= 4 and (sa in sb or sb in sa):
        return True
    if len(sa) >= 3 and len(sb) >= 3:
        return difflib.SequenceMatcher(None, sa, sb).ratio() >= float(ratio_thr)
    return False


def _dedupe_evidence(items: Sequence[TextEvidence], iou_thr: float = 0.45) -> List[TextEvidence]:
    """同一处文字被多次读到（多尺度 / 多角度 / 全局+局部）时只保留最好的那条。"""
    out: List[TextEvidence] = []
    # 排序技巧：先按"量化到 0.1 的置信度"排序，再优先保留**接近水平**的框。
    # 原因：全局旋转扫描会把同一个水平水印也读出来，映射回来的框是斜的；
    # 若按浮点分数排序，斜框可能因为分数高一点点而挤掉正确的水平框。
    for it in sorted(items, key=lambda x: (-round(float(x.score), 1), abs(_norm_angle(x.theta)), -x.height)):
        dup = None
        for o in out:
            same_box = _iou_rect(it.rect, o.rect)
            if same_box >= 0.80:
                dup = o                      # 同一块地方，不管读到什么，只留最好的
                break
            if not _similar_text(it.text, o.text):
                continue
            if same_box >= iou_thr:
                dup = o
                break
            d = math.hypot(it.center[0] - o.center[0], it.center[1] - o.center[1])
            if d <= 0.6 * max(it.height, o.height):
                dup = o
                break
            # 一个框的中心落在另一个框里 → 同一处文字的两种读法
            if _point_in_rect(it.center, o.rect) or _point_in_rect(o.center, it.rect):
                dup = o
                break
        if dup is None:
            out.append(it)
        else:
            # 保留读得更准的那条，但把另一条的证据并进来
            for r in it.reasons:
                if r not in dup.reasons:
                    dup.reasons.append(r)
    return out


def _point_in_rect(pt: Tuple[float, float], rect: Sequence[float], margin: float = 0.0) -> bool:
    dx, dy = _local_frame(rect[0], rect[1], rect[4], pt[0], pt[1])
    return abs(dx) <= rect[2] / 2.0 + margin and abs(dy) <= rect[3] / 2.0 + margin


# --------------------------------------------------------------------------
# 笔画聚类 → 逐块（含旋转）OCR
# --------------------------------------------------------------------------
@dataclass
class _Cluster:
    rect: Tuple[float, float, float, float, float]
    strokes: int
    weight: float


def _oriented_line_kernels(length: int, thickness: int,
                           angles: Sequence[float]) -> List[np.ndarray]:
    """生成若干"不同方向的线段"结构元。

    为什么需要：水印文字不一定是水平的。用**单一横向**的闭运算只能把横排字符
    连成一行，旋转 30~45° 的文字会被拆成一个个孤立字符，OCR 自然读不出来。
    这里同时用 0°/±30°/±45°/±60°/90° 的线段做闭运算再取并集，
    任何方向的"文字行"都能连成一块，然后再由最小外接矩形估计倾角、转正后 OCR。
    """
    length = int(max(5, length))
    thickness = int(max(1, thickness))
    size = length + 4
    c = size // 2.0
    kernels: List[np.ndarray] = []
    for a in angles:
        k = np.zeros((size, size), np.uint8)
        rad = math.radians(float(a))
        dx, dy = math.cos(rad), math.sin(rad)
        p1 = (int(round(c - dx * length / 2.0)), int(round(c - dy * length / 2.0)))
        p2 = (int(round(c + dx * length / 2.0)), int(round(c + dy * length / 2.0)))
        cv2.line(k, p1, p2, 1, thickness)
        if k.sum() > 0:
            kernels.append(k)
    return kernels


def _stroke_clusters(tn: np.ndarray, min_strokes: int, max_clusters: int) -> List[_Cluster]:
    """把零散笔画按"文字行"聚成候选块（用于小字 / 旋转文字的区域 OCR）。

    连通用**多方向线段闭运算**实现，因此竖排、倾斜 30~60° 的文字同样能连成一块。
    """
    if tn.size == 0:
        return []
    h, w = tn.shape[:2]
    thr = max(0.16, float(np.percentile(tn, 96.0)) * 0.45)
    binary = (tn > thr).astype(np.uint8)
    if int(binary.sum()) < min_strokes:
        return []
    # 先用小圆盘把同一笔画的断续处接上（文字抗锯齿、压缩噪点）
    disk_r = int(np.clip(min(h, w) * 0.004, 1, 3))
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * disk_r + 1, 2 * disk_r + 1))
    base = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, disk)
    line_len = int(np.clip(min(h, w) * 0.045, 9, 45))
    kernels = _oriented_line_kernels(line_len, 3, (0, 30, -30, 45, -45, 60, -60, 90))
    joined = np.zeros_like(base)
    for k in kernels:
        joined = cv2.bitwise_or(joined, cv2.morphologyEx(base, cv2.MORPH_CLOSE, k))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(joined, 8)
    out: List[_Cluster] = []
    for i in range(1, n):
        x, y, bw, bh = (int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP]),
                        int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT]))
        if bw < 6 or bh < 6:
            continue
        if bw > 0.98 * w and bh > 0.98 * h:
            continue                        # 整图 = 交给全局 OCR
        if bw * bh < 0.00002 * float(h * w):
            continue
        if bw * bh > 0.75 * float(h * w):
            continue                        # 近乎整图的大连通块，信息量低且很费 OCR
        comp = (labels[y:y + bh, x:x + bw] == i).astype(np.uint8)
        strokes = int(np.count_nonzero(comp & (binary[y:y + bh, x:x + bw] > 0)))
        if strokes < min_strokes:
            continue
        ys, xs = np.nonzero(comp)
        pts = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
        rect_local = _rect_from_points(pts)             # 以局部像素点估计方向
        cx = rect_local[0] + x
        cy = rect_local[1] + y
        rect = (cx, cy, rect_local[2], rect_local[3], rect_local[4])
        density = strokes / max(1.0, float(bw * bh))
        weight = float(strokes) * (1.0 + min(1.0, density * 3.0))
        out.append(_Cluster(rect=rect, strokes=strokes, weight=weight))
    out.sort(key=lambda c: -c.weight)
    return out[:max_clusters]


def collect_evidence(
    detector: wd.WatermarkDetector,
    img_full: np.ndarray,
    tn: np.ndarray,
    scale: float,
    det_img: np.ndarray,
    ocr_min_score: float,
    max_region_ocr: int = 20,
    use_region_ocr: bool = True,
    sweep_angles: Sequence[float] = (),
    sweep_side: int = 1100,
    sweep_budget_ms: float = 4500.0,
    region_budget_ms: float = 6000.0,
    hi_res_ocr: bool = True,
    hi_tiles: int = 6,
    hi_budget_ms: float = 8000.0,
    hi_min_score: float = 0.35,
    roi_boxes: Sequence[Tuple[int, int, int, int]] = (),
    local_ocr: bool = True,
    local_ocr_scale: float = 3.0,
    local_ocr_pad: float = 0.60,
    local_ocr_budget_ms: float = 4500.0,
    local_ocr_max: int = 6,
) -> Tuple[List[TextEvidence], Dict[str, float]]:
    """收集 OCR 证据，共三遍：

    ① **全局正读**：检测分辨率整图 OCR（正常方向的中等/大字号文字）；
    ② **全局旋转扫描**：整图旋转 ±25°/±45° 再 OCR（这一步解决"倾斜水印整体漏检"）；
    ③ **逐块去倾斜放大**：笔画聚类 → 每块按自身倾角转正、放大后再 OCR
       （小字、竖排、旋转 30~45°、复杂背景上的文字都靠这一步）。
    ④ **高清兜底**（新增）：缩图 OCR 没解释掉的文字区域，按需在**原分辨率**
       （大图则分块）再读一次 —— 这是"大图上的小字/半透明英文水印"不再丢失的关键。
    ⑤ **局部高分辨率 OCR（LOHR，本轮新增）**：当证据仍然不足时，只在**很小的 ROI**
       （人脸框 / 笔画聚类块）上以 2~3 倍采样再读一次。
       依据：512×1080 人像上 24px 的小字，1x/1.5x/2x **整图** OCR 都读不出，
       但把人脸区域放大 3 倍就能稳定读到（实测 confidence 0.58~0.99）；
       而对**没有水印的同区域**读数为 0 条 —— 因此不会因为"放大"而把皮肤/头发当成文字。
       **不降低全局阈值**：局部 OCR 的阈值只作用在这几个小 ROI 上。
    """
    t_global = time.time()
    h, w = det_img.shape[:2]
    items = _ocr_patch(detector, det_img, w / 2.0, h / 2.0, w, h, 0.0,
                       ocr_min_score, "OCR", force_scale=1.0, pad_ratio=0.0)
    ocr_ms = (time.time() - t_global) * 1000.0

    # ---- ② 全局旋转扫描（降采样后做，成本可控）----
    t_sweep = time.time()
    sweep_hits = 0
    if sweep_angles:
        sh, sw = det_img.shape[:2]
        s2 = min(1.0, float(sweep_side) / float(max(1, max(sh, sw))))
        if s2 < 1.0:
            small = cv2.resize(det_img, (max(1, int(round(sw * s2))), max(1, int(round(sh * s2)))),
                               interpolation=cv2.INTER_AREA)
        else:
            small = det_img
        sh2, sw2 = small.shape[:2]
        back = 1.0 / s2 if s2 > 0 else 1.0          # 小图坐标 → 检测坐标
        for ang in sweep_angles:
            if (time.time() - t_sweep) * 1000.0 > sweep_budget_ms:
                break                        # 时间预算：避免在密集文字图上"无限扫描"
            got = _ocr_patch(detector, small, sw2 / 2.0, sh2 / 2.0, sw2, sh2, float(ang),
                             max(0.35, ocr_min_score), f"旋转 OCR（{ang:+.0f}°）",
                             force_scale=1.0, pad_ratio=0.0)
            if got:
                got = _scale_evidence(got, back)
                sweep_hits += len(got)
                items.extend(got)
    sweep_ms = (time.time() - t_sweep) * 1000.0

    # ---- ③ 逐块去倾斜放大 OCR ----
    t_region = time.time()
    attempts = 0
    n_region = 0
    region_items: List[TextEvidence] = []
    if use_region_ocr and scale > 0 and max_region_ocr > 0:
        min_strokes = int(max(24, 0.00006 * float(h * w)))
        clusters = _stroke_clusters(tn, min_strokes, max_clusters=max(24, max_region_ocr * 2))
        inv = 1.0 / scale
        covered = [it for it in items if it.score >= 0.50]
        for cl in clusters:
            if attempts >= max_region_ocr:
                break
            if (time.time() - t_region) * 1000.0 > region_budget_ms:
                break
            rect = cl.rect
            # 已经有高置信度 OCR 覆盖这块 → 跳过（省时间）
            skip = False
            for it in covered:
                ir = it.rect
                if _iou_rect(_clip_rect(rect, tn.shape), ir) > 0.55:
                    skip = True
                    break
            if skip:
                continue
            cx, cy, cw, ch, theta = rect
            # 坐标换算到原图分辨率后，**直接从原图裁剪**（不损失细节）
            fx, fy = cx * inv, cy * inv
            fw, fh = max(8.0, cw * inv), max(8.0, ch * inv)
            if fw * fh > 0.9 * float(img_full.shape[0] * img_full.shape[1]):
                continue
            angles = [theta]
            if ch > 1.6 * max(1.0, cw):          # 竖排文字：再试一次转 90°
                angles.append(_norm_angle(theta + 90.0))
            got: List[TextEvidence] = []
            for ang in angles:
                attempts += 1
                got = _ocr_patch(detector, img_full, fx, fy, fw, fh, ang,
                                 max(0.30, ocr_min_score * 0.85),
                                 "局部放大 OCR" if len(angles) == 1 else "旋转/竖排 OCR",
                                 target_h=88.0, min_scale=0.5, max_scale=4.0, pad_ratio=0.22)
                if got:
                    break
                if attempts >= max_region_ocr:
                    break
            if got:
                got = _scale_evidence(got, scale)   # 原图坐标 → 检测坐标
                n_region += 1
                region_items.extend(got)

    region_ms = (time.time() - t_region) * 1000.0
    merged = _dedupe_evidence(list(items) + region_items)

    # ---- ④ 高清兜底 OCR（只在"缩图没读全"时才跑，避免拖慢批处理）----
    t_hi = time.time()
    hi_items: List[TextEvidence] = []
    hi_tiles_used = 0
    # 触发条件：① 图片被缩小过（缩图 OCR 可能把字缩小到读不出）；或
    #           ② 整条快路径**一条文字都没读到**（此时值得用原图再确认一次，
    #              对本来就小于 1500 的图，这次就等于"再看一眼原图"，代价很小）。
    if hi_res_ocr and img_full is not None and (scale < 0.995 or not merged):
        need = _hi_res_needed(merged, tn, scale)
        if need:
            for box in _hi_res_regions(img_full, tn, scale, hi_tiles):
                if (time.time() - t_hi) * 1000.0 > hi_budget_ms:
                    break
                x0, y0, x1, y1 = box
                w_b, h_b = x1 - x0, y1 - y0
                if w_b < 32 or h_b < 24:
                    continue
                got = _ocr_patch(detector, img_full, x0 + w_b / 2.0, y0 + h_b / 2.0,
                                 w_b, h_b, 0.0, float(hi_min_score), "高清 OCR",
                                 force_scale=1.0, pad_ratio=0.0)
                hi_tiles_used += 1
                if got:
                    # 坐标必须是"检测坐标系"才能与其它证据合并（原图坐标 × scale）
                    hi_items.extend(_scale_evidence(got, scale))
            if hi_items:
                # 高清结果与已有结果合并去重（坐标已经是原图坐标系）
                merged = _dedupe_evidence(merged + hi_items)
    hi_ms = (time.time() - t_hi) * 1000.0

    # ---- ⑤ 局部高分辨率 OCR：只在"证据不足 + 小 ROI"时启用 ----
    t_lo = time.time()
    lo_items: List[TextEvidence] = []
    lo_rois = 0
    if local_ocr and img_full is not None and int(local_ocr_max) > 0:
        need_local = (not merged) or _hi_res_needed(merged, tn, scale, 0.35)
        if need_local:
            rois: List[Tuple[float, float, float, float]] = []
            for (rx, ry, rw, rh) in (roi_boxes or ()):
                if rw >= 8 and rh >= 8:
                    # 注意：上下文余量交给 _ocr_patch 的 pad_ratio 处理（默认 0.6），
                    # 这里再额外加 padding 会把 ROI 撑得过大、小字相对更小 → 反而读不到。
                    px, py = rw * 0.10, rh * 0.10
                    rois.append((rx - px, ry - py, rw + 2 * px, rh + 2 * py))
            # 证据没覆盖到的笔画聚类（按权重排序，已经在前面的区域 OCR 里试过 1x）
            try:
                clusters = _stroke_clusters(tn, int(max(24, 0.00006 * float(tn.size))), 12)
            except Exception:
                clusters = []
            covered = [it for it in merged if it.score >= 0.30]
            for cl in clusters:
                if len(rois) >= int(local_ocr_max):
                    break
                cx_, cy_, cw_, ch_, _th = cl.rect
                if any(_iou_rect(_clip_rect(cl.rect, tn.shape), it.rect) > 0.5 for it in covered):
                    continue
                inv = 1.0 / scale if scale else 1.0
                rois.append((cx_ * inv - cw_ * inv * 0.3, cy_ * inv - ch_ * inv * 0.3,
                             cw_ * inv * 1.6, ch_ * inv * 1.6))
            for (rx, ry, rw, rh) in rois[: int(local_ocr_max)]:
                if (time.time() - t_lo) * 1000.0 > float(local_ocr_budget_ms):
                    break
                x0, y0 = max(0.0, rx), max(0.0, ry)
                x1 = min(float(img_full.shape[1]), rx + rw)
                y1 = min(float(img_full.shape[0]), ry + rh)
                if x1 - x0 < 24 or y1 - y0 < 24:
                    continue
                lo_rois += 1
                # **多尺度一致性**：同一 ROI 用 2x 与 3x 各读一次，只保留**两次都读到**
                # 的文字。理由：放大后偶然把纹理读成"字"的假阳性通常只在一个倍率出现，
                # 而真实小字在 2x/3x 都能被读到（实测 24px/20px/16px 均如此）。
                base_scale = float(local_ocr_scale)
                multi: List[List[TextEvidence]] = []
                for sc in (max(2.0, base_scale - 1.0), base_scale):
                    got = _ocr_patch(detector, img_full, (x0 + x1) / 2.0, (y0 + y1) / 2.0,
                                     x1 - x0, y1 - y0, 0.0,
                                     max(0.25, float(hi_min_score) - 0.05),
                                     "局部高清 OCR", force_scale=sc,
                                     pad_ratio=float(local_ocr_pad))
                    multi.append([it for it in got if len(_text_shape(it.text)) >= 2
                                  and it.height * scale >= 6.0])
                if len(multi) == 2 and multi[0] and multi[1]:
                    agreed: List[TextEvidence] = []
                    for it in multi[1]:                      # 以 3x 的结果为准
                        for o in multi[0]:
                            if _iou_rect(it.rect, o.rect) > 0.25 or _similar_text(it.text, o.text):
                                agreed.append(it)
                                break
                    if agreed:
                        lo_items.extend(_scale_evidence(agreed, scale))
            if lo_items:
                merged = _dedupe_evidence(merged + lo_items)
    lo_ms = (time.time() - t_lo) * 1000.0
    return merged, {
        "ocr_ms": round(ocr_ms, 1),
        "sweep_ms": round(sweep_ms, 1),
        "sweep_hits": float(sweep_hits),
        "region_ocr_ms": round(region_ms, 1),
        "region_ocr_tries": float(attempts),
        "region_ocr_hits": float(n_region),
        "hi_ocr_ms": round(hi_ms, 1),
        "hi_ocr_tiles": float(hi_tiles_used),
        "hi_ocr_items": float(len(hi_items)),
        "local_ocr_ms": round(lo_ms, 1),
        "local_ocr_rois": float(lo_rois),
        "local_ocr_items": float(len(lo_items)),
        "ocr_items": float(len(merged)),
    }


def _hi_res_needed(items: Sequence[TextEvidence], tn: np.ndarray, scale: float,
                   uncovered_ratio: float = 0.50) -> bool:
    """判断是否需要高清兜底：缩图 OCR 没有解释掉足够多的"文字笔画能量"。

    做法：把已有 OCR 框覆盖的区域在笔画图上抹掉，看**剩下的笔画能量**占比。
    剩下的多 → 说明图里还有没读出来的文字（小字/半透明），值得花时间读原图。
    """
    if tn is None or tn.size == 0:
        return False
    if not items:
        return True
    h, w = tn.shape[:2]
    covered = np.zeros((h, w), np.uint8)
    for it in items:
        if it.score < 0.30:
            continue
        x, y, bw, bh = _envelope(it.rect)
        px, py = int(max(2, bw * 0.25)), int(max(2, bh * 0.25))
        x0, y0 = max(0, x - px), max(0, y - py)
        x1, y1 = min(w, x + bw + px), min(h, y + bh + py)
        if x1 > x0 and y1 > y0:
            covered[y0:y1, x0:x1] = 1
    total = float(tn.sum())
    if total <= 1e-6:
        return False
    rest = float(tn[covered == 0].sum())
    return (rest / total) > float(uncovered_ratio)


def _hi_res_regions(img_full: np.ndarray, tn: np.ndarray, scale: float,
                    max_tiles: int) -> List[Tuple[int, int, int, int]]:
    """挑出需要高清 OCR 的区域：小图直接整图；大图按"文字笔画能量"挑若干块。"""
    H, W = img_full.shape[:2]
    if max(H, W) <= 2600 or scale <= 0:
        return [(0, 0, W, H)]
    gh = gw = 4
    scored: List[Tuple[float, Tuple[int, int, int, int]]] = []
    th, tw = tn.shape[:2]
    for gy in range(gh):
        for gx in range(gw):
            x0, x1 = int(gx * W / gw), int((gx + 1) * W / gw)
            y0, y1 = int(gy * H / gh), int((gy + 1) * H / gh)
            tx0, tx1 = int(x0 * scale), max(int(x0 * scale) + 1, int(x1 * scale))
            ty0, ty1 = int(y0 * scale), max(int(y0 * scale) + 1, int(y1 * scale))
            sub = tn[max(0, ty0):min(th, ty1), max(0, tx0):min(tw, tx1)]
            scored.append((float(sub.mean()) if sub.size else 0.0, (x0, y0, x1, y1)))
    scored.sort(key=lambda t: -t[0])
    tiles = [box for s, box in scored[:max(1, int(max_tiles))] if s > 0.02]
    return tiles or [(0, 0, W, H)]


# --------------------------------------------------------------------------
# 行合并 → 水印块合并
# --------------------------------------------------------------------------
@dataclass
class WatermarkGroup:
    """一个"水印块"：可能由多行文字 / 账号 / 相邻小图标组成。"""

    rect: Tuple[float, float, float, float, float]
    items: List[TextEvidence] = field(default_factory=list)
    text: str = ""
    source: str = "OCR"
    template_id: int = -1
    instances: int = 1
    lattice: bool = False
    from_tiling: bool = False
    confidence: float = 0.0
    tier: str = "low"
    kind: str = ""
    reasons: List[str] = field(default_factory=list)
    on_face: bool = False
    selected: bool = False

    @property
    def center(self) -> Tuple[float, float]:
        return (self.rect[0], self.rect[1])

    @property
    def width(self) -> float:
        return self.rect[2]

    @property
    def height(self) -> float:
        return self.rect[3]

    @property
    def theta(self) -> float:
        return self.rect[4]

    @property
    def poly(self) -> np.ndarray:
        return _poly_of_rect(self.rect)

    @property
    def area(self) -> float:
        return float(self.width * self.height)


def _merge_two_rects(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float, float, float, float]:
    pts = np.vstack([_poly_of_rect(a), _poly_of_rect(b)])
    return _rect_from_points(pts)


def _same_line(line_rect: Sequence[float], line_items: Sequence[TextEvidence],
               it: TextEvidence) -> bool:
    lr = line_rect
    if abs(_norm_angle(it.theta - lr[4])) > 16.0:
        return False
    h = max(4.0, min(lr[3], it.height))
    dx, dy = _local_frame(lr[0], lr[1], lr[4], it.center[0], it.center[1])
    if abs(dy) > 0.85 * h:
        return False
    gap = abs(dx) - (lr[2] + it.width) / 2.0
    return gap <= 2.4 * h


def _same_block(block_rect: Sequence[float], line_rect: Sequence[float]) -> bool:
    if abs(_norm_angle(line_rect[4] - block_rect[4])) > 16.0:
        return False
    h_avg = max(4.0, (block_rect[3] + line_rect[3]) / 2.0)
    ratio = line_rect[3] / max(4.0, block_rect[3])
    if not (0.35 <= ratio <= 2.8):
        return False
    dx, dy = _local_frame(block_rect[0], block_rect[1], block_rect[4],
                          line_rect[0], line_rect[1])
    if abs(dy) > 2.0 * h_avg:
        return False
    # 垂直相邻（不是同一条线）或高度重叠
    if abs(dy) < 0.55 * h_avg:
        return False
    overlap = min(block_rect[0] + block_rect[2] / 2.0, line_rect[0] + line_rect[2] / 2.0) - \
              max(block_rect[0] - block_rect[2] / 2.0, line_rect[0] - line_rect[2] / 2.0)
    if overlap <= 0:
        return False
    overlap_ratio = overlap / max(1.0, min(block_rect[2], line_rect[2]))
    center_off = abs(dx)
    return overlap_ratio >= 0.30 or center_off <= 0.55 * max(block_rect[2], line_rect[2])


def _group_order_text(items: Sequence[TextEvidence]) -> str:
    ordered = sorted(items, key=lambda i: (round(i.center[1] / max(1.0, i.height)), i.center[0]))
    return " ｜ ".join(t.text for t in ordered if t.text.strip())[:80]


def group_evidence(items: Sequence[TextEvidence], shape: Tuple[int, int]) -> List[WatermarkGroup]:
    """先把同一行的碎块并成行，再把相邻行并成"水印块"。"""
    H, W = int(shape[0]), int(shape[1])
    lines: List[Dict[str, Any]] = []
    for it in sorted(items, key=lambda x: -x.score):
        placed = False
        for ln in lines:
            if _same_line(ln["rect"], ln["items"], it):
                ln["items"].append(it)
                ln["rect"] = _merge_two_rects(ln["rect"], it.rect)
                placed = True
                break
        if not placed:
            lines.append({"rect": it.rect, "items": [it]})

    blocks: List[Dict[str, Any]] = []
    for ln in sorted(lines, key=lambda l: -l["rect"][2]):
        placed = False
        for blk in blocks:
            if _same_block(blk["rect"], ln["rect"]):
                merged = _merge_two_rects(blk["rect"], ln["rect"])
                if merged[3] <= 0.5 * H and merged[2] <= 1.15 * W:
                    blk["rect"] = merged
                    blk["lines"].append(ln)
                    placed = True
                    break
        if not placed:
            blocks.append({"rect": ln["rect"], "lines": [ln]})

    groups: List[WatermarkGroup] = []
    for blk in blocks:
        line_items = [it for ln in blk["lines"] for it in ln["items"]]
        rect = _clip_rect(blk["rect"], (H, W))
        src = "OCR"
        if any("局部" in it.source for it in line_items):
            src = "OCR（局部放大）"
        if any("旋转" in it.source for it in line_items):
            src = "OCR（旋转/竖排）"
        groups.append(WatermarkGroup(
            rect=rect, items=line_items,
            text=_group_order_text(line_items), source=src,
        ))
    return groups


# --------------------------------------------------------------------------
# 模板聚类 + 重复实例展开
# --------------------------------------------------------------------------
def _template_similar(a: WatermarkGroup, b: WatermarkGroup) -> bool:
    if a.text.strip() and b.text.strip():
        # 模板聚类要比"同一处重复读到的去重"更严格：
        # "左上角标记" 与 "右上角标记" 只差一个字，但它们是**两个不同的水印**，
        # 相似度放宽到 0.80 会把它们当成同一模板，进而在整幅图上乱匹配。
        if not _similar_text(a.text, b.text, ratio_thr=0.92):
            return False
    elif bool(a.text.strip()) != bool(b.text.strip()):
        return False
    rw = a.width / max(1.0, b.width)
    rh = a.height / max(1.0, b.height)
    if not (0.55 <= rw <= 1.8 and 0.55 <= rh <= 1.8):
        return False
    return abs(_norm_angle(a.theta - b.theta)) <= 20.0


def cluster_templates(groups: Sequence[WatermarkGroup]) -> List[List[int]]:
    """按"同一水印模板"聚类，返回每类的下标列表。"""
    parent = list(range(len(groups)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            if _template_similar(groups[i], groups[j]):
                union(i, j)
    buckets: Dict[int, List[int]] = {}
    for i in range(len(groups)):
        buckets.setdefault(find(i), []).append(i)
    return list(buckets.values())


def _binary_patch(tn: np.ndarray, rect: Sequence[float], thr: float, pad: float = 0.25
                  ) -> Tuple[Optional[np.ndarray], Tuple[int, int]]:
    """取旋转矩形对应的轴对齐笔画块（用于模板匹配）。"""
    x, y, w, h = _envelope(rect)
    H, W = tn.shape[:2]
    px = int(round(w * pad))
    py = int(round(h * pad))
    x0, y0 = max(0, x - px), max(0, y - py)
    x1, y1 = min(W, x + w + px), min(H, y + h + py)
    if x1 - x0 < 6 or y1 - y0 < 6:
        return None, (0, 0)
    patch = (tn[y0:y1, x0:x1] > thr).astype(np.float32)
    if float(patch.sum()) < 12:
        return None, (0, 0)
    return patch, (x0, y0)


def _nms_peaks(scores: np.ndarray, thr: float, template_shape: Tuple[int, int],
               limit: int = 60) -> List[Tuple[int, int, float]]:
    th, tw = template_shape
    h, w = scores.shape
    peaks: List[Tuple[int, int, float]] = []
    work = scores.copy()
    for _ in range(limit):
        idx = int(np.argmax(work))
        py, px = np.unravel_index(idx, work.shape)
        val = float(work[py, px])
        if val < thr:
            break
        peaks.append((int(px), int(py), val))
        y0 = max(0, py - th // 2)
        y1 = min(h, py + th - th // 2)
        x0 = max(0, px - tw // 2)
        x1 = min(w, px + tw - tw // 2)
        work[y0:y1, x0:x1] = -1.0
    return peaks


def _lattice_vectors(centers: Sequence[Tuple[float, float]], scale_ref: float
                     ) -> List[np.ndarray]:
    """从实例中心点估计点阵基向量（最近邻位移中出现次数最多的两个方向）。"""
    if len(centers) < 3:
        return []
    offsets: List[Tuple[float, float]] = []
    for i, a in enumerate(centers):
        for j, b in enumerate(centers):
            if i >= j:
                continue
            dx, dy = b[0] - a[0], b[1] - a[1]
            if 0 < math.hypot(dx, dy) <= 1.6 * scale_ref:
                offsets.append((dx, dy))
    if not offsets:
        return []
    clusters: List[List[Tuple[float, float]]] = []
    for off in sorted(offsets, key=lambda o: math.hypot(*o)):
        placed = False
        for cl in clusters:
            cx = sum(o[0] for o in cl) / len(cl)
            cy = sum(o[1] for o in cl) / len(cl)
            if math.hypot(off[0] - cx, off[1] - cy) <= 0.25 * scale_ref:
                cl.append(off)
                placed = True
                break
        if not placed:
            clusters.append([off])
    clusters.sort(key=lambda cl: -len(cl))
    vecs: List[np.ndarray] = []
    for cl in clusters:
        if len(cl) < 2:
            continue
        v = np.array([sum(o[0] for o in cl) / len(cl), sum(o[1] for o in cl) / len(cl)], np.float32)
        norm = float(np.linalg.norm(v))
        if norm < 6:
            continue
        if any(abs(float(np.dot(v / norm, u / max(1e-6, float(np.linalg.norm(u)))))) > 0.93
               for u in vecs):
            continue
        vecs.append(v)
        if len(vecs) >= 2:
            break
    return vecs


def expand_repeated_instances(
    tn: np.ndarray,
    thr: float,
    groups: List[WatermarkGroup],
    clusters: Sequence[Sequence[int]],
    shape: Tuple[int, int],
    max_new: int = 40,
) -> Tuple[List[WatermarkGroup], Dict[str, float]]:
    """模板匹配 + 点阵展开：把同一模板的**全部实例**补齐。"""
    t0 = time.time()
    H, W = int(shape[0]), int(shape[1])
    added: List[WatermarkGroup] = []
    matched = 0
    lattice_added = 0

    for cl in clusters:
        if len(cl) < 2 or len(added) >= max_new:
            continue
        idx = sorted(cl, key=lambda i: -groups[i].confidence) or list(cl)
        best = groups[idx[0]]
        patch, (px0, py0) = _binary_patch(tn, best.rect, thr)
        if patch is None:
            continue
        ph, pw = patch.shape
        if ph < 6 or pw < 6:
            continue
        if ph >= H or pw >= W:
            continue

        # ---- ① 模板匹配：找同模板的其它位置 ----
        try:
            res = cv2.matchTemplate(tn, patch, cv2.TM_CCORR_NORMED)
        except Exception as exc:  # pragma: no cover
            logger.debug("模板匹配失败：%s", exc)
            res = None
        rect_wh = (best.width, best.height)
        tpl_bin = patch > 0
        tpl_count = float(np.count_nonzero(tpl_bin)) or 1.0
        tpl_mean = float(tn[int(py0):int(py0) + ph, int(px0):int(px0) + pw].mean())
        if res is not None:
            for px, py, val in _nms_peaks(res, 0.55, patch.shape, limit=40):
                if len(added) >= max_new:
                    break
                cx = px + pw / 2.0
                cy = py + ph / 2.0
                cand_rect = (cx, cy, rect_wh[0], rect_wh[1], best.theta)
                if any(_iou_rect(cand_rect, g.rect) > 0.35 for g in groups):
                    continue
                if any(_iou_rect(cand_rect, g.rect) > 0.35 for g in added):
                    continue
                region = tn[max(0, py):py + ph, max(0, px):px + pw]
                if region.shape != patch.shape:
                    continue
                reg_bin = region > thr
                # 真实实例：模板笔画应大部分被该处的笔画覆盖，且笔画量相当
                overlap = float(np.count_nonzero(tpl_bin & reg_bin)) / tpl_count
                amount = float(np.count_nonzero(reg_bin)) / tpl_count
                if overlap < 0.55 or amount < 0.60:
                    continue
                if float(region.mean()) < 0.40 * max(1e-6, tpl_mean):
                    continue
                matched += 1
                added.append(WatermarkGroup(
                    rect=_clip_rect(cand_rect, (H, W)),
                    text=best.text, source="重复模板匹配", template_id=best.template_id,
                    instances=1, from_tiling=True,
                    reasons=[f"与已识别水印模板一致（相似度 {val:.2f}）"],
                ))

        # ---- ② 点阵展开：按最近邻位移把网格补齐 ----
        centers = [(groups[i].center) for i in cl]
        vecs = _lattice_vectors(centers, max(best.width, best.height) * 1.6)
        if vecs:
            ref_support = float(tn[int(py0):int(py0) + ph, int(px0):int(px0) + pw].mean())
            origin = np.array(best.center, np.float32)
            for e1 in vecs:
                for k in range(1, 8):
                    v1 = e1 * k
                    for e2 in (vecs if len(vecs) > 1 else [np.zeros(2, np.float32)]):
                        for m in range(0, 4):
                            if len(added) >= max_new:
                                break
                            center = origin + v1 + (e2 * m if e2 is not None else 0)
                            cx, cy = float(center[0]), float(center[1])
                            if not (0 <= cx < W and 0 <= cy < H):
                                continue
                            cand_rect = (cx, cy, rect_wh[0], rect_wh[1], best.theta)
                            if any(_iou_rect(cand_rect, g.rect) > 0.35 for g in groups):
                                continue
                            if any(_iou_rect(cand_rect, g.rect) > 0.35 for g in added):
                                continue
                            x, y, w, h = _envelope(cand_rect)
                            x0, y0 = max(0, x), max(0, y)
                            x1, y1 = min(W, x + w), min(H, y + h)
                            if x1 - x0 < 6 or y1 - y0 < 6:
                                continue
                            support = float(tn[y0:y1, x0:x1].mean())
                            if support < 0.55 * max(1e-6, ref_support):
                                continue
                            lattice_added += 1
                            added.append(WatermarkGroup(
                                rect=_clip_rect(cand_rect, (H, W)),
                                text=best.text, source="点阵展开", template_id=best.template_id,
                                instances=1, lattice=True, from_tiling=True,
                                reasons=["沿同一水印的排列周期补齐"],
                            ))
    return added, {
        "template_ms": round((time.time() - t0) * 1000.0, 1),
        "template_matches": float(matched),
        "lattice_matches": float(lattice_added),
    }


# --------------------------------------------------------------------------
# 打分
# --------------------------------------------------------------------------
_KIND_LABEL = {
    "url": "URL/网址水印", "date": "日期水印", "time": "时间水印",
    "copyright": "版权文字水印", "platform": "平台名水印",
    "at": "@账号水印", "digits": "数字水印",
}


def _text_like_visual(group: WatermarkGroup, area_total: float) -> bool:
    """判断"没有 OCR 证据的视觉候选"是否长得像一行字。

    只保留：**细长**（长短边比 ≥ 2.0）且**小**（不超过画面 1.5%）的结构。
    皮肤高光、衣服褶皱、云层渐变都不满足这个形状条件，因此会被挡在候选之外；
    真正被漏读的平铺文字水印仍然是"细长的行"，不会因此被误杀。
    """
    w, h = max(1.0, float(group.width)), max(1.0, float(group.height))
    long_side, short_side = max(w, h), max(1.0, min(w, h))
    if long_side / short_side < 2.0:
        return False
    return (w * h) / max(1.0, float(area_total)) <= 0.015


def _position_bonus(rect: Sequence[float], shape: Tuple[int, int]) -> Tuple[str, float]:
    H, W = int(shape[0]), int(shape[1])
    cx, cy = rect[0] / max(1.0, W), rect[1] / max(1.0, H)
    ex, ey = min(cx, 1 - cx), min(cy, 1 - cy)
    if ex < 0.16 and ey < 0.20:
        return "角落", 0.16
    if ex < 0.18 or ey < 0.18:
        return "边缘带", 0.11
    if 0.28 < cx < 0.72 and 0.22 < cy < 0.78:
        # 居中只做轻微降权：实际使用中"压在人物/画面中间的平铺水印"非常常见，
        # 降权过重会导致这类水印被漏掉（第六轮反馈的主要问题之一）。
        return "画面中央", -0.05
    return "画面中部", 0.0


def score_group(
    group: WatermarkGroup,
    shape: Tuple[int, int],
    select_threshold: float,
    translucent_overlap: float = 0.0,
) -> None:
    """计算 Watermark Score（就地写入 ``confidence / tier / kind / reasons``）。"""
    H, W = int(shape[0]), int(shape[1])
    area_total = float(H * W)
    items = group.items
    texts = [it.text for it in items if it.text.strip()]
    mean_score = float(np.mean([it.score for it in items])) if items else 0.35
    prior = float(max((it.prior for it in items), default=0.45)) if items else 0.45

    kinds = [classify_text(t)[0] for t in texts] or ["text"]
    kind_counts: Dict[str, int] = {}
    for k in kinds:
        kind_counts[k] = kind_counts.get(k, 0) + 1
    kind = max(kind_counts.items(), key=lambda kv: kv[1])[0]

    conf = 0.0
    reasons: List[str] = []

    conf += 0.42 * max(0.35, prior)
    if prior >= 0.7:
        reasons.append("文字类型高度符合水印特征")

    conf += 0.16 * max(0.0, min(1.0, mean_score))

    pos_name, bonus = _position_bonus(group.rect, (H, W))
    conf += bonus
    reasons.append(f"位于{pos_name}")

    h_ratio = group.height / max(1.0, H)
    if h_ratio < 0.05:
        conf += 0.07
        reasons.append("字号偏小（常见水印尺寸）")
    elif h_ratio < 0.09:
        conf += 0.03
    elif h_ratio > 0.22:
        conf -= 0.08
        reasons.append("字号很大（更像画面正文）")

    n_lines = len({round(it.center[1] / max(1.0, it.height)) for it in items}) or 1
    if len(items) >= 2 and n_lines >= 2:
        conf += 0.07
        reasons.append("多行文字块（账号型水印常见）")

    if group.instances >= 2:
        conf += 0.16
        reasons.append(f"同模板出现 {group.instances} 次")
    if group.instances >= 4:
        conf += 0.08
        reasons.append("平铺/重复出现")
    if group.lattice:
        conf += 0.05
        reasons.append("符合周期性排列")
    if group.from_tiling and not items:
        conf -= 0.05
        reasons.append("仅由重复结构推断（无 OCR 证据）")

    if translucent_overlap > 0.25:
        conf += 0.06
        reasons.append("检测到半透明叠加层")

    if abs(_norm_angle(group.theta)) > 8.0:
        conf += 0.02
        reasons.append(f"文字倾斜 {abs(_norm_angle(group.theta)):.0f}°")

    joined = " ".join(texts)
    shape_len = len(_text_shape(joined))

    # ---- 明确文字证据（本轮新增）：OCR 已经清楚读出来的文字不允许被"水印评分"丢掉 ----
    # 依据：英文字母/数字占比高、OCR 置信度高、字符数足够 —— 这是文字水印最可靠的证据。
    strong_text = bool(texts) and mean_score >= 0.60 and shape_len >= 3
    latin_like = _alnum_ratio(joined) >= 0.6
    cjk_len = len(_RE_CJK.findall(joined))
    # "散文样长句"（≥8 个汉字）更像海报标题/字幕，不做保底、仍按水印特征分排序
    prose_like = _is_cjk_dominant(joined) and cjk_len >= 8
    if strong_text:
        conf += 0.10
        reasons.append(f"OCR 明确识别到文字（置信度 {mean_score:.2f}）")
        if latin_like and len(texts) >= 1:
            conf += 0.04
            reasons.append("英文/数字为主的字符串（常见水印）")

    # 长句惩罚只针对**中日韩长句**（海报标题、字幕）；
    # 英文/数字水印（如 "GALAXY RI TA"、"www.demo.com"）本身就可能很长，不能再惩罚。
    if (kind == "text" and shape_len >= 10 and group.instances < 2 and bonus < 0
            and prose_like):
        conf -= 0.10
        reasons.append("中文长句且不在边缘，更像画面正文")

    # ---- 文字保底：读出来的文字至少给到"中置信度"，避免 A 模式整体漏检 ----
    # 仅对"水印样字符串"生效：英文/数字为主（URL、账号、日期、GALAXY RI TA…）
    # 或很短的标签文字；**长中文句子不做保底**，避免把海报标题当成水印。
    if strong_text and (latin_like or not prose_like) and conf < 0.62:
        conf = 0.62
        reasons.append("文字保底：OCR 已确认是文字，不因水印特征分偏低而丢弃")
    if group.area / area_total > 0.22:
        conf -= 0.15
        reasons.append("区域过大（更像画面内容）")

    if group.on_face:
        conf *= 0.88
        reasons.append("与人脸/五官重叠，需人工确认")

    conf = float(np.clip(conf, 0.0, 0.98))
    group.confidence = conf
    group.reasons = reasons
    tier = "high" if conf >= select_threshold else ("medium" if conf >= TIER_MEDIUM_MIN else "low")
    group.kind = _KIND_LABEL.get(kind, "水印文字候选")
    if group.instances >= 3:
        group.kind = "重复/平铺" + group.kind
    # 只有"确实低置信度"的普通文字才标注为"普通图片文字"，
    # 早期写法读的是上一次的 tier（默认 low），会把高置信度水印也标成低置信度文字。
    if tier == "low" and kind == "text" and prior < 0.5 and group.instances < 2:
        group.kind = "普通图片文字（低置信度）"
    group.tier = tier


# --------------------------------------------------------------------------
# Mask 生成（贴合笔画，而不是大矩形）
# --------------------------------------------------------------------------
def _stroke_mask_in_poly(img_full: np.ndarray, poly: np.ndarray, dilate: int) -> np.ndarray:
    """在旋转矩形内取"文字笔画"，得到贴合水印的 Mask。"""
    H, W = img_full.shape[:2]
    x, y, w, h = _envelope(poly)
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    mask = np.zeros((H, W), np.uint8)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return mask
    roi = np.ascontiguousarray(img_full[y0:y1, x0:x1])
    if roi.ndim == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        # 彩色对比：橙色/红色水印在灰度上可能和皮肤几乎一样亮，
        # 只看灰度会把这类像素整片漏掉（实测：橙色小图标 + 红字阴影没进 Mask）。
        gray_soft = cv2.GaussianBlur(gray, (3, 3), 0)
        k_c = _odd(int(max(7, min(gray.shape[:2]) * 0.08)), 5)
        d_gray = local_contrast(gray_soft, k_c)
        d_col = np.zeros_like(d_gray)
        for ch in range(3):
            d_col = np.maximum(d_col, local_contrast(cv2.GaussianBlur(roi[..., ch], (3, 3), 0), k_c))
        d = np.maximum(d_gray, d_col)
    else:
        gray = cv2.GaussianBlur(roi, (3, 3), 0)
        k_c = _odd(int(max(7, min(gray.shape[:2]) * 0.08)), 5)
        d = local_contrast(gray, k_c)
    gray = cv2.GaussianBlur(cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY) if roi.ndim == 3 else roi,
                            (3, 3), 0)
    k = _odd(int(max(7, min(gray.shape[:2]) * 0.08)), 5)
    region = np.zeros(gray.shape[:2], np.uint8)
    local_poly = np.round(poly - np.array([x0, y0], np.float32)).astype(np.int32)
    cv2.fillPoly(region, [local_poly], 255)
    inside = region > 0
    region_px = int(np.count_nonzero(inside))
    if region_px <= 0:
        return mask
    p99 = float(np.percentile(d[inside], 99.0)) if region_px > 0 else 0.0
    thr = max(4.0, 0.32 * p99)
    strokes = (d > thr) & inside
    stroke_px = int(np.count_nonzero(strokes))
    if stroke_px >= max(16, int(0.045 * region_px)):
        local = (strokes.astype(np.uint8)) * 255

        # ---- 填"字身空洞"（本轮修残留的第一步，也是最关键的一步）----
        # 局部对比度只能标记笔画的**边缘**：一个大字号粗笔画（例如 44px 的白字）
        # 内部是均匀色块，|灰度 − 中值背景| ≈ 0，于是字身中间会留下空洞，
        # LaMa 只拿到"字的外框"，字身就残留在结果里（实测：只覆盖水印脚印的 72%）。
        # 做法：小半径闭运算把笔画边缘连起来，再把**被包围的内部空洞**填实。
        # 半径取文字高度的 4%（1~4px），远小于字间距，因此不会把相邻字粘成一块矩形。
        r = int(np.clip(round(0.06 * min(gray.shape[:2])), 1, 6))
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        closed = cv2.morphologyEx(local, cv2.MORPH_CLOSE, kern)
        closed = cv2.bitwise_and(closed, region)          # 严格限制在 OCR 框内
        filled = _fill_holes(closed)
        filled = cv2.bitwise_and(filled, region)
        local = filled

        # ---- 半透明/描边文字的"边缘余量"（本轮修正残留问题的关键）----
        # 文字水印常见：抗锯齿边、描边外沿、投影、半透明羽化边。
        # 它们的对比度明显低于笔画核心，只取"高对比笔画"会漏掉这些像素，
        # 修复后就会留下黑色/彩色描边残影（实测只覆盖水印脚印的 72%）。
        # 做法：在中低对比阈值上加一圈**紧贴笔画**的余量，余量宽度按文字高度
        # 取 12%（上限 12px），因此仍然是"贴着文字"，而不是变成一个大矩形。
        reach = int(np.clip(round(0.10 * min(gray.shape[:2])), 2, 10))
        halo_thr = max(3.5, 0.22 * p99)
        halo = (d > halo_thr) & inside
        near = mp.dilate_mask(local, reach) > 0
        local = np.where(halo & near, 255, local).astype(np.uint8)
    else:
        local = region                      # 对比度太弱（极淡水印）→ 退回整块
    if dilate:
        local = mp.dilate_mask(local, int(dilate))
    mask[y0:y1, x0:x1] = local
    return mask


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    """把二值掩膜里"被前景包围的空洞"填实（例如粗笔画的字身、字母 o/a 的内孔）。"""
    m = mp.ensure_binary(mask)
    if mp.is_empty(m):
        return m
    h, w = m.shape[:2]
    ff = m.copy()
    cv2.floodFill(ff, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 255)
    holes = cv2.bitwise_not(ff)
    return cv2.bitwise_or(m, holes)


def group_mask(img_full: np.ndarray, group: WatermarkGroup, dilate: int = 3) -> np.ndarray:
    """一个水印块的 Mask：每条文字各自的贴合笔画 + 相邻小图标。"""
    H, W = img_full.shape[:2]
    mask = np.zeros((H, W), np.uint8)
    if group.items:
        for it in group.items:
            mask = cv2.bitwise_or(mask, _stroke_mask_in_poly(img_full, it.poly, 0))
        # 水印块内、靠近文字的小图标 / 勾选符号：用块的旋转矩形补上（限制在块内）
        block = np.zeros((H, W), np.uint8)
        cv2.fillPoly(block, [np.round(group.poly).astype(np.int32)], 255)
        near = mp.dilate_mask(mask, int(max(4, round(0.8 * max(6.0, group.height / max(1, len(group.items)))))))
        extra = cv2.bitwise_and(block, near)
        mask = cv2.bitwise_or(mask, extra)
    else:
        mask = _stroke_mask_in_poly(img_full, group.poly, 0)
    if mp.is_empty(mask):
        mask = np.zeros((H, W), np.uint8)
        cv2.fillPoly(mask, [np.round(group.poly).astype(np.int32)], 255)
    if dilate:
        mask = mp.dilate_mask(mask, int(dilate))
    return mask


# --------------------------------------------------------------------------
# 对外结果结构（与旧版 watermark_candidate 完全兼容）
# --------------------------------------------------------------------------
@dataclass
class WatermarkCandidate:
    bbox: Tuple[int, int, int, int]
    kind: str
    confidence: float
    source: str
    text: str = ""
    reasons: List[str] = field(default_factory=list)
    on_face: bool = False
    selected: bool = False
    tier: str = "medium"
    template_id: int = -1
    instances: int = 1
    angle: float = 0.0
    poly: Optional[np.ndarray] = None
    from_tiling: bool = False

    @property
    def area(self) -> int:
        return int(self.bbox[2] * self.bbox[3])

    def to_row(self) -> List[object]:
        x, y, w, h = self.bbox
        tier = {"high": "高", "medium": "中", "low": "低"}.get(self.tier, self.tier)
        return [
            "选中" if self.selected else "",
            self.kind,
            tier,
            f"{self.confidence * 100:.0f}%",
            self.source,
            (self.text[:24] + "...") if len(self.text) > 24 else self.text,
            f"({x},{y}) {w}x{h}",
            f"{self.area / 1000:.1f}k px",
            "；".join(self.reasons[:3]),
        ]


@dataclass
class WatermarkReport:
    candidates: List[WatermarkCandidate] = field(default_factory=list)
    mask_all: Optional[np.ndarray] = None
    mask_selected: Optional[np.ndarray] = None
    summary: str = ""
    warnings: List[str] = field(default_factory=list)
    stats: Dict[str, object] = field(default_factory=dict)
    face_message: str = ""
    timings: Dict[str, float] = field(default_factory=dict)
    settings: Dict[str, object] = field(default_factory=dict)

    @property
    def high_confidence(self) -> List[WatermarkCandidate]:
        return [c for c in self.candidates if c.selected]

    def counts(self) -> Dict[str, int]:
        out = {"high": 0, "medium": 0, "low": 0}
        for c in self.candidates:
            if c.tier in out:
                out[c.tier] += 1
        return out


#: 兼容旧名字（界面 / 测试里用的是 CandidateReport）
CandidateReport = WatermarkReport


@dataclass
class DetectSettings:
    """自动检测参数。"""

    sensitivity: str = DEFAULT_SENSITIVITY
    max_side: int = 1500
    ocr_min_score: float = 0.40
    min_confidence: float = 0.22
    select_threshold: float = 0.50
    select_tiers: Tuple[str, ...] = ("high", "medium")
    max_area_ratio: float = 0.35
    max_region_ocr: int = 20
    max_candidates: int = 96
    dilate: int = 3
    protect_faces: bool = True
    use_region_ocr: bool = True
    use_tiling: bool = True
    use_translucent: bool = True
    sweep_angles: Tuple[float, ...] = (-25.0, 25.0, -45.0, 45.0)
    sweep_side: int = 1100
    sweep_budget_ms: float = 4500.0
    region_budget_ms: float = 6000.0
    #: 高清 OCR 兜底：缩图 OCR 没读全时，按需在原分辨率（或分块）再读一次
    hi_res_ocr: bool = True
    #: 大图分块时最多读几块（按文字笔画能量挑最像文字的块）
    hi_tiles: int = 6
    #: 高清 OCR 的时间预算
    hi_budget_ms: float = 8000.0
    #: 高清 OCR 的置信度阈值（比整图略低，但不过度放宽）
    hi_min_score: float = 0.35
    #: 是否允许"没有 OCR 证据的视觉候选"进入默认勾选（默认否：避免皮肤/衣服误检）
    allow_visual_only: bool = False
    #: 半透明响应覆盖率超过该比例时，判定响应不可信（皮肤/天空渐变会大面积触发）
    visual_max_ratio: float = 0.25
    #: 局部高分辨率 OCR（LOHR）：只在人脸框 / 笔画聚类等**小 ROI** 上放大 2~4 倍再读，
    #: 用于解决"512×1080 人像上 24px 小字"这类漏检；不影响全局阈值与任何 Mask。
    local_ocr: bool = True
    local_ocr_scale: float = 3.0
    local_ocr_pad: float = 0.60
    local_ocr_budget_ms: float = 4500.0
    local_ocr_max: int = 6

    def as_dict(self) -> Dict[str, object]:
        return {
            "sensitivity": self.sensitivity,
            "max_side": self.max_side,
            "ocr_min_score": self.ocr_min_score,
            "min_confidence": self.min_confidence,
            "select_threshold": self.select_threshold,
            "select_tiers": list(self.select_tiers),
            "max_area_ratio": self.max_area_ratio,
            "dilate": self.dilate,
            "protect_faces": self.protect_faces,
            "sweep_angles": [float(a) for a in self.sweep_angles],
            "hi_res_ocr": self.hi_res_ocr,
            "allow_visual_only": self.allow_visual_only,
            "local_ocr": self.local_ocr,
            "local_ocr_scale": self.local_ocr_scale,
        }


def resolve_settings(
    sensitivity: Optional[str] = None,
    min_confidence: Optional[float] = None,
    select_threshold: Optional[float] = None,
    max_area_ratio: Optional[float] = None,
    dilate: Optional[int] = None,
    ocr_min_score: Optional[float] = None,
    protect_faces: bool = True,
    max_side: Optional[int] = None,
    **extra: Any,
) -> DetectSettings:
    """按敏感度预设生成设置；显式给出的参数覆盖预设。"""
    key = str(sensitivity or DEFAULT_SENSITIVITY).strip().lower()
    if key not in SENSITIVITY_PRESETS:
        for name, preset in SENSITIVITY_PRESETS.items():
            if key == str(preset["label"]):
                key = name
                break
    preset = SENSITIVITY_PRESETS.get(key, SENSITIVITY_PRESETS[DEFAULT_SENSITIVITY])
    st = DetectSettings(
        sensitivity=key if key in SENSITIVITY_PRESETS else DEFAULT_SENSITIVITY,
        min_confidence=float(preset["min_confidence"]),
        select_threshold=float(preset["select_threshold"]),
        select_tiers=tuple(preset["select_tiers"]),
        max_area_ratio=float(preset["max_area_ratio"]),
        max_region_ocr=int(preset["max_region_ocr"]),
        max_candidates=int(preset["max_candidates"]),
        sweep_angles=tuple(float(a) for a in preset.get("sweep_angles", ())),
    )
    if min_confidence is not None:
        st.min_confidence = float(min_confidence)
    if select_threshold is not None:
        st.select_threshold = float(select_threshold)
    if max_area_ratio is not None:
        st.max_area_ratio = float(max_area_ratio)
    if dilate is not None:
        st.dilate = int(dilate)
    if ocr_min_score is not None:
        st.ocr_min_score = float(ocr_min_score)
    if max_side:
        st.max_side = int(max_side)
    st.protect_faces = bool(protect_faces)
    for k, v in extra.items():
        if hasattr(st, k) and v is not None:
            setattr(st, k, v)
    return st


# --------------------------------------------------------------------------
# 主入口
# --------------------------------------------------------------------------
_DETECTOR: Optional[wd.WatermarkDetector] = None

# --------------------------------------------------------------------------
# 检测结果缓存（本轮性能优化）
# --------------------------------------------------------------------------
#: 同一张图（内容未变）+ 同一套参数 → 直接复用上次的检测结果。
#: 作用：Mask 预览与"开始修复"共用一次检测；用户重复点"自动检测"不再重跑；
#: 连续处理里"结果没变"的图也命中缓存。图片内容一变（指纹变化）自动失效。
_REPORT_CACHE: "OrderedDict[str, WatermarkReport]" = OrderedDict()
_REPORT_CACHE_MAX = 6


def _image_key(img: np.ndarray) -> str:
    """图片内容指纹：48×48 灰度缩略图（量化到 16 级）+ 尺寸。

    量化是必要的：同一张图经过"上传归一化 / PIL 往返 / 重新编码"后个别像素会差 1~2，
    不量化会永远命不中缓存；量化到 16 级后仍然能可靠区分"内容真的变了"
    （例如修复后的结果图，差异远大于 16 灰阶）。
    """
    small = cv2.resize(img, (48, 48), interpolation=cv2.INTER_AREA)
    if small.ndim == 3:
        small = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    q = (small.astype(np.uint16) >> 4).astype(np.uint8)
    return f"{img.shape[0]}x{img.shape[1]}:{hashlib.md5(q.tobytes()).hexdigest()}"


def _settings_key(st: "DetectSettings", mode: str) -> str:
    return (f"{mode}|{st.sensitivity}|{st.min_confidence:.3f}|{st.select_threshold:.3f}|"
            f"{st.dilate}|{st.ocr_min_score:.2f}|{int(st.protect_faces)}|{int(st.hi_res_ocr)}|"
            f"{int(st.local_ocr)}|{st.local_ocr_scale}|{int(st.use_tiling)}|"
            f"{int(st.use_translucent)}|{int(st.use_region_ocr)}")


def _clone_report(rep: WatermarkReport) -> WatermarkReport:
    """复制一份报告（掩膜与候选都要独立，避免调用方修改影响缓存）。"""
    cands: List[WatermarkCandidate] = []
    for c in rep.candidates:
        cands.append(WatermarkCandidate(
            bbox=c.bbox, kind=c.kind, confidence=c.confidence, source=c.source,
            text=c.text, reasons=list(c.reasons), on_face=c.on_face, selected=c.selected,
            tier=c.tier, template_id=c.template_id, instances=c.instances,
            angle=c.angle, poly=None if c.poly is None else c.poly.copy(),
            from_tiling=c.from_tiling,
        ))
    return WatermarkReport(
        candidates=cands,
        mask_all=None if rep.mask_all is None else rep.mask_all.copy(),
        mask_selected=None if rep.mask_selected is None else rep.mask_selected.copy(),
        summary=rep.summary, warnings=list(rep.warnings), stats=dict(rep.stats),
        face_message=rep.face_message, timings=dict(rep.timings),
        settings=dict(rep.settings),
    )


def clear_report_cache() -> None:
    """清空检测缓存（测试与"换了模型/参数"时用）。"""
    _REPORT_CACHE.clear()


def get_detector() -> wd.WatermarkDetector:
    global _DETECTOR
    if _DETECTOR is None:
        _DETECTOR = wd.WatermarkDetector()
    return _DETECTOR


def detect_watermarks(
    image_rgb: np.ndarray,
    settings: Optional[DetectSettings] = None,
    mode: str = "auto",
    **overrides: Any,
) -> WatermarkReport:
    """自动寻找水印候选（**绝不进入修复**）。

    ``mode``：

    * ``"auto"`` —— A 模式「智能自动识别」：OCR + 重复性 + 位置 + 半透明 + 人脸冲突
      综合评分，按敏感度分层勾选；
    * ``"text"`` —— B 模式「文字全部去除」：**OCR 读到的文字默认全部进入 Mask**，
      不做水印评分过滤，只按识别置信度排序、仍然先给预览与人工确认。
    """
    t_start = time.time()
    st = settings or resolve_settings(**overrides)
    mode = "text" if str(mode).lower() in ("text", "all_text", "text_all") else "auto"
    # ---- 检测结果缓存：同一张图 + 同一套参数直接复用（不影响任何算法）----
    cache_key = ""
    try:
        _img0 = mp.to_uint8(image_rgb)
        cache_key = f"{_image_key(_img0)}#{_settings_key(st, mode)}"
        hit = _REPORT_CACHE.get(cache_key)
        if hit is not None:
            _REPORT_CACHE.move_to_end(cache_key)
            rep = _clone_report(hit)
            rep.timings = dict(rep.timings)
            rep.timings["cache_hit"] = 1.0
            rep.timings["total_ms"] = round((time.time() - t_start) * 1000.0, 1)
            return rep
    except Exception as exc:  # 缓存出错绝不影响主流程
        logger.debug("检测缓存查询失败：%s", exc)
    det = get_detector()
    report = WatermarkReport(settings=st.as_dict())
    report.settings["mode"] = mode
    timings: Dict[str, float] = {}

    img = mp.to_uint8(image_rgb)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    H, W = img.shape[:2]
    area_total = float(H * W)

    # ---- ① 人脸（保护 + 降权提示）----
    t0 = time.time()
    face_mask = critical_mask = None
    face_boxes: List[Tuple[int, int, int, int]] = []
    if st.protect_faces:
        try:
            faces = fp.detect_faces(img)
        except Exception as exc:
            logger.warning("人脸检测失败：%s", exc)
            faces = []
        if faces:
            # 人脸框（按面积从大到小）作为"局部高分辨率 OCR"的优先 ROI：
            # 脸部的小字水印是当前最主要的漏检场景。
            face_boxes = [tuple(int(v) for v in f.box) for f in
                          sorted(faces, key=lambda f: -(f.box[2] * f.box[3]))]
            face_mask, critical_mask = fp.build_face_masks((H, W), faces)
            report.face_message = (
                f"检测到 {len(faces)} 张人脸：这些区域默认受保护，"
                f"与水印候选重叠时会降权并提示人工确认"
            )
    timings["face_ms"] = round((time.time() - t0) * 1000.0, 1)

    # ---- ② 文字笔画图 ----
    t0 = time.time()
    try:
        tn, scale, det_img = textness_map(img, max_side=st.max_side)
    except Exception as exc:
        logger.exception("文字笔画图失败：%s", exc)
        report.summary = "自动检测失败：无法分析图像（请查看日志）"
        report.warnings.append("图像分析失败，请改用手动画笔涂抹。")
        timings["total_ms"] = round((time.time() - t_start) * 1000.0, 1)
        report.timings = timings
        return report
    timings["textness_ms"] = round((time.time() - t0) * 1000.0, 1)
    timings["detect_scale"] = round(float(scale), 4)

    # ---- ③ OCR 证据（全局 + 逐块去倾斜放大）----
    try:
        items, ocr_stats = collect_evidence(
            det, img, tn, scale, det_img,
            ocr_min_score=st.ocr_min_score,
            max_region_ocr=st.max_region_ocr,
            use_region_ocr=st.use_region_ocr,
            sweep_angles=st.sweep_angles,
            sweep_side=st.sweep_side,
            sweep_budget_ms=st.sweep_budget_ms,
            region_budget_ms=st.region_budget_ms,
            hi_res_ocr=st.hi_res_ocr,
            hi_tiles=st.hi_tiles,
            hi_budget_ms=st.hi_budget_ms,
            hi_min_score=st.hi_min_score,
            roi_boxes=face_boxes,
            local_ocr=st.local_ocr,
            local_ocr_scale=st.local_ocr_scale,
            local_ocr_pad=st.local_ocr_pad,
            local_ocr_budget_ms=st.local_ocr_budget_ms,
            local_ocr_max=st.local_ocr_max,
        )
    except Exception as exc:
        logger.exception("OCR 证据收集失败：%s", exc)
        items, ocr_stats = [], {}
    timings.update(ocr_stats)

    # ---- ④ 分组（行 → 水印块）----
    t0 = time.time()
    groups = group_evidence(items, (H, W))
    timings["group_ms"] = round((time.time() - t0) * 1000.0, 1)

    # ---- ⑤ 半透明响应（作为独立证据 + 打分用）----
    trans_mask = None
    if st.use_translucent:
        t0 = time.time()
        try:
            res = det.translucent.detect(det_img, dilate=0)
            if res is not None and res.found and not mp.is_empty(res.mask):
                merge_px = max(3, int(0.012 * min(det_img.shape[:2])))
                trans_mask = mp.dilate_mask(res.mask, merge_px)
        except Exception as exc:
            logger.warning("半透明检测失败：%s", exc)
        timings["translucent_ms"] = round((time.time() - t0) * 1000.0, 1)

    # ---- ⑥ 无 OCR 证据的诊断（复杂背景 / 极淡水印兜底）----
    extra_groups: List[WatermarkGroup] = []
    visual_unreliable = bool(
        trans_mask is not None and not mp.is_empty(trans_mask)
        and mp.area_ratio(trans_mask) > float(st.visual_max_ratio)
    )
    if visual_unreliable:
        # 半透明响应覆盖了整幅图的很大一片（皮肤、天空渐变、布料都会这样），
        # 说明这个"响应"不是水印叠加层——此时**不再**用它生成视觉候选，
        # 否则人体/衣服会被成片框住（本轮实测：格纹布料 56 个高置信度误检）。
        logger.info("半透明响应覆盖率 %.1f%% 超过上限 %.0f%%，本轮不使用视觉兜底候选",
                    mp.area_ratio(trans_mask) * 100.0, st.visual_max_ratio * 100.0)
    if st.use_tiling and not visual_unreliable:
        t0 = time.time()
        try:
            rep = det.repeated.detect(det_img, dilate=0)
        except Exception as exc:
            logger.warning("重复水印检测失败：%s", exc)
            rep = None
        if rep is not None and rep.found and not mp.is_empty(rep.mask):
            n, labels, stats, _ = cv2.connectedComponentsWithStats((rep.mask > 0).astype(np.uint8), 8)
            comps = []
            for i in range(1, n):
                a = int(stats[i, cv2.CC_STAT_AREA])
                x, y, w, h = (int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP]),
                              int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT]))
                if a < max(30, int(0.00008 * area_total * (scale ** 2) if scale else 1)):
                    continue
                if min(w, h) < 4:
                    continue
                if h > 0.35 * det_img.shape[0]:
                    continue
                comps.append((a, (x, y, w, h)))
            comps.sort(key=lambda t: -t[0])
            base_conf = float(getattr(rep, "confidence", 0.0) or 0.0)
            for a, (x, y, w, h) in comps[:16]:
                rect = (x + w / 2.0, y + h / 2.0, float(w), float(h), 0.0)
                extra_groups.append(WatermarkGroup(
                    rect=rect, text="", source="重复结构（无 OCR 证据）",
                    reasons=[f"与其它 {len(comps)} 处结构一致（周期性重复）"],
                ))
        timings["repeated_ms"] = round((time.time() - t0) * 1000.0, 1)

    # ---- ⑦ 模板聚类 + 重复实例展开 ----
    all_groups: List[WatermarkGroup] = list(groups) + list(extra_groups)
    # 同一位置的组去重（OCR 块 vs 重复结构块）
    deduped: List[WatermarkGroup] = []
    for g in sorted(all_groups, key=lambda x: -len(x.items)):
        if any(_iou_rect(g.rect, o.rect) > 0.6 for o in deduped):
            continue
        deduped.append(g)
    all_groups = deduped

    clusters = cluster_templates(all_groups)
    for tid, cl in enumerate(clusters):
        for i in cl:
            all_groups[i].template_id = tid
            all_groups[i].instances = len(cl)

    tpl_stats: Dict[str, float] = {}
    if st.use_tiling and all_groups:
        thr = max(0.16, float(np.percentile(tn, 96.0)) * 0.45)
        try:
            added, tpl_stats = expand_repeated_instances(
                tn, thr, all_groups, clusters, det_img.shape[:2],
            )
        except Exception as exc:
            logger.warning("重复实例展开失败：%s", exc)
            added = []
        for g in added:
            g.instances = max(1, max((all_groups[i].instances for i in clusters[g.template_id]), default=1)
                              if 0 <= g.template_id < len(clusters) else 1)
        all_groups.extend(added)
    timings.update(tpl_stats)

    # ---- ⑧ 打分 ----
    t0 = time.time()
    inv_scale = 1.0 / scale if scale else 1.0
    for g in all_groups:
        overlap = 0.0
        if trans_mask is not None:
            try:
                x, y, w, h = _envelope(g.rect)
                x0, y0 = max(0, x), max(0, y)
                x1, y1 = min(trans_mask.shape[1], x + w), min(trans_mask.shape[0], y + h)
                if x1 > x0 and y1 > y0:
                    sub = trans_mask[y0:y1, x0:x1] > 0
                    overlap = float(np.count_nonzero(sub)) / float(max(1, sub.size))
            except Exception:
                overlap = 0.0
        if critical_mask is not None:
            # 人脸冲突判断要在同一坐标系下比较：把检测坐标放大到原图坐标
            poly = g.poly * inv_scale
            x, y, w, h = _envelope(poly)
            x0, y0 = max(0, x), max(0, y)
            x1, y1 = min(W, x + w), min(H, y + h)
            if x1 > x0 and y1 > y0:
                sub = critical_mask[y0:y1, x0:x1] > 0
                face_ratio = float(np.count_nonzero(sub)) / float(max(1, sub.size))
                g.on_face = face_ratio > 0.18
        score_group(g, det_img.shape[:2], st.select_threshold, overlap)
    timings["score_ms"] = round((time.time() - t0) * 1000.0, 1)

    # ---- ⑧' B 模式（文字全部去除）：不做水印评分过滤 ----
    if mode == "text":
        kept: List[WatermarkGroup] = []
        for g in all_groups:
            # B 模式只处理"OCR 可识别的文字"：没有文字内容的重复结构（例如纯纹理
            # 的周期性响应）不算文字，直接排除，避免出现一堆空候选。
            if not g.text.strip():
                continue                    # B 模式只处理"OCR 可识别的文字"
            scores = [it.score for it in g.items] or [0.55]
            mean = float(np.mean(scores))
            g.confidence = float(np.clip(0.35 + 0.65 * mean, 0.0, 0.99))
            g.tier = "high" if mean >= 0.70 else ("medium" if mean >= 0.45 else "low")
            g.kind = "文字区域（B 模式：全部去除）"
            g.reasons = ["OCR 检出文字 → 默认进入待去除 Mask（B 模式不做水印评分过滤）"] + g.reasons[:2]
            kept.append(g)
        all_groups = kept
        timings["text_mode_groups"] = float(len(kept))

    # ---- ⑨ 候选汇总 / 分层 / 选择 ----
    cands: List[WatermarkCandidate] = []
    visual_only_count = 0
    for g in sorted(all_groups, key=lambda x: -x.confidence):
        if mode != "text" and g.confidence < st.min_confidence:
            continue

        # ---- 没有 OCR 证据的"视觉候选"（重复结构/半透明响应）----
        # 这类候选是皮肤、衣服、布料纹理误检的主要来源，所以：
        #   * B 模式（文字全部去除）完全不用它；
        #   * A 模式只接受"细长的小型结构"，并且默认**不勾选**（除非显式允许）。
        if not g.text.strip():
            if mode == "text":
                continue
            if not _text_like_visual(g, area_total):
                continue
            if not st.allow_visual_only:
                g.tier = "low"
            visual_only_count += 1

        poly = (g.poly * inv_scale).astype(np.float32)
        x, y, w, h = _envelope(poly)
        x0, y0 = max(0, x), max(0, y)
        ww, hh = min(W, x + w) - x0, min(H, y + h) - y0
        if ww < 3 or hh < 3:
            continue
        cands.append(WatermarkCandidate(
            bbox=(int(x0), int(y0), int(ww), int(hh)),
            kind=g.kind, confidence=g.confidence, source=g.source, text=g.text,
            reasons=list(g.reasons), on_face=g.on_face, tier=g.tier,
            template_id=g.template_id, instances=g.instances,
            angle=float(_norm_angle(g.theta)), poly=poly, from_tiling=g.from_tiling,
        ))
    # 去重分两步（顺序很重要）：
    # ① **先按面积从大到小**：被更大候选框"包住"的碎块直接并入大框
    #    （平铺/重复结构检测经常在同一个水印内部再切出很多小框，留着只会污染界面）；
    # ② 再按置信度做 IoU 去重，处理互相重叠但不是包含关系的框。
    kept_large: List[WatermarkCandidate] = []
    for c in sorted(cands, key=lambda x: -x.area):
        covered = False
        for o in kept_large:
            if o.area < c.area:
                continue
            inter = _intersection_area(c.bbox, o.bbox)
            if inter <= 0:
                continue
            if inter / max(1.0, float(c.area)) > 0.75:
                covered = True            # 这个小框属于同一个水印 → 并入大框
                break
        if not covered:
            kept_large.append(c)
    deduped_c: List[WatermarkCandidate] = []
    for c in sorted(kept_large, key=lambda x: -x.confidence):
        if any(_iou_rect(c.bbox, o.bbox) > 0.55 for o in deduped_c):
            continue
        deduped_c.append(c)
    if logger.isEnabledFor(10):     # DEBUG：只在排查时输出，正常使用零开销
        logger.debug("候选去重前 %d 个：%s", len(cands),
                     [(round(c.confidence, 3), c.bbox, c.text[:10]) for c in cands])
        logger.debug("候选去重后 %d 个：%s", len(deduped_c),
                     [(round(c.confidence, 3), c.bbox, c.text[:10]) for c in deduped_c])
    cands = deduped_c[: st.max_candidates]
    report.candidates = cands

    # ---- ⑩ Mask（原图分辨率，贴合笔画）----
    # 先算 Mask 再决定勾选：面积守卫必须用**真实 Mask 面积**，
    # 而不是"旋转矩形的外接框面积"（平铺水印的外接框可能很大，但笔画面积很小）。
    t0 = time.time()
    mask_all = np.zeros((H, W), np.uint8)
    for g in all_groups:
        if g.confidence < st.min_confidence * 0.9:
            continue
        poly_full = (g.poly * inv_scale).astype(np.float32)
        m = _stroke_mask_in_poly(img, poly_full, st.dilate)
        mask_all = cv2.bitwise_or(mask_all, m)

    cand_masks: List[np.ndarray] = []
    for c in cands:
        m = None
        if c.poly is not None:
            m = _stroke_mask_in_poly(img, c.poly, st.dilate)
        if mp.is_empty(m):
            m = np.zeros((H, W), np.uint8)
            cv2.fillPoly(m, [np.round(c.poly).astype(np.int32)], 255)
        cand_masks.append(m)
    timings["mask_ms"] = round((time.time() - t0) * 1000.0, 1)

    # ---- ⑩' 勾选 + 面积守卫（用真实 Mask 面积）----
    for c in cands:
        c.selected = True if mode == "text" else (c.tier in st.select_tiers)
    budget = float(st.max_area_ratio) * area_total
    if mode == "text":
        budget = float(max(st.max_area_ratio, 0.75)) * area_total   # B 模式优先"不漏"
    used = 0.0
    order = sorted(range(len(cands)), key=lambda i: -cands[i].confidence)
    dropped = 0
    for i in order:
        c = cands[i]
        if not c.selected:
            continue
        a = float(np.count_nonzero(cand_masks[i]))
        if used + a > budget:
            c.selected = False
            dropped += 1
            report.warnings.append(
                f"自动检测面积过大（超过画面 {st.max_area_ratio * 100:.0f}%），"
                f"已把置信度较低的「{(c.text[:12] or '某区域')}」取消默认勾选，请人工确认"
            )
        else:
            used += a
    if mode == "text" and dropped:
        report.warnings.append("B 模式下检测到的文字区域过多，已取消部分默认勾选，请人工确认后修复。")

    mask_sel = np.zeros((H, W), np.uint8)
    for i, c in enumerate(cands):
        if c.selected:
            mask_sel = cv2.bitwise_or(mask_sel, cand_masks[i])
    report.mask_all, report.mask_selected = mask_all, mask_sel
    timings["selected_mask_ratio"] = round(mp.area_ratio(mask_sel), 5)
    # ---- ⑪ 人脸提示 / 汇总文案 ----
    if critical_mask is not None and not mp.is_empty(critical_mask) and not mp.is_empty(mask_sel):
        inter = float(np.count_nonzero(cv2.bitwise_and(mask_sel, critical_mask)))
        ratio = inter / max(1.0, float(np.count_nonzero(mask_sel)))
        if ratio > 0.15:
            report.warnings.append(
                f"检测到水印可能覆盖人脸（占候选区域 {ratio * 100:.0f}%），"
                f"请确认 Mask 后再修复；修复时会自动进入面部保护模式"
            )

    counts = report.counts()
    n_sel = len(report.high_confidence)
    sel_ratio = mp.area_ratio(mask_sel)
    if mode == "text":
        if not cands:
            report.summary = (
                "**未检测到可识别文字**（OCR 文字区域 0 个）→ 已跳过：原图不做任何修改。\n\n"
                "如果这张图确实有文字水印，说明它可能是极低对比度、严重模糊、纯图形 Logo "
                "或特殊字体；请改用「智能自动识别」或「手动选择」。"
            )
            report.warnings.append("OCR 未检测到文字（文字区域 0 个），已跳过修复，原图未被修改。")
        else:
            report.summary = (
                f"OCR 检测到 **{len(cands)}** 个文字区域，Mask 覆盖画面 **{sel_ratio * 100:.2f}%**"
                f"（B 模式：文字默认全部去除，可取消个别误检）。请先看 Mask 预览再修复。"
            )
    elif not cands:
        report.summary = "未发现水印候选：请用画笔手动涂抹要去除的区域。"
    else:
        report.summary = (
            f"发现 **{len(cands)}** 个候选水印区域"
            f"（高置信度 {counts['high']}｜中 {counts['medium']}｜低 {counts['low']}）"
            f"，默认已勾选 **{n_sel}** 个。"
            f"请检查红色 Mask 预览，确认无误后再点「✨ AI 去除」。"
        )
    if cands and n_sel == 0:
        report.warnings.append("候选置信度偏低，已默认不勾选：请人工确认或改用手动画笔。")

    report.stats = {
        "mode": mode,
        "candidates": len(cands),
        "selected": n_sel,
        "high": counts["high"],
        "medium": counts["medium"],
        "low": counts["low"],
        "ocr_items": int(len(items)),
        "visual_only_candidates": int(visual_only_count),
        "visual_response_unreliable": bool(visual_unreliable),
        "ocr_regions": int(len(cands)) if mode == "text" else int(len(items)),
        "templates": int(sum(1 for cl in clusters if len(cl) >= 2)),
        "selected_area_ratio": round(sel_ratio, 5),
        "all_area_ratio": round(mp.area_ratio(mask_all), 5),
        "face_protected": bool(face_mask is not None),
        "sensitivity": st.sensitivity,
    }
    timings["total_ms"] = round((time.time() - t_start) * 1000.0, 1)
    report.timings = timings
    # 写入缓存（LRU 上限 6 张图）
    if cache_key:
        try:
            _REPORT_CACHE[cache_key] = _clone_report(report)
            while len(_REPORT_CACHE) > _REPORT_CACHE_MAX:
                _REPORT_CACHE.popitem(last=False)
        except Exception as exc:
            logger.debug("检测缓存写入失败：%s", exc)
    return report


def candidates_to_mask(report: WatermarkReport, only_selected: bool = True,
                       dilate: int = 0) -> Optional[np.ndarray]:
    """把候选转成掩膜（供"确认检测结果并修复"使用）。"""
    m = report.mask_selected if only_selected else report.mask_all
    if m is None or mp.is_empty(m):
        return None
    return mp.dilate_mask(m, int(dilate)) if dilate else m


def detect_text_regions(
    image_rgb: np.ndarray,
    settings: Optional[DetectSettings] = None,
    **overrides: Any,
) -> WatermarkReport:
    """**B 模式**「文字全部去除」：OCR 读到的文字区域**全部**作为待去除候选。

    与水印评分无关（不做 Watermark Score 过滤），只按 OCR 识别置信度排序；
    仍然只生成"候选 + Mask 预览"，由用户确认后才进入第五轮修复流程。
    """
    st = settings or resolve_settings(**overrides)
    return detect_watermarks(image_rgb, settings=st, mode="text")


def mask_from_candidates(image_rgb: np.ndarray, candidates: Sequence[WatermarkCandidate],
                         selected: Optional[Sequence[bool]] = None, dilate: int = 3) -> np.ndarray:
    """按"用户勾选的候选"重新生成 Mask（表格里改选后调用）。"""
    img = mp.to_uint8(image_rgb)
    mask = np.zeros(img.shape[:2], np.uint8)
    for i, c in enumerate(candidates):
        if selected is not None and i < len(selected) and not selected[i]:
            continue
        if c.poly is None:
            continue
        m = _stroke_mask_in_poly(img, c.poly, 0)
        if mp.is_empty(m):
            m = np.zeros(img.shape[:2], np.uint8)
            cv2.fillPoly(m, [np.round(c.poly).astype(np.int32)], 255)
        mask = cv2.bitwise_or(mask, m)
    return mp.dilate_mask(mask, int(dilate)) if dilate else mask


def timings_text(timings: Dict[str, float]) -> str:
    """把耗时字典渲染成一行中文摘要。"""
    if not timings:
        return "无耗时数据"
    order = [
        ("ocr_ms", "OCR"), ("sweep_ms", "旋转扫描"),
        ("region_ocr_ms", "局部放大 OCR"), ("hi_ocr_ms", "高清兜底 OCR"),
        ("textness_ms", "笔画分析"),
        ("face_ms", "人脸检测"), ("group_ms", "区域合并"), ("translucent_ms", "半透明分析"),
        ("repeated_ms", "重复结构"), ("template_ms", "重复实例展开"), ("score_ms", "打分"),
        ("mask_ms", "Mask 生成"), ("total_ms", "检测总计"),
    ]
    parts = [f"{label} {timings[key]:.0f}ms" for key, label in order if key in timings]
    return " ｜ ".join(parts)
