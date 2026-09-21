"""Mask 自动收紧（Brush → Watermark Core）。

核心思想（对应第四轮问题一）：

    用户的画笔只是"告诉程序水印在哪里"，**不是**"要求把这一整块重新生成"。

因此流程是：

    用户涂抹范围（安全边界，绝不超出）
        ↓  用多种证据估计真正的水印像素
    水印核心区 = OCR/文字框 ∩ 涂抹 ∪ 半透明检测 ∩ 涂抹 ∪ 局部对比度残差 ∩ 涂抹
        ↓  形态学清理 + 安全兜底（证据不足时不冒险过度收缩）
    交给 LaMa 重建的最小区域

证据来源（按可靠度排序）：
1. OCR / 文字检测框 —— 最可靠（水印多为文字）；
2. 半透明水印检测结果 —— 中（颜色/对比度分析）；
3. 局部对比度残差（相对中值背景的亮度偏移）—— 兜底，自适应阈值；
4. 以上都不可靠时：对涂抹区域做**温和内缩**（而不是激进收缩），并标记低置信度。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from . import mask_processor as mp


@dataclass
class RefineResult:
    """收紧结果。"""

    mask: np.ndarray                     # 收紧后的"水印核心区"（不含额外扩张）
    user_mask: np.ndarray                # 用户原始涂抹范围（安全边界）
    method: str = "none"
    confidence: float = 0.0
    stats: Dict[str, float] = field(default_factory=dict)
    note: str = ""
    note_extra: str = ""


def estimate_brush_width(mask: np.ndarray) -> float:
    """用距离变换估计笔触宽度（像素）。"""
    m = mp.ensure_binary(mask)
    if mp.is_empty(m):
        return 0.0
    dist = cv2.distanceTransform((m > 0).astype(np.uint8), cv2.DIST_L2, 3)
    return float(dist.max() * 2.0)


def _overlay_response(image_rgb: np.ndarray, bg_kernel: int) -> np.ndarray:
    """相对中值背景的亮度残差（半透明水印会表现为稳定的正/负偏移）。

    注意：OpenCV 的 ``medianBlur`` 对 8 位图的大核有限制（k≥16 会断言失败），
    因此大核走"降采样 → 小核中值 → 升采样"的等价实现，既稳定又快。
    """
    img = mp.to_uint8(image_rgb)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
    k = int(max(9, bg_kernel) | 1)
    k = min(k, (min(gray.shape) // 2) * 2 + 1)
    bg = _median_blur_safe(gray, k)
    return gray.astype(np.float32) - bg.astype(np.float32)


def _median_blur_safe(gray: np.ndarray, k: int) -> np.ndarray:
    """对任意奇数核都安全的中值滤波实现。"""
    k = int(k) | 1
    k = max(3, min(k, (min(gray.shape) // 2) * 2 + 1))
    if k <= 15:
        return cv2.medianBlur(gray, k)
    s = max(2, int(np.ceil(k / 15.0)))
    small = cv2.resize(gray, (max(8, gray.shape[1] // s), max(8, gray.shape[0] // s)),
                       interpolation=cv2.INTER_AREA)
    kk = max(3, min(15, (k // s) | 1))
    bg_small = cv2.medianBlur(small, kk)
    return cv2.resize(bg_small, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_LINEAR)


def refine(
    image_rgb: np.ndarray,
    user_mask: np.ndarray,
    ocr_boxes: Optional[Sequence[Sequence[int]]] = None,
    evidence_mask: Optional[np.ndarray] = None,
    min_keep_ratio: float = 0.10,
    max_grow_ratio: float = 1.0,
    sensitivity: float = 1.0,
    adaptive: bool = True,
) -> RefineResult:
    """把用户涂抹范围收紧为"水印核心区"。

    Parameters
    ----------
    image_rgb : 原图（RGB）
    user_mask : 用户涂抹的二值 Mask（安全边界）
    ocr_boxes : OCR 文字框 [(x, y, w, h), ...]（可选，最可靠证据）
    evidence_mask : 其它证据掩膜（如半透明水印检测结果，可选）
    min_keep_ratio : 收紧后至少保留用户涂抹面积的比例；低于该值则改用"温和内缩"兜底
    max_grow_ratio : 收紧结果相对用户涂抹的最大面积比（默认不允许超过用户涂抹范围）
    sensitivity : 残差阈值灵敏度（越大越容易判定为水印）
    """
    user = mp.ensure_binary(user_mask)
    result = RefineResult(mask=user.copy(), user_mask=user)
    if mp.is_empty(user):
        result.note = "涂抹区域为空"
        return result

    user_px = float(np.count_nonzero(user))
    brush_w = estimate_brush_width(user)
    result.stats.update({"user_px": user_px, "brush_width": round(brush_w, 1)})
    # 用户涂得很大（例如整张平铺水印都涂上）时自动进入"严格模式"：
    #   不做形态学粘连、不加证据外扩、提高残差阈值 —— 保持水印碎片彼此分离，
    #   这样后续可以逐块局部修复，而不是让 LaMA 大面积重画（头发/布料会被改写）。
    user_ratio = mp.area_ratio(user)
    strict = bool(adaptive and user_ratio > 0.30)
    close_px = 0 if strict else 2
    evidence_dilate = 0 if strict else 2
    if strict:
        sensitivity = float(sensitivity) * 0.8
        result.stats["strict_mode"] = 1.0

    # ---------- 证据 1：OCR / 文字框 ----------
    core = np.zeros_like(user)
    used: List[str] = []
    if ocr_boxes:
        box_mask = np.zeros_like(user)
        for b in ocr_boxes:
            try:
                x, y, w, h = [int(v) for v in b[:4]]
            except Exception:
                continue
            if w <= 1 or h <= 1:
                continue
            cv2.rectangle(box_mask, (x, y), (x + w - 1, y + h - 1), 255, -1)
        box_mask = mp.dilate_mask(cv2.bitwise_and(box_mask, user), evidence_dilate)
        if not mp.is_empty(box_mask):
            core = cv2.bitwise_or(core, box_mask)
            used.append("OCR文字框")

    # ---------- 证据 2：半透明水印检测 ----------
    if evidence_mask is not None and not mp.is_empty(evidence_mask):
        ev = cv2.bitwise_and(mp.ensure_binary(evidence_mask, user.shape), user)
        if not mp.is_empty(ev):
            core = cv2.bitwise_or(core, mp.clean_mask(ev, min_area=4, close_px=1))
            used.append("半透明检测")

    # ---------- 证据 3：局部对比度残差 ----------
    resid = _overlay_response(image_rgb, bg_kernel=int(max(11, brush_w * 1.2)))
    inside = user > 0
    outside_ring = (cv2.dilate(user, np.ones((9, 9), np.uint8)) > 0) & ~inside
    noise = float(np.median(np.abs(resid[outside_ring] - np.median(resid[outside_ring])))) if int(outside_ring.sum()) > 50 else 2.0
    sigma = max(1.0, 1.4826 * noise)
    thr = max(3.0, 2.6 * sigma / max(0.4, float(sensitivity)))
    overlay = (np.abs(resid) > thr) & inside
    overlay_u8 = (overlay.astype(np.uint8)) * 255
    if not strict:
        overlay_u8 = cv2.morphologyEx(overlay_u8, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    overlay_u8 = cv2.morphologyEx(overlay_u8, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    overlay_u8 = mp.clean_mask(overlay_u8, min_area=max(6, int(user_px * 0.0008)), close_px=1)
    overlay_px = float(np.count_nonzero(overlay_u8))
    result.stats.update({"overlay_px": overlay_px, "threshold": round(thr, 2)})
    # 严格模式（用户把整张图都涂上）下，对比度残差在头发/布料纹理上会大量误判，
    # 因此只信任 OCR 与半透明检测这两类"结构化"证据。
    if overlay_px > 0 and not strict:
        core = cv2.bitwise_or(core, overlay_u8)
        used.append("对比度残差")
    elif strict and overlay_px > 0:
        used.append("对比度残差(已忽略)")

    # ---------- 清理 & 安全边界 ----------
    core = cv2.bitwise_and(core, user)          # 绝不超出用户涂抹范围
    core = mp.clean_mask(core, min_area=max(6, int(user_px * 0.0008)), close_px=close_px)
    core_px = float(np.count_nonzero(core))
    keep_ratio = core_px / max(1.0, user_px)

    # ---------- 兜底：证据不足时不冒险收缩 ----------
    if core_px <= 0 or keep_ratio < float(min_keep_ratio):
        shrink = max(1, int(round(brush_w * 0.18)))
        fallback = mp.erode_mask(user, shrink)
        if mp.is_empty(fallback):
            fallback = user.copy()
        result.mask = fallback
        result.method = "shrink-fallback"
        result.confidence = 0.35
        result.stats["keep_ratio"] = round(mp.area_ratio(fallback) / max(1e-9, mp.area_ratio(user)), 3)
        result.note = (
            f"水印核心不明显（可用证据：{'、'.join(used) if used else '无'}），"
            f"已对涂抹区域做温和内缩 {shrink}px 作为兜底（未过度收缩，避免水印残留）"
        )
        return result

    result.mask = core
    result.stats["keep_ratio"] = round(keep_ratio, 3)
    result.method = "+".join(used) if used else "对比度残差"

    # 面积上限：水印核心不应超过画面的一定比例，否则 LaMa 会"重画"整幅图
    # （头发/布料纹理会被改写）。超出时逐步内缩，并如实告知用户。
    cap = 0.25
    ratio_now = mp.area_ratio(core)
    if ratio_now > cap:
        shrunk = core
        for k in (2, 3, 4, 5, 6, 8, 10, 12, 16):
            cand = mp.erode_mask(core, k)
            if mp.is_empty(cand):
                break
            shrunk = cand
            if mp.area_ratio(cand) <= cap:
                break
        result.note_extra = (
            f"；水印核心面积达 {ratio_now * 100:.0f}%，已内缩到 {mp.area_ratio(shrunk) * 100:.0f}%"
            f"（超出 {int(cap * 100)}% 时 LaMa 会重画整幅图、改写头发与布料纹理）"
            f"，建议分 2~4 次分块修复"
        )
        core = shrunk
        result.mask = core
        result.stats["area_capped"] = 1.0
    else:
        result.note_extra = ""
    if "OCR文字框" in used:
        result.confidence = 0.9
    elif "半透明检测" in used:
        result.confidence = 0.75
    else:
        result.confidence = 0.6
    result.note = (
        f"已把涂抹范围收紧为该范围内真实的水印像素（保留 {keep_ratio * 100:.0f}%，"
        f"依据：{result.method}），避免把整块正常背景交给 AI 重建"
    ) + getattr(result, "note_extra", "")
    return result


def clip_to_user(final_mask: np.ndarray, user_mask: np.ndarray, allow_px: int = 2) -> np.ndarray:
    """把最终 Mask 限制在"用户涂抹范围 + 极小余量"之内，防止笔刷形状外扩。"""
    allowed = mp.dilate_mask(mp.ensure_binary(user_mask), max(0, int(allow_px)))
    return cv2.bitwise_and(mp.ensure_binary(final_mask), allowed)


def changed_outside_ratio(before: np.ndarray, after: np.ndarray, mask: np.ndarray, tol: int = 2) -> float:
    """量化"结果是否超出给定 Mask 改动"（用于质量自检）。"""
    b = mp.to_uint8(before).astype(np.int16)
    a = mp.to_uint8(after).astype(np.int16)
    diff = np.abs(a - b).max(axis=2) if a.ndim == 3 else np.abs(a - b)
    outside = mp.ensure_binary(mask, diff.shape) == 0
    if not np.any(outside):
        return 0.0
    return float(np.mean(diff[outside] > tol))
