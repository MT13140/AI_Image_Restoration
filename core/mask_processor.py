"""Mask（修复区域）处理模块。

职责：
* 把 UI（Gradio ImageEditor）返回的图层转换为二值 Mask；
* Mask 扩张 / 羽化 / 合并 / 统计；
* Mask 可视化预览（红/绿的半透明覆盖层）；
* 矩形选区生成 Mask；
* 马赛克区域启发式检测（供"马赛克遮挡"修复场景使用）。
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from config import get_logger

logger = get_logger("ai_restore.mask")

BBox = Tuple[int, int, int, int]  # x, y, w, h


# --------------------------------------------------------------------------
# 基础工具
# --------------------------------------------------------------------------
def to_uint8(image: np.ndarray) -> np.ndarray:
    """把任意数值图像安全转换为 uint8。"""
    arr = np.asarray(image)
    if arr.dtype == np.uint8:
        return arr
    if arr.dtype == bool:
        return (arr.astype(np.uint8)) * 255
    arr = arr.astype(np.float32)
    if arr.size and float(np.nanmax(arr)) <= 1.0001:
        arr = arr * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


def pil_to_numpy(img) -> np.ndarray:
    """PIL.Image -> numpy（保留通道数）。"""
    if isinstance(img, np.ndarray):
        return img
    return np.asarray(img)


def to_gray(image: np.ndarray) -> np.ndarray:
    """转灰度 uint8（输入可以是 RGB / RGBA / 灰度）。"""
    arr = to_uint8(pil_to_numpy(image))
    if arr.ndim == 2:
        return arr
    if arr.shape[2] == 4:
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2GRAY)
    if arr.shape[2] == 3:
        return cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    return arr[..., 0]


def ensure_binary(mask: np.ndarray, target_shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """把各种形态的 Mask 统一成 uint8 的 {0, 255}、形状 (H, W)。

    支持的输入：bool、灰度、RGBA（使用 alpha 通道）、三通道（取最大值）。
    """
    arr = pil_to_numpy(mask)
    if arr is None:
        raise ValueError("mask 为空")
    arr = np.asarray(arr)
    if arr.ndim == 3:
        if arr.shape[2] == 4:
            arr = arr[..., 3]
        elif arr.shape[2] == 1:
            arr = arr[..., 0]
        else:
            arr = arr.max(axis=2)
    if target_shape is not None and arr.shape[:2] != tuple(target_shape):
        arr = cv2.resize(arr.astype(np.float32), (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
    arr = to_uint8(arr)
    binary = np.where(arr > 8, 255, 0).astype(np.uint8)
    return binary


def is_empty(mask: np.ndarray) -> bool:
    return int(np.count_nonzero(mask)) == 0


def area_ratio(mask: np.ndarray) -> float:
    """Mask 覆盖面积占比。"""
    m = ensure_binary(mask)
    return float(np.count_nonzero(m)) / float(m.size) if m.size else 0.0


# --------------------------------------------------------------------------
# Mask 组合 / 变换
# --------------------------------------------------------------------------
def combine_masks(*masks: Optional[np.ndarray]) -> np.ndarray:
    """把多个 Mask 求并集（自动对齐形状）。"""
    valid = [ensure_binary(m) for m in masks if m is not None]
    if not valid:
        raise ValueError("至少需要一个有效 Mask")
    shape = valid[0].shape
    out = np.zeros(shape, np.uint8)
    for m in valid:
        if m.shape != shape:
            m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        out = cv2.bitwise_or(out, m)
    return out


def dilate_mask(mask: np.ndarray, pixels: int) -> np.ndarray:
    """Mask 扩张（水印边缘通常需要多覆盖一些像素）。"""
    m = ensure_binary(mask)
    pixels = int(pixels)
    if pixels <= 0:
        return m.copy()
    k = 2 * pixels + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.dilate(m, kernel, iterations=1)


def erode_mask(mask: np.ndarray, pixels: int) -> np.ndarray:
    m = ensure_binary(mask)
    pixels = int(pixels)
    if pixels <= 0:
        return m.copy()
    k = 2 * pixels + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.erode(m, kernel, iterations=1)


def feather_mask(mask: np.ndarray, pixels: int) -> np.ndarray:
    """羽化：返回 float32 的 0..1 权重图，用于无缝融合。"""
    m = ensure_binary(mask)
    pixels = int(pixels)
    if pixels <= 0:
        return (m.astype(np.float32) / 255.0)
    k = 2 * pixels + 1
    blur = cv2.GaussianBlur(m.astype(np.float32) / 255.0, (k, k), 0)
    core = m.astype(np.float32) / 255.0
    return np.clip(np.maximum(blur, core), 0.0, 1.0)


def clean_mask(mask: np.ndarray, min_area: int = 8, close_px: int = 2) -> np.ndarray:
    """去除噪点、填补小孔洞，让 Mask 更干净。"""
    m = ensure_binary(mask)
    if close_px > 0:
        k = 2 * int(close_px) + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=1)
    if min_area > 0:
        n, labels, stats, _ = cv2.connectedComponentsWithStats((m > 0).astype(np.uint8), connectivity=8)
        keep = np.zeros_like(m)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                keep[labels == i] = 255
        m = keep
    return m


def mask_to_boxes(mask: np.ndarray, min_area: int = 12) -> List[BBox]:
    """Mask 连通域外接矩形列表。"""
    m = ensure_binary(mask)
    n, labels, stats, _ = cv2.connectedComponentsWithStats((m > 0).astype(np.uint8), connectivity=8)
    boxes: List[BBox] = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        boxes.append((x, y, w, h))
    boxes.sort(key=lambda b: (b[1], b[0]))
    return boxes


# --------------------------------------------------------------------------
# 矩形选区
# --------------------------------------------------------------------------
def rects_to_mask(
    shape: Tuple[int, int],
    rects: Sequence[Sequence[float]],
    percent: bool = True,
) -> np.ndarray:
    """把矩形列表转换为 Mask。

    rects 中每个元素为 (x1, y1, x2, y2)。percent=True 时按 0~100 百分比解析。
    """
    h, w = int(shape[0]), int(shape[1])
    mask = np.zeros((h, w), np.uint8)
    for rect in rects:
        if rect is None or len(rect) != 4:
            continue
        x1, y1, x2, y2 = (float(v) for v in rect)
        if percent:
            x1, x2 = x1 / 100.0 * w, x2 / 100.0 * w
            y1, y2 = y1 / 100.0 * h, y2 / 100.0 * h
        xs, xe = sorted((x1, x2))
        ys, ye = sorted((y1, y2))
        xi, yi = int(math.floor(xs)), int(math.floor(ys))
        xe_i, ye_i = int(math.ceil(xe)), int(math.ceil(ye))
        # 至少 1 像素，避免零面积
        xe_i = max(xe_i, xi + 1)
        ye_i = max(ye_i, yi + 1)
        cv2.rectangle(mask, (xi, yi), (xe_i - 1, ye_i - 1), 255, thickness=-1)
    return mask


def points_to_rect(p1: Sequence[float], p2: Sequence[float], convert_percent: bool = False) -> List[float]:
    """两个点 -> 百分比矩形。convert_percent=True 时输入为百分比坐标。"""
    (x1, y1), (x2, y2) = p1, p2
    xs, xe = sorted((float(x1), float(x2)))
    ys, ye = sorted((float(y1), float(y2)))
    return [xs, ys, xe, ye]


def boxes_from_points(points: Sequence[Sequence[float]], shape: Tuple[int, int]) -> List[List[float]]:
    """把用户在预览图上点击的两个像素点转换成百分比矩形。"""
    h, w = int(shape[0]), int(shape[1])
    if len(points) < 2:
        return []
    (x1, y1), (x2, y2) = points[0], points[1]
    xs, xe = sorted((float(x1), float(x2)))
    ys, ye = sorted((float(y1), float(y2)))
    return [
        [
            max(0.0, xs / w * 100.0),
            max(0.0, ys / h * 100.0),
            min(100.0, xe / w * 100.0),
            min(100.0, ye / h * 100.0),
        ]
    ]


# --------------------------------------------------------------------------
# 与 Gradio ImageEditor 交互
# --------------------------------------------------------------------------
def _layer_to_alpha(layer) -> Optional[np.ndarray]:
    """从 ImageEditor 的图层提取 alpha 通道。"""
    if layer is None:
        return None
    arr = pil_to_numpy(layer)
    if arr is None:
        return None
    arr = np.asarray(arr)
    if arr.ndim == 2:
        return to_uint8(arr)
    if arr.shape[2] == 4:
        return to_uint8(arr[..., 3])
    if arr.shape[2] == 3:
        # 没有 alpha 的图层：只要不是纯黑（未绘制）就视为已绘制
        rgb = to_uint8(arr)
        return np.where(rgb.max(axis=2) > 3, 255, 0).astype(np.uint8)
    return to_uint8(arr[..., 0])


def editor_to_mask(editor_value, shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """把 Gradio ImageEditor 的值转换为二值 Mask。

    优先解析 layers 的 alpha 通道；若 layers 为空，则用 composite 与
    background 的差异推断（兼容不同前端版本的行为差异）。
    """
    if editor_value is None:
        raise ValueError("尚未载入图片")

    # 兼容：某些前端/版本会以 [background, layers, composite] 数组形式传值
    if isinstance(editor_value, (list, tuple)):
        if len(editor_value) >= 3:
            editor_value = {
                "background": editor_value[0],
                "layers": list(editor_value[1] or []),
                "composite": editor_value[2],
            }
        elif len(editor_value) >= 1:
            editor_value = {"background": editor_value[0], "layers": [], "composite": None}
        else:
            raise ValueError("尚未载入图片")

    bg = None
    layers: List[np.ndarray] = []

    if isinstance(editor_value, dict):
        bg = editor_value.get("background")
        raw_layers = editor_value.get("layers") or []
        for layer in raw_layers:
            alpha = _layer_to_alpha(layer)
            if alpha is not None:
                layers.append(alpha)
        composite = editor_value.get("composite")
    else:
        composite = editor_value

    bg_arr = pil_to_numpy(bg) if bg is not None else None
    comp_arr = pil_to_numpy(composite) if composite is not None else None

    if bg_arr is not None:
        bg_arr = np.asarray(bg_arr)
        if bg_arr.ndim == 2:
            target = bg_arr.shape
        else:
            target = bg_arr.shape[:2]
    elif comp_arr is not None:
        comp_arr = np.asarray(comp_arr)
        target = comp_arr.shape[:2] if comp_arr.ndim >= 2 else None
    else:
        target = shape

    if shape is not None:
        target = tuple(shape)

    if target is None:
        raise ValueError("无法确定图片尺寸")

    mask = np.zeros(target, np.uint8)
    for alpha in layers:
        if alpha.shape != target:
            alpha = cv2.resize(alpha, (target[1], target[0]), interpolation=cv2.INTER_NEAREST)
        mask = cv2.bitwise_or(mask, np.where(alpha > 8, 255, 0).astype(np.uint8))

    if not is_empty(mask):
        return mask

    # 回退：用 composite 与 background 的差异推断 Mask
    if bg_arr is not None and comp_arr is not None:
        bg_n = np.asarray(bg_arr)
        comp_n = np.asarray(comp_arr)
        if bg_n.ndim == 3 and bg_n.shape[2] == 4:
            bg_n = bg_n[..., :3]
        if comp_n.ndim == 3 and comp_n.shape[2] == 4:
            comp_n = comp_n[..., :3]
        if bg_n.shape[:2] != comp_n.shape[:2]:
            bg_n = cv2.resize(bg_n, (comp_n.shape[1], comp_n.shape[0]), interpolation=cv2.INTER_AREA)
        diff = cv2.absdiff(to_uint8(bg_n), to_uint8(comp_n))
        if diff.ndim == 3:
            diff = diff.max(axis=2)
        mask = np.where(diff > 12, 255, 0).astype(np.uint8)
        mask = cv2.resize(mask, (target[1], target[0]), interpolation=cv2.INTER_NEAREST)
        # 安全阀（第六轮）：这条"差异推断"只是兼容不同前端版本的兜底路径。
        # 实测某些前端会把画布按显示尺寸重新渲染（甚至补边距），此时
        # composite 与 background 整幅都"不一样"，差异推断就会得出"整张图都涂抹过"。
        # 那会破坏"用户没涂到的地方绝对不修改"的底线，所以这里直接丢弃该推断结果。
        if area_ratio(mask) > 0.9:
            logger.warning(
                "编辑器 composite 与 background 不可比（推断掩膜覆盖 %.1f%%），已忽略该推断结果",
                area_ratio(mask) * 100.0,
            )
            mask = np.zeros(target, np.uint8)

    return mask


def mask_overlay(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int] = (255, 64, 64),
    alpha: float = 0.45,
    outline: bool = True,
) -> np.ndarray:
    """把 Mask 以半透明色叠加到原图上，生成可视化预览。

    输入/输出均为 RGB。
    """
    img = to_uint8(image_rgb)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    m = ensure_binary(mask, target_shape=img.shape[:2])
    out = img.copy()
    sel = m > 0
    if np.any(sel):
        overlay = out.copy()
        overlay[sel] = color
        out = cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0)
    if outline:
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, (255, 255, 0), 1)
    return out


# --------------------------------------------------------------------------
# Gradio ImageEditor 值构造（关键兼容点）
# --------------------------------------------------------------------------
def _alpha_over(base_rgb: np.ndarray, rgba: np.ndarray) -> np.ndarray:
    """把 RGBA 图层按 alpha 合成到 RGB 底图上。"""
    base = base_rgb.astype(np.float32)
    if rgba.shape[2] == 4:
        rgb = rgba[..., :3].astype(np.float32)
        a = rgba[..., 3:4].astype(np.float32) / 255.0
        out = rgb * a + base * (1.0 - a)
    else:
        out = rgba[..., :3].astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def flatten_layers(background_rgb: np.ndarray, layers: Sequence[np.ndarray]) -> np.ndarray:
    """把背景 + 若干 RGBA 图层压平成一张 RGB 图（作为 ImageEditor 的 composite）。"""
    out = to_uint8(background_rgb)
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2RGB)
    for layer in layers:
        if layer is None:
            continue
        arr = to_uint8(pil_to_numpy(layer))
        if arr.ndim == 2:
            continue
        if arr.shape[2] == 3:
            arr = np.dstack([arr, np.full(arr.shape[:2], 255, np.uint8)])
        if arr.shape[:2] != out.shape[:2]:
            arr = cv2.resize(arr, (out.shape[1], out.shape[0]), interpolation=cv2.INTER_NEAREST)
        out = _alpha_over(out, arr)
    return out


def make_editor_value(
    background,
    layers: Optional[Sequence] = None,
    composite=None,
) -> Dict[str, object]:
    """构造 Gradio 4.44 的 ``gr.ImageEditor`` 值：background / layers / composite。

    ⚠️ 关键兼容点：Gradio 4.44.x 的 ``ImageEditor.postprocess()`` 会**无条件**
    读取 ``value["composite"]``（见 gradio/components/image_editor.py:421），
    如果只返回 ``{"background": ..., "layers": [...]}``，事件会直接抛出
    ``KeyError: 'composite'``，网页上表现为编辑器 / 预览 / 放大区域全部显示"错误"。
    因此整个项目必须一律通过本函数构造编辑器值。
    """
    bg_arr = to_uint8(pil_to_numpy(background))
    if bg_arr.ndim == 2:
        bg_arr = cv2.cvtColor(bg_arr, cv2.COLOR_GRAY2RGB)
    elif bg_arr.ndim == 3 and bg_arr.shape[2] == 4:
        bg_arr = _alpha_over(np.full(bg_arr.shape[:2] + (3,), 255, np.uint8), bg_arr)
    elif bg_arr.ndim == 3 and bg_arr.shape[2] > 4:
        bg_arr = bg_arr[..., :3]

    layer_list: List[np.ndarray] = []
    for layer in layers or []:
        if layer is None:
            continue
        arr = to_uint8(pil_to_numpy(layer))
        if arr.ndim == 2:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGBA)
        elif arr.shape[2] == 3:
            arr = np.dstack([arr, np.full(arr.shape[:2], 255, np.uint8)])
        elif arr.shape[2] > 4:
            arr = arr[..., :4]
        if arr.shape[:2] != bg_arr.shape[:2]:
            arr = cv2.resize(arr, (bg_arr.shape[1], bg_arr.shape[0]), interpolation=cv2.INTER_NEAREST)
        layer_list.append(arr)

    if composite is None:
        comp = flatten_layers(bg_arr, layer_list)
    else:
        comp = to_uint8(pil_to_numpy(composite))
        if comp.ndim == 2:
            comp = cv2.cvtColor(comp, cv2.COLOR_GRAY2RGB)
        elif comp.shape[2] == 4:
            comp = _alpha_over(bg_arr, comp)
        elif comp.shape[2] > 4:
            comp = comp[..., :3]
    return {"background": bg_arr, "layers": layer_list, "composite": comp}


# --------------------------------------------------------------------------
# 马赛克启发式检测
# --------------------------------------------------------------------------
def _grid_strength(profile: np.ndarray, period: int) -> float:
    """周期网格线位置的响应强度与整体响应的比值。"""
    if profile.size < period * 2 or profile.mean() <= 1e-6:
        return 0.0
    idx = np.arange(period, profile.size - 1, period)
    if idx.size < 3:
        return 0.0
    # 取网格线邻域的最大响应，避免因 1px 偏移漏检
    vals = []
    for i in idx:
        lo, hi = max(0, i - 1), min(profile.size, i + 2)
        vals.append(float(profile[lo:hi].max()))
    return float(np.mean(vals)) / float(profile.mean())


def detect_mosaic_regions(
    image_rgb: np.ndarray,
    block_sizes: Iterable[int] = (8, 12, 16, 20, 24, 32, 40, 48),
    grid_ratio_threshold: float = 1.45,
    flat_ratio_threshold: float = 0.65,
    min_blocks: int = 4,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """启发式检测马赛克/像素化遮挡区域。

    原理：人工马赛克会留下固定周期的块状边界，因此
    1) 在水平/垂直梯度图上搜索"周期网格"响应；
    2) 结合块内平坦度（块内方差显著低于全图）判定被像素化的块；
    3) 保留连成片的区域作为候选。

    注意：这是启发式检测，不等于可靠的马赛克分割，UI 中允许用户手动修正。
    """
    img = to_uint8(image_rgb)
    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
    gray_f = gray.astype(np.float32)
    h, w = gray.shape

    gx = np.abs(cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3))
    gy = np.abs(cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3))
    col_profile = gx.mean(axis=0)
    row_profile = gy.mean(axis=1)

    best = None
    for size in block_sizes:
        size = int(size)
        if size < 4 or size * 3 >= min(h, w):
            continue
        rs = 0.5 * (_grid_strength(col_profile, size) + _grid_strength(row_profile, size))
        if best is None or rs > best[1]:
            best = (size, rs)

    if best is None or best[1] < grid_ratio_threshold:
        info = {
            "found": False,
            "block_size": None,
            "grid_ratio": round(float(best[1]), 3) if best else 0.0,
            "blocks": 0,
            "message": "未检测到明显的马赛克块状纹理（阈值：%.2f）" % grid_ratio_threshold,
        }
        return np.zeros((h, w), np.uint8), info

    size, ratio = best
    # 块内标准差
    mean = cv2.boxFilter(gray_f, -1, (size, size), normalize=True, borderType=cv2.BORDER_REFLECT)
    sq = cv2.boxFilter(gray_f * gray_f, -1, (size, size), normalize=True, borderType=cv2.BORDER_REFLECT)
    std = np.sqrt(np.clip(sq - mean * mean, 0, None))
    med_std = float(np.median(std)) or 1.0

    mask = np.zeros((h, w), np.uint8)
    blocks = 0
    for y in range(0, h - size + 1, size):
        for x in range(0, w - size + 1, size):
            patch_std = float(std[y : y + size, x : x + size].mean())
            if patch_std < flat_ratio_threshold * med_std:
                mask[y : y + size, x : x + size] = 255
                blocks += 1

    # 只保留成片的区域，去掉零星平坦块（天空、墙面等）
    n, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    keep = np.zeros_like(mask)
    kept_components = 0
    for i in range(1, n):
        if int(stats[i, cv2.CC_STAT_AREA]) >= min_blocks * size * size:
            keep[labels == i] = 255
            kept_components += 1

    found = kept_components > 0
    info = {
        "found": bool(found),
        "block_size": int(size),
        "grid_ratio": round(float(ratio), 3),
        "blocks": int(blocks),
        "regions": int(kept_components),
        "message": (
            f"检测到疑似马赛克区域：块大小约 {size}px，网格响应 {ratio:.2f}，"
            f"区域数 {kept_components}，块数 {blocks}"
            if found
            else f"检测到块状纹理但未形成有效区域（块大小 {size}px，响应 {ratio:.2f}）"
        ),
    }
    return keep, info
