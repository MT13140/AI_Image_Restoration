"""多类型水印检测模块。

包含彼此独立、可单独调用的检测器：

* :class:`OCRWatermarkDetector`  —— 文字 / URL / 日期 / 时间戳（基于 OCR）
* :class:`TranslucentWatermarkDetector` —— 半透明文字/图案水印（对比度 + 颜色分析）
* :class:`RepeatedWatermarkDetector` —— 重复 / 平铺式水印（自相关 + 复制验证）
* :class:`LogoDetector` —— Logo 水印候选（启发式色块，需要人工确认）
* :class:`MosaicDetector` —— 马赛克/像素化遮挡区域（块状纹理分析）
* :class:`WatermarkDetector` —— 统一调度入口

设计原则：任何自动检测都只是"候选"，UI 允许用户手动修正；
绝不虚构不存在的深度模型。Logo 检测明确标注为启发式候选。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from . import mask_processor as mp
from .ocr_detector import OCRDetector, TextItem, get_default_detector


@dataclass
class DetectionResult:
    """一次检测的输出。"""

    name: str
    mask: np.ndarray
    message: str
    details: List[Dict[str, object]] = field(default_factory=list)
    seconds: float = 0.0
    confidence: float = 0.0

    @property
    def found(self) -> bool:
        return not mp.is_empty(self.mask)


def _as_rgb(image: np.ndarray) -> np.ndarray:
    img = mp.to_uint8(image)
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    return img


# --------------------------------------------------------------------------
# 1. OCR 文字类水印
# --------------------------------------------------------------------------
class OCRWatermarkDetector:
    """文字 / URL / 日期 / 时间戳水印检测。"""

    name = "OCR 文字检测"

    def __init__(self, detector: Optional[OCRDetector] = None, dilate: int = 4):
        self.detector = detector or get_default_detector()
        self.dilate = int(dilate)
        self.last_items: List[TextItem] = []

    @property
    def available(self) -> bool:
        return self.detector.available or self.detector.ensure_loaded()

    def detect(
        self,
        image_rgb: np.ndarray,
        min_score: float = 0.5,
        dilate: Optional[int] = None,
        kinds: Optional[Sequence[str]] = None,
        watermark_only: bool = False,
        watermark_threshold: float = 0.6,
    ) -> DetectionResult:
        t0 = time.time()
        img = _as_rgb(image_rgb)
        items = self.detector.detect(img, min_score=min_score)
        self.last_items = items

        used = list(items)
        if watermark_only:
            used = [it for it in items if it.watermark_score >= watermark_threshold]
        if kinds is not None:
            allowed = set(kinds)
            used = [it for it in used if it.kind in allowed]

        mask = self.detector.build_mask(img.shape[:2], used, dilate=self.dilate if dilate is None else int(dilate))

        if used:
            kinds_count: Dict[str, int] = {}
            for it in used:
                kinds_count[it.kind] = kinds_count.get(it.kind, 0) + 1
            kind_desc = "、".join(f"{k}×{v}" for k, v in sorted(kinds_count.items(), key=lambda kv: -kv[1]))
            message = f"识别到 {len(used)} 处文字（{kind_desc}），共 {len(items)} 条 OCR 结果"
        else:
            message = "未识别到符合条件的文字区域"

        conf = float(np.mean([it.watermark_score for it in used])) if used else 0.0
        return DetectionResult(
            name=self.name,
            mask=mask,
            message=message,
            details=[it.to_dict() for it in used],
            seconds=time.time() - t0,
            confidence=conf,
        )


# --------------------------------------------------------------------------
# 2. 半透明水印
# --------------------------------------------------------------------------
class TranslucentWatermarkDetector:
    """半透明文字/图案水印检测。

    思路（经典图像处理，而非"模糊遮挡"）：
    1. 用中值滤波估计背景，得到残差图（半透明叠加会形成稳定的正/负亮度偏移）；
    2. 用残差的鲁棒统计量确定阈值，分别提取"更亮"和"更暗"的叠加层；
    3. 用饱和度比较过滤：白色/灰色水印所在像素的饱和度通常低于周围；
    4. 用连通域几何（细笔画、分散、重复出现的短结构）过滤掉正常图像内容。
    """

    name = "半透明水印检测"

    def __init__(self, bg_kernel: int = 31):
        self.bg_kernel = int(bg_kernel) | 1

    def detect(
        self,
        image_rgb: np.ndarray,
        sensitivity: float = 1.0,
        min_area: int = 12,
        dilate: int = 3,
        max_area_ratio: float = 0.06,
    ) -> DetectionResult:
        t0 = time.time()
        img = _as_rgb(image_rgb)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        sat = hsv[..., 1].astype(np.float32)

        bg = cv2.medianBlur(gray.astype(np.uint8), self.bg_kernel).astype(np.float32)
        diff = gray - bg

        # 鲁棒噪声水平（MAD），用于自适应阈值
        mad = float(np.median(np.abs(diff - np.median(diff))))
        sigma = max(1.4826 * mad, 1.0)
        thr = max(4.0, 2.6 * sigma) / max(0.4, float(sensitivity))

        total = float(gray.size)
        mask = np.zeros(gray.shape, np.uint8)
        detail: List[Dict[str, object]] = []

        for polarity in (1, -1):
            band = (diff > thr) if polarity > 0 else (diff < -thr)
            if not np.any(band):
                continue
            band_u8 = (band.astype(np.uint8)) * 255
            # 细笔画：轻度闭运算连接笔画，再开运算去掉孤立噪点
            band_u8 = cv2.morphologyEx(band_u8, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
            band_u8 = cv2.morphologyEx(band_u8, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

            n, labels, stats, _ = cv2.connectedComponentsWithStats((band_u8 > 0).astype(np.uint8), connectivity=8)
            for i in range(1, n):
                area = int(stats[i, cv2.CC_STAT_AREA])
                if area < min_area:
                    continue
                if area > max_area_ratio * total:
                    continue
                x = int(stats[i, cv2.CC_STAT_LEFT])
                y = int(stats[i, cv2.CC_STAT_TOP])
                w = int(stats[i, cv2.CC_STAT_WIDTH])
                hh = int(stats[i, cv2.CC_STAT_HEIGHT])
                if w < 2 or hh < 2:
                    continue
                comp = labels == i
                comp_sat = float(sat[comp].mean())
                neigh = sat[max(0, y - 4): y + hh + 4, max(0, x - 4): x + w + 4]
                neigh_sat = float(neigh.mean()) if neigh.size else comp_sat
                fill = area / float(w * hh)
                # 半透明叠加层常见的三个特征
                low_sat = comp_sat <= max(12.0, neigh_sat * 0.92)
                texture_like = fill >= 0.12
                thin = (hh <= 0.06 * gray.shape[0]) or (w <= 0.25 * gray.shape[1])
                if (low_sat and texture_like) or (texture_like and thin and comp_sat <= neigh_sat * 1.05):
                    mask[labels == i] = 255
                    detail.append(
                        {
                            "bbox": [x, y, w, hh],
                            "area": area,
                            "polarity": "亮于背景" if polarity > 0 else "暗于背景",
                            "sat": round(comp_sat, 1),
                            "neighbor_sat": round(neigh_sat, 1),
                        }
                    )

        mask = mp.clean_mask(mask, min_area=min_area, close_px=2)
        mask = mp.dilate_mask(mask, dilate)

        found = not mp.is_empty(mask)
        message = (
            f"检测到 {len(detail)} 处疑似半透明叠加区域（阈值 {thr:.1f}）"
            if found
            else f"未检测到明显的半透明叠加层（阈值 {thr:.1f}）"
        )
        return DetectionResult(
            name=self.name,
            mask=mask,
            message=message,
            details=detail[:200],
            seconds=time.time() - t0,
            confidence=float(np.clip(1.0 - thr / 32.0, 0.1, 0.95)) if found else 0.0,
        )


# --------------------------------------------------------------------------
# 3. 重复 / 平铺水印
# --------------------------------------------------------------------------
class RepeatedWatermarkDetector:
    """重复 / 平铺式水印检测。

    思路：
    1. 先得到"叠加层响应图"（优先使用半透明检测的结果，否则现算一次）；
    2. 对响应图做 FFT 自相关，找到非零位移上的周期性峰值，得到平移周期；
    3. 取响应最强的一块作为种子图案，按周期平移复制，并用原始响应图验证
       （重叠率 IoU 达标才保留），从而避免"凭空生成"区域。
    """

    name = "重复水印检测"

    def __init__(self, translucent: Optional[TranslucentWatermarkDetector] = None):
        self.translucent = translucent or TranslucentWatermarkDetector()

    @staticmethod
    def _top_periods(mask: np.ndarray, top_k: int = 6, min_period: int = 20) -> List[Tuple[int, int, float]]:
        h, w = mask.shape
        resp = (mask > 0).astype(np.float32)
        if resp.sum() < 40:
            return []
        f = np.fft.rfft2(resp)
        ac = np.fft.irfft2(f * np.conj(f), s=resp.shape)
        if ac[0, 0] <= 1e-6:
            return []
        ac = ac / ac[0, 0]
        ac[0, 0] = 0.0
        # 抑制中心邻域，避免把"整体自相关"当成周期
        yy, xx = np.mgrid[0:h, 0:w]
        dy = np.minimum(yy, h - yy)
        dx = np.minimum(xx, w - xx)
        near = (dy < min_period) & (dx < min_period)
        ac = np.where(near, 0.0, ac)

        flat = ac.copy()
        peaks: List[Tuple[int, int, float]] = []
        for _ in range(top_k):
            idx = int(np.argmax(flat))
            py, px = np.unravel_index(idx, flat.shape)
            val = float(flat[py, px])
            if val < 0.12:
                break
            sy = py if py <= h - py else py - h
            sx = px if px <= w - px else px - w
            if abs(sx) >= min_period or abs(sy) >= min_period:
                peaks.append((sx, sy, val))
            # 抑制峰值邻域
            y0, y1 = max(0, py - min_period), min(h, py + min_period + 1)
            x0, x1 = max(0, px - min_period), min(w, px + min_period + 1)
            flat[y0:y1, x0:x1] = 0.0
            flat[(py - h) % h, (px - w) % w] = 0.0 if val > 0 else flat[(py - h) % h, (px - w) % w]
            flat[(py) % h, (px) % w] = 0.0
        return peaks

    @staticmethod
    def _seed_pattern(
        resp: np.ndarray, period: Tuple[int, int], max_block: int = 48
    ) -> Optional[Tuple[np.ndarray, int, int]]:
        """找出最能代表重复单元的种子块。

        块尺寸上限为 ``max_block``：过大的种子块会混入大量非水印像素，
        使平移匹配的 IoU 被稀释导致漏检；小方块只要求"图案片段可重复"，
        对半透明、边缘残缺的水印更稳健。

        返回 ``(种子图案, 左上角 x, 左上角 y)``，便于沿周期展开验证。
        """
        h, w = resp.shape
        sx, sy = abs(int(period[0])), abs(int(period[1]))
        bw = int(max(8, min(sx if sx else 32, w, max_block)))
        bh = int(max(8, min(sy if sy else 32, h, max_block)))
        best = (0.0, None)
        step_x = max(4, bw // 2)
        step_y = max(4, bh // 2)
        binary = (resp > 0).astype(np.float32)
        for y in range(0, max(1, h - bh + 1), step_y):
            for x in range(0, max(1, w - bw + 1), step_x):
                patch = binary[y:y + bh, x:x + bw]
                s = float(patch.sum())
                if s > best[0]:
                    best = (s, (x, y, bw, bh))
        if best[1] is None:
            return None
        x, y, bw, bh = best[1]
        seed = (binary[y:y + bh, x:x + bw] > 0).astype(np.uint8)
        if seed.sum() < 10:
            return None
        return seed, x, y

    @staticmethod
    def _profile_periods(profile: np.ndarray, min_period: int, top_k: int = 3) -> List[Tuple[int, float]]:
        """一维投影自相关，用于估计平铺水印的行/列间距。

        相比直接在二维响应图上找自相关峰，行/列能量投影对"稀疏且微弱"的
        平铺文字更敏感：同一行/列上的重复图案会叠加成明显的周期峰。
        """
        p = np.asarray(profile, dtype=np.float64).ravel()
        if p.size < max(8, min_period * 2) or not np.any(p):
            return []
        p = p - p.mean()
        if float(np.abs(p).sum()) <= 1e-9:
            return []
        ac = np.correlate(p, p, mode="full")[p.size - 1:]
        if ac[0] <= 1e-9:
            return []
        ac = ac / ac[0]
        upper = min(len(ac) - 1, int(0.9 * p.size))
        cands: List[Tuple[int, float]] = []
        for lag in range(max(2, int(min_period)), upper):
            if ac[lag] >= ac[lag - 1] and ac[lag] >= ac[lag + 1] and ac[lag] > 0.10:
                cands.append((lag, float(ac[lag])))
        cands.sort(key=lambda t: -t[1])
        out: List[Tuple[int, float]] = []
        for lag, v in cands:
            if all(abs(lag - l2) > max(3, min_period // 2) for l2, _ in out):
                out.append((lag, v))
            if len(out) >= top_k:
                break
        return out

    def _candidates(
        self, mask: np.ndarray, min_period: int, top_k: int = 10
    ) -> List[Tuple[int, int, float]]:
        """综合二维自相关峰与行/列投影周期，给出待验证的点阵周期候选。"""
        binary = (mask > 0).astype(np.float32)
        x_periods = self._profile_periods(binary.sum(axis=0), min_period, top_k=5)   # 列能量 -> 水平周期
        y_periods = self._profile_periods(binary.sum(axis=1), min_period, top_k=5)   # 行能量 -> 垂直周期
        peaks = self._top_periods(mask, top_k=12, min_period=min_period)

        cands: List[Tuple[int, int, float]] = []
        for lag, v in x_periods:
            cands.append((lag, 0, v))
        for lag, v in y_periods:
            cands.append((0, lag, v))
        for lx, vx in x_periods[:2]:
            for ly, vy in y_periods[:2]:
                cands.append((lx, ly, (vx + vy) / 2.0))
        cands.extend((sx, sy, float(v)) for sx, sy, v in peaks)

        seen = set()
        out: List[Tuple[int, int, float]] = []
        for sx, sy, v in sorted(cands, key=lambda c: -c[2]):
            # 位移必须为 0 或达到最小周期，否则只是"相邻窗口"造成的伪重复
            if 0 < abs(sx) < min_period or 0 < abs(sy) < min_period:
                continue
            key = (abs(sx), abs(sy))
            if key == (0, 0) or key in seen:
                continue
            seen.add(key)
            out.append((sx, sy, v))
        return out[:top_k]

    @staticmethod
    def _stroke_filter(mask: np.ndarray) -> np.ndarray:
        """只保留"细笔画"结构，用于重复水印分析。

        水印文字通常是细长的笔画，而马赛克边缘、贴纸边界等是大块结构。
        先做这一步过滤，可以避免把块状边缘当成"重复图案"。
        """
        m = mp.ensure_binary(mask)
        h, w = m.shape
        total = float(h * w)
        n, labels, stats, _ = cv2.connectedComponentsWithStats((m > 0).astype(np.uint8), connectivity=8)
        keep = np.zeros_like(m)
        for i in range(1, n):
            bw = int(stats[i, cv2.CC_STAT_WIDTH])
            bh = int(stats[i, cv2.CC_STAT_HEIGHT])
            area = int(stats[i, cv2.CC_STAT_AREA])
            if bh <= max(12, 0.08 * h) and bw <= max(12, 0.35 * w) and area <= 0.01 * total:
                keep[labels == i] = 255
        return keep

    def detect(
        self,
        image_rgb: np.ndarray,
        response_mask: Optional[np.ndarray] = None,
        dilate: int = 3,
        iou_threshold: float = 0.25,
        max_copies: int = 24,
    ) -> DetectionResult:
        t0 = time.time()
        img = _as_rgb(image_rgb)
        h, w = img.shape[:2]

        base = response_mask
        base_msg = ""
        if base is None or mp.is_empty(base):
            res = self.translucent.detect(img, dilate=0)
            base = res.mask
            base_msg = "（响应图来自半透明检测）"
        base = self._stroke_filter(mp.ensure_binary(base, (h, w)))

        # 周期下限：过小的周期通常来自笔画/纹理自身的重复，而不是平铺水印的点阵间距
        min_period = max(24, int(0.06 * min(h, w)))
        periods = self._candidates(base, min_period)
        if not periods:
            info = "未发现明显的重复周期"
            return DetectionResult(self.name, np.zeros((h, w), np.uint8), info, [], time.time() - t0, 0.0)

        binary = (base > 0).astype(np.uint8)
        best_mask = np.zeros((h, w), np.uint8)
        used_period = None
        accepted_total = 0
        best_score = 0.0
        best_spread = 0.0

        for sx, sy, peak in periods[:8]:
            if sx == 0 and sy == 0:
                continue
            found = self._seed_pattern(binary, (sx, sy))
            if found is None:
                continue
            seed, seed_x, seed_y = found
            bh, bw = seed.shape
            pattern = seed > 0
            if int(pattern.sum()) < 10:
                continue
            accum = np.zeros((h, w), np.uint8)
            accepted = 0
            positions: List[Tuple[int, int]] = []
            span = int(max(1, max_copies // 2))
            # 以"种子块所在位置"为原点，沿周期向四周展开，逐块用 IoU 验证
            for ky in range(-span, span + 1):
                for kx in range(-span, span + 1):
                    if accepted >= max_copies:
                        break
                    y0 = seed_y + ky * sy
                    x0 = seed_x + kx * sx
                    if y0 < 0 or x0 < 0 or y0 + bh > h or x0 + bw > w:
                        continue
                    region = binary[y0:y0 + bh, x0:x0 + bw]
                    inter = int(np.logical_and(region > 0, pattern).sum())
                    union = int(np.logical_or(region > 0, pattern).sum())
                    iou = inter / union if union else 0.0
                    if iou >= iou_threshold:
                        block = accum[y0:y0 + bh, x0:x0 + bw]
                        np.maximum(block, (region > 0).astype(np.uint8) * 255, out=block)
                        accum[y0:y0 + bh, x0:x0 + bw] = block
                        accepted += 1
                        positions.append((x0, y0))
                if accepted >= max_copies:
                    break

            # 打分同时考虑"命中数量"和"覆盖范围"：
            # 真正的平铺水印会散布到整幅图，而文字笔画内部产生的伪周期只会集中在很小的一块区域。
            if positions:
                xs = [p[0] for p in positions]
                ys = [p[1] for p in positions]
                spread = float((max(xs) - min(xs)) * (max(ys) - min(ys)))
            else:
                spread = 0.0
            score = accepted * (1.0 + (spread ** 0.5) / 60.0)
            if score > best_score:
                best_score = score
                accepted_total = accepted
                best_spread = spread
                best_mask = accum
                used_period = (sx, sy)

        mask = mp.dilate_mask(best_mask, dilate)
        if not mp.is_empty(mask):
            message = (
                f"检测到重复水印：水平周期 {used_period[0]}px，垂直周期 {used_period[1]}px，"
                f"命中 {accepted_total} 个重复单元，覆盖范围 {int(best_spread ** 0.5)}px {base_msg}"
            )
            if best_spread <= 0 or accepted_total < 6:
                message += "（置信度较低：疑似重复结构较少，请人工确认或改用矩形/画笔手动框选）"
        else:
            message = "发现周期性但未能可靠定位重复水印单元，建议使用手动框选"
        conf = 0.0
        if accepted_total:
            conf = float(np.clip(accepted_total / 12.0, 0.0, 0.9)) * (0.5 if best_spread <= 0 else 1.0)
        return DetectionResult(self.name, mask, message, [{"period": used_period, "copies": accepted_total}], time.time() - t0, conf)


# --------------------------------------------------------------------------
# 4. Logo 候选
# --------------------------------------------------------------------------
class LogoDetector:
    """Logo 水印候选检测（启发式，需人工确认）。

    说明：本项目**不包含**通用的深度学习 Logo 识别模型，因此这里只做
    "彩色/高对比小块区域"的候选提取（常见于角落 Logo），并明确标注为启发式，
    由用户在界面中确认或补充框选。接口保持独立，方便以后接入真实 Logo 检测模型。
    """

    name = "Logo 候选检测（启发式）"
    is_deep_model = False

    def detect(
        self,
        image_rgb: np.ndarray,
        dilate: int = 4,
        max_candidates: int = 6,
        min_area_ratio: float = 0.0004,
        max_area_ratio: float = 0.08,
        prefer_corners: bool = True,
    ) -> DetectionResult:
        t0 = time.time()
        img = _as_rgb(image_rgb)
        h, w = img.shape[:2]
        total = float(h * w)
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        sat = hsv[..., 1]
        val = hsv[..., 2]

        # 彩色且非极暗/极亮的区域
        colored = ((sat > 60) & (val > 50)).astype(np.uint8) * 255
        colored = cv2.morphologyEx(colored, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        colored = cv2.morphologyEx(colored, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        n, labels, stats, _ = cv2.connectedComponentsWithStats((colored > 0).astype(np.uint8), connectivity=8)
        cands = []
        for i in range(1, n):
            area = int(stats[i, cv2.CC_STAT_AREA])
            ratio = area / total
            if ratio < min_area_ratio or ratio > max_area_ratio:
                continue
            x = int(stats[i, cv2.CC_STAT_LEFT])
            y = int(stats[i, cv2.CC_STAT_TOP])
            bw = int(stats[i, cv2.CC_STAT_WIDTH])
            bh = int(stats[i, cv2.CC_STAT_HEIGHT])
            if bw < 8 or bh < 8:
                continue
            fill = area / float(bw * bh)
            if fill < 0.25 or fill > 0.98:
                continue
            aspect = bw / float(bh)
            if aspect > 8 or aspect < 0.12:
                continue
            cx, cy = x + bw / 2.0, y + bh / 2.0
            corner_bonus = 0.0
            if prefer_corners:
                nx = abs(cx / w - 0.5) * 2
                ny = abs(cy / h - 0.5) * 2
                corner_bonus = 0.25 * ((nx * ny) ** 1.5)
            edge_bonus = 0.1 if (x < 0.12 * w or y < 0.12 * h or x + bw > 0.88 * w or y + bh > 0.88 * h) else 0.0
            score = float(np.clip(0.35 + corner_bonus + edge_bonus + min(0.2, ratio * 4), 0, 1))
            cands.append({"bbox": [x, y, bw, bh], "area": area, "score": round(score, 3), "fill": round(fill, 3)})

        cands.sort(key=lambda c: -c["score"])
        cands = cands[:max_candidates]

        mask = np.zeros((h, w), np.uint8)
        for c in cands:
            x, y, bw, bh = c["bbox"]
            pad = int(0.08 * max(bw, bh)) + 2
            cv2.rectangle(
                mask,
                (max(0, x - pad), max(0, y - pad)),
                (min(w - 1, x + bw + pad), min(h - 1, y + bh + pad)),
                255,
                -1,
            )
        mask = mp.dilate_mask(mask, dilate)

        found = bool(cands)
        message = (
            f"给出 {len(cands)} 个 Logo 候选区域（启发式，非深度学习模型，请人工确认）"
            if found
            else "未找到明显的 Logo 候选区域，请手动框选"
        )
        return DetectionResult(self.name, mask, message, cands, time.time() - t0, 0.45 if found else 0.0)


# --------------------------------------------------------------------------
# 5. 马赛克
# --------------------------------------------------------------------------
class MosaicDetector:
    """马赛克/像素化遮挡区域检测（见 mask_processor.detect_mosaic_regions）。"""

    name = "马赛克区域检测"

    def detect(self, image_rgb: np.ndarray, dilate: int = 0, **kwargs) -> DetectionResult:
        t0 = time.time()
        mask, info = mp.detect_mosaic_regions(image_rgb, **kwargs)
        if dilate:
            mask = mp.dilate_mask(mask, dilate)
        return DetectionResult(
            name=self.name,
            mask=mask,
            message=str(info.get("message", "")),
            details=[{k: v for k, v in info.items() if k != "message"}],
            seconds=time.time() - t0,
            confidence=float(np.clip((info.get("grid_ratio") or 0) / 3.0, 0, 0.9)),
        )


# --------------------------------------------------------------------------
# 统一入口
# --------------------------------------------------------------------------
class WatermarkDetector:
    """统一的水印检测调度器。"""

    def __init__(self, ocr: Optional[OCRDetector] = None):
        self.ocr = OCRWatermarkDetector(detector=ocr)
        self.translucent = TranslucentWatermarkDetector()
        self.repeated = RepeatedWatermarkDetector(self.translucent)
        self.logo = LogoDetector()
        self.mosaic = MosaicDetector()

    def detect_text(
        self,
        image_rgb: np.ndarray,
        min_score: float = 0.5,
        watermark_only: bool = False,
        kinds: Optional[Sequence[str]] = None,
        dilate: int = 4,
    ) -> DetectionResult:
        return self.ocr.detect(
            image_rgb, min_score=min_score, watermark_only=watermark_only, kinds=kinds, dilate=dilate
        )

    def detect_translucent(self, image_rgb: np.ndarray, **kwargs) -> DetectionResult:
        return self.translucent.detect(image_rgb, **kwargs)

    def detect_repeated(self, image_rgb: np.ndarray, response_mask=None, **kwargs) -> DetectionResult:
        return self.repeated.detect(image_rgb, response_mask=response_mask, **kwargs)

    def detect_logo(self, image_rgb: np.ndarray, **kwargs) -> DetectionResult:
        return self.logo.detect(image_rgb, **kwargs)

    def detect_mosaic(self, image_rgb: np.ndarray, **kwargs) -> DetectionResult:
        return self.mosaic.detect(image_rgb, **kwargs)

    def detect_all(self, image_rgb: np.ndarray, dilate: int = 4) -> List[DetectionResult]:
        """依次执行所有检测器（OCR 失败不会中断其它检测器）。"""
        results: List[DetectionResult] = []
        try:
            results.append(self.detect_text(image_rgb, dilate=dilate))
        except Exception as exc:
            results.append(DetectionResult("OCR 文字检测", np.zeros(_as_rgb(image_rgb).shape[:2], np.uint8), f"OCR 不可用：{exc}", [], 0.0, 0.0))
        translucent = self.detect_translucent(image_rgb, dilate=dilate)
        results.append(translucent)
        results.append(self.detect_repeated(image_rgb, response_mask=translucent.mask, dilate=dilate))
        results.append(self.detect_logo(image_rgb, dilate=dilate))
        results.append(self.detect_mosaic(image_rgb, dilate=dilate))
        return results


def format_report(results: Sequence[DetectionResult]) -> str:
    """把检测结果整理成中文报告文本。"""
    lines = []
    for r in results:
        tag = "✅" if r.found else "➖"
        lines.append(f"{tag} **{r.name}**：{r.message}（{r.seconds:.2f}s）")
    return "\n\n".join(lines)
