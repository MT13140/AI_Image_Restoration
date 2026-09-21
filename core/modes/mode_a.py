"""A 模式：智能识别自动水印（独立模块）。

职责边界：

* 调用 `core/auto_detect` 的 **A 打分链路**（类型 + 位置 + 重复性 + 半透明 + 人脸冲突），
  得到候选水印 Mask；
* 允许叠加用户涂抹（"自动 + 手动修正"）；
* 不做任何修复：Mask 交给 `core/pipeline.run_repair()`。

以后要优化 A 的检测，只改本文件（或 `core/auto_detect` 里的打分部分），
不会影响 B（走 mode_b 的文本模式）与 C（完全不碰检测）。
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from .. import auto_detect as ad
from .. import mask_processor as mp
from .base import (
    Mode,
    ModeRequest,
    ModeResult,
    combine_masks,
    manual_mask_from_editor,
    mask_stats,
)


class AutoWatermarkMode(Mode):
    """智能识别：给出带置信度的候选，默认勾选高分候选。"""

    id = "auto"
    label = "智能识别"
    hint = "自动找出文字/水印候选并给出置信度，确认后再修复；也可以用手画笔补充。"
    uses_auto_detection = True

    # ---------------- 检测 ----------------
    def detect(self, image: np.ndarray, req: ModeRequest) -> ad.WatermarkReport:
        """只做检测，返回报告（界面用它来显示"发现 N 个候选"）。"""
        settings = ad.resolve_settings(
            sensitivity=req.sensitivity,
            dilate=max(2, int(req.dilate)),
            ocr_min_score=float(req.ocr_min_score) if req.ocr_min_score else None,
            protect_faces=bool(req.protect_faces),
        )
        return ad.detect_watermarks(image, settings=settings, mode="auto")

    # ---------------- 产出 Mask ----------------
    def run(self, req: ModeRequest) -> ModeResult:
        manual = manual_mask_from_editor(req.editor_value, req.image.shape[:2])
        confirmed = req.confirmed_mask
        if confirmed is not None and not mp.is_empty(confirmed):
            # 用户已经在预览里确认过 Mask → 不重复检测（省时间、也保证"所见即所修"）
            auto = mp.ensure_binary(confirmed, req.image.shape[:2])
            report = ad.WatermarkReport()
            report.summary = "使用你已经确认的 Mask（未重新检测）"
        else:
            report = self.detect(req.image, req)
            auto = report.mask_selected

        mask = combine_masks(auto, manual)
        stats = mask_stats(mask)
        counts = report.counts()
        details = dict(report.stats or {})
        details.update({
            "high": counts["high"], "medium": counts["medium"], "low": counts["low"],
            "mask_regions": stats["regions"], "mask_ratio": stats["ratio"],
            "mode_label": self.label,
        })
        lines: List[str] = [report.summary]
        if report.face_message:
            lines.append(report.face_message)
        lines.append(f"高置信度 {counts['high']} ｜ 中 {counts['medium']} ｜ 低 {counts['low']}")
        if stats["regions"]:
            lines.append(f"待修复 Mask：{stats['regions']} 处，占画面 {stats['ratio'] * 100:.2f}%")
        lines.append(f"检测耗时：{ad.timings_text(report.timings)}")

        if stats["regions"] == 0:
            return ModeResult(
                mode=self.id, mask=None, ok=False,
                message="没有发现可修复的区域：可以换用「文字全部去除」，或直接用手画笔涂抹。",
                details=details, candidates=list(report.candidates),
                warnings=list(report.warnings), timings=dict(report.timings), lines=lines,
            )
        return ModeResult(
            mode=self.id, mask=mask, ok=True,
            message=(f"发现 {len(report.candidates)} 个候选水印区域"
                     f"（高置信度 {counts['high']} 个），Mask 覆盖画面 {stats['ratio'] * 100:.2f}%"),
            details=details, candidates=list(report.candidates),
            warnings=list(report.warnings), timings=dict(report.timings), lines=lines,
        )


MODE = AutoWatermarkMode()
