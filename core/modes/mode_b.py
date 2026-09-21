"""B 模式：文字全部去除（独立模块，批量主力）。

职责边界：

* 用 **OCR** 找出图片里所有可识别的文字，**默认全部**作为待去除区域（不做水印评分过滤）；
* 允许叠加用户涂抹（自动结果 + 手动补充）；
* OCR 一个文字都没找到时明确返回"没有可修区域"，由界面/批量流程显示"跳过"，绝不假装成功；
* 不做任何修复：Mask 交给 `core/pipeline.run_repair()`。

批处理也走这里：`detect_mask()` 是"单张图 → Mask"的唯一入口，
每张图片都独立调用（绝不复制上一张的 Mask）。
"""

from __future__ import annotations

from typing import List, Optional, Tuple

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


class TextRemovalMode(Mode):
    """文字全部去除：OCR 命中的文字默认全部划入待去除区域。"""

    id = "text"
    label = "文字全部去除"
    hint = "把图片里所有能识别出来的文字默认全部去掉（批量处理水印照片推荐）。"
    uses_auto_detection = True
    #: B 专用修复提示：**只把送进 LaMa 的孔洞扩开 6px**（写回范围一点不变）。
    #: 依据：文字笔画掩膜是"零余量孔洞"时，LaMa 会顺着孔洞两侧像素把原文字抄回来
    #: （实测残留 21.4%、红字变金色块）；孔洞加 4~6px 后残留 0.2%，
    #: 而实际修改范围（mask_used / 融合边界）保持与之前完全一致。
    #: A/C 不带这个提示 → 行为完全不变。
    repair_hints = {"hole_grow": 6, "keep_user_instances": True}

    # ---------------- 检测 ----------------
    def detect(self, image: np.ndarray, req: ModeRequest) -> ad.WatermarkReport:
        settings = ad.resolve_settings(
            sensitivity=req.sensitivity,
            dilate=max(2, int(req.dilate)),
            ocr_min_score=float(req.ocr_min_score) if req.ocr_min_score else None,
            protect_faces=bool(req.protect_faces),
        )
        return ad.detect_watermarks(image, settings=settings, mode="text")

    def detect_mask(self, image: np.ndarray, req: Optional[ModeRequest] = None,
                    ) -> Tuple[ad.WatermarkReport, Optional[np.ndarray]]:
        """批量处理用的最小接口：``(报告, 自动 Mask)``，每张图独立调用。"""
        r = req or ModeRequest(image=image)
        report = self.detect(image, r)
        return report, report.mask_selected

    # ---------------- 产出 Mask ----------------
    def run(self, req: ModeRequest) -> ModeResult:
        manual = manual_mask_from_editor(req.editor_value, req.image.shape[:2])
        confirmed = req.confirmed_mask
        if confirmed is not None and not mp.is_empty(confirmed):
            auto = mp.ensure_binary(confirmed, req.image.shape[:2])
            report = ad.WatermarkReport()
            report.summary = "使用你已经确认的 Mask（未重新检测）"
        else:
            report = self.detect(req.image, req)
            auto = report.mask_selected
        mask = combine_masks(auto, manual)
        stats = mask_stats(mask)
        ocr_regions = int(report.stats.get("ocr_regions") or len(report.candidates) or 0)
        details = dict(report.stats or {})
        details.update({
            "ocr_regions": ocr_regions, "mask_regions": stats["regions"],
            "mask_ratio": stats["ratio"], "mode_label": self.label,
        })
        lines: List[str] = [report.summary]
        lines.append(f"OCR 文字区域：{ocr_regions} 个")
        if report.face_message:
            lines.append(report.face_message)
        if stats["regions"]:
            lines.append(f"待修复 Mask：{stats['regions']} 处，占画面 {stats['ratio'] * 100:.2f}%")
        lines.append(f"检测耗时：{ad.timings_text(report.timings)}")

        if stats["regions"] == 0:
            return ModeResult(
                mode=self.id, mask=None, ok=False,
                message=("未检测到可识别文字（OCR 文字区域 0 个）→ 已跳过，原图不做任何修改。"),
                details=details, candidates=list(report.candidates),
                warnings=list(report.warnings), timings=dict(report.timings), lines=lines,
            )
        return ModeResult(
            mode=self.id, mask=mask, ok=True,
            repair_hints=dict(self.repair_hints),
            message=(f"检测到 {ocr_regions} 个文字区域，Mask 覆盖画面 "
                     f"{stats['ratio'] * 100:.2f}%"),
            details=details, candidates=list(report.candidates),
            warnings=list(report.warnings), timings=dict(report.timings), lines=lines,
        )


MODE = TextRemovalMode()
