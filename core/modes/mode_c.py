"""C 模式：手动选择（**稳定基线，独立模块**）。

这个文件**只做一件事**：把用户涂抹的区域变成 Mask。

* 不 import 自动检测（`auto_detect` / `watermark_detector`）——以后改 A/B 的检测算法，
  物理上碰不到这里；
* 不改任何修复参数：产出的 Mask 原样交给第五轮的 `core/pipeline.run_repair()`；
* 行为与之前 UI 里的手动路径**完全一致**（Mask 不额外扩张，交给修复引擎自己收紧）。

`tests/test_mode_isolation.py` 会校验：本模块不引用自动检测相关模块，
并且同一张图 + 同一个涂抹，产出结果与第五轮手动路径逐像素一致。
"""

from __future__ import annotations

from typing import List

from .base import Mode, ModeRequest, ModeResult, manual_mask_from_editor, mask_stats


class ManualMode(Mode):
    """手动选择：用户画哪里就只修哪里。"""

    id = "manual"
    label = "手动选择"
    hint = "用画笔涂掉要去除的内容，只修你涂到的地方（最可控，推荐处理单个水印）"
    uses_auto_detection = False

    def run(self, req: ModeRequest) -> ModeResult:
        mask = manual_mask_from_editor(req.editor_value, req.image.shape[:2])
        stats = mask_stats(mask)
        if stats["regions"] == 0:
            return ModeResult(
                mode=self.id, mask=None, ok=False,
                message="还没有选中要去掉的区域：请用画笔在图片上涂一下。",
                details=stats,
            )
        lines: List[str] = [
            f"涂抹区域：{stats['regions']} 处，占画面 {stats['ratio'] * 100:.2f}%",
            "手动模式不会自动增减任何区域；Mask 外像素保持原样。",
        ]
        return ModeResult(
            mode=self.id, mask=mask, ok=True,
            message=f"已选中 {stats['regions']} 处区域，可以开始修复。",
            details=stats, lines=lines,
        )


MODE = ManualMode()
