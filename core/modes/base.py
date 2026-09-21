"""A / B / C 三种处理模式的统一接口（模块隔离层）。

为什么要有这一层
----------------

任务要求：**三种模式必须模块化、相互隔离**——
以后要单独优化 A 或 B，都不能牵动已经验证有效的 C 手动模式。

所以这里把"界面"和"算法"之间切开，每个模式**只负责一件事：产出 Mask**：

    A 智能识别      → core/modes/mode_a.py（自动水印识别，可叠加用户涂抹）
    B 文字全部去除  → core/modes/mode_b.py（OCR 全文字，可叠加用户涂抹）
    C 手动选择      → core/modes/mode_c.py（只用用户涂抹，**不引用任何自动检测**）
                              ↓  Mask
                     core/pipeline.run_repair()  ← 第五轮修复引擎（三模式**共用**，保持冻结）
                              ↓
                             结果

依赖方向是**单向**的、可以机械校验：

    ui/interface.py、core/batch_processor.py
        → core/modes/mode_{a,b,c}.py
            → core/auto_detect.py（仅 A/B 使用） / core/mask_processor.py（仅 C 使用）
                → core/pipeline.py（修复引擎，模式层不修改它）

`mode_c.py` **不允许** import `auto_detect` / `watermark_detector`；
`tests/test_mode_isolation.py` 会用子进程实测这一点（防止以后被无意破坏）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from config import get_logger
from .. import mask_processor as mp

logger = get_logger("ai_restore.modes")

#: 用户涂抹覆盖率超过该比例时，判定为"前端画布回传异常"而不是真的涂满整张图
MANUAL_MASK_MAX_RATIO = 0.95


@dataclass
class ModeRequest:
    """一次"产出 Mask"的请求（所有模式共用同一份输入结构）。"""

    image: np.ndarray                                  # 全分辨率 RGB 原图
    editor_value: Any = None                           # 用户在编辑器里的涂抹（Gradio ImageEditor 值）
    confirmed_mask: Optional[np.ndarray] = None         # A/B：已经确认过的自动 Mask
    sensitivity: str = "aggressive"                     # A/B：检测敏感度
    dilate: int = 3                                     # A/B：Mask 扩张像素
    ocr_min_score: float = 0.40                         # A/B：OCR 置信度阈值
    protect_faces: bool = True                          # A/B：人脸冲突提示
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModeResult:
    """模式产出的结果（**只有 Mask 是给修复引擎的**，其余都是给界面/日志看的）。"""

    mode: str
    mask: Optional[np.ndarray] = None
    ok: bool = False
    message: str = ""                                   # 主界面显示的一句话状态
    details: Dict[str, Any] = field(default_factory=dict)
    candidates: List[Any] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    timings: Dict[str, float] = field(default_factory=dict)
    lines: List[str] = field(default_factory=list)       # 给"检测详情"的多行说明
    #: 该模式对修复阶段的**建议参数**（例如 B 的 "hole_pad"）。
    #: 由模式层给出、修复层读取，A/B/C 因此可以各自不同而互不影响。
    repair_hints: Dict[str, Any] = field(default_factory=dict)


class Mode:
    """模式基类：只定义"输入图片 → 输出 Mask"，不涉及任何修复算法。"""

    id: str = ""
    label: str = ""
    hint: str = ""
    #: 该模式是否会用到自动检测（C 恒为 False，可据此校验隔离性）
    uses_auto_detection: bool = False
    #: 该模式默认的修复建议（见 :attr:`ModeResult.repair_hints`）
    repair_hints: Dict[str, Any] = {}

    def run(self, req: ModeRequest) -> ModeResult:      # pragma: no cover - 接口定义
        raise NotImplementedError


# --------------------------------------------------------------------------
# 各模式共用的"用户涂抹"解析（C 的唯一来源；A/B 的补充区域）
# --------------------------------------------------------------------------
def manual_mask_from_editor(editor_value, shape: Tuple[int, int]) -> np.ndarray:
    """把编辑器里的涂抹解析成原图分辨率的二值 Mask。

    * 只读"用户画的内容"，不调用任何自动检测；
    * 覆盖率 > 95% 时判为前端回传异常并忽略（保护"没涂到的地方不修改"这条底线）。
    """
    h, w = int(shape[0]), int(shape[1])
    mask = np.zeros((h, w), np.uint8)
    if editor_value is None:
        return mask
    try:
        preview = mp.editor_to_mask(editor_value)
    except Exception as exc:                       # noqa: BLE001 - 解析失败当作没涂
        logger.warning("解析编辑器涂抹失败：%s", exc)
        return mask
    if preview is None or mp.is_empty(preview):
        return mask
    if preview.shape[:2] != (h, w):
        import cv2
        preview = cv2.resize(preview, (w, h), interpolation=cv2.INTER_NEAREST)
    mask = mp.ensure_binary(preview, (h, w))
    ratio = float(np.count_nonzero(mask)) / float(mask.size) if mask.size else 0.0
    if ratio > MANUAL_MASK_MAX_RATIO:
        logger.warning("涂抹掩膜覆盖整张图的 %.1f%%，判定为前端回传异常，已忽略", ratio * 100.0)
        return np.zeros((h, w), np.uint8)
    return mask


def combine_masks(*masks: Optional[np.ndarray]) -> np.ndarray:
    """若干掩膜取并集；全空时返回**同形状的空掩膜**（调用方永远拿到 ndarray）。"""
    valid = [m for m in masks if m is not None and not mp.is_empty(m)]
    if valid:
        return mp.combine_masks(*valid)
    for m in masks:                     # 保持尺寸信息，避免下游拿到 1×1
        if m is not None:
            return np.zeros(np.asarray(m).shape[:2], np.uint8)
    return np.zeros((1, 1), np.uint8)


def mask_stats(mask: Optional[np.ndarray]) -> Dict[str, float]:
    """给界面/日志用的掩膜统计（区域数、覆盖率）。"""
    if mask is None or mp.is_empty(mask):
        return {"regions": 0, "ratio": 0.0}
    boxes = mp.mask_to_boxes(mask, min_area=8)
    return {"regions": int(len(boxes)), "ratio": float(mp.area_ratio(mask))}


def empty_result(mode: str, message: str, lines: Optional[Sequence[str]] = None,
                 warnings: Optional[Sequence[str]] = None) -> ModeResult:
    """构造"没有可修区域"的结果（统一文案，界面上就是一句友好提示）。"""
    return ModeResult(mode=mode, mask=None, ok=False, message=message,
                      warnings=list(warnings or []), lines=list(lines or []))
