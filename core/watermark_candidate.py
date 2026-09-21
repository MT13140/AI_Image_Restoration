"""自动水印**候选**检测 —— 兼容层（第六轮）。

第六轮把检测算法整体重写进了 :mod:`core.auto_detect`（多尺度 OCR +
逐块去倾斜放大 OCR + 文字块合并 + 模板聚类 + 点阵展开）。本文件只做两件事：

1. 保持旧接口 ``detect_candidates`` / ``candidates_to_mask`` / 数据类可用，
   这样界面、批量处理与既有测试不需要改动调用方式；
2. 把"敏感度预设"翻译成具体的置信度阈值。

检测结果**永远只是候选**：不会自动修复、不会自动删除任何内容。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from .auto_detect import (                     # noqa: F401  （对外再导出）
    DEFAULT_SENSITIVITY,
    SENSITIVITY_PRESETS,
    CandidateReport,
    DetectSettings,
    TextEvidence,
    WatermarkCandidate,
    WatermarkGroup,
    WatermarkReport,
    candidates_to_mask,
    cluster_templates,
    detect_text_regions,
    detect_watermarks,
    get_detector,
    group_evidence,
    mask_from_candidates,
    resolve_settings,
    textness_map,
    timings_text,
)

#: 兼容旧默认值（"高置信度"的默认勾选阈值）
DEFAULT_SELECT_THRESHOLD = 0.50
#: 兼容旧默认值（候选区域总面积上限）
DEFAULT_MAX_AREA_RATIO = 0.35


def detect_candidates(
    image_rgb: np.ndarray,
    settings: Optional[DetectSettings] = None,
    min_confidence: Optional[float] = None,
    select_threshold: Optional[float] = None,
    protect_faces: bool = True,
    max_area_ratio: Optional[float] = None,
    dilate: int = 3,
    ocr_min_score: Optional[float] = None,
    sensitivity: Optional[str] = None,
    **extra: Any,
) -> WatermarkReport:
    """自动寻找"常见水印"的**候选区域**（绝不自动进入修复）。

    ``sensitivity``：``conservative`` / ``balanced`` / ``aggressive``（默认积极）。
    显式传入的 ``min_confidence`` / ``select_threshold`` 会覆盖预设阈值。
    ``settings`` 直接给出完整的 :class:`DetectSettings` 时，其它参数不再生效。
    """
    if settings is not None:
        return detect_watermarks(np.asarray(image_rgb), settings=settings, mode="auto")
    settings = resolve_settings(
        sensitivity=sensitivity,
        min_confidence=min_confidence if min_confidence is not None else extra.pop("min_conf", None),
        select_threshold=select_threshold if select_threshold is not None else extra.pop("select_conf", None),
        max_area_ratio=max_area_ratio,
        dilate=dilate,
        ocr_min_score=ocr_min_score,
        protect_faces=protect_faces,
        **extra,
    )
    return detect_watermarks(np.asarray(image_rgb), settings=settings)


def sensitivity_choices() -> List[tuple]:
    """给界面用的下拉项：``(显示名, 内部值)``。"""
    order = [("aggressive", "减少漏检优先（推荐）"),
             ("balanced", "均衡"),
             ("conservative", "只保留很有把握的")]
    out: List[tuple] = []
    for key, note in order:
        preset = SENSITIVITY_PRESETS.get(key)
        if not preset:
            continue
        out.append((f"{preset['label']}：{note}", key))
    return out


def describe_sensitivity(key: str) -> Dict[str, Any]:
    """返回某个敏感度对应的阈值说明（用于界面提示）。"""
    preset = SENSITIVITY_PRESETS.get(str(key), SENSITIVITY_PRESETS[DEFAULT_SENSITIVITY])
    return dict(preset)
