"""三种处理模式的注册表（A / B / C 相互隔离）。

界面与批处理**只通过这里**拿模式对象，不直接 import 具体检测实现：

    from core import modes
    result = modes.get_mode("text").run(modes.ModeRequest(image=img, editor_value=ev))
    out = pipe.run_repair(img, result.mask, options=...)      # 第五轮修复引擎

**关键设计：懒加载。**
三个模式的实现在这里**不预先 import**，而是等 `get_mode()` 真正被调用时才按需加载。
这样"只想用手动模式"的运行路径里，A/B 的自动检测与 OCR 模块**根本不会被 import**
（`tests/test_mode_isolation.py` 会用子进程实测 `sys.modules` 来证明这一点）。
以后要改 A，只动 `mode_a.py`（+ auto_detect 的 A 打分部分）；改 B 只动 `mode_b.py`；
两者都不会牵动 `mode_c.py`。
"""

from __future__ import annotations

import importlib
from typing import Dict, Iterator, List, Sequence, Tuple

from .base import (
    MANUAL_MASK_MAX_RATIO,
    Mode,
    ModeRequest,
    ModeResult,
    combine_masks,
    empty_result,
    manual_mask_from_editor,
    mask_stats,
)

#: 模式 id → (模块名, 对象名)。顺序即界面显示顺序：A、B、C。
_REGISTRY: Dict[str, Tuple[str, str]] = {
    "auto": ("core.modes.mode_a", "MODE"),
    "text": ("core.modes.mode_b", "MODE"),
    "manual": ("core.modes.mode_c", "MODE"),
}

#: 显示名（静态表，避免为了显示名字而 import 实现）
_LABELS: Dict[str, str] = {
    "auto": "智能识别",
    "text": "文字全部去除",
    "manual": "手动选择",
}

#: 默认模式（与之前版本保持一致）
DEFAULT_MODE = "auto"

_CACHE: Dict[str, Mode] = {}


def get_mode(mode_id: str) -> Mode:
    """按 id 取模式对象（首次调用时才 import 对应实现）。未知 id 回退默认模式。"""
    key = str(mode_id)
    if key not in _REGISTRY:
        key = DEFAULT_MODE
    if key not in _CACHE:
        module_name, attr = _REGISTRY[key]
        module = importlib.import_module(module_name)
        _CACHE[key] = getattr(module, attr)
    return _CACHE[key]


class _LazyRegistry:
    """dict 风格的只读视图（`modes.MODES["text"]` / `.values()` 都能用，但懒加载）。"""

    def __getitem__(self, key: str) -> Mode:
        return get_mode(key)

    def __contains__(self, key: object) -> bool:
        return str(key) in _REGISTRY

    def __len__(self) -> int:
        return len(_REGISTRY)

    def __iter__(self) -> Iterator[str]:
        return iter(_REGISTRY)

    def keys(self) -> List[str]:
        return list(_REGISTRY)

    def values(self) -> List[Mode]:
        return [get_mode(k) for k in _REGISTRY]

    def items(self):
        return [(k, get_mode(k)) for k in _REGISTRY]

    def get(self, key: str, default=None):
        return get_mode(key) if str(key) in _REGISTRY else default

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return f"<modes: {list(_REGISTRY)}>"


#: 模式 id → 模式对象（懒加载视图）
MODES = _LazyRegistry()


def mode_choices() -> List[Tuple[str, str]]:
    """给 `gr.Radio` 用的选项：``(显示名, id)``（不触发任何 import）。"""
    return [(_LABELS.get(k, k), k) for k in _REGISTRY]


def mode_label(mode_id: str) -> str:
    return _LABELS.get(str(mode_id), get_mode(mode_id).label)


def mode_ids() -> Sequence[str]:
    return list(_REGISTRY)


__all__ = [
    "MANUAL_MASK_MAX_RATIO", "MODES", "DEFAULT_MODE", "Mode", "ModeRequest", "ModeResult",
    "combine_masks", "empty_result", "get_mode", "manual_mask_from_editor",
    "mask_stats", "mode_choices", "mode_ids", "mode_label",
]
