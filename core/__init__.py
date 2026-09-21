"""核心算法包：Mask 处理 / OCR / 水印检测 / AI 修复 / 图像后处理 / 三种模式。

**这里刻意不做"提前 import"。**

早期版本在本文件里一次性 import 了所有子模块，后果是：
只要用到任何一个子模块（哪怕是纯手动的 Mask 处理），
都会把 OCR、水印检测、LaMa 推理全部加载进来 —— 这既是启动慢的原因，
也让"A / B / C 三种模式相互隔离"变成空话（改 A 很容易牵动 C）。

现在改成按需加载（PEP 562 的模块级 ``__getattr__``）：

* ``from core import mask_processor`` 之类的写法**完全不受影响**；
* 只有真正用到某个子模块时，它才会被 import；
* 手动模式（C）的运行路径里不会再出现 OCR / 自动检测模块，
  可以被 ``tests/test_mode_isolation.py`` 用子进程实测证明。
"""

from __future__ import annotations

import importlib
from typing import List

#: 允许用 ``core.xxx`` 形式按需访问的子模块
_SUBMODULES = (
    "auto_detect",
    "batch_processor",
    "error_handler",
    "face_protector",
    "folder_picker",
    "image_processor",
    "inpainting",
    "mask_processor",
    "mask_refiner",
    "modes",
    "ocr_detector",
    "pipeline",
    "watermark_candidate",
    "watermark_detector",
)

__all__: List[str] = list(_SUBMODULES)


def __getattr__(name: str):
    """按需 import 子模块（``core.image_processor`` / ``core.modes`` …）。"""
    if name in _SUBMODULES:
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module          # 缓存，后续访问不再走 __getattr__
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> List[str]:
    return sorted(list(globals().keys()) + list(_SUBMODULES))
