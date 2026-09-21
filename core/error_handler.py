"""统一错误处理模块。

目标：
1. 任何异常都写入日志（含完整 traceback），方便排查；
2. 网页上只显示简短、友好的中文提示，绝不把 Python 堆栈丢给用户；
3. 回调出错时返回安全值，避免 Gradio 组件进入 "Error" 状态。

用法（Gradio 回调）::

    @guarded(fallback=[None, None, "（未开始）"], action="载入图片")
    def on_upload(image):
        ...
        return editor_value, result, "已载入"

约定：**回调的最后一个输出是状态文本（Markdown）**，出错时会被替换为友好提示。
"""

from __future__ import annotations

import functools
import logging
import traceback
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence

from config import LOG_DIR, ensure_dirs, get_logger, setup_logging

logger = get_logger("ai_restore.error")


class AppError(Exception):
    """带"用户可见文案"的业务异常。"""

    def __init__(self, user_message: str, detail: str = ""):
        super().__init__(user_message)
        self.user_message = user_message
        self.detail = detail


# --------------------------------------------------------------------------
# 异常 -> 中文提示
# --------------------------------------------------------------------------
_IMAGE_READ_HINT = "图片读取失败，请重新上传 JPG、PNG 或 WEBP 格式的图片。"
_IMAGE_DATA_HINT = "图片数据格式异常，请重新上传图片。"
_DEVICE_HINT = "显卡显存或内存不足，请尝试降低“处理分辨率上限”后重试。"


def _text_of(exc: BaseException) -> str:
    return f"{exc.__class__.__name__}: {exc}".lower()


def friendly_message(exc: BaseException) -> str:
    """把异常翻译成用户能看懂的中文提示。"""
    if isinstance(exc, AppError):
        return exc.user_message

    text = _text_of(exc)
    name = exc.__class__.__name__

    # 1. 文件系统类
    if isinstance(exc, FileNotFoundError):
        if "lama" in text or "big-lama" in text or "模型" in str(exc):
            return "AI 修复模型加载失败，请检查 models/big-lama.pt 是否存在。"
        return "文件不存在或已被移动，请重新上传图片。"
    if isinstance(exc, (PermissionError, IsADirectoryError)):
        return "文件无法读取（权限不足），请换一张图片或检查文件是否被占用。"

    # 2. 图片读取类
    if name in {"UnidentifiedImageError", "DecompressionBombError"} or "cannot identify image file" in text:
        return _IMAGE_READ_HINT
    if "truncated" in text or "image file is truncated" in text:
        return "图片文件不完整（可能下载/传输中断），请重新上传。"
    if isinstance(exc, OSError) and ("image" in text or "png" in text or "jpeg" in text):
        return _IMAGE_READ_HINT

    # 3. 显存 / 内存
    if "out of memory" in text or "cuda oom" in text or isinstance(exc, MemoryError):
        return _DEVICE_HINT
    if "cuda" in text and ("error" in text or "failed" in text):
        return "CUDA 调用失败，已可尝试改用 CPU 模式（高级设置 → 修复后端）。"

    # 4. 模型
    if "lama" in text and ("加载" in str(exc) or "load" in text or "open file failed" in text):
        return "AI 修复模型加载失败，请检查 models/big-lama.pt 是否存在。"
    if "no such file" in text and "big-lama" in text:
        return "AI 修复模型加载失败，请检查 models/big-lama.pt 是否存在。"

    # 5. 业务校验类（我们主动抛出的中文提示，直接沿用）
    if isinstance(exc, ValueError):
        msg = str(exc)
        if msg and any("\u4e00" <= ch <= "\u9fff" for ch in msg):
            return msg
        return "参数不正确，请调整设置后重试。"

    # 6. OpenCV / 数据格式
    if name == "error" and "opencv" in str(type(exc)).lower():
        return "图像处理失败，请换一张图片重试。"
    if isinstance(exc, (IndexError, KeyError, TypeError, AttributeError)):
        return _IMAGE_DATA_HINT

    return "处理失败，请重试；如果一直失败，请把 logs/app.log 发给开发者。"


# --------------------------------------------------------------------------
# 日志
# --------------------------------------------------------------------------
def log_exception(exc: BaseException, context: str = "", extra: Optional[dict] = None) -> str:
    """记录完整 traceback，返回给用户看的中文提示。"""
    try:
        ensure_dirs()
        # 确保日志文件已初始化（例如在脚本/测试中直接调用本模块时）
        if not logging.getLogger("ai_restore").handlers:
            setup_logging()
        detail = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        head = f"[{context}] {exc.__class__.__name__}: {exc}"
        if extra:
            head += f" | extra={extra}"
        logger.error("%s\n%s", head, detail)
        try:
            (LOG_DIR / "app.log").parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
    except Exception:  # 日志本身不能影响主流程
        pass
    return friendly_message(exc)


# --------------------------------------------------------------------------
# Gradio 回调装饰器
# --------------------------------------------------------------------------
def guarded(fallback: Sequence[Any], action: str = "处理"):
    """回调保护装饰器：出错时写日志并返回安全值（最后一个输出为中文提示）。

    fallback 的长度必须与回调的输出个数一致。
    """
    fallback = list(fallback)

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001 - 这里就是要兜住所有异常
                hint = log_exception(exc, context=f"{action} / {fn.__name__}")
                outs = list(fallback)
                if not outs:
                    return f"❌ {hint}"
                outs[-1] = f"❌ {hint}"
                return tuple(outs)

        return wrapper

    return decorator


def safe_call(fn: Callable[[], Any], default: Any = None, context: str = "") -> Any:
    """执行一个可能失败的小操作，失败时返回默认值（内部已记录日志）。"""
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001
        log_exception(exc, context=context or getattr(fn, "__name__", "call"))
        return default


def check_image_size(width: int, height: int, max_pixels: int = 60_000_000, min_side: int = 8) -> None:
    """校验图片尺寸是否合法（过小或过大都给出中文提示）。"""
    if width <= 0 or height <= 0:
        raise AppError("图片尺寸无效，请重新上传图片。", detail=f"{width}x{height}")
    if min(width, height) < min_side:
        raise AppError(f"图片太小（{width}×{height}），无法进行修复处理。")
    if width * height > max_pixels:
        raise AppError(
            f"图片尺寸过大（{width}×{height}），请先缩小到 {max_pixels // 1_000_000} 百万像素以内。"
        )
