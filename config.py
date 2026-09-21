"""AI 图片智能修复工具 - 全局配置。

该模块只做三件事：
1. 统一定义项目内的目录与默认参数；
2. 检测运行设备（CUDA / CPU）并给出可读描述；
3. 提供日志初始化。

注意：本文件不直接 import torch，设备探测在函数内延迟导入，
这样即使 PyTorch 尚未安装，项目其余模块（如 OCR、Mask 处理）依然可用。
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

# --------------------------------------------------------------------------
# 目录
# --------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
TEMP_DIR = BASE_DIR / "temp"
LOG_DIR = BASE_DIR / "logs"

_ALL_DIRS = (MODELS_DIR, OUTPUTS_DIR, TEMP_DIR, LOG_DIR)


def ensure_dirs() -> None:
    """确保运行所需目录存在（幂等）。"""
    for d in _ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# 默认参数
# --------------------------------------------------------------------------
#: 修复时 Mask 默认扩张像素（够覆盖水印边缘即可；扩张过大会让 LaMa"凭空生成"更多内容，
#: 从而更容易出现色块/补丁感）
DEFAULT_MASK_DILATE = 4
#: 处理分辨率上限（长边），过大图片会先缩放，避免显存/内存溢出
DEFAULT_MAX_SIDE = 2560
#: 羽化宽度，用于修复结果与周围图像融合
DEFAULT_FEATHER = 6
#: Web 服务
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 7860

#: 中文字体候选（用于绘制对比图上的中文标注）
FONT_CANDIDATES = (
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\msyhbd.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\simsun.ttc",
    r"C:\Windows\Fonts\Deng.ttf",
)


# --------------------------------------------------------------------------
# 设备检测
# --------------------------------------------------------------------------
@dataclass
class DeviceInfo:
    """运行环境描述。"""

    device: str = "cpu"            # "cuda" 或 "cpu"
    name: str = "CPU"              # 人类可读名称
    torch_available: bool = False  # torch 是否可用
    torch_version: str = ""
    cuda_available: bool = False
    cuda_version: str = ""         # torch 编译时使用的 CUDA 版本
    gpu_name: str = ""
    vram_gb: float = 0.0
    note: str = ""

    @property
    def is_gpu(self) -> bool:
        return self.device == "cuda"

    def summary(self) -> str:
        """一行中文摘要，用于 UI 顶部展示。"""
        if self.cuda_available and self.torch_available:
            return (
                f"Device: CUDA ｜ GPU: {self.gpu_name} ｜ 显存: {self.vram_gb:.1f} GB "
                f"｜ PyTorch {self.torch_version} (CUDA {self.cuda_version})"
            )
        if self.torch_available:
            return (
                f"Device: CPU ｜ 未检测到可用的 NVIDIA CUDA ｜ "
                f"PyTorch {self.torch_version}（CPU 版）"
            )
        return f"Device: CPU ｜ 未安装 PyTorch（{self.note}）"

    def to_dict(self) -> dict:
        return {
            "device": self.device,
            "name": self.name,
            "torch_available": self.torch_available,
            "torch_version": self.torch_version,
            "cuda_available": self.cuda_available,
            "cuda_version": self.cuda_version,
            "gpu_name": self.gpu_name,
            "vram_gb": round(self.vram_gb, 2),
            "note": self.note,
            "summary": self.summary(),
        }


@lru_cache(maxsize=1)
def detect_device() -> DeviceInfo:
    """检测运行设备。结果会被缓存。"""
    info = DeviceInfo()
    try:
        import torch  # 延迟导入，避免强制依赖
    except Exception as exc:  # pragma: no cover - 环境相关
        info.note = f"import torch 失败: {exc.__class__.__name__}"
        return info

    info.torch_available = True
    info.torch_version = getattr(torch, "__version__", "")
    try:
        info.cuda_available = bool(torch.cuda.is_available())
    except Exception as exc:  # pragma: no cover
        info.cuda_available = False
        info.note = f"CUDA 检测失败: {exc.__class__.__name__}"

    if info.cuda_available:
        try:
            info.gpu_name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            info.vram_gb = float(props.total_memory) / (1024 ** 3)
            info.device = "cuda"
            info.name = info.gpu_name
        except Exception as exc:  # pragma: no cover
            info.device = "cpu"
            info.name = "CPU"
            info.note = f"GPU 信息读取失败: {exc.__class__.__name__}"
    info.cuda_version = getattr(getattr(torch, "version", None), "cuda", None) or ""
    return info


def torch_device() -> str:
    """返回 torch 可用设备字符串。"""
    return detect_device().device


def clear_cuda_cache() -> None:
    """释放 CUDA 缓存（在 torch 不可用时静默跳过）。"""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# --------------------------------------------------------------------------
# 日志
# --------------------------------------------------------------------------
_LOGGER_READY = False


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """初始化全局日志（控制台 + 文件）。"""
    global _LOGGER_READY
    logger = logging.getLogger("ai_restore")
    if _LOGGER_READY:
        return logger

    logger.setLevel(level)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s", "%H:%M:%S")

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    logger.addHandler(stream)

    try:
        ensure_dirs()
        file_handler = logging.FileHandler(LOG_DIR / "app.log", encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
    except Exception:
        pass

    logger.propagate = False
    _LOGGER_READY = True
    return logger


def get_logger(name: str = "ai_restore") -> logging.Logger:
    return logging.getLogger(name)


def env_flag(name: str, default: bool = False) -> bool:
    """读取布尔环境变量。"""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on", "y"}


if __name__ == "__main__":  # 便于快速自检
    ensure_dirs()
    dev = detect_device()
    print(f"项目目录: {BASE_DIR}")
    print(dev.summary())
    print(dev.to_dict())
