"""AI 图片智能修复工具 — 程序入口。

普通用户直接双击 ``start.bat`` 即可；也可以手动运行：

    .venv\\Scripts\\python.exe app.py                # 默认 127.0.0.1:7860，端口占用时自动换端口
    .venv\\Scripts\\python.exe app.py --port 7861
    .venv\\Scripts\\python.exe app.py --no-browser    # 不自动打开浏览器
    .venv\\Scripts\\python.exe app.py --preload       # 启动时预加载 AI 模型
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
from pathlib import Path

# 本地工具：关闭 Gradio / HuggingFace 遥测，避免无网络时的启动等待
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="AI 图片智能修复工具（本地 Web 应用）")
    parser.add_argument("--host", default=None, help="监听地址，默认 127.0.0.1")
    parser.add_argument("--port", type=int, default=None, help="监听端口，默认 7860（被占用时自动+1）")
    parser.add_argument("--strict-port", action="store_true", help="端口被占用时直接报错，不自动换端口")
    parser.add_argument("--no-browser", action="store_true", help="启动后不自动打开浏览器")
    parser.add_argument("--share", action="store_true", help="生成 Gradio 公网分享链接（默认关闭）")
    parser.add_argument("--preload", action="store_true", help="启动时预先加载 AI 模型")
    return parser.parse_args(argv)


def port_available(host: str, port: int) -> bool:
    """检测端口是否可用。"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
            return True
        except OSError:
            return False


def pick_port(host: str, port: int, strict: bool = False, tries: int = 20) -> int:
    """端口被占用时自动向后寻找可用端口（7860 → 7861 → …）。"""
    if port_available(host, port):
        return port
    if strict:
        raise SystemExit(f"端口 {port} 已被占用，请用 --port 指定其它端口。")
    for offset in range(1, tries + 1):
        candidate = port + offset
        if port_available(host, candidate):
            print(f"  端口 {port} 已被占用，自动改用 {candidate}")
            return candidate
    raise SystemExit(f"从 {port} 开始的 {tries} 个端口都不可用，请关闭占用程序后重试。")


def main(argv=None) -> int:
    args = parse_args(argv)

    from config import (
        DEFAULT_HOST,
        DEFAULT_PORT,
        clear_cuda_cache,
        detect_device,
        ensure_dirs,
        get_logger,
        setup_logging,
    )

    setup_logging()
    logger = get_logger("ai_restore.app")
    ensure_dirs()

    device = detect_device()
    print("=" * 68)
    print("  AI 图片智能修复  |  AI Image Restoration")
    print("=" * 68)
    print(f"  AI 加速：{'CUDA' if device.is_gpu else 'CPU'}  |  {device.summary()}")
    if device.torch_available and not device.cuda_available:
        print("  提示：当前为 CPU 模式，速度较慢；建议在「高级设置」把处理分辨率上限调到 1280~1920。")
    print(f"  项目目录：{ROOT}")
    print("=" * 68)

    if args.preload:
        from core import inpainting as inp

        engine = inp.get_engine(backend="lama")
        ok, reason = engine.ensure_loaded()
        print(f"  模型预加载：{'成功' if ok else '失败 -> ' + reason}")
        clear_cuda_cache()

    from ui.interface import build_interface

    demo = build_interface()
    host = args.host or DEFAULT_HOST
    port = pick_port(host, args.port or DEFAULT_PORT, strict=args.strict_port)
    url = f"http://{'127.0.0.1' if host in ('0.0.0.0', '') else host}:{port}"
    print(f"  正在启动，请在浏览器打开： {url}")
    print("  关闭本窗口或按 Ctrl+C 可停止服务。")
    print("=" * 68)
    logger.info("启动 Web 服务 %s (device=%s)", url, device.device)

    try:
        demo.queue(max_size=16).launch(
            server_name=host,
            server_port=port,
            share=bool(args.share),
            inbrowser=not args.no_browser,
            # 网页上不显示 Python 报错（需求：错误只写 logs/app.log，页面只给中文提示）
            show_error=False,
            quiet=False,
            allowed_paths=[str(ROOT / "outputs"), str(ROOT / "temp")],
        )
    except KeyboardInterrupt:
        print("\n  已停止。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
