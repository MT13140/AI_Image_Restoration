"""AI 图片智能修复工具 - 全流程自动测试。

覆盖任务要求的 8 项测试：
  1. Python 环境
  2. 依赖 import
  3. 项目语法
  4. 启动 Gradio
  5. 图片读取
  6. Mask 创建
  7. Inpainting 模型加载
  8. 完整图片修复流程

运行：
    .venv\\Scripts\\python.exe tests\\test_all.py
    .venv\\Scripts\\python.exe tests\\test_all.py --skip-launch   # 跳过 Web 启动测试
"""

from __future__ import annotations

import argparse
import importlib
import os
import platform
import socket
import sys
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, List, Tuple

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Windows 控制台默认可能是 GBK，这里统一为 UTF-8 并允许替换无法编码的字符，
# 避免测试报告因为符号（✅ / ▶）而抛 UnicodeEncodeError。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

TEMP_TEST_DIR = ROOT / "temp" / "test"
OUT_TEST_DIR = ROOT / "outputs" / "test"
RESULTS: List[Tuple[str, bool, str, float]] = []


# --------------------------------------------------------------------------
# 输出工具
# --------------------------------------------------------------------------
def banner(text: str) -> None:
    print("\n" + "=" * 74)
    print(text)
    print("=" * 74)


def run_test(name: str, fn: Callable[[], str]) -> bool:
    print(f"\n▶ {name}")
    t0 = time.time()
    try:
        detail = fn() or "OK"
        ok = True
    except Exception as exc:
        detail = f"{exc.__class__.__name__}: {exc}"
        ok = False
        traceback.print_exc()
    dt = time.time() - t0
    RESULTS.append((name, ok, detail, dt))
    print(f"  {'✅ 通过' if ok else '❌ 失败'}  {detail}  ({dt:.2f}s)")
    return ok


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


# --------------------------------------------------------------------------
# 测试素材
# --------------------------------------------------------------------------
def _font(size: int):
    for cand in (
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\segoeui.ttf",
    ):
        try:
            return ImageFont.truetype(cand, size)
        except Exception:
            continue
    return ImageFont.load_default()


def make_test_image(width: int = 720, height: int = 520) -> np.ndarray:
    """构造一张含多种"待修复目标"的测试图（真实像素内容，非随机噪声）。"""
    y, x = np.mgrid[0:height, 0:width]
    base = np.zeros((height, width, 3), np.float32)
    base[..., 0] = 70 + 90 * (x / width)
    base[..., 1] = 90 + 70 * (y / height)
    base[..., 2] = 150 - 60 * (x / width) + 40 * np.sin(y / 40.0)
    base += 12 * np.sin((x + y) / 17.0)[..., None]
    img = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), "RGB")
    draw = ImageDraw.Draw(img)

    # 画面主体：建筑轮廓 + 天空 + 地面，让修复有真实的上下文纹理
    draw.rectangle([40, 300, 260, 470], fill=(196, 178, 150), outline=(120, 100, 80), width=3)
    draw.rectangle([80, 340, 140, 400], fill=(90, 130, 170), outline=(50, 70, 100), width=2)
    draw.rectangle([180, 340, 240, 400], fill=(90, 130, 170), outline=(50, 70, 100), width=2)
    draw.ellipse([300, 120, 420, 240], fill=(250, 235, 180))
    draw.rectangle([0, 470, width, height], fill=(86, 120, 76))
    for i in range(20):
        xi = int(i * width / 20)
        draw.line([xi, 470, xi + 6, height], fill=(64, 100, 58), width=3)
    for i in range(7):
        cx = 470 + i * 32
        draw.ellipse([cx, 360 - i * 4, cx + 40, 430], fill=(60 + i * 6, 110 + i * 5, 60))

    # 1) 半透明文字水印（右下角网站 URL）
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    od.text((392, 452), "www.example.com", font=_font(30), fill=(255, 255, 255, 105))
    od.text((404, 486), "2024-05-01 12:30", font=_font(24), fill=(255, 255, 255, 95))
    # 2) 半透明重复水印（中心斜向的小标记）
    for gy in range(60, 300, 96):
        for gx in range(60, width - 60, 150):
            od.text((gx, gy), "DEMO © 2024", font=_font(20), fill=(255, 255, 255, 70))
    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")

    # 3) 完全不透明遮挡（模拟贴纸/文字块），需要 AI 重建
    arr = np.asarray(img).copy()
    arr[250:300, 300:470] = (18, 18, 22)

    # 4) 马赛克区域（真实像素化遮挡）
    y0, y1, x0, x1 = 200, 268, 520, 660
    patch = arr[y0:y1, x0:x1]
    small = np.asarray(Image.fromarray(patch).resize((8, 6), Image.BILINEAR))
    mosaic = np.asarray(Image.fromarray(small).resize((x1 - x0, y1 - y0), Image.NEAREST))
    arr[y0:y1, x0:x1] = mosaic

    TEMP_TEST_DIR.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(TEMP_TEST_DIR / "synthetic_source.png")
    return arr


def make_manual_mask(shape) -> np.ndarray:
    """手工 Mask：覆盖遮挡块 + 两个水印区域。"""
    import cv2

    mask = np.zeros(shape[:2], np.uint8)
    cv2.rectangle(mask, (296, 246), (474, 304), 255, -1)      # 不透明遮挡块
    cv2.rectangle(mask, (386, 446), (700, 512), 255, -1)      # 右下角文字水印
    cv2.rectangle(mask, (515, 195), (665, 273), 255, -1)      # 马赛克区域
    return mask


# --------------------------------------------------------------------------
# 测试实现
# --------------------------------------------------------------------------
def test_1_python_env() -> str:
    info = [
        f"Python {platform.python_version()}",
        f"实现 {platform.python_implementation()}",
        f"系统 {platform.system()} {platform.release()}",
        f"解释器 {sys.executable}",
    ]
    if sys.version_info[:2] not in ((3, 10), (3, 11), (3, 12)):
        raise RuntimeError(f"Python 版本不受推荐: {platform.python_version()}（建议 3.10 / 3.11）")
    if platform.system() != "Windows":
        info.append("注意：当前不是 Windows 环境")
    venv = os.environ.get("VIRTUAL_ENV") or (".venv" in sys.executable.lower() and str(ROOT) in sys.executable)
    info.append(f"虚拟环境: {'是' if venv else '否（直接使用系统解释器）'}")
    return " ｜ ".join(info)


def test_2_imports() -> str:
    modules = [
        ("numpy", "数值计算"),
        ("cv2", "OpenCV 图像处理"),
        ("PIL", "Pillow 图像读写"),
        ("gradio", "Web 界面"),
        ("torch", "深度学习框架"),
        ("torchvision", "PyTorch 视觉库"),
        ("rapidocr_onnxruntime", "OCR（PP-OCRv4）"),
        ("onnxruntime", "ONNXRuntime 推理后端"),
        ("simple_lama_inpainting", "LaMa 修复模型封装"),
    ]
    ok_list, missing = [], []
    for name, desc in modules:
        try:
            mod = importlib.import_module(name)
            version = getattr(mod, "__version__", "")
            ok_list.append(f"{name}{'=' + str(version) if version else ''}")
        except Exception as exc:
            missing.append(f"{name}({desc}) -> {exc.__class__.__name__}")
    if missing:
        raise RuntimeError("缺失依赖: " + "; ".join(missing))
    return "已导入 " + ", ".join(ok_list)


def test_3_syntax() -> str:
    import py_compile

    files = sorted(
        p
        for p in ROOT.rglob("*.py")
        if ".venv" not in p.parts and "__pycache__" not in p.parts
    )
    errors = []
    for f in files:
        try:
            py_compile.compile(str(f), doraise=True, cfile=str(TEMP_TEST_DIR / (f.stem + ".pyc")))
        except py_compile.PyCompileError as exc:
            errors.append(f"{f.name}: {exc.msg}")
    if errors:
        raise RuntimeError("语法错误: " + " | ".join(errors))
    return f"全部 {len(files)} 个 Python 文件编译通过"


def test_4_gradio_launch() -> str:
    import gradio as gr

    from ui.interface import build_interface

    port = find_free_port()
    demo = build_interface()
    if not isinstance(demo, gr.Blocks):
        raise RuntimeError("build_interface() 未返回 gr.Blocks 实例")

    demo.launch(
        server_name="127.0.0.1",
        server_port=port,
        prevent_thread_lock=True,
        inbrowser=False,
        show_error=True,
        quiet=True,
    )
    try:
        status = None
        last_err = None
        for _ in range(40):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=5) as resp:
                    status = resp.status
                    body = resp.read(4000).decode("utf-8", "ignore")
                break
            except Exception as exc:  # 服务可能尚未就绪
                last_err = exc
                time.sleep(0.5)
        if status != 200:
            raise RuntimeError(f"HTTP 状态异常: {status} ({last_err})")
        if "gradio" not in body.lower() and "AI 图片智能修复" not in body:
            raise RuntimeError("返回内容不是 Gradio 页面")
        return f"Gradio {gr.__version__} 服务启动成功，HTTP 200，端口 {port}"
    finally:
        try:
            demo.close()
        except Exception:
            pass


def test_5_image_read() -> str:
    from core import image_processor as ip

    src = TEMP_TEST_DIR / "synthetic_source.png"
    img = ip.load_image(src)
    assert img.ndim == 3 and img.shape[2] == 3, f"读取结果通道异常: {img.shape}"
    assert img.dtype == np.uint8
    small, scale = ip.resize_max_side(img, 360)
    assert max(small.shape[:2]) <= 360 and 0 < scale < 1
    Path(OUT_TEST_DIR).mkdir(parents=True, exist_ok=True)
    saved = ip.save_image(OUT_TEST_DIR / "test_original.png", img)
    assert Path(saved).exists() and Path(saved).stat().st_size > 1000
    st = ip.image_stats(img)
    return f"读取 {st['width']}×{st['height']}，缩放测试 {small.shape[1]}×{small.shape[0]}，保存 {Path(saved).name}"


def test_6_mask_create() -> str:
    from core import image_processor as ip
    from core import mask_processor as mp

    img = ip.load_image(TEMP_TEST_DIR / "synthetic_source.png")
    h, w = img.shape[:2]

    manual = make_manual_mask(img.shape)
    assert mp.area_ratio(manual) > 0.02, "手工 Mask 面积异常"

    rects = [[10, 10, 40, 30], [60, 60, 90, 90]]
    rect_mask = mp.rects_to_mask((h, w), rects, percent=True)
    assert mp.area_ratio(rect_mask) > 0.02, "矩形 Mask 生成失败"

    dilated = mp.dilate_mask(manual, 10)
    assert np.count_nonzero(dilated) > np.count_nonzero(manual), "Mask 扩张无效"

    editor_value = {"background": img, "layers": [mp_to_layer(manual)]}
    from_editor = mp.editor_to_mask(editor_value, shape=(h, w))
    assert mp.area_ratio(from_editor) > 0.02, "ImageEditor 图层解析失败"

    combined = mp.combine_masks(manual, rect_mask)
    assert np.count_nonzero(combined) >= max(np.count_nonzero(manual), np.count_nonzero(rect_mask))

    overlay = mp.mask_overlay(img, combined)
    ip.save_image(OUT_TEST_DIR / "test_mask_overlay.png", overlay)
    ip.save_image(OUT_TEST_DIR / "test_mask.png", combined)

    mosaic_mask, mosaic_info = mp.detect_mosaic_regions(img)
    return (
        f"手工 Mask {mp.area_ratio(manual) * 100:.2f}%，矩形 {mp.area_ratio(rect_mask) * 100:.2f}%，"
        f"扩张后 {mp.area_ratio(dilated) * 100:.2f}%，编辑器解析 {mp.area_ratio(from_editor) * 100:.2f}%；"
        f"马赛克检测: {mosaic_info.get('message')}（Mask {mp.area_ratio(mosaic_mask) * 100:.2f}%）"
    )


def mp_to_layer(mask: np.ndarray) -> np.ndarray:
    m = (np.asarray(mask) > 0).astype(np.uint8)
    layer = np.zeros((*m.shape, 4), np.uint8)
    layer[..., 0] = 255
    layer[..., 3] = m * 255
    return layer


def test_7_model_load() -> str:
    from config import detect_device
    from core import inpainting as inp

    dev = detect_device()
    engine = inp.get_engine(backend="lama")
    t0 = time.time()
    ok, reason = engine.ensure_loaded()
    dt = time.time() - t0
    if not ok:
        raise RuntimeError(f"LaMa 模型加载失败: {reason}")
    backend = engine._get_backend()
    mode = getattr(backend, "mode", "?")
    weights = str(getattr(backend, "model_path", "") or "包内自动缓存")
    return (
        f"LaMa 加载成功（{dt:.2f}s）｜ 模式 {mode} ｜ 设备 {dev.device.upper()} ｜ 权重 {weights}"
    )


def test_8_full_pipeline() -> str:
    from config import detect_device
    from core import image_processor as ip
    from core import inpainting as inp
    from core import mask_processor as mp

    dev = detect_device()
    img = ip.load_image(TEMP_TEST_DIR / "synthetic_source.png")
    mask = make_manual_mask(img.shape)

    engine = inp.get_engine(backend="lama")
    outcome = engine.inpaint(
        img,
        mask,
        dilate=6,
        feather=6,
        color_match=0.5,
        sharpen=0.2,
        denoise=0,
        max_side=0,
        fallback=True,
    )
    assert outcome.backend == "lama", f"未使用 LaMa 后端（实际 {outcome.backend}）"
    assert outcome.image.shape == img.shape

    used = outcome.mask_used
    inside = used > 0
    outside = mp.dilate_mask(used, 8) == 0
    diff = np.abs(outcome.image.astype(np.int16) - img.astype(np.int16)).max(axis=2)
    inside_delta = float(diff[inside].mean())
    outside_delta = float(diff[outside].max()) if outside.any() else 0.0
    if inside_delta < 4.0:
        raise RuntimeError(f"修复区域几乎没有变化（平均差异 {inside_delta:.2f}），可能未真正执行 Inpainting")
    if outside_delta > 0.0:
        raise RuntimeError(f"Mask 之外的像素被改动（最大差异 {outside_delta:.1f}）")

    result_path = ip.save_image(OUT_TEST_DIR / "test_result.png", outcome.image)
    compare = ip.side_by_side(img, outcome.image, labels=("原图（含水印/遮挡）", "LaMa 修复结果"))
    compare_path = ip.save_image(OUT_TEST_DIR / "test_compare.png", compare)
    ip.save_image(OUT_TEST_DIR / "test_result_mask.png", outcome.mask_used)
    ip.save_image(OUT_TEST_DIR / "test_difference.png", ip.difference_map(img, outcome.image))

    return (
        f"LaMa 修复完成：{outcome.seconds:.2f}s ｜ 设备 {dev.device.upper()} ｜ "
        f"Mask {outcome.info['mask_ratio'] * 100:.2f}% ｜ 区内平均变化 {inside_delta:.1f} ｜ 区外无改动 ｜ "
        f"输出 {Path(result_path).name} / {Path(compare_path).name}"
    )


def test_9_ocr_pipeline() -> str:
    """附加测试：OCR 自动检测文字并生成 Mask。"""
    from core import image_processor as ip
    from core import mask_processor as mp
    from core import watermark_detector as wd

    img = ip.load_image(TEMP_TEST_DIR / "synthetic_source.png")
    det = wd.WatermarkDetector()
    res = det.detect_text(img, min_score=0.4, dilate=4)
    texts = [d.get("text", "") for d in res.details]
    if not texts:
        raise RuntimeError("OCR 未检测到任何文字，请检查 OCR 依赖")
    joined = " ".join(texts)
    kinds = {d.get("kind") for d in res.details}
    mask_ratio = mp.area_ratio(res.mask)
    overlay = mp.mask_overlay(img, res.mask)
    ip.save_image(OUT_TEST_DIR / "test_ocr_overlay.png", overlay)
    return f"识别 {len(texts)} 条文字（类型 {sorted(kinds)}），Mask {mask_ratio * 100:.2f}% ｜ 示例: {joined[:80]}"


def test_10_watermark_detectors() -> str:
    """附加测试：半透明 / 重复 / Logo / 马赛克检测器可运行。"""
    from core import image_processor as ip
    from core import mask_processor as mp
    from core import watermark_detector as wd

    img = ip.load_image(TEMP_TEST_DIR / "synthetic_source.png")
    det = wd.WatermarkDetector()
    trans = det.detect_translucent(img, dilate=2)
    rep = det.detect_repeated(img, response_mask=trans.mask, dilate=2)
    logo = det.detect_logo(img, dilate=2)
    mosaic = det.detect_mosaic(img)
    parts = [
        f"半透明 {mp.area_ratio(trans.mask) * 100:.2f}%（{trans.message}）",
        f"重复 {mp.area_ratio(rep.mask) * 100:.2f}%（{rep.message}）",
        f"Logo 候选 {mp.area_ratio(logo.mask) * 100:.2f}%（{logo.message}）",
        f"马赛克 {mp.area_ratio(mosaic.mask) * 100:.2f}%（{mosaic.message}）",
    ]
    combined = mp.combine_masks(
        trans.mask,
        rep.mask if not mp.is_empty(rep.mask) else np.zeros(img.shape[:2], np.uint8),
        logo.mask if not mp.is_empty(logo.mask) else np.zeros(img.shape[:2], np.uint8),
    )
    ip.save_image(OUT_TEST_DIR / "test_watermark_detection.png", mp.mask_overlay(img, combined))
    return "; ".join(parts)


def test_11_opencv_fallback() -> str:
    """附加测试：LaMa 不可用时的 OpenCV 兜底后端可用。"""
    from core import image_processor as ip
    from core import inpainting as inp

    img = ip.load_image(TEMP_TEST_DIR / "synthetic_source.png")
    mask = make_manual_mask(img.shape)
    engine = inp.get_engine(backend="opencv")
    outcome = engine.inpaint(img, mask, dilate=4, color_match=0.4, max_side=0)
    assert outcome.backend == "opencv"
    assert outcome.image.shape == img.shape
    ip.save_image(OUT_TEST_DIR / "test_result_opencv.png", outcome.image)
    return f"OpenCV Telea 后端可用（{outcome.seconds:.2f}s）"


def test_12_ui_handlers() -> str:
    """附加测试：界面回调链路 + Gradio 输出契约（防止网页再次出现 Error）。

    关键：所有返回 ImageEditor 值的回调必须包含 background/layers/composite，
    并且每个回调的返回值个数、类型都要能通过对应组件的 postprocess（这正是
    之前网页显示 "Error / tuple index out of range" 的地方）。
    """
    import gradio as gr

    from core import image_processor as ip
    from core import mask_processor as mp
    from ui import interface as ui

    img = ip.load_image(TEMP_TEST_DIR / "synthetic_source.png")

    editor_c = gr.ImageEditor(type="numpy")
    image_c = gr.Image(type="numpy")
    image_ro = gr.Image(type="numpy", interactive=False)
    state_c = gr.State()
    md_c = gr.Markdown()
    file_c = gr.File()
    df_c = gr.Dataframe()
    slider_c = gr.Slider(minimum=0, maximum=100)

    def contract(label: str, values, comps, status_index: int = -1) -> str:
        if len(values) != len(comps):
            raise RuntimeError(f"{label}: 返回值 {len(values)} 个与输出组件 {len(comps)} 个不匹配")
        problems = []
        for i, (val, comp) in enumerate(zip(values, comps)):
            try:
                comp.postprocess(val)
            except Exception as exc:  # noqa: BLE001
                problems.append(f"第{i + 1}个输出({comp.__class__.__name__}) {exc.__class__.__name__}: {exc}")
        if problems:
            raise RuntimeError(f"{label} 输出契约失败: " + "；".join(problems))
        status = str(values[status_index]) if values else ""
        if "❌" in status:
            raise RuntimeError(f"{label} 返回了错误提示: {status[:180]}")
        return status

    # 1) 上传：编辑器值必须带 background/layers/composite 三个键
    editor, stored, stored_preview, preview, auto0, info_md, status = ui.on_upload(img, None)
    if not (isinstance(editor, dict) and {"background", "layers", "composite"} <= set(editor)):
        keys = list(editor) if isinstance(editor, dict) else type(editor).__name__
        raise RuntimeError(f"编辑器值缺少必要键（Gradio 会抛 KeyError）: {keys}")
    contract("on_upload", [editor, stored, stored_preview, preview, auto0, info_md, status],
             [editor_c, state_c, state_c, image_c, state_c, md_c, md_c])

    # 2) 涂抹状态
    preview2, note = ui.on_editor_change(editor, stored, stored_preview, None)
    contract("on_editor_change", [preview2, note], [image_c, md_c])

    # 3) 自动检测（A 模式：候选 + Mask 预览）
    auto, cands, dpreview, dinfo, rows, dstatus = ui.on_auto_detect(
        editor, stored, stored_preview, "auto", "aggressive", 8, 0.4
    )
    if auto is None or mp.is_empty(auto):
        raise RuntimeError(f"自动检测未生成候选区域: {dstatus[:180]}")
    contract("on_auto_detect", [auto, cands, dpreview, dinfo, rows, dstatus],
             [state_c, state_c, image_c, md_c, df_c, md_c])

    # 3b) 表格勾选 → 重建 Mask（"应用表格修改"）
    table = [c.to_row() for c in cands]
    for row in table:
        row[0] = "选中"
    auto_t, dpreview_t, s_t, rows_t = ui.on_apply_table(
        table, cands, editor, stored, stored_preview, 8
    )
    contract("on_apply_table", [auto_t, dpreview_t, s_t, rows_t], [state_c, image_c, md_c, df_c])

    # 4) 确认自动 Mask（合并进编辑器）
    editor2, auto2, dpreview2, s2 = ui.on_confirm_detection(editor, stored, stored_preview, auto)
    contract("on_confirm_detection", [editor2, auto2, dpreview2, s2],
             [editor_c, state_c, image_c, md_c])

    # 5) AI 去除（真实 LaMa 推理 + 输出文件）—— 新签名：含模式与输出目录
    out_dir = str(TEMP_TEST_DIR / "ui_out")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    result, compare, path, rstatus, qc_md, log_md = ui.on_remove(
        editor2, stored, stored_preview, auto2, "auto", out_dir,
        8, 0, "lama", 6, 0.5, 0.2, 0, False
    )
    if result is None or compare is None or not path:
        raise RuntimeError(f"修复回调异常: {rstatus[:200]}")
    out_file = Path(str(path))
    if not out_file.exists() or out_file.stat().st_size < 5000:
        raise RuntimeError(f"输出文件缺失或过小: {out_file}")
    contract("on_remove", [result, compare, path, rstatus, qc_md, log_md],
             [image_ro, image_ro, file_c, md_c, md_c, md_c])

    # 5b) 批量处理（B 模式：逐张独立 OCR → Mask → 修复）
    batch_rows = ui.on_scan_batch("", [str(TEMP_TEST_DIR / "synthetic_source.png")],
                                  "text", "aggressive", 8, 0.4)
    contract("on_scan_batch", [batch_rows], [df_c])
    prog, bstatus, brows, blog = ui.on_batch_run(
        "", [str(TEMP_TEST_DIR / "synthetic_source.png")], str(TEMP_TEST_DIR / "batch_out"),
        "text", "png",
        True, True, True, "aggressive", 8, 1600, "lama", 6, 0.35, 0.1, 0, False,
        True, True, True, 0.4,
    )
    contract("on_batch_run", [prog, bstatus, brows, blog], [slider_c, md_c, df_c, md_c])

    # 6) 其余回调
    ed3, st3, pv3, disp3, auto3, cands3, info3, table3, s3 = ui.on_use_result(result, stored)
    if auto3 is not None or cands3 is not None:
        raise RuntimeError("继续处理时没有清空上一张图的 Mask/候选状态（第二次修复会沿用旧 Mask）")
    contract("on_use_result", [ed3, st3, pv3, disp3, auto3, cands3, info3, table3, s3],
             [editor_c, state_c, state_c, image_c, state_c, state_c, md_c, df_c, md_c])

    ed4, pv4, s4 = ui.on_clear_paint(ed3, st3, pv3)
    contract("on_clear_paint", [ed4, pv4, s4], [editor_c, image_c, md_c])

    au4, pv5, info5, s5 = ui.on_clear_detection(ed4, st3, pv3)
    contract("on_clear_detection", [au4, pv5, info5, s5], [state_c, image_c, md_c, md_c])

    ms = ui.on_preload_model("lama")
    contract("on_preload_model", [ms], [md_c])

    return (
        f"回调链路与输出契约全部通过 ｜ 检测候选 {len(mp.mask_to_boxes(auto, 12))} 个区域 ｜ "
        f"输出 {out_file.name}（{out_file.stat().st_size / 1024:.0f} KB）"
    )


def test_13_large_image_preview_mapping() -> str:
    """附加测试：大图缩略预览 + Mask 坐标映射（"画哪里就修哪里"）。"""
    import re

    from core import image_processor as ip
    from core import mask_processor as mp
    from ui import interface as ui

    cases = [
        ("900×640", 900, 640),
        ("1920×1080", 1920, 1080),
        ("2560×1440", 2560, 1440),
        ("3840×2160", 3840, 2160),
        ("竖图1280×1632", 1280, 1632),
    ]
    details = []
    for label, w, h in cases:
        photo = make_test_image(w, h)
        editor, original, preview, _disp, _auto, _info, status = ui.on_upload(photo, None)
        if original.shape[:2] != (h, w):
            raise RuntimeError(f"{label}: 原图分辨率被改动 {original.shape[:2]} != {(h, w)}")
        if max(preview.shape[:2]) > ui.EDITOR_MAX_SIDE:
            raise RuntimeError(f"{label}: 预览未缩略 {preview.shape[:2]}")

        # 在"预览坐标"里画一个已知矩形（右下角水印位置）
        ph, pw = preview.shape[:2]
        px0, py0, px1, py1 = int(pw * 0.45), int(ph * 0.80), int(pw * 0.98), int(ph * 0.95)
        layer = np.zeros((ph, pw, 4), np.uint8)
        layer[py0:py1, px0:px1, 0] = 255
        layer[py0:py1, px0:px1, 3] = 255
        painted = mp.make_editor_value(preview, layers=[layer])

        # 期望映射回原图的矩形
        sx, sy = w / float(pw), h / float(ph)
        exp = (px0 * sx, py0 * sy, px1 * sx, py1 * sy)

        result, compare, path, rstatus, _qc_md, _log_md = ui.on_remove(
            painted, original, preview, None, "manual", str(TEMP_TEST_DIR / "ui_out_13"),
            0, 2560, "lama", 6, 0.5, 0.2, 0, False
        )
        if result is None or not path:
            raise RuntimeError(f"{label}: 修复失败 {rstatus[:140]}")
        if result.shape[:2] != (h, w):
            raise RuntimeError(f"{label}: 输出分辨率异常 {result.shape[:2]} != {(h, w)}")
        # 掩膜必须落在"画的位置"上
        # 坐标映射要验证的是"用户涂抹范围" → 原图坐标（不含 Mask 自动收紧的影响）
        # 主状态只给一句话；路径等细节在「处理日志」里（UI 简化后的分工）
        m = re.search(r"(?:用户涂抹范围|掩膜图)：`([^`]+)`", str(rstatus) + "\n" + str(_log_md))
        if not m:
            raise RuntimeError(f"{label}: 状态与日志里都没有涂抹范围路径")
        mask_img = ip.load_image(Path(m.group(1)))
        mask_bin = mp.ensure_binary(mask_img)
        ys, xs = np.nonzero(mask_bin)
        if xs.size == 0:
            raise RuntimeError(f"{label}: 掩膜为空")
        got = (xs.min(), ys.min(), xs.max(), ys.max())
        offset = max(abs(got[i] - exp[i]) for i in range(4))
        if offset > 12:
            raise RuntimeError(
                f"{label}: Mask 位置偏移 {offset:.0f}px（期望 {tuple(round(v) for v in exp)}，实际 {got}）"
            )
        details.append(f"{label}: 原图{original.shape[1]}×{original.shape[0]} 预览{preview.shape[1]}×{preview.shape[0]} 偏移{offset:.0f}px")

    return "大图完整预览 + 坐标映射正确 ｜ " + "；".join(details)


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="AI 图片智能修复工具 - 自动测试")
    parser.add_argument("--skip-launch", action="store_true", help="跳过 Gradio 启动测试")
    parser.add_argument("--skip-ocr", action="store_true", help="跳过 OCR 测试")
    parser.add_argument("--only", default="", help="只运行名称包含该关键字的测试")
    args = parser.parse_args(argv)

    TEMP_TEST_DIR.mkdir(parents=True, exist_ok=True)
    OUT_TEST_DIR.mkdir(parents=True, exist_ok=True)

    banner("AI 图片智能修复工具 - 自动测试")
    print(f"项目根目录: {ROOT}")

    print("\n[准备] 生成测试图片…")
    make_test_image()
    print(f"  测试图: {TEMP_TEST_DIR / 'synthetic_source.png'}")

    tests: List[Tuple[str, Callable[[], str]]] = [
        ("测试 1：Python 环境", test_1_python_env),
        ("测试 2：依赖 import", test_2_imports),
        ("测试 3：项目语法", test_3_syntax),
    ]
    if not args.skip_launch:
        tests.append(("测试 4：启动 Gradio", test_4_gradio_launch))
    tests += [
        ("测试 5：图片读取", test_5_image_read),
        ("测试 6：Mask 创建", test_6_mask_create),
        ("测试 7：Inpainting 模型加载", test_7_model_load),
        ("测试 8：完整图片修复流程", test_8_full_pipeline),
    ]
    if not args.skip_ocr:
        tests.append(("测试 9（附加）：OCR 自动检测", test_9_ocr_pipeline))
    tests += [
        ("测试 10（附加）：水印检测器", test_10_watermark_detectors),
        ("测试 11（附加）：OpenCV 兜底后端", test_11_opencv_fallback),
        ("测试 12（附加）：界面回调链路与输出契约", test_12_ui_handlers),
        ("测试 13（附加）：大图预览与坐标映射", test_13_large_image_preview_mapping),
    ]

    for name, fn in tests:
        if args.only and args.only not in name:
            continue
        run_test(name, fn)

    banner("测试结果汇总")
    passed = sum(1 for _, ok, _, _ in RESULTS if ok)
    for name, ok, detail, dt in RESULTS:
        print(f"{'✅' if ok else '❌'} {name}  ({dt:.2f}s)\n    {detail}")
    print(f"\n通过 {passed}/{len(RESULTS)}")
    print(f"输出目录: {OUT_TEST_DIR}")

    failed = [name for name, ok, _, _ in RESULTS if not ok]
    if failed:
        print("\n❌ 失败项：" + "、".join(failed))
        return 1
    print("\n🎉 全部测试通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
