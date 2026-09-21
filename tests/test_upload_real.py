"""真实上传场景测试：覆盖各种图片格式、命名与尺寸，走完整修复链路。

每个用例都会执行：

    生成/写入真实文件 → Gradio 组件预处理 → normalize_image →
    Mask 生成 → LaMa 修复 → 结果与对比图保存 → 输出文件校验

同时验证异常输入（损坏文件、非图片、空文件、None、过小/过大）不会让程序崩溃，
而是给出友好的中文提示。

运行：
    .venv\\Scripts\\python.exe tests\\test_upload_real.py
    .venv\\Scripts\\python.exe tests\\test_upload_real.py --no-lama   # 只测格式与 Mask
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
import traceback
import zlib
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np  # noqa: E402
import cv2  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from core import image_processor as ip  # noqa: E402
from core import inpainting as inp  # noqa: E402
from core import mask_processor as mp  # noqa: E402
from core.error_handler import AppError  # noqa: E402

CASE_DIR = ROOT / "temp" / "upload_cases"
OUT_DIR = ROOT / "outputs" / "upload_cases"


# --------------------------------------------------------------------------
# 素材
# --------------------------------------------------------------------------
def _font(size: int):
    for cand in (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\arial.ttf"):
        try:
            return ImageFont.truetype(cand, size)
        except Exception:
            continue
    return ImageFont.load_default()


def make_photo(width: int, height: int) -> Image.Image:
    """生成一张带水印文字的“照片”（渐变 + 景物 + 半透明水印）。"""
    y, x = np.mgrid[0:height, 0:width]
    base = np.zeros((height, width, 3), np.float32)
    base[..., 0] = 70 + 90 * (x / max(1, width))
    base[..., 1] = 95 + 70 * (y / max(1, height))
    base[..., 2] = 150 - 60 * (x / max(1, width))
    base += 10 * np.sin((x + y) / 19.0)[..., None]
    img = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), "RGB")
    d = ImageDraw.Draw(img)
    d.rectangle([int(width * 0.05), int(height * 0.6), int(width * 0.35), int(height * 0.92)],
                fill=(198, 180, 152), outline=(120, 100, 80), width=3)
    d.ellipse([int(width * 0.55), int(height * 0.1), int(width * 0.75), int(height * 0.32)],
              fill=(250, 235, 180))

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    fs = max(14, int(min(width, height) * 0.06))
    od.text((int(width * 0.45), int(height * 0.8)), "www.example.com", font=_font(fs),
            fill=(255, 255, 255, 115))
    od.text((int(width * 0.45), int(height * 0.88)), "2024-05-01 12:30", font=_font(int(fs * 0.8)),
            fill=(255, 255, 255, 105))
    return Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")


def watermark_box(width: int, height: int) -> Tuple[int, int, int, int]:
    return (int(width * 0.43), int(height * 0.76), int(width * 0.99), int(height * 0.96))


# --------------------------------------------------------------------------
# 用例定义
# --------------------------------------------------------------------------
def build_cases() -> List[Dict[str, object]]:
    CASE_DIR.mkdir(parents=True, exist_ok=True)
    cases: List[Dict[str, object]] = []

    def add(name: str, filename: str, image: Image.Image, save_kwargs: dict | None = None, note: str = ""):
        path = CASE_DIR / filename
        image.save(path, **(save_kwargs or {}))
        cases.append({
            "name": name,
            "path": path,
            "size": image.size,
            "note": note or f"{image.mode} {image.size[0]}×{image.size[1]}",
        })

    photo = make_photo(720, 520)
    add("JPG", "case01.jpg", photo, {"format": "JPEG", "quality": 92})
    add("JPEG(扩展名)", "case02.jpeg", photo, {"format": "JPEG", "quality": 88})
    add("PNG", "case03.png", photo, {"format": "PNG"})
    add("WEBP", "case04.webp", photo, {"format": "WEBP", "quality": 90})

    # RGBA 透明 PNG：左上角一块完全透明
    rgba = photo.convert("RGBA")
    rgba_np = np.asarray(rgba).copy()
    rgba_np[:120, :180, 3] = 0
    add("RGBA 透明 PNG", "case05_rgba.png", Image.fromarray(rgba_np, "RGBA"), {"format": "PNG"})

    add("灰度 PNG", "case06_gray.png", photo.convert("L"), {"format": "PNG"})
    add("CMYK JPG", "case07_cmyk.jpg", photo.convert("CMYK"), {"format": "JPEG", "quality": 90})
    add("中文文件名", "中文水印测试图.jpg", photo, {"format": "JPEG", "quality": 90})
    add("文件名带空格", "my photo with spaces 2024.png", photo, {"format": "PNG"})
    add("小图片", "case10_tiny.png", make_photo(48, 36), {"format": "PNG"}, note="48×36 小尺寸")
    add("大图片", "case11_big.jpg", make_photo(2400, 1600), {"format": "JPEG", "quality": 85},
        note="2400×1600")
    add("高分辨率 PNG", "case12_hires.png", make_photo(3200, 2000), {"format": "PNG"},
        note="3200×2000")
    return cases


def build_bad_cases() -> List[Dict[str, object]]:
    CASE_DIR.mkdir(parents=True, exist_ok=True)
    bad: List[Dict[str, object]] = []

    broken = CASE_DIR / "坏文件_损坏的JPEG.jpg"
    broken.write_bytes(b"\xff\xd8\xff\xe0" + b"this is not a real jpeg" * 40)
    bad.append({"name": "损坏的 JPEG", "value": str(broken)})

    textfile = CASE_DIR / "其实是文本.txt"
    textfile.write_text("hello, this is not an image", encoding="utf-8")
    bad.append({"name": "文本文件冒充图片", "value": str(textfile)})

    empty = CASE_DIR / "空文件.png"
    empty.write_bytes(b"")
    bad.append({"name": "0 字节空文件", "value": str(empty)})

    bad.append({"name": "None（未上传）", "value": None})
    bad.append({"name": "不存在的路径", "value": str(CASE_DIR / "不存在.png")})
    bad.append({"name": "空列表", "value": []})
    bad.append({"name": "空 dict", "value": {}})
    return bad


# --------------------------------------------------------------------------
# 单个用例执行
# --------------------------------------------------------------------------
def run_case(case: Dict[str, object], use_lama: bool, max_side: int) -> str:
    from gradio.data_classes import FileData
    import gradio as gr

    from ui import interface as ui

    path: Path = case["path"]  # type: ignore[assignment]
    name = case["name"]

    # 1) 走 Gradio 组件预处理（与浏览器上传完全一致：FileData -> numpy）
    image_comp = gr.Image(type="numpy")
    payload = FileData(path=str(path), orig_name=path.name,
                       mime_type=None, size=path.stat().st_size)
    arr = image_comp.preprocess(payload)
    if not isinstance(arr, np.ndarray):
        raise RuntimeError(f"组件预处理返回类型异常: {type(arr)}")

    # 2) 走统一入口 normalize_image（文件路径分支，验证 EXIF/CMYK/透明处理）
    arr2, info = ip.normalize_image(str(path), max_side=max_side)
    if arr2.shape[2] != 3:
        raise RuntimeError(f"normalize 后通道数异常: {arr2.shape}")

    # 3) 界面回调：上传（必须返回带 composite 的编辑器值）
    editor, stored, stored_preview, preview, _auto, info_md, status = ui.on_upload(arr, None)
    if not (isinstance(editor, dict) and {"background", "layers", "composite"} <= set(editor)):
        raise RuntimeError(f"编辑器值缺少必要键: {list(editor) if isinstance(editor, dict) else type(editor)}")
    if "❌" in str(status):
        raise RuntimeError(f"上传回调返回错误: {status}")

    # 4) 生成 Mask（模拟用户在"编辑器预览"上涂抹水印区域）
    h, w = stored_preview.shape[:2]
    x0, y0, x1, y1 = watermark_box(w, h)
    layer = np.zeros((h, w, 4), np.uint8)
    layer[y0:y1, x0:x1, 0] = 255
    layer[y0:y1, x0:x1, 3] = 255
    editor_with_mask = mp.make_editor_value(stored_preview, layers=[layer])
    mask = mp.editor_to_mask(editor_with_mask, shape=(h, w))
    if mp.is_empty(mask):
        raise RuntimeError("Mask 生成失败")

    # 5) AI 去除（LaMa）
    t0 = time.time()
    if use_lama:
        result, compare, out_path, rstatus, _qc_md, _log_md = ui.on_remove(
            editor_with_mask, stored, stored_preview, None, "manual",
            str(ROOT / "temp" / "upload_out"),
            8, max_side, "lama", 6, 0.5, 0.2, 0, False
        )
        if result is None or not out_path:
            raise RuntimeError(f"修复失败: {rstatus[:160]}")
        out_file = Path(str(out_path))
        if not out_file.exists() or out_file.stat().st_size < 200:
            raise RuntimeError(f"输出文件异常: {out_file}")
        # 尺寸一致性校验（小图片的 PNG 体积本来就可能很小，所以按尺寸而不是体积判断）
        with Image.open(out_file) as chk:
            if chk.size != (stored.shape[1], stored.shape[0]):
                raise RuntimeError(
                    f"输出尺寸与输入不一致: {chk.size} vs {stored.shape[1]}×{stored.shape[0]}"
                )
        # Mask 需要按比例映射回原图分辨率后再校验修复区域
        if stored.shape[:2] != (h, w):
            full_mask = cv2.resize(mp.ensure_binary(mask, (h, w)), (stored.shape[1], stored.shape[0]),
                                   interpolation=cv2.INTER_NEAREST)
        else:
            full_mask = mp.ensure_binary(mask, stored.shape[:2])
        diff = np.abs(result.astype(np.int16) - stored.astype(np.int16)).max(axis=2)
        inside = float(diff[full_mask > 0].mean())
        if inside < 3.0:
            raise RuntimeError(f"修复区域几乎没有变化（平均差异 {inside:.2f}）")
        elapsed = time.time() - t0
        saved = f"结果 {out_file.name}（{out_file.stat().st_size / 1024:.0f} KB，区内变化 {inside:.1f}）"
    else:
        engine = inp.get_engine(backend="opencv")
        full_mask = mask if stored.shape[:2] == (h, w) else cv2.resize(
            mp.ensure_binary(mask, (h, w)), (stored.shape[1], stored.shape[0]), interpolation=cv2.INTER_NEAREST
        )
        outcome = engine.inpaint(stored, full_mask, dilate=0, max_side=max_side)
        saved = f"仅 Mask 测试（OpenCV 兜底 {outcome.seconds:.2f}s）"
        elapsed = time.time() - t0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ip.save_image(OUT_DIR / f"{path.stem}_result.png", result if use_lama else stored)
    return f"{ip.image_info_text(info)} ｜ 上传/掩膜/修复全部通过（{elapsed:.1f}s）｜ {saved}"


def run_bad_case(case: Dict[str, object]) -> str:
    name = case["name"]
    value = case["value"]
    try:
        ip.normalize_image(value)
    except AppError as exc:
        return f"已按预期拦截 → “{exc.user_message}”"
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"抛出的不是友好错误（{exc.__class__.__name__}: {exc}）") from exc
    raise RuntimeError("异常输入没有被拦截（应当报错）")


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="真实上传场景测试")
    parser.add_argument("--no-lama", action="store_true", help="跳过 LaMa 推理，只验证格式与 Mask")
    parser.add_argument("--max-side", type=int, default=1600, help="测试时的处理分辨率上限")
    args = parser.parse_args(argv)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if CASE_DIR.exists():
        shutil.rmtree(CASE_DIR, ignore_errors=True)

    print("=" * 78)
    print("真实上传场景测试：格式 / 命名 / 尺寸 / 异常输入")
    print("=" * 78)

    cases = build_cases()
    failures: List[str] = []
    passed = 0

    for case in cases:
        name = str(case["name"])
        try:
            detail = run_case(case, use_lama=not args.no_lama, max_side=args.max_side)
            print(f"✅ {name:<14} {detail}")
            passed += 1
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{name}: {exc}")
            print(f"❌ {name:<14} {exc.__class__.__name__}: {exc}")
            traceback.print_exc()

    print("\n" + "-" * 78)
    print("异常输入（必须被友好拦截，不能崩溃）")
    print("-" * 78)
    for case in build_bad_cases():
        name = str(case["name"])
        try:
            detail = run_bad_case(case)
            print(f"✅ {name:<16} {detail}")
            passed += 1
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{name}: {exc}")
            print(f"❌ {name:<16} {exc}")

    print("\n" + "=" * 78)
    total = len(cases) + len(build_bad_cases())
    print(f"通过 {passed}/{total}")
    print(f"测试图片目录：{CASE_DIR}")
    print(f"结果输出目录：{OUT_DIR}")
    if failures:
        print("\n失败项：")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("🎉 全部上传场景通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
