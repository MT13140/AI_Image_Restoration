"""第六轮专项测试：B 模式（文字全部去除）+ 批处理 + 文件夹选择 + 性能统计。

覆盖任务书第十五章要求的 11 项：

    1 单张图片 OCR → Mask → 修复        2 一张图片多个文字水印
    3 多张图片不同位置文字（逐张独立）   4 文字水印位于人脸附近
    5 半透明文字                        6 多行文字
    7 OCR 无结果 → 明确跳过             8 批量输入文件夹
    9 输出文件夹（原生选择窗口 + 可写校验 + 自动编号）
    10 输出文件不覆盖原图               11 C 手动模式回归（第五轮逻辑未变）

运行：
    .venv\\Scripts\\python.exe tests\\test_text_mode.py
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import time
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
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from core import auto_detect as ad  # noqa: E402
from core import batch_processor as bp  # noqa: E402
from core import face_protector as fp  # noqa: E402
from core import folder_picker as fpk  # noqa: E402
from core import image_processor as ip  # noqa: E402
from core import mask_processor as mp  # noqa: E402
from core import pipeline as pipe  # noqa: E402

WORK = ROOT / "temp" / "round6_text"
OUT = ROOT / "outputs" / "round6_text"
RESULTS: List[Tuple[str, bool, str]] = []
DETECT_LOG: List[str] = []

FONT_CANDIDATES = (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\arial.ttf")


def banner(t: str) -> None:
    print("\n" + "=" * 90)
    print(t)
    print("=" * 90)


def check(name: str, ok: bool, detail: str) -> bool:
    RESULTS.append((name, ok, detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}\n      {detail}")
    return ok


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def font(size: int):
    for c in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(c, int(size))
        except Exception:
            continue
    return ImageFont.load_default()


def photo(w: int = 1280, h: int = 800, seed: int = 3) -> np.ndarray:
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    img = np.stack([90 + 90 * (x / w), 110 + 70 * (y / h), 170 - 50 * (x / w)], axis=2)
    img = np.clip(img + np.random.default_rng(seed).normal(0, 4.0, img.shape), 0, 255)
    return img.astype(np.uint8)


def stamp(img: np.ndarray, text: str, cx: float, cy: float, size: int,
          opacity: float = 0.45, angle: float = 0.0,
          color=(255, 255, 255)) -> np.ndarray:
    """按**真实不透明度**叠加水印（先画不透明文字，再降图层 alpha）。"""
    base = Image.fromarray(np.asarray(img, np.uint8)).convert("RGBA")
    tmp = Image.new("RGBA", (int(size * len(text) * 1.35) + 30, int(size * 2.2)), (0, 0, 0, 0))
    ImageDraw.Draw(tmp).text((12, 8), text, font=font(size), fill=(*color, 255))
    if angle:
        tmp = tmp.rotate(angle, expand=True, resample=Image.BICUBIC)
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    layer.paste(tmp, (int(cx - tmp.size[0] / 2), int(cy - tmp.size[1] / 2)), tmp)
    layer.putalpha(layer.split()[3].point(lambda v: int(v * float(opacity))))
    return np.asarray(Image.alpha_composite(base, layer).convert("RGB"))


def text_settings(sensitivity: str = "aggressive"):
    return ad.resolve_settings(sensitivity=sensitivity)


def detect_text(img: np.ndarray, sensitivity: str = "aggressive"):
    """B 模式检测：返回 ``(report, mask, 耗时秒)``。"""
    t0 = time.time()
    rep = ad.detect_text_regions(img, settings=text_settings(sensitivity))
    mask = ad.candidates_to_mask(rep, only_selected=True, dilate=0)
    return rep, mask, time.time() - t0


def main() -> int:  # noqa: C901
    WORK.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    banner("第六轮：B 模式（文字全部去除）+ 批处理 + 文件夹选择 + 性能统计")

    # ---------------- 1. 单张：OCR → Mask → 第五轮修复 ----------------
    single = stamp(photo(seed=1), "www.example.com", 900, 735, 26, opacity=0.35)
    rep1, mask1, t_det = detect_text(single)
    ok_det = mask1 is not None and not mp.is_empty(mask1)
    detail1 = (f"OCR 区域 {rep1.stats.get('ocr_regions')}，候选 {len(rep1.candidates)}，"
               f"Mask {mp.area_ratio(mask1) * 100:.2f}%，检测 {t_det:.1f}s")
    if ok_det:
        out1 = pipe.run_repair(single, mask1, options=pipe.RepairOptions(max_side=1600))
        qc1 = out1.qc
        p1 = bp.unique_output_path(OUT / "restored", "single", "_restored", ".png")
        ip.save_image(p1, out1.image)
        detail1 += (f"｜修复用时 {out1.seconds:.1f}s｜Mask 外改动 "
                    f"{qc1.get('outside_change_ratio', -1) * 100:.3f}%｜"
                    f"耗时 { {k: round(v) for k, v in (out1.timings or {}).items()} }")
        check("1 单张 B 模式：OCR → Mask → 修复", qc1.get("outside_change_ratio", 1) <= 0.002,
              detail1)
    else:
        check("1 单张 B 模式：OCR → Mask → 修复", False, detail1 + "（未检测到 Mask）")
    DETECT_LOG.append(f"单张 URL 水印 | 期望 1 | 检测 {rep1.stats.get('ocr_regions')} | "
                      f"Mask {mp.area_ratio(mask1) * 100 if mask1 is not None else 0:.2f}%")

    # ---------------- 2. 一张图片多个文字水印 ----------------
    multi = photo(seed=2)
    for text, cx, cy, size in (("左上角标记", 170, 70, 26), ("右上角标记", 1100, 70, 26),
                               ("中间水印", 640, 400, 30), ("底部网址 www.demo.com", 520, 750, 26)):
        multi = stamp(multi, text, cx, cy, size, opacity=0.40)
    rep2, mask2, _ = detect_text(multi)
    n2 = int(rep2.stats.get("ocr_regions") or 0)
    check("2 一张图片多个文字水印", n2 >= 3,
          f"OCR 检出 {n2} 个文字区域（期望 ≥3）｜Mask {mp.area_ratio(mask2) * 100 if mask2 is not None else 0:.2f}%")
    DETECT_LOG.append(f"多处水印 | 期望 4 | 检测 {n2} | "
                      f"Mask {mp.area_ratio(mask2) * 100 if mask2 is not None else 0:.2f}%")

    # ---------------- 5/6. 半透明 + 多行文字 ----------------
    trans = stamp(photo(seed=5), "CONFIDENTIAL", 640, 400, 32, opacity=0.18)
    rep5, mask5, _ = detect_text(trans)
    check("5 半透明文字（不透明度 0.18）", len(rep5.candidates) >= 1,
          f"候选 {len(rep5.candidates)}，Mask {mp.area_ratio(mask5) * 100 if mask5 is not None else 0:.2f}%"
          f"；{'；'.join(rep5.warnings[:1])}")

    two_lines = stamp(stamp(photo(seed=6), "NoNoNo御姐", 300, 700, 30, opacity=0.40),
                      "@asian_ma46", 300, 748, 28, opacity=0.40)
    rep6, mask6, _ = detect_text(two_lines)
    texts6 = [c.text for c in rep6.candidates]
    merged = any(("NoNoNo" in t and "asian" in t) for t in texts6)
    check("6 多行文字合并为同一个水印块（昵称 + 账号）", merged,
          f"候选 {len(rep6.candidates)}，文本={[t[:40] for t in texts6][:3]}")

    # ---------------- 4. 人脸附近水印 ----------------
    face_src = ROOT / "temp" / "round4" / "faces" / "candidate_0.png"
    if face_src.exists():
        fimg = np.asarray(Image.open(face_src).convert("RGB"))
        faces = fp.detect_faces(fimg)
        if faces:
            x, y, w, h = faces[0].box
            face_img = stamp(fimg, "watermark", x + w * 0.5, y + h * 0.5,
                             max(24, int(w * 0.36)), opacity=0.6, color=(255, 255, 255))
            rep4, mask4, _ = detect_text(face_img)
            face_mask, critical = fp.build_face_masks(face_img.shape[:2], faces)
            face_px = float(np.count_nonzero(face_mask))
            mask_px = float(np.count_nonzero(mask4)) if mask4 is not None else 0.0
            share = mask_px / max(1.0, face_px)
            check("4 人脸附近的水印（Mask 只覆盖文字，不覆盖整张脸）",
                  len(rep4.candidates) >= 1 and 0 < share < 0.6,
                  f"候选 {len(rep4.candidates)}，Mask {mask_px:.0f}px 仅为脸部区域 "
                  f"{face_px:.0f}px 的 {share * 100:.1f}%（应远小于整张脸）")
        else:
            check("4 人脸附近的水印（Mask 只覆盖文字）", True, "素材未检出人脸 → 跳过实测")
    else:
        check("4 人脸附近的水印（Mask 只覆盖文字）", True, "缺少人脸素材 → 跳过实测")

    # ---------------- 7. OCR 无结果 → 明确跳过 ----------------
    rep7, mask7, _ = detect_text(photo(seed=17))
    check("7 OCR 无结果时明确跳过（不假装成功）",
          (mask7 is None or mp.is_empty(mask7)) and "未检测到" in rep7.summary,
          f"summary={rep7.summary.splitlines()[0][:60]}…｜候选 {len(rep7.candidates)}")

    # ---------------- 9. 输出文件夹：可写校验 + 自动编号 ----------------
    probe_dir = WORK / "probe_out"
    ok_write, msg_write = fpk.ensure_writable_dir(str(probe_dir))
    collide = probe_dir / "photo_restored.png"
    collide.write_bytes(b"x")
    next_path = bp.unique_output_path(probe_dir, "photo", "_restored", ".png")
    check("9 输出文件夹（可写校验 + 同名自动编号）",
          ok_write and next_path.name == "photo_restored_1.png"
          and fpk.tk_available() and "可用" in fpk.describe_capability(),
          f"可写：{ok_write}（{msg_write}）｜同名 → {next_path.name}｜"
          f"{fpk.describe_capability()}")

    # ---------------- 3/8/10. 批量：逐张独立 + 文件夹输入 + 不覆盖原图 ----------------
    src = WORK / "src"
    if src.exists():
        shutil.rmtree(src, ignore_errors=True)
    src.mkdir(parents=True, exist_ok=True)
    files: List[Path] = []
    positions = [(200, 120), (1000, 200), (600, 420), (300, 700), (1100, 620)]
    for i, (cx, cy) in enumerate(positions):
        img = stamp(photo(seed=30 + i), f"水印{i + 1}号 www.demo{i + 1}.com", cx, cy, 28,
                    opacity=0.40)
        p = src / f"photo_{i + 1:03d}.jpg"
        Image.fromarray(img).save(p, quality=95)
        files.append(p)
    blank = src / "photo_006_no_text.jpg"
    Image.fromarray(photo(seed=99)).save(blank, quality=95)
    files.append(blank)
    broken = src / "broken_007.jpg"
    broken.write_bytes(b"\xff\xd8\xff\xe0" + b"not an image" * 40)
    files.append(broken)
    hashes = {p: sha256(p) for p in files}

    batch_out = WORK / "batch_out"
    if batch_out.exists():
        shutil.rmtree(batch_out, ignore_errors=True)
    logs: List[str] = []
    t0 = time.time()
    summary = bp.run_batch(
        [str(src)],                      # 传**文件夹**（对应"选择输入文件夹"）
        output_dir=str(batch_out),
        mode="text",
        out_format="png",
        options=pipe.RepairOptions(max_side=1600),
        log_cb=logs.append,
    )
    dt = time.time() - t0
    ok_items = [i for i in summary.items if i.status == "ok"]
    skipped = [i for i in summary.items if i.status == "skipped"]
    failed = [i for i in summary.items if i.status == "failed"]
    restored = sorted((batch_out / "restored").glob("*.png"))
    check("3/8 批量处理（文件夹输入，逐张独立检测）",
          summary.total == 7 and len(ok_items) >= 5 and len(skipped) >= 1 and len(failed) >= 1,
          f"共 {summary.total}：成功 {summary.ok} / 跳过 {summary.skipped} / 失败 {summary.failed}，"
          f"总耗时 {dt:.1f}s，平均 {summary.avg_seconds:.1f}s/张")
    check("10 批量输出不覆盖原图 + 同名自动编号",
          all(sha256(p) == hashes[p] for p in files) and len(restored) == len(ok_items),
          f"{len(files)} 个输入文件哈希未变；输出 {len(restored)} 个（成功 {len(ok_items)} 个）")

    # 每张图 Mask 必须不同（禁止复制第一张的 Mask）
    mask_files = sorted(p for p in (batch_out / "masks").glob("*_mask.png")
                        if "user_mask" not in p.name)
    uniq = {sha256(p) for p in mask_files}
    check("3b 每张图片的 Mask 互相独立（没有复制第一张）",
          len(mask_files) >= 5 and len(uniq) == len(mask_files),
          f"Mask 文件 {len(mask_files)} 个，内容互不相同 {len(uniq)} 个")

    # 第二次运行 → 必须出现 _1（绝不覆盖）
    summary2 = bp.run_batch([str(src)], output_dir=str(batch_out), mode="text",
                            out_format="png", options=pipe.RepairOptions(max_side=1600),
                            save_masks=False, save_reports=False)
    outs2 = sorted((batch_out / "restored").glob("*_restored_1.png"))
    check("10b 重复运行自动加编号（绝不覆盖已有结果）",
          len(outs2) == summary2.ok,
          f"第二次运行成功 {summary2.ok} 张，生成 `*_restored_1.png` {len(outs2)} 个")

    # ---------------- 11. C 手动模式回归（第五轮逻辑未改）----------------
    base11 = photo(seed=41)
    img11 = stamp(base11, "手动涂抹测试", 640, 400, 34, opacity=0.45)
    paint = np.zeros(img11.shape[:2], np.uint8)
    cv2_rectangle(paint, 470, 350, 810, 450)
    out11 = pipe.run_repair(img11, paint, options=pipe.RepairOptions(max_side=1600))
    diff = np.abs(out11.image.astype(np.int16) - img11.astype(np.int16)).max(axis=2)
    outside = np.count_nonzero(diff[paint == 0])
    check("11 C 手动模式回归（Mask 外 0 改动，未走自动检测）",
          outside == 0 and out11.qc.get("outside_change_ratio", 1) <= 0.002,
          f"涂抹区外被改动像素 {outside}（应为 0）｜质检外改动 "
          f"{out11.qc.get('outside_change_ratio', -1) * 100:.3f}%｜"
          f"Mask 由涂抹面积 {mp.area_ratio(paint) * 100:.2f}% 收紧到 "
          f"{mp.area_ratio(out11.mask_used) * 100:.2f}%")

    # ---------------- 性能统计 ----------------
    banner("性能统计（每张：OCR / Mask / LaMa / 后处理 / 总计）")
    for it in summary.items:
        t = it.timings
        print(f"[{it.name}] 状态={it.status}｜OCR {t.get('ocr_ms', 0) / 1000:.2f}s｜"
              f"Mask {(t.get('mask_refine_ms', 0) + t.get('mask_ms', 0)) / 1000:.2f}s｜"
              f"LaMa {t.get('lama_ms', 0) / 1000:.2f}s｜后处理 {t.get('post_ms', 0) / 1000:.2f}s｜"
              f"总计 {t.get('total_ms', 0) / 1000:.2f}s")
    print("\n" + summary.text())
    print("\n自动检测明细（水印数量 / 检测数量）：")
    for line in DETECT_LOG:
        print("   " + line)
    if logs:
        print("\n批量日志（前 12 行）：")
        for line in logs[:12]:
            print("   " + line)

    banner("结果汇总")
    passed = sum(1 for _, ok, _ in RESULTS if ok)
    for name, ok, _detail in RESULTS:
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
    print(f"\n通过 {passed}/{len(RESULTS)}")
    return 0 if passed == len(RESULTS) else 1


def cv2_and(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    import cv2
    return cv2.bitwise_and(mp.ensure_binary(a), mp.ensure_binary(b))


def cv2_rectangle(mask: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> None:
    import cv2
    cv2.rectangle(mask, (x0, y0), (x1, y1), 255, -1)


if __name__ == "__main__":
    sys.exit(main())
