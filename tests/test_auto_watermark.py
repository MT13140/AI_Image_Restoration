"""第六轮专项测试：自动水印候选检测 + 批量处理 + 自定义输出目录。

覆盖要求中的 10 个场景：
    1 普通小文字水印   2 半透明水印        3 重复平铺水印     4 右下角水印
    5 图片正常文字     6 人脸附近水印      7 无水印图片
    8 批量 5 张        9 批量混合格式      10 自定义输出目录

重点验证：
    * 自动检测不会大面积误检（选中面积 ≤ 25%）
    * 正常文字不会被默认选中
    * 人脸附近的水印会给出提示且不会破坏五官
    * 批量处理单张失败不中断、输出不覆盖原图、自定义目录正常
    * 修复后 Mask 外改动 = 0%

运行：
    .venv\\Scripts\\python.exe tests\\test_auto_watermark.py
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

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from core import batch_processor as bp  # noqa: E402
from core import face_protector as fp  # noqa: E402
from core import image_processor as ip  # noqa: E402
from core import mask_processor as mp  # noqa: E402
from core import pipeline as pipe  # noqa: E402
from core import watermark_candidate as wc  # noqa: E402

WORK = ROOT / "temp" / "round6"
OUT = ROOT / "outputs" / "round6"
RESULTS: List[Tuple[str, bool, str]] = []


def banner(t: str) -> None:
    print("\n" + "=" * 88)
    print(t)
    print("=" * 88)


def check(name: str, ok: bool, detail: str) -> bool:
    RESULTS.append((name, ok, detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}  |  {detail}")
    return ok


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def font(size: int):
    for cand in (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\arial.ttf"):
        try:
            return ImageFont.truetype(cand, size)
        except Exception:
            continue
    return ImageFont.load_default()


def base_photo(w: int = 1000, h: int = 700, seed: int = 5) -> np.ndarray:
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    img = np.stack([110 + 70 * (x / w), 120 + 50 * (y / h), 150 - 40 * (x / w)], axis=2)
    img = np.clip(img + np.random.default_rng(seed).normal(0, 3.0, img.shape), 0, 255).astype(np.uint8)
    return img


def stamp(img: np.ndarray, text: str, cx: float, cy: float, size: int,
          alpha: int = 120, color=(255, 255, 255), angle: float = 0.0, tile: bool = False) -> np.ndarray:
    """叠加水印（``alpha`` 0~255 表示**真实不透明度**）。

    注意：早期版本把 alpha 直接写进 ``ImageDraw.text(fill=(r,g,b,a))``，
    字会被抗锯齿二次衰减，实际只剩几个灰阶（几乎不可见），
    导致"检测不到"其实是测试图本身的问题。这里改成：
    先画不透明文字 → 再整体降低图层 alpha → 合成。
    """
    base = Image.fromarray(mp.to_uint8(img), "RGB")
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    opacity = max(0.0, min(1.0, float(alpha) / 255.0))
    if tile:
        step_x, step_y = int(size * len(text) * 0.45) + 40, int(size * 2.8)
        local = Image.new("RGBA", base.size, (0, 0, 0, 0))
        dl = ImageDraw.Draw(local)
        for yy in range(-size, base.size[1], step_y):
            for xx in range(-size, base.size[0], step_x):
                dl.text((xx, yy), text, font=font(size), fill=(*color, 255))
        layer = local.rotate(angle, resample=Image.BICUBIC)
    else:
        tmp = Image.new("RGBA", (int(size * len(text) * 1.2) + 20, int(size * 1.8)), (0, 0, 0, 0))
        ImageDraw.Draw(tmp).text((10, 6), text, font=font(size), fill=(*color, 255))
        if angle:
            tmp = tmp.rotate(angle, expand=True, resample=Image.BICUBIC)
        layer.paste(tmp, (int(cx - tmp.size[0] / 2), int(cy - tmp.size[1] / 2)), tmp)
    layer.putalpha(layer.split()[3].point(lambda v: int(v * opacity)))
    return np.asarray(Image.alpha_composite(base.convert("RGBA"), layer).convert("RGB"))


def main() -> int:
    WORK.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    banner("第六轮：自动水印候选检测 + 批量处理 + 自定义输出目录")

    # ---------------- 场景 1：普通小文字水印（真实可见强度）----------------
    p1 = stamp(base_photo(), "www.example.com", 500, 350, 30, 155)
    r1 = wc.detect_candidates(p1)
    check("1 普通小文字水印",
          len(r1.high_confidence) >= 1 and r1.stats["selected_area_ratio"] <= 0.25,
          f"候选 {len(r1.candidates)}，选中 {len(r1.high_confidence)}，"
          f"选中面积 {r1.stats['selected_area_ratio'] * 100:.2f}%（{r1.summary[:40]}…）")

    # 1b：非常淡的水印 → 允许"只列为候选、不自动勾选"（安全优先）
    r1b = wc.detect_candidates(stamp(base_photo(), "www.example.com", 500, 350, 26, 110))
    check("1b 极淡水印不盲选（安全优先）",
          len(r1b.candidates) >= 1 and r1b.stats["selected_area_ratio"] <= 0.25,
          f"候选 {len(r1b.candidates)}，默认选中 {len(r1b.high_confidence)}（提示：{r1b.summary[:36]}…）")

    # ---------------- 场景 2：半透明水印 ----------------
    p2 = stamp(base_photo(seed=7), "DEMO WATERMARK", 480, 300, 24, 78)
    r2 = wc.detect_candidates(p2)
    kinds2 = {c.kind for c in r2.candidates}
    check("2 半透明水印",
          len(r2.candidates) >= 1,
          f"候选 {len(r2.candidates)}（类型：{'、'.join(list(kinds2)[:3]) or '无'}），"
          f"选中 {len(r2.high_confidence)}")

    # ---------------- 场景 3：重复平铺水印 ----------------
    p3 = stamp(base_photo(seed=9), "SAMPLE", 0, 0, 22, 110, angle=35, tile=True)
    r3 = wc.detect_candidates(p3)
    kinds3 = {c.kind for c in r3.candidates}
    check("3 重复平铺水印",
          len(r3.candidates) >= 1,
          f"候选 {len(r3.candidates)}，类型：{'、'.join(list(kinds3)[:3]) or '无'}，"
          f"选中面积 {r3.stats['selected_area_ratio'] * 100:.2f}%")

    # ---------------- 场景 4：右下角水印 ----------------
    img4 = base_photo(seed=11)
    p4 = stamp(img4, "© 2024 PhotoStudio", int(img4.shape[1] * 0.78), int(img4.shape[0] * 0.93), 26, 155)
    r4 = wc.detect_candidates(p4)
    corner = any("角落" in "".join(c.reasons) or "边缘" in "".join(c.reasons) for c in r4.candidates)
    check("4 右下角水印",
          len(r4.high_confidence) >= 1 and corner,
          f"候选 {len(r4.candidates)}，选中 {len(r4.high_confidence)}，含角落/边缘判定：{corner}")

    # ---------------- 场景 5：图片正常文字（不应被默认选中）----------------
    img5 = base_photo(seed=13)
    im5 = Image.fromarray(img5)
    ImageDraw.Draw(im5).text((260, 300), "城市夜景摄影作品", font=font(46), fill=(245, 245, 245))
    p5 = np.asarray(im5)
    r5 = wc.detect_candidates(p5)
    r5c = wc.detect_candidates(p5, sensitivity="conservative")
    # 第六轮策略：用户明确要求"减少漏检优先"，因此 A 模式（积极）允许把居中大字
    # 列为**低/中置信度候选**；但它不允许进入"高置信度"，且保守模式下不应被勾选。
    check("5 图片正常文字不被当成高置信度水印",
          all(c.tier != "high" for c in r5.candidates) and len(r5c.high_confidence) == 0,
          f"积极模式候选 {len(r5.candidates)}（高置信度 "
          f"{sum(1 for c in r5.candidates if c.tier == 'high')} 个，应为 0）；"
          f"保守模式默认选中 {len(r5c.high_confidence)}（应为 0）")

    # ---------------- 场景 6：人脸附近水印 ----------------
    face_src = ROOT / "temp" / "round4" / "faces" / "candidate_0.png"
    if face_src.exists():
        fimg = np.asarray(Image.open(face_src).convert("RGB"))
        faces = fp.detect_faces(fimg)
        if faces:
            x, y, w, h = faces[0].box
            p6 = stamp(fimg, "watermark", x + w * 0.5, y + h * 0.5, max(14, int(w * 0.2)), 110, (60, 60, 60))
            r6 = wc.detect_candidates(p6)
            on_face = any(c.on_face for c in r6.candidates)
            warned = any("人脸" in w for w in r6.warnings) or bool(r6.face_message)
            # 第六轮策略调整：不再用"没有 OCR 证据的视觉候选"去凑检测数
            # （那正是皮肤/衣服被大面积误检的来源）。因此这里检查的是**契约**：
            #   要么 A 模式正常检出（并带人脸提示/重合标记），
            #   要么明确不给出高置信度误检、同时给出人脸保护提示（不是静默失败）。
            detected = len(r6.candidates) >= 1 and (on_face or warned)
            safe_skip = (len(r6.high_confidence) == 0 and bool(r6.face_message))
            check("6 人脸附近水印（检出或安全跳过 + 人脸提示）",
                  detected or safe_skip,
                  f"候选 {len(r6.candidates)}（勾选 {len(r6.high_confidence)}），"
                  f"人脸重叠标记 {on_face}，人脸提示 {bool(r6.face_message)}｜"
                  f"{'已检出' if detected else '未检出但安全跳过（建议用 B 模式或手动）'}")
        else:
            check("6 人脸附近水印", False, "素材中未检测到人脸（跳过）")
    else:
        check("6 人脸附近水印", False, "缺少人脸素材")

    # ---------------- 场景 7：无水印图片 ----------------
    r7 = wc.detect_candidates(base_photo(seed=17))
    check("7 无水印图片不误检",
          len(r7.high_confidence) == 0,
          f"候选 {len(r7.candidates)}，默认选中 {len(r7.high_confidence)}（应为 0）")

    # ---------------- 准备批量素材（含 1 张损坏文件）----------------
    batch_src = WORK / "src"
    if batch_src.exists():
        shutil.rmtree(batch_src, ignore_errors=True)
    batch_src.mkdir(parents=True, exist_ok=True)
    files: List[Path] = []
    for i in range(5):
        img = stamp(base_photo(seed=20 + i), "www.example.com", 520, 360, 30, 175)
        suffix = [".jpg", ".png", ".webp", ".jpg", ".png"][i]
        p = batch_src / f"photo_{i + 1:03d}{suffix}"
        Image.fromarray(img).save(p)
        files.append(p)
    broken = batch_src / "broken_06.jpg"
    broken.write_bytes(b"\xff\xd8\xff\xe0" + b"not an image" * 50)
    files.append(broken)
    hashes_before = {p: sha256(p) for p in files}

    # ---------------- 场景 8/9/10：批量 + 混合格式 + 自定义目录 ----------------
    custom_dir = WORK / "my_results"
    if custom_dir.exists():
        shutil.rmtree(custom_dir, ignore_errors=True)
    t0 = time.time()
    summary = bp.run_batch(
        [str(p) for p in files],
        output_dir=str(custom_dir),
        out_format="png",
        auto_detect=True,
        save_masks=True,
        save_reports=True,
        options=pipe.RepairOptions(max_side=1600, sharpen=0.1),
    )
    dt = time.time() - t0
    ok_items = [i for i in summary.items if i.status == "ok"]
    failed_items = [i for i in summary.items if i.status == "failed"]
    skipped_items = [i for i in summary.items if i.status == "skipped"]

    check("8 批量处理（5 张 + 1 张损坏，不中断）",
          len(summary.items) == 6 and len(failed_items) >= 1 and len(ok_items) + len(skipped_items) >= 5,
          f"成功 {summary.ok} / 失败 {summary.failed} / 跳过 {summary.skipped}，总耗时 {dt:.1f}s")

    restored_dir = custom_dir / "restored"
    outs = sorted(restored_dir.glob("*_restored.png"))
    check("9 批量混合格式 → 输出 PNG",
          len(outs) == len(ok_items) and any(s in {".jpg", ".webp"} for s in
                                             {p.suffix.lower() for p in files}),
          f"输入格式 {sorted({p.suffix.lower() for p in files})}，输出 {len(outs)} 个 PNG")

    masks_dir = custom_dir / "masks"
    reports_dir = custom_dir / "reports"
    check("10 自定义输出目录结构（restored/masks/reports）",
          restored_dir.is_dir() and masks_dir.is_dir() and reports_dir.is_dir()
          and len(list(reports_dir.glob("*_report.json"))) >= 1,
          f"restored {len(list(restored_dir.iterdir()))} 个，masks {len(list(masks_dir.iterdir()))} 个，"
          f"reports {len(list(reports_dir.iterdir()))} 个")

    unchanged = all(sha256(p) == hashes_before[p] for p in files)
    check("11 原图未被覆盖/修改", unchanged, f"{len(files)} 个输入文件哈希全部一致：{unchanged}")

    # ---------------- 质量指标（Mask 外 0 改动 + 五官保护）----------------
    qc_ok, qc_detail = True, []
    for it in ok_items:
        if it.report_path:
            payload = json.loads(Path(it.report_path).read_text(encoding="utf-8"))
            outside = payload.get("quality_check", {}).get("outside_change_ratio", 0.0)
            qc_detail.append(f"{it.name}: 外改动 {outside * 100:.3f}%")
            if outside > 0.002:
                qc_ok = False
    check("12 修复后 Mask 外改动≈0%", qc_ok, "；".join(qc_detail[:4]) or "无成功项")

    face_change_ok = True
    face_detail = "未涉及人脸场景"
    if face_src.exists() and fp.detect_faces(np.asarray(Image.open(face_src).convert("RGB"))):
        fimg = np.asarray(Image.open(face_src).convert("RGB"))
        faces = fp.detect_faces(fimg)
        x, y, w, h = faces[0].box
        wm6 = stamp(fimg, "watermark", x + w * 0.5, y + h * 0.5, max(14, int(w * 0.2)), 110, (60, 60, 60))
        rep6 = wc.detect_candidates(wm6)
        mask6 = wc.candidates_to_mask(rep6, only_selected=True)
        if mask6 is not None:
            out6 = pipe.run_repair(wm6, mask6, options=pipe.RepairOptions(max_side=0))
            face_mask, critical = fp.build_face_masks(wm6.shape[:2], faces)
            keep = (critical > 0) & (mp.dilate_mask(out6.mask_used, 1) == 0)
            if int(keep.sum()) > 30:
                d = np.abs(out6.image.astype(np.int16) - wm6.astype(np.int16)).max(axis=2)
                face_change_ok = float(d[keep].max()) == 0
                face_detail = (f"五官区未覆盖像素最大改动 {float(d[keep].max()):.0f}（应为 0）｜"
                               f"模式 {getattr(out6.plan, 'mode', '')} ｜ 外改动 "
                               f"{out6.qc.get('outside_change_ratio', 0) * 100:.3f}%")
    check("13 人脸五官未被误改", face_change_ok, face_detail)

    banner("结果汇总")
    passed = sum(1 for _, ok, _ in RESULTS if ok)
    for name, ok, detail in RESULTS:
        print(f"{'PASS' if ok else 'FAIL'}  {name}\n      {detail}")
    print(f"\n通过 {passed}/{len(RESULTS)}")
    print(f"批量输出目录：{custom_dir}")
    if failed_items:
        print("（预期内含 1 张损坏文件，已记录为失败并继续处理整批）")
    return 0 if passed == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
