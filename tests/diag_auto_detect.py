"""自动检测"逐步诊断"工具（第六轮新增）。

用途：当某张图的水印"应该能检测到却没检测到"时，用它把**每一步的中间结果**
都保存成图片，一眼看出是在哪一步被丢掉的。

输出（默认写到 ``outputs/debug/``）::

    01_original.png       原图
    02_ocr_input.png      缩图 OCR 真正"看到"的画面（含缩放比例）
    03_ocr_boxes.png      缩图 OCR 的原始检测框（画在原图上）
    04_hi_ocr_boxes.png   高清兜底 / 局部放大 OCR 的框（画在原图上）
    05_filtered_boxes.png 几何+文字特征过滤后的文字框
    06_final_candidates.png 最终候选（红=默认勾选，橙=不勾选）
    07_final_mask.png     最终 Mask（贴文字的笔画掩膜）
    08_overlay.png        原图 + 最终 Mask 半透明叠加

用法::

    .venv\\Scripts\\python.exe tests\\diag_auto_detect.py                 # 用内置合成样例（GALAXY RI TA）
    .venv\\Scripts\\python.exe tests\\diag_auto_detect.py --image D:\\x.jpg
    .venv\\Scripts\\python.exe tests\\diag_auto_detect.py --image x.jpg --target GALAXY
    .venv\\Scripts\\python.exe tests\\diag_auto_detect.py --mode text     # 诊断 B 模式
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

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

from core import auto_detect as ad  # noqa: E402
from core import image_processor as ip  # noqa: E402
from core import mask_processor as mp  # noqa: E402

OUT_DIR = ROOT / "outputs" / "debug"


# --------------------------------------------------------------------------
# 画框工具
# --------------------------------------------------------------------------
def _draw_boxes(img: np.ndarray, boxes: Sequence[Tuple[float, float, float, float, float]],
                color: Tuple[int, int, int] = (255, 40, 40), label: Optional[str] = None,
                thickness: int = 3) -> np.ndarray:
    out = img.copy()
    for rect in boxes:
        pts = np.round(ad._poly_of_rect(rect)).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [pts], True, color, thickness, cv2.LINE_AA)
    if label:
        out = ip.draw_label(out, label, (14, 14), size=max(18, img.shape[1] // 45))
    return out


def _text_evidence_rects(items) -> List[Tuple[float, float, float, float, float]]:
    return [tuple(float(v) for v in it.rect) for it in items]


def _det_to_full(items, scale: float):
    """把"检测坐标系"的证据换算回原图坐标（仅用于画框展示）。"""
    if abs(scale - 1.0) < 1e-9:
        return list(items)
    return ad._scale_evidence(items, 1.0 / scale)


# --------------------------------------------------------------------------
# 合成样例（英文半透明水印，模拟"GALAXY RI TA"这类照片）
# --------------------------------------------------------------------------
def demo_image(w: int = 3000, h: int = 2000, opacity: float = 0.35) -> np.ndarray:
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    band = (y / h)[..., None]
    sky = np.stack([110 + 90 * (y / h), 150 + 70 * (y / h), 235 - 40 * (y / h)], axis=2)
    sea = np.stack([40 + 20 * (y / h), 90 + 30 * (y / h), 150 + 30 * (y / h)], axis=2)
    sand = np.stack([200 + 30 * (y / h), 185 + 25 * (y / h), 150 + 20 * (y / h)], axis=2)
    img = np.where(band < 0.45, sky, np.where(band < 0.62, sea, sand)).astype(np.float32)
    img += np.random.default_rng(5).normal(0, 3.0, img.shape)
    base = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).convert("RGBA")
    text = "GALAXY RI TA"
    tmp = Image.new("RGBA", (int(46 * len(text) * 0.62) + 40, 92), (0, 0, 0, 0))
    ImageDraw.Draw(tmp).text((10, 6), text, fill=(255, 255, 255, 255),
                             font=ImageFont.truetype(r"C:\Windows\Fonts\arialbd.ttf", 46))
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    layer.paste(tmp, (int(w * 0.5 - tmp.size[0] / 2), int(h * 0.625 - tmp.size[1] / 2)), tmp)
    layer.putalpha(layer.split()[3].point(lambda v: int(v * float(opacity))))
    return np.asarray(Image.alpha_composite(base, layer).convert("RGB"))


# --------------------------------------------------------------------------
# 主诊断流程
# --------------------------------------------------------------------------
def diagnose(img: np.ndarray, target: str = "GALAXY", mode: str = "auto",
             out_dir: Path = OUT_DIR, sensitivity: str = "aggressive") -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    H, W = img.shape[:2]
    print("=" * 88)
    print(f"自动检测逐步诊断 ｜ 图片 {W}×{H} ｜ 模式 {mode} ｜ 目标文字 {target!r}")
    print("=" * 88)

    tokens = [t for t in ad._text_shape(target).split() if len(t) >= 3] or ["galaxy"]

    def hit(items_or_cands) -> bool:
        """模糊命中判定：OCR 常把 "GALAXY RI TA" 读成 "GALAX ｜ RLTA" 这样的碎片，
        所以只要**目标词的一段**出现（去掉空格/标点后）就算命中。"""
        for it in items_or_cands:
            text = getattr(it, "text", "") or ""
            # 逐段比较：OCR 可能读成 "GALAX"、也可能合并成 "RLTA ｜ GALAX"，
            # 所以按分隔符拆开，逐个词做**双向包含**判断。
            for word in re.split(r"[\s｜|,;，；]+", text):
                shape = ad._text_shape(word)
                if len(shape) < 4:
                    continue
                if any(tok in shape or shape in tok for tok in tokens):
                    return True
        return False

    ip.save_image(out_dir / "01_original.png", img)
    print(f"[1] 原图 → 01_original.png")

    # ---- ② OCR 真正"看到"的画面 ----
    tn, scale, det_img = ad.textness_map(img, max_side=1500)
    ip.save_image(out_dir / "02_ocr_input.png", det_img)
    print(f"[2] 缩图 OCR 输入：{det_img.shape[1]}×{det_img.shape[0]}"
          f"（缩放 {scale:.3f}）→ 02_ocr_input.png")

    det = ad.get_detector()
    det.ocr.detector.ensure_loaded()

    # ---- ③ 缩图整图 OCR ----
    t0 = time.time()
    base_items = ad._ocr_patch(det, det_img, det_img.shape[1] / 2.0, det_img.shape[0] / 2.0,
                               det_img.shape[1], det_img.shape[0], 0.0, 0.40, "OCR",
                               force_scale=1.0, pad_ratio=0.0)
    base_ms = (time.time() - t0) * 1000.0
    img3 = _draw_boxes(img, _text_evidence_rects(_det_to_full(base_items, scale)),
                       (255, 60, 60), f"缩图 OCR：{len(base_items)} 个框")
    ip.save_image(out_dir / "03_ocr_boxes.png", img3)
    print(f"[3] 缩图 OCR：{len(base_items)} 个框（{base_ms:.0f}ms）"
          f"｜命中目标：{'是' if hit(base_items) else '否'} → 03_ocr_boxes.png")
    for it in base_items[:6]:
        print(f"      · {it.text[:30]!r} score={it.score:.2f}")

    # ---- ④ 局部放大 / 高清兜底 OCR ----
    t0 = time.time()
    items, stats = ad.collect_evidence(
        det, img, tn, scale, det_img,
        ocr_min_score=0.40, max_region_ocr=20, use_region_ocr=True,
        sweep_angles=(-25.0, 25.0, -45.0, 45.0), sweep_side=1100,
        hi_res_ocr=True, hi_tiles=6,
    )
    all_ms = (time.time() - t0) * 1000.0
    extra = [it for it in items if "OCR" == it.source or not it.source.startswith("OCR ")]
    zoom_items = [it for it in items if "局部放大" in it.source or "旋转" in it.source
                  or "高清" in it.source]
    img4 = _draw_boxes(img, _text_evidence_rects(_det_to_full(zoom_items, scale)),
                       (60, 140, 255), f"局部放大/旋转/高清 OCR：{len(zoom_items)} 个框")
    ip.save_image(out_dir / "04_hi_ocr_boxes.png", img4)
    print(f"[4] 全部 OCR 证据：{len(items)} 条（{all_ms:.0f}ms）"
          f"｜命中目标：{'是' if hit(items) else '否'} → 04_hi_ocr_boxes.png")
    print(f"      阶段耗时：{ad.timings_text(stats)}")
    for it in items:
        if hit([it]):
            print(f"      ★ 命中：{it.text!r} score={it.score:.2f} 来源={it.source} "
                  f"rect={tuple(round(v) for v in it.rect)}")

    # ---- ⑤ 分组 + 过滤 ----
    groups = ad.group_evidence(items, (H, W))
    st = ad.resolve_settings(sensitivity=sensitivity)
    for g in groups:
        ad.score_group(g, det_img.shape[:2], st.select_threshold, 0.0)
    rects_full = []
    for g in groups:
        r = g.rect
        rects_full.append((r[0] / scale, r[1] / scale, r[2] / scale, r[3] / scale, r[4]))
    img5 = _draw_boxes(img, rects_full, (255, 160, 30),
                       f"分组后的文字块：{len(groups)} 个")
    ip.save_image(out_dir / "05_filtered_boxes.png", img5)
    print(f"[5] 分组：{len(groups)} 个文字块｜命中目标：{'是' if hit(groups) else '否'}"
          f" → 05_filtered_boxes.png")
    for g in groups[:8]:
        print(f"      · conf={g.confidence:.2f} tier={g.tier:6s} text={g.text[:26]!r}")

    # ---- ⑥ 最终候选 + Mask ----
    rep = ad.detect_watermarks(img, settings=st, mode=mode)
    sel = [c for c in rep.candidates if c.selected]
    img6 = img.copy()
    for c in rep.candidates:
        x, y, w, h = c.bbox
        col = (255, 40, 40) if c.selected else (255, 170, 40)
        cv2.rectangle(img6, (x, y), (x + w, y + h), col, 3, cv2.LINE_AA)
    img6 = ip.draw_label(img6, f"最终候选 {len(rep.candidates)} 个（默认勾选 {len(sel)}）",
                         (14, 14), size=max(18, W // 45))
    ip.save_image(out_dir / "06_final_candidates.png", img6)
    print(f"[6] 最终候选：{len(rep.candidates)} 个（默认勾选 {len(sel)}）"
          f"｜命中目标：{'是' if hit(rep.candidates) else '否'} → 06_final_candidates.png")

    mask = rep.mask_selected if rep.mask_selected is not None else np.zeros((H, W), np.uint8)
    ip.save_image(out_dir / "07_final_mask.png", mask)
    ip.save_image(out_dir / "08_overlay.png", mp.mask_overlay(img, mask))
    print(f"[7] 最终 Mask：覆盖画面 {mp.area_ratio(mask) * 100:.3f}%"
          f"（{len(mp.mask_to_boxes(mask, min_area=8))} 处）→ 07_final_mask.png / 08_overlay.png")

    # ---- 结论：到底哪一步丢的 ----
    print("\n" + "-" * 88)
    if hit(items):
        if hit(groups) and hit(rep.candidates) and hit([c for c in rep.candidates if c.selected]):
            print(f"结论：{target} 全程畅通，最终已默认勾选。")
        elif hit(groups) and hit(rep.candidates):
            print(f"结论：{target} 被检测到但**未默认勾选**（置信度/分层把它压下去了）。"
                  f"→ 需要调整打分策略或手动勾选。")
        else:
            print(f"结论：{target} 在 OCR 阶段读到，但在**分组/过滤或候选构建**阶段被丢弃。")
    elif hit(base_items):
        print(f"结论：{target} 只有缩图 OCR 读到，后续被过滤。")
    else:
        print(f"结论：**所有 OCR 都没读到** {target}。"
              f"下一步应检查：① 水印在缩图后的像素高度；② 是否需要更高分辨率/更强预处理。")
    print("-" * 88)
    return 0 if hit(rep.candidates) else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="自动检测逐步诊断")
    ap.add_argument("--image", default="", help="要诊断的图片路径；留空用内置合成样例")
    ap.add_argument("--target", default="GALAXY", help="关心的目标文字（用于判断是否命中）")
    ap.add_argument("--mode", default="auto", choices=["auto", "text"], help="按哪个模式诊断")
    ap.add_argument("--sensitivity", default="aggressive",
                    choices=["aggressive", "balanced", "conservative"])
    ap.add_argument("--out", default="", help="输出目录（默认 outputs/debug）")
    args = ap.parse_args(argv)

    if args.image:
        img = ip.load_image(args.image)
    else:
        img = demo_image()
        print("（未指定图片，使用内置合成样例：3000×2000 海滩照 + 半透明英文水印 GALAXY RI TA）")
    out_dir = Path(args.out) if args.out else OUT_DIR
    return diagnose(img, target=args.target, mode=args.mode, out_dir=out_dir,
                    sensitivity=args.sensitivity)


if __name__ == "__main__":
    sys.exit(main())
