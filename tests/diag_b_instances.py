"""B 模式「逐实例」诊断：把每个文字水印实例单独走一遍链路并量化失败点。

对应任务书第三/六/七/十一阶段的要求：

    对每个水印实例分别输出 OCR → text mask → repair_mask → blend_mask → LaMa raw → final
    并判断它是"成功去除"还是"残留"，从而回答
    "为什么有些实例成功、有些实例失败"。

判定方法（无需人工标注真值）：

    * ``text_like``   ：原图里"文字结构像素"（局部对比度高的笔画/边缘）
    * ``covered``     ：被 text mask / repair mask 覆盖的比例
    * ``resid_final`` ：text_like 像素里，**最终图与原图几乎没有差别**（<4 灰阶）的比例
                        —— 没被改过 ⇒ 文字还在（残留）
    * ``resid_raw``   ：同上，但看 LaMa **原始输出**，用来判断残留是模型产生的还是融合产生的

用法::

    .venv\\Scripts\\python.exe tests\\diag_b_instances.py --image D:\\x.jpg
    .venv\\Scripts\\python.exe tests\\diag_b_instances.py --dir temp\\b_cases     # 批量诊断
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

from core import auto_detect as ad  # noqa: E402
from core import image_processor as ip  # noqa: E402
from core import mask_processor as mp  # noqa: E402
from core import modes  # noqa: E402
from core import pipeline as pipe  # noqa: E402

OUT_ROOT = ROOT / "outputs" / "debug"


def _local_contrast(gray: np.ndarray, k: Optional[int] = None) -> np.ndarray:
    h, w = gray.shape[:2]
    if k is None:
        k = ad._odd(int(max(7, min(h, w) * 0.10)), 5)
    k = min(k, max(5, (min(h, w) // 2) * 2 - 1))
    bg = cv2.medianBlur(gray, ad._odd(k, 5))
    return cv2.GaussianBlur(cv2.absdiff(gray, bg).astype(np.float32), (0, 0), 1.0)


def instance_metrics(orig: np.ndarray, final: np.ndarray, raw: Optional[np.ndarray],
                     mask_used: np.ndarray, text_mask: np.ndarray,
                     repair_mask: np.ndarray, bbox: Tuple[int, int, int, int]
                     ) -> Dict[str, float]:
    """在单个实例的框内量化：文字像素覆盖率、残留比例。"""
    x, y, w, h = bbox
    H, W = orig.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x1 - x0 < 6 or y1 - y0 < 6:
        return {}
    o = orig[y0:y1, x0:x1]
    f = final[y0:y1, x0:x1]
    r = raw[y0:y1, x0:x1] if raw is not None else None
    gray = cv2.cvtColor(o, cv2.COLOR_RGB2GRAY)
    d = _local_contrast(gray)
    p99 = float(np.percentile(d, 99.0)) if d.size else 0.0
    text_like = d > max(5.0, 0.45 * p99)
    n_text = int(np.count_nonzero(text_like))
    if n_text < 20:
        return {"text_px": n_text}
    sub = lambda m: (m[y0:y1, x0:x1] > 0) if m is not None else np.zeros_like(text_like)
    cov_text = float(np.count_nonzero(sub(text_mask) & text_like)) / n_text
    cov_repair = float(np.count_nonzero(sub(repair_mask) & text_like)) / n_text
    cov_used = float(np.count_nonzero(sub(mask_used) & text_like)) / n_text
    changed = np.abs(f.astype(np.int16) - o.astype(np.int16)).max(axis=2) > 4
    resid_final = float(np.count_nonzero(text_like & ~changed)) / n_text
    resid_raw = float("nan")
    if r is not None:
        changed_raw = np.abs(r.astype(np.int16) - o.astype(np.int16)).max(axis=2) > 4
        resid_raw = float(np.count_nonzero(text_like & ~changed_raw)) / n_text

    # ---- 更客观的"水印是否还在"：结构相关度 ----
    # 取"原图 − 局部平滑"的高频结构，与"结果 − 局部平滑"的高频结构做相关：
    #   相关度高 ⇒ 原来的文字结构还在（残留）；接近 0 ⇒ 已被重建掉。
    # 这样不会把头发/草叶等背景纹理误判成残留。
    def _structure(im: np.ndarray) -> np.ndarray:
        g = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY).astype(np.float32)
        return g - cv2.GaussianBlur(g, (0, 0), 2.0)
    s_o = _structure(o)[text_like]
    s_f = _structure(f)[text_like]
    s_r = _structure(r)[text_like] if r is not None else None
    def _corr(a, b):
        if a.size < 20 or float(a.std()) < 1e-6 or float(b.std()) < 1e-6:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])
    corr_final = _corr(s_o, s_f)
    corr_raw = _corr(s_o, s_r) if s_r is not None else float("nan")

    # ---- "修复痕迹"指标：区域内外是否出现明显色差/纹理差/接缝 ----
    m = sub(mask_used)
    ring = cv2.dilate(m.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    ring &= ~m
    art: Dict[str, float] = {}
    if int(m.sum()) >= 40 and int(ring.sum()) >= 40:
        for tag, img_roi in (("final", f), ("raw", r)):
            if img_roi is None:
                continue
            lab_in = cv2.cvtColor(img_roi, cv2.COLOR_RGB2LAB).astype(np.float32)
            inside_mu = lab_in[m].mean(axis=0)
            ring_mu = lab_in[ring].mean(axis=0)
            art[f"dE_{tag}"] = float(np.linalg.norm((inside_mu - ring_mu) * np.array([1.0, 1.3, 1.3])))
            g_in = cv2.cvtColor(img_roi, cv2.COLOR_RGB2GRAY).astype(np.float32)
            hp = g_in - cv2.GaussianBlur(g_in, (0, 0), 1.2)
            hf_in = float(np.std(hp[m]))
            hf_ring = float(np.std(hp[ring]))
            art[f"hf_{tag}"] = hf_in / max(1e-6, hf_ring)
    out = {
        "text_px": n_text,
        "mask_ratio": float(sub(mask_used).mean()),
        "cov_text_mask": cov_text,
        "cov_repair_mask": cov_repair,
        "cov_used_mask": cov_used,
        "resid_final": resid_final,
        "resid_raw": resid_raw,
        "corr_final": corr_final,
        "corr_raw": corr_raw,
    }
    out.update({k: float(v) for k, v in art.items()})
    return out


def diagnose_image(img: np.ndarray, slug: str, out_root: Path = OUT_ROOT,
                   save_artifacts: bool = True) -> List[Dict[str, object]]:
    out_dir = out_root / slug
    if save_artifacts:
        out_dir.mkdir(parents=True, exist_ok=True)
    det = ad.get_detector()

    # ① OCR（B 模式：文字全部去除）
    res = modes.get_mode("text").run(modes.ModeRequest(image=img, sensitivity="aggressive"))
    cands = list(res.candidates)
    print(f"\n=== {slug} | {img.shape[1]}×{img.shape[0]} | "
          f"OCR 实例 {res.details.get('ocr_regions', len(cands))} 个 / 候选 {len(cands)} 个 ===")
    if save_artifacts:
        ip.save_image(out_dir / "01_original.png", img)
        boxes_img = img.copy()
        for c in cands:
            x, y, w, h = c.bbox
            cv2.rectangle(boxes_img, (x, y), (x + w, y + h), (255, 40, 40), 2)
        ip.save_image(out_dir / "02_ocr_boxes.png", boxes_img)
    if not cands:
        print("    未检测到文字（跳过）")
        return []

    # ② 修复（走**真实 B 路径**：带上模式层给出的修复建议，同时取 LaMa 原始输出）
    import dataclasses
    opt = pipe.RepairOptions(max_side=0)
    hints = {k: v for k, v in (res.repair_hints or {}).items() if hasattr(opt, k)}
    if hints:
        opt = dataclasses.replace(opt, **hints)
        print(f"    修复建议（B 模式）：{hints}")
    out = pipe.run_repair(img, res.mask, options=opt, return_raw=True)
    raw = out.raw_image if out.raw_image is not None else out.image
    if save_artifacts:
        ip.save_image(out_dir / "03_text_mask.png", res.mask)
        ip.save_image(out_dir / "04_repair_mask.png", out.repair_mask)
        ip.save_image(out_dir / "05_blend_mask.png", out.blend_mask)
        ip.save_image(out_dir / "06_lama_raw.png", raw)
        ip.save_image(out_dir / "07_final.png", out.image)
        ip.save_image(out_dir / "08_before_after.png",
                      ip.side_by_side(img, out.image, labels=("原图", "修复结果")))

    rows: List[Dict[str, object]] = []
    for i, c in enumerate(cands):
        m = instance_metrics(img, out.image, raw, out.mask_used, res.mask,
                             out.repair_mask, c.bbox)
        if not m or m.get("text_px", 0) < 20:
            continue
        # 判定：结构相关度 < 0.5（水印结构已基本消失）**且** 残留像素 ≤ 15%
        cf = float(m.get("corr_final", float("nan")))
        ok = ((cf != cf) or cf < 0.5) and float(m["resid_final"]) <= 0.15
        row = dict(index=i, text=c.text, bbox=c.bbox, conf=float(c.confidence),
                   src=c.source, tier=c.tier, ok=ok, **m)
        rows.append(row)
        if save_artifacts:
            x, y, w, h = c.bbox
            pad = 6
            x0, y0 = max(0, x - pad), max(0, y - pad)
            x1, y1 = min(img.shape[1], x + w + pad), min(img.shape[0], y + h + pad)
            crop = img[y0:y1, x0:x1]
            ip.save_image(out_dir / f"i{i:02d}_crop.png", crop)
            ip.save_image(out_dir / f"i{i:02d}_mask.png", res.mask[y0:y1, x0:x1])
            ip.save_image(out_dir / f"i{i:02d}_raw.png", raw[y0:y1, x0:x1])
            ip.save_image(out_dir / f"i{i:02d}_final.png", out.image[y0:y1, x0:x1])

    ok_n = sum(1 for r in rows if r["ok"])
    print(f"    实例结果：成功 {ok_n} / 残留 {len(rows) - ok_n}（判定：残留 ≤5% 视为成功）")
    print(f"    {'#':>2} {'文本':<18}{'conf':>5}{'覆盖率':>8}{'结构相关':>9}{'ΔE(final)':>10}"
          f"{'纹理比':>8}  判定")
    for r in rows:
        cf = float(r.get("corr_final", float("nan")))
        de = float(r.get("dE_final", float("nan")))
        hf = float(r.get("hf_final", float("nan")))
        print(f"    {r['index']:>2} {str(r['text'])[:16]:<18}{float(r['conf']):>5.2f}"
              f"{float(r['cov_used_mask']) * 100:>7.1f}%"
              f"{('—' if cf != cf else f'{cf:>9.2f}'):>9}"
              f"{('—' if de != de else f'{de:>9.1f}'):>10}"
              f"{('—' if hf != hf else f'{hf:>7.2f}'):>8}  "
              f"{'✓成功' if r['ok'] else '✗残留'}")
    print(f"    整图：Mask {mp.area_ratio(out.mask_used) * 100:.2f}% ｜ "
          f"质检外改动 {out.qc.get('outside_change_ratio', -1) * 100:.3f}% ｜ "
          f"残留消除 {out.qc.get('residue_cut', 0) * 100:.0f}% ｜ {out.seconds:.1f}s")
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="B 模式逐实例诊断")
    ap.add_argument("--image", default="")
    ap.add_argument("--dir", default="")
    ap.add_argument("--glob", default="*_original.png")
    args = ap.parse_args(argv)

    targets: List[Path] = []
    if args.image:
        targets = [Path(args.image)]
    elif args.dir:
        targets = sorted(Path(args.dir).glob(args.glob))
    else:
        default = ROOT / "temp" / "b_cases"
        targets = sorted(default.glob("*_original.png"))
        print(f"（未指定输入，使用 {default} 里的真实样例）")

    all_rows: List[Dict[str, object]] = []
    for p in targets:
        try:
            img = ip.load_image(p)
        except Exception as exc:  # noqa: BLE001
            print(f"读取失败 {p}: {exc}")
            continue
        slug = re.sub(r"[^\w\-]+", "_", p.stem)[:40]
        all_rows += diagnose_image(img, slug)

    print("\n" + "=" * 96)
    total = len(all_rows)
    ok = sum(1 for r in all_rows if r["ok"])
    print(f"总计：{len(targets)} 张图 ｜ 实例 {total} 个 ｜ 成功 {ok} ｜ 残留 {total - ok}")
    if total:
        print(f"平均：Mask 覆盖文字像素 "
              f"{np.mean([float(r['cov_used_mask']) for r in all_rows]) * 100:.1f}% ｜ "
              f"结构相关度 {np.nanmean([float(r.get('corr_final', np.nan)) for r in all_rows]):.2f}"
              f"（0=水印已消失，1=原样保留）｜"
              f"平均 ΔE {np.nanmean([float(r.get('dE_final', np.nan)) for r in all_rows]):.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
