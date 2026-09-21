"""修复质量报告：量化"阴影 / 色块 / 补丁 / 硬边 / 纹理差异"，并对比新旧融合策略。

每个场景都会保存完整链路，方便肉眼复核：

    00_ground_truth.png  无水印的真值（合成场景才有，用于客观评分）
    01_original.png      原图（含水印）
    02_mask.png          实际使用的 Mask
    03_lama_raw.png      LaMa 模型原始输出（未做任何融合）
    04_final_old.png     旧融合（全局均值/方差匹配 + 高斯羽化）
    05_final_new.png     新融合（低频光照场延续 + 颗粒匹配 + 平滑羽化）
    06_zoom_compare.png  局部放大：原图 ｜ 旧融合 ｜ 新融合

指标说明：
    MAE / PSNR : 与真值的误差（只在修复区内计算）——**越低/越高越好**
    dL/da/db   : 修复区与周围邻域的亮度/色彩关系相对原图的变化
    texture    : 修复区与邻域的高频(颗粒/纹理)比值，越接近 1.0 越自然
    seam       : 修复边界梯度相对原图的倍数，>1.3 表示有硬边/光晕

运行：
    .venv\\Scripts\\python.exe tests\\quality_report.py
    .venv\\Scripts\\python.exe tests\\quality_report.py --only user_photo
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
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
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from core import image_processor as ip  # noqa: E402
from core import inpainting as inp  # noqa: E402
from core import mask_processor as mp  # noqa: E402

OUT_ROOT = ROOT / "outputs" / "quality"


# --------------------------------------------------------------------------
# 指标
# --------------------------------------------------------------------------
def _lab(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(mp.to_uint8(img), cv2.COLOR_RGB2LAB).astype(np.float32)


def _hf(img: np.ndarray, sigma: float = 1.6) -> np.ndarray:
    gray = cv2.cvtColor(mp.to_uint8(img), cv2.COLOR_RGB2GRAY).astype(np.float32)
    return gray - cv2.GaussianBlur(gray, (0, 0), sigma)


def _hf_std(img: np.ndarray, where: np.ndarray) -> float:
    vals = _hf(img)[where]
    return float(vals.std()) if vals.size else 0.0


def _boundary_grad(img: np.ndarray, mask: np.ndarray) -> float:
    gray = cv2.cvtColor(mp.to_uint8(img), cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    band = ((cv2.dilate(mask, np.ones((3, 3), np.uint8)) - cv2.erode(mask, np.ones((3, 3), np.uint8))) > 0) & (mask > 0)
    return float(mag[band].mean()) if int(band.sum()) else 0.0


def evaluate(original: np.ndarray, result: np.ndarray, mask: np.ndarray,
             ground_truth: Optional[np.ndarray] = None) -> Dict[str, float]:
    m = mp.ensure_binary(mask, original.shape[:2])
    sel = m > 0
    ring = (cv2.dilate(m, np.ones((17, 17), np.uint8)) > 0) & (m == 0)
    if int(sel.sum()) < 30:
        return {}

    lab_o, lab_r = _lab(original), _lab(result)
    shift = []
    for c in range(3):
        base = float(lab_o[..., c][sel].mean() - lab_o[..., c][ring].mean())
        now = float(lab_r[..., c][sel].mean() - lab_r[..., c][ring].mean())
        shift.append(now - base)

    out: Dict[str, float] = {
        "dL": shift[0], "da": shift[1], "db": shift[2],
        "texture_before": _hf_std(original, sel) / max(1e-6, _hf_std(original, ring)),
        "texture_after": _hf_std(result, sel) / max(1e-6, _hf_std(result, ring)),
        "seam_ratio": _boundary_grad(result, m) / max(1e-6, _boundary_grad(original, m)),
        "changed_pixels": float(sel.sum()),
    }
    out["texture_delta"] = out["texture_after"] - out["texture_before"]

    if ground_truth is not None and ground_truth.shape == result.shape:
        gt = mp.to_uint8(ground_truth).astype(np.float32)
        res = mp.to_uint8(result).astype(np.float32)
        diff = np.abs(gt - res)
        mae = float(diff[sel].mean())
        mse = float((diff ** 2)[sel].mean())
        out["mae_gt"] = mae
        out["psnr_gt"] = float(10 * np.log10((255.0 ** 2) / max(mse, 1e-6)))
        gx1 = cv2.Sobel(cv2.cvtColor(gt.astype(np.uint8), cv2.COLOR_RGB2GRAY), cv2.CV_32F, 1, 0)
        gy1 = cv2.Sobel(cv2.cvtColor(gt.astype(np.uint8), cv2.COLOR_RGB2GRAY), cv2.CV_32F, 0, 1)
        gx2 = cv2.Sobel(cv2.cvtColor(res.astype(np.uint8), cv2.COLOR_RGB2GRAY), cv2.CV_32F, 1, 0)
        gy2 = cv2.Sobel(cv2.cvtColor(res.astype(np.uint8), cv2.COLOR_RGB2GRAY), cv2.CV_32F, 0, 1)
        e1 = np.sqrt(gx1 * gx1 + gy1 * gy1)
        e2 = np.sqrt(gx2 * gx2 + gy2 * gy2)
        out["grad_err_gt"] = float(np.abs(e1 - e2)[sel].mean())
        # 纹理/颗粒与真值的接近程度（1.0 = 颗粒感一致，<1 表示偏平滑/偏糊）
        out["texture_gt"] = _hf_std(result, sel) / max(1e-6, _hf_std(ground_truth, sel))
    return out


def fmt(name: str, m: Dict[str, float]) -> str:
    if not m:
        return f"{name}: 指标不可用"
    line = (f"{name}: dL={m['dL']:+.2f} da={m['da']:+.2f} db={m['db']:+.2f} ｜ "
            f"纹理 {m['texture_before']:.2f}→{m['texture_after']:.2f} ｜ 接缝 {m['seam_ratio']:.2f}×")
    if "mae_gt" in m:
        line += (f" ｜ 与真值 MAE={m['mae_gt']:.2f} PSNR={m['psnr_gt']:.1f}dB "
                 f"梯度差={m['grad_err_gt']:.2f} 颗粒={m.get('texture_gt', 0):.2f}")
    return line


# --------------------------------------------------------------------------
# 场景素材（返回 真值, 含水印图, Mask）
# --------------------------------------------------------------------------
def _grain(img: np.ndarray, sigma: float = 4.0, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.clip(img.astype(np.float32) + rng.normal(0, sigma, img.shape).astype(np.float32), 0, 255).astype(np.uint8)


def _font(size: int):
    for cand in (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\arial.ttf"):
        try:
            return ImageFont.truetype(cand, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _add_watermark(clean: np.ndarray, xy: Tuple[int, int], text: str, size: int,
                   alpha: int = 115, color=(255, 255, 255)) -> np.ndarray:
    img = Image.fromarray(clean, "RGB")
    ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ImageDraw.Draw(ov).text(xy, text, font=_font(size), fill=(*color, alpha))
    return np.asarray(Image.alpha_composite(img.convert("RGBA"), ov).convert("RGB"))


def scene_gradient() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """渐变背景 + 半透明文字水印（最容易出现"色块/补丁"）。"""
    w, h = 1200, 800
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    base = np.zeros((h, w, 3), np.float32)
    base[..., 0] = 120 + 90 * (x / w)
    base[..., 1] = 110 + 70 * (y / h)
    base[..., 2] = 160 - 50 * (x / w) + 30 * (y / h)
    clean = _grain(base, 3.0)
    wm = _add_watermark(clean, (int(w * 0.36), int(h * 0.44)), "www.example.com 2024", max(20, w // 26))
    mask = mp.dilate_mask(mp.rects_to_mask((h, w), [[35, 43, 69, 52]], percent=True), 4)
    return clean, wm, mask


def scene_shadow() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """带硬阴影边界的照片：检验修复区是否会出现"阴影块/光照不一致"。"""
    w, h = 1200, 800
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    lit = (x + y * 0.4) < (w * 0.62)
    base = np.zeros((h, w, 3), np.float32)
    base[..., 0] = np.where(lit, 208, 118)
    base[..., 1] = np.where(lit, 192, 110)
    base[..., 2] = np.where(lit, 176, 102)
    base += (18 * np.sin(y / 40.0))[..., None]
    clean = _grain(base, 4.0, seed=11)
    wm = _add_watermark(clean, (int(w * 0.53), int(h * 0.44)), "粉馒头 @demo", max(20, w // 28), 150, (20, 20, 20))
    mask = mp.dilate_mask(mp.rects_to_mask((h, w), [[52, 42, 79, 52]], percent=True), 4)
    return clean, wm, mask


def scene_solid() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """纯色背景：最容易看出色块与颜色偏差。"""
    w, h = 1000, 700
    base = np.zeros((h, w, 3), np.float32)
    base[..., :] = (232, 226, 218)
    clean = _grain(base, 3.0, seed=3)
    wm = _add_watermark(clean, (int(w * 0.31), int(h * 0.5)), "DEMO 水印 2024", max(20, w // 24), 120, (90, 90, 90))
    mask = mp.dilate_mask(mp.rects_to_mask((h, w), [[29, 47, 71, 57]], percent=True), 4)
    return clean, wm, mask


def scene_outdoor() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """室外场景：天空渐变 + 地面 + 太阳，检验大面积渐变区域的过渡。"""
    w, h = 1280, 720
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    horizon = h * 0.62
    sky = y < horizon
    base = np.zeros((h, w, 3), np.float32)
    base[..., 0] = np.where(sky, 120 + 80 * (1 - y / horizon), 96)
    base[..., 1] = np.where(sky, 160 + 60 * (1 - y / horizon), 122)
    base[..., 2] = np.where(sky, 225, 70)
    base += (14 * np.sin(x / 70.0) * sky)[..., None]
    sun = ((x - w * 0.78) ** 2 + (y - h * 0.22) ** 2) < (w * 0.06) ** 2
    base[sun] = (255, 244, 200)
    clean = _grain(base, 3.5, seed=5)
    im = Image.fromarray(clean, "RGB")
    ImageDraw.Draw(im).rectangle([w * 0.08, h * 0.55, w * 0.3, h * 0.8], fill=(150, 120, 90))
    clean = _grain(np.asarray(im).astype(np.float32), 3.0, seed=9)
    wm = _add_watermark(clean, (int(w * 0.42), int(h * 0.72)), "© DEMO 2024-05-01", max(18, w // 30))
    mask = mp.dilate_mask(mp.rects_to_mask((h, w), [[41, 70, 76, 80]], percent=True), 4)
    return clean, wm, mask


def scene_logo() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Logo 水印（实心色块），检验是否残留矩形补丁。"""
    w, h = 1100, 760
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    base = np.zeros((h, w, 3), np.float32)
    base[..., 0] = 150 + 60 * (x / w)
    base[..., 1] = 140 + 40 * (y / h)
    base[..., 2] = 130 + 30 * (x / w)
    clean = _grain(base, 3.5, seed=13)
    wm_img = Image.fromarray(clean, "RGB")
    d = ImageDraw.Draw(wm_img, "RGBA")
    d.rectangle([w * 0.7, h * 0.78, w * 0.7 + w * 0.16, h * 0.78 + h * 0.1], fill=(40, 90, 200, 205))
    d.ellipse([w * 0.72, h * 0.8, w * 0.72 + h * 0.06, h * 0.8 + h * 0.06], fill=(250, 250, 250, 235))
    wm = np.asarray(wm_img)
    mask = mp.dilate_mask(mp.rects_to_mask((h, w), [[69, 77, 87, 89]], percent=True), 4)
    return clean, wm, mask


def scene_user_photo() -> Optional[Tuple[Optional[np.ndarray], np.ndarray, np.ndarray, np.ndarray]]:
    """用户真实照片：从历史对比图中取回原图，并用"实际被改动的区域"还原 Mask。"""
    comp_path = ROOT / "outputs" / "compare_20260913_154200.png"
    if not comp_path.exists():
        return None
    comp = np.asarray(Image.open(comp_path).convert("RGB"))
    h, w = comp.shape[:2]
    half = (w - 16) // 2
    original = comp[:, :half].copy()
    old_result = comp[:, half + 16:half + 16 + half].copy()
    diff = np.abs(original.astype(np.int16) - old_result.astype(np.int16)).max(axis=2)
    region = (diff > 10).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(region, 8)
    if n > 1:
        best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        region = (labels == best).astype(np.uint8)
    mask = mp.clean_mask(region * 255, min_area=200, close_px=3)
    return None, original, mask, old_result


# --------------------------------------------------------------------------
# 场景执行
# --------------------------------------------------------------------------
def run_scenario(name: str, gt: Optional[np.ndarray], wm: np.ndarray, mask: np.ndarray,
                 old_result: Optional[np.ndarray] = None, max_side: int = 2048) -> Dict[str, Dict[str, float]]:
    out_dir = OUT_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    if gt is not None:
        ip.save_image(out_dir / "00_ground_truth.png", gt)
    ip.save_image(out_dir / "01_original.png", wm)
    ip.save_image(out_dir / "02_mask.png", mask)

    engine = inp.get_engine(backend="lama")
    outcome = engine.inpaint(wm, mask, dilate=0, feather=6, color_match=0.35, sharpen=0.0,
                             denoise=0, max_side=max_side, grain=True, legacy_blend=False, return_raw=True)
    if outcome.raw_image is not None:
        ip.save_image(out_dir / "03_lama_raw.png", outcome.raw_image)
    ip.save_image(out_dir / "05_final_new.png", outcome.image)

    metrics: Dict[str, Dict[str, float]] = {}
    metrics["new"] = evaluate(wm, outcome.image, outcome.mask_used, gt)
    if outcome.raw_image is not None:
        metrics["raw"] = evaluate(wm, outcome.raw_image, outcome.mask_used, gt)

    raw = outcome.raw_image if outcome.raw_image is not None else outcome.image
    old, _ = ip.finalize_result(wm, raw, outcome.mask_used, feather=6, color_match=0.5,
                               sharpen=0.0, denoise=0, legacy=True)
    ip.save_image(out_dir / "04_final_old.png", old)
    metrics["old"] = evaluate(wm, old, outcome.mask_used, gt)

    # 仅"平滑羽化 + 颗粒匹配"（不做低频校正）——用于判断校正是否真的有益
    nocorr, _ = ip.finalize_result(wm, raw, outcome.mask_used, feather=6, color_match=0.0,
                                   sharpen=0.0, denoise=0, grain=True)
    metrics["v2_nocorr"] = evaluate(wm, nocorr, outcome.mask_used, gt)
    if old_result is not None and old_result.shape == wm.shape:
        metrics["user_history"] = evaluate(wm, old_result, outcome.mask_used, gt)

    boxes = mp.mask_to_boxes(outcome.mask_used, min_area=100)
    x, y, ww, hh = max(boxes, key=lambda b: b[2] * b[3])
    pad = 80
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(wm.shape[1], x + ww + pad), min(wm.shape[0], y + hh + pad)
    crops = [wm[y0:y1, x0:x1], raw[y0:y1, x0:x1], old[y0:y1, x0:x1], outcome.image[y0:y1, x0:x1]]
    if gt is not None:
        crops.append(gt[y0:y1, x0:x1])
    gap = np.full((crops[0].shape[0], 8, 3), 255, np.uint8)
    stacked = crops[0]
    for c in crops[1:]:
        stacked = np.hstack([stacked, gap, c])
    ip.save_image(out_dir / "06_zoom_compare.png", stacked)

    print(f"\n=== 场景: {name} （{wm.shape[1]}×{wm.shape[0]}，Mask {mp.area_ratio(outcome.mask_used)*100:.2f}%，"
          f"LaMa {outcome.seconds:.1f}s / {outcome.info.get('device')}）===")
    for key in ("user_history", "raw", "old", "v2_nocorr", "new"):
        if key in metrics:
            label = {"user_history": "你历史结果   ", "raw": "LaMa 原始输出 ",
                     "old": "旧融合(改前)  ", "v2_nocorr": "新融合·无校正 ",
                     "new": "新融合(默认) "}[key]
            print("   " + fmt(label, metrics[key]))
    return metrics


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="修复质量报告")
    parser.add_argument("--only", default="", help="只跑名称包含该关键字的场景")
    parser.add_argument("--max-side", type=int, default=2048)
    args = parser.parse_args(argv)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print("=" * 92)
    print("修复质量报告：阴影 / 色块 / 纹理 / 接缝 量化对比")
    print("dL/da/db 越接近 0 越好 ｜ 纹理比接近 1.0 ｜ 接缝接近 1.0 ｜ MAE 越低越好 ｜ PSNR 越高越好")
    print("=" * 92)

    scenarios: List[Tuple[str, object]] = [
        ("user_photo", scene_user_photo),
        ("gradient_text", scene_gradient),
        ("shadow_edge", scene_shadow),
        ("solid_bg", scene_solid),
        ("outdoor_scene", scene_outdoor),
        ("logo_box", scene_logo),
    ]

    summary: List[str] = []
    for name, builder in scenarios:
        if args.only and args.only not in name:
            continue
        try:
            built = builder()
            if built is None:
                print(f"\n=== 场景: {name} ===\n   跳过（素材不存在）")
                continue
            if len(built) == 4:
                gt, wm, mask, old_result = built
            else:
                gt, wm, mask = built
                old_result = None
            m = run_scenario(name, gt, wm, mask, old_result, max_side=args.max_side)
            if "old" in m and "new" in m:
                line = (f"{name:<14} dL {m['old']['dL']:+.2f}→{m['new']['dL']:+.2f} ｜ "
                        f"纹理Δ {m['old']['texture_delta']:+.2f}→{m['new']['texture_delta']:+.2f} ｜ "
                        f"接缝 {m['old']['seam_ratio']:.2f}×→{m['new']['seam_ratio']:.2f}×")
                if "mae_gt" in m["new"]:
                    line += f" ｜ MAE {m['old'].get('mae_gt', float('nan')):.2f}→{m['new']['mae_gt']:.2f}"
                summary.append(line)
        except Exception as exc:  # noqa: BLE001
            print(f"\n=== 场景: {name} ===\n   ❌ {exc.__class__.__name__}: {exc}")
            traceback.print_exc()

    print("\n" + "=" * 92)
    print("汇总（旧融合 → 新融合）")
    print("=" * 92)
    for line in summary:
        print("  " + line)
    print(f"\n所有阶段图片已保存到：{OUT_ROOT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
