"""第四轮专项质量测试：Mask 收紧（笔刷印记） + 人脸保护（五官不变形）。

每个场景都会保存完整链路，便于肉眼复核：

    00_ground_truth.png   无水印真值（真实照片场景=未加水印的原图）
    01_original.png       原图（含水印）
    02_user_mask.png      用户"涂大"的画笔范围（故意比水印大）
    03_refined_mask.png   Mask 自动收紧后的"水印核心区"
    04_lama_raw.png       LaMa 原始输出
    05_final.png          最终融合结果
    06_zoom.png           局部放大：原图 | 用户涂抹 | 最终结果 | 真值

关键指标：
    outside_change : 用户涂抹范围**之外**被改动的像素比例（应 ≈ 0）
    changed_ratio  : 改动像素 / 用户涂抹像素（最小修改，越小说明越没"整块重建"）
    brush_leak     : 改动像素中落在"收紧后 Mask + 羽化余量之外"的比例（笔刷形状泄漏，应 ≈ 0）
    face_change    : 人脸五官核心区（未被水印覆盖部分）相对真值的改变量（应 ≈ 0）
    mae_in / psnr  : 收紧区域内相对真值的误差 / 峰值信噪比（水印是否真被去掉）
    seam           : 修复边界梯度倍数（1.0 = 自然）

运行：
    .venv\\Scripts\\python.exe tests\\quality_round4.py
    .venv\\Scripts\\python.exe tests\\quality_round4.py --only face_eyes
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

from core import face_protector as fp  # noqa: E402
from core import image_processor as ip  # noqa: E402
from core import inpainting as inp  # noqa: E402
from core import mask_processor as mp  # noqa: E402
from core import mask_refiner as mr  # noqa: E402
from core import pipeline as pipe  # noqa: E402

OUT_ROOT = ROOT / "outputs" / "round4"
MAT_DIR = ROOT / "temp" / "round4"


# --------------------------------------------------------------------------
# 素材
# --------------------------------------------------------------------------
def _font(size: int):
    for cand in (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf", r"C:\Windows\Fonts\arial.ttf"):
        try:
            return ImageFont.truetype(cand, size)
        except Exception:
            continue
    return ImageFont.load_default()


def load_id_photo() -> Optional[np.ndarray]:
    """真实人像样例（含平铺水印）：从对比图左半部分取回。"""
    p = MAT_DIR / "id_original.png"
    if not p.exists():
        return None
    img = np.asarray(Image.open(p).convert("RGB"))
    return img[50:545, 215:605].copy()          # 只取人像样例本体


def load_id_photo_clean() -> Optional[np.ndarray]:
    """同一张人像样例的"无水印"参考（取上一轮修复结果的同区域）。"""
    p = MAT_DIR / "id_prev_result.png"
    if not p.exists():
        return None
    img = np.asarray(Image.open(p).convert("RGB"))
    return img[50:545, 215:605].copy()


def load_cloth_photo() -> Optional[np.ndarray]:
    """真实衣服照片的**无水印**版本（取历史修复对比图的右半部分 = 已去除水印的结果）。"""
    p = ROOT / "outputs" / "compare_20260913_154200.png"
    if not p.exists():
        return None
    comp = np.asarray(Image.open(p).convert("RGB"))
    h, w = comp.shape[:2]
    half = (w - 16) // 2
    right = comp[:, half + 16:half + 16 + half].copy()
    return right[620:1000, 600:1230].copy()      # 衣服区域（无水印）


def load_clean_face() -> Optional[np.ndarray]:
    """一张**完全无水印**的真实人脸照（本地素材，作为人脸测试的真值）。"""
    p = MAT_DIR / "faces" / "candidate_0.png"
    if not p.exists():
        return None
    img = np.asarray(Image.open(p).convert("RGB"))
    faces = fp.detect_faces(img)
    if not faces:
        return None
    x, y, bw, bh = faces[0].box
    x0 = max(0, int(x - bw * 1.9)); x1 = min(img.shape[1], int(x + bw * 2.9))
    y0 = max(0, int(y - bh * 1.5)); y1 = min(img.shape[0], int(y + bh * 2.2))
    return img[y0:y1, x0:x1].copy()


def stamp_text(img: np.ndarray, text: str, cx: float, cy: float, size: int,
               alpha: int = 120, color=(255, 255, 255), angle: float = 0.0, tile: bool = False) -> np.ndarray:
    """在图上盖一个半透明文字水印（可旋转、可平铺）。"""
    base = Image.fromarray(mp.to_uint8(img), "RGB")
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    font = _font(size)
    if tile:
        step_x, step_y = int(size * text.__len__() * 0.42) + 30, int(size * 2.6)
        for y in range(-size, base.size[1], step_y):
            for x in range(-size, base.size[0], step_x):
                d.text((x, y), text, font=font, fill=(*color, alpha))
        layer = layer.rotate(angle, resample=Image.BICUBIC, expand=False)
    else:
        tmp = Image.new("RGBA", (int(size * len(text) * 1.2) + 20, int(size * 1.8)), (0, 0, 0, 0))
        ImageDraw.Draw(tmp).text((10, 6), text, font=font, fill=(*color, alpha))
        if angle:
            tmp = tmp.rotate(angle, resample=Image.BICUBIC, expand=True)
        layer.paste(tmp, (int(cx - tmp.size[0] / 2), int(cy - tmp.size[1] / 2)), tmp)
        layer = Image.alpha_composite(Image.new("RGBA", base.size, (0, 0, 0, 0)), layer)
    return np.asarray(Image.alpha_composite(base.convert("RGBA"), layer).convert("RGB"))


def brush_stroke_mask(shape: Tuple[int, int], strokes: List[Tuple[int, int, int, int]],
                      width: int = 34) -> np.ndarray:
    """模拟用户"涂大"的画笔：粗笔触覆盖水印并吞掉周围正常像素。"""
    m = np.zeros((shape[0], shape[1]), np.uint8)
    for (x1, y1, x2, y2) in strokes:
        cv2.line(m, (int(x1), int(y1)), (int(x2), int(y2)), 255, int(width), cv2.LINE_AA)
    return (m > 0).astype(np.uint8) * 255


def gradient_scene() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    w, h = 900, 600
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    gt = np.zeros((h, w, 3), np.float32)
    gt[..., 0] = 120 + 80 * (x / w)
    gt[..., 1] = 115 + 60 * (y / h)
    gt[..., 2] = 150 - 40 * (x / w)
    gt = np.clip(gt + np.random.default_rng(3).normal(0, 3.0, gt.shape), 0, 255).astype(np.uint8)
    wm = stamp_text(gt, "www.example.com 2024", w * 0.5, h * 0.5, 34, 110)
    user = brush_stroke_mask((h, w), [(int(w * 0.18), int(h * 0.46), int(w * 0.82), int(h * 0.48)),
                                      (int(w * 0.18), int(h * 0.54), int(w * 0.82), int(h * 0.52))], 40)
    return gt, wm, user


def logo_scene(cloth: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    gt = cloth.copy()
    h, w = gt.shape[:2]
    im = Image.fromarray(gt, "RGB")
    d = ImageDraw.Draw(im, "RGBA")
    x0, y0 = int(w * 0.55), int(h * 0.35)
    d.rounded_rectangle([x0, y0, x0 + 150, y0 + 70], radius=14, fill=(30, 80, 190, 210))
    d.ellipse([x0 + 14, y0 + 14, x0 + 56, y0 + 56], fill=(250, 250, 250, 235))
    d.text((x0 + 64, y0 + 22), "LOGO", font=_font(26), fill=(255, 255, 255, 235))
    wm = np.asarray(im)
    user = brush_stroke_mask((h, w), [(x0 - 30, y0 - 20, x0 + 180, y0 + 90),
                                      (x0 - 30, y0 + 25, x0 + 180, y0 + 45)], 46)
    return gt, wm, user


def large_area_scene(cloth: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    gt = np.ascontiguousarray(cloth).copy()
    h, w = gt.shape[:2]
    wm = np.array(stamp_text(gt, "真实测试样例", w * 0.5, h * 0.45, 46, 120, tile=False), copy=True)
    cv2.rectangle(wm, (int(w * 0.12), int(h * 0.20)), (int(w * 0.88), int(h * 0.66)), (250, 250, 250), -1)
    wm = np.ascontiguousarray((wm.astype(np.float32) * 0.82 + gt.astype(np.float32) * 0.18).astype(np.uint8))
    user = brush_stroke_mask((h, w), [(int(w * 0.08), int(h * 0.22), int(w * 0.92), int(h * 0.24)),
                                      (int(w * 0.08), int(h * 0.44), int(w * 0.92), int(h * 0.46)),
                                      (int(w * 0.08), int(h * 0.64), int(w * 0.92), int(h * 0.62))], 60)
    return gt, wm, user


def face_scene_with_text(base: Optional[np.ndarray], place: str) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """真实人脸场景：在眼睛 / 鼻子 / 嘴巴 / 脸颊上盖半透明水印，并"涂大"。

    ``base`` 必须是**无水印**的真实人脸照（本次用本地素材），因此可以做真值评分。
    """
    if base is None:
        return None
    base = np.ascontiguousarray(base)
    h, w = base.shape[:2]
    faces = fp.detect_faces(base)
    if not faces:
        return None
    f = faces[0]
    x, y, bw, bh = f.box
    if f.landmarks is not None:
        eye_r, eye_l = f.landmarks[0], f.landmarks[1]
        nose, mouth_r, mouth_l = f.landmarks[2], f.landmarks[3], f.landmarks[4]
        anchors = {
            "eyes": ((eye_r[0] + eye_l[0]) / 2.0, (eye_r[1] + eye_l[1]) / 2.0),
            "nose": (nose[0], nose[1]),
            "mouth": ((mouth_r[0] + mouth_l[0]) / 2.0, (mouth_r[1] + mouth_l[1]) / 2.0),
            "cheek": (x + bw * 0.30, y + bh * 0.62),
        }
    else:
        anchors = {
            "eyes": (x + bw * 0.5, y + bh * 0.40), "nose": (x + bw * 0.5, y + bh * 0.60),
            "mouth": (x + bw * 0.5, y + bh * 0.78), "cheek": (x + bw * 0.30, y + bh * 0.62),
        }
    cx, cy = anchors[place]
    size = max(16, int(bw * 0.16))
    wm = stamp_text(base, "水印DEMO", cx, cy, size, 115, (60, 60, 60))
    half_w = max(40, int(bw * 0.42))
    half_h = max(14, int(size * 0.9))
    user = brush_stroke_mask((h, w), [(cx - half_w, cy - half_h * 0.5, cx + half_w, cy + half_h * 0.2),
                                      (cx - half_w, cy + half_h * 0.6, cx + half_w, cy + half_h * 0.9)], 30)
    return base, wm, user


def face_and_cloth_scene(base: Optional[np.ndarray]) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """水印横跨"脸的下缘 + 衣领"（边界场景）。"""
    if base is None:
        return None
    base = np.ascontiguousarray(base)
    h, w = base.shape[:2]
    faces = fp.detect_faces(base)
    if not faces:
        return None
    x, y, bw, bh = faces[0].box
    cy = int(y + bh * 1.02)
    size = max(16, int(bw * 0.15))
    wm = stamp_text(base, "水印跨脸与衣服", x + bw * 0.5, cy, size, 118, (60, 60, 60))
    user = brush_stroke_mask((h, w), [(x + bw * 0.02, cy - 16, x + bw * 0.98, cy - 8),
                                      (x + bw * 0.02, cy + 6, x + bw * 0.98, cy + 14)], 30)
    return base, wm, user


def tiled_translucent_scene(base: Optional[np.ndarray]) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """平铺半透明水印（复现真实测试样例场景 · 30% 不透明度 · 42° 平铺）。"""
    if base is None:
        return None
    base = np.ascontiguousarray(base)
    h, w = base.shape[:2]
    wm = stamp_text(base, "真实测试样例", 0, 0, 18, 95, (255, 255, 255), angle=42, tile=True)
    strokes = [(16, y, w - 16, y + 6) for y in range(24, h - 24, 52)]
    user = brush_stroke_mask((h, w), strokes, 40)
    return base, wm, user


# --------------------------------------------------------------------------
# 指标
# --------------------------------------------------------------------------
def _hf(img: np.ndarray, sigma: float = 1.6) -> np.ndarray:
    g = cv2.cvtColor(mp.to_uint8(img), cv2.COLOR_RGB2GRAY).astype(np.float32)
    return g - cv2.GaussianBlur(g, (0, 0), sigma)


def metrics(gt: np.ndarray, before: np.ndarray, after: np.ndarray,
            user_mask: np.ndarray, refined: np.ndarray,
            used_mask: np.ndarray, feather: int, faces: List[fp.FaceRegion]) -> Dict[str, float]:
    gt = mp.to_uint8(gt) if gt is not None else None
    diff = np.abs(mp.to_uint8(after).astype(np.int16) - mp.to_uint8(before).astype(np.int16)).max(axis=2)
    changed = diff > 2
    user = mp.ensure_binary(user_mask)
    refined_b = mp.ensure_binary(refined)
    used = mp.ensure_binary(used_mask)
    allowed = mp.dilate_mask(used, max(2, feather + 2))

    out: Dict[str, float] = {}
    outside = (user == 0)
    out["outside_change"] = float(np.mean(changed[outside])) if np.any(outside) else 0.0
    hard_out = mp.dilate_mask(user, 3) == 0
    out["outside_hard"] = float(np.mean(changed[hard_out])) if np.any(hard_out) else 0.0
    user_px = float(np.count_nonzero(user)) or 1.0
    out["changed_ratio"] = float(np.count_nonzero(changed)) / user_px
    out["refined_ratio"] = float(np.count_nonzero(refined_b)) / user_px
    changed_px = float(np.count_nonzero(changed)) or 1.0
    out["brush_leak"] = float(np.count_nonzero(changed & (allowed == 0))) / changed_px

    sel = refined_b > 0
    if gt is not None and np.any(sel):
        d = np.abs(mp.to_uint8(after).astype(np.float32) - gt.astype(np.float32))
        mae = float(d[sel].mean())
        mse = float((d ** 2)[sel].mean())
        out["mae_in"] = mae
        out["psnr_in"] = float(10 * np.log10((255.0 ** 2) / max(mse, 1e-6)))

    # 人脸五官核心区：未被水印覆盖的部分必须保持原样
    if faces:
        face_mask, critical = fp.build_face_masks(before.shape[:2], faces)
        # 五官区里"实际未被修改"的像素：以真正使用的 Mask（含扩张）为准
        keep = (critical > 0) & (mp.dilate_mask(used, 1) == 0)
        if int(keep.sum()) > 50:
            d = np.abs(mp.to_uint8(after).astype(np.float32) - mp.to_uint8(before).astype(np.float32))
            out["face_change"] = float(d[keep].max())
            out["face_change_mean"] = float(d[keep].mean())
        # 面部结构变化（与真值比较的梯度差异，越小越说明五官没被改写）
        if gt is not None:
            g1 = cv2.Laplacian(cv2.cvtColor(gt, cv2.COLOR_RGB2GRAY), cv2.CV_32F)
            g2 = cv2.Laplacian(cv2.cvtColor(mp.to_uint8(after), cv2.COLOR_RGB2GRAY), cv2.CV_32F)
            fb = (face_mask > 0)
            if int(fb.sum()) > 50:
                out["face_struct_delta"] = float(np.abs(np.abs(g1) - np.abs(g2))[fb].mean())
    return out


def fmt(name: str, m: Dict[str, float]) -> str:
    parts = [f"{name}"]
    if "outside_change" in m:
        parts.append(f"范围外改动 {m['outside_change'] * 100:.3f}%")
    if "changed_ratio" in m:
        parts.append(f"改动/涂抹 {m['changed_ratio'] * 100:.0f}%")
    if "refined_ratio" in m:
        parts.append(f"收紧后 {m['refined_ratio'] * 100:.0f}%")
    if "brush_leak" in m:
        parts.append(f"笔刷泄漏 {m['brush_leak'] * 100:.2f}%")
    if "mae_in" in m:
        parts.append(f"MAE {m['mae_in']:.2f}")
    if "face_change" in m:
        parts.append(f"五官区最大变化 {m['face_change']:.1f}")
    if "face_struct_delta" in m:
        parts.append(f"面部结构差 {m['face_struct_delta']:.2f}")
    if "edge_residual_cut" in m:
        parts.append(f"边缘残留消除 {m['edge_residual_cut'] * 100:.0f}%")
    return " ｜ ".join(parts)


# --------------------------------------------------------------------------
# 场景执行
# --------------------------------------------------------------------------
def run_case(name: str, gt: np.ndarray, wm: np.ndarray, user_mask: np.ndarray,
             gt_clean: bool = True, dilate: int = 4, feather: int = 6,
             max_side: int = 1600) -> Dict[str, float]:
    out_dir = OUT_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    if gt_clean and gt is not None:
        ip.save_image(out_dir / "00_ground_truth.png", gt)
    ip.save_image(out_dir / "01_original.png", wm)
    ip.save_image(out_dir / "02_user_mask.png", user_mask)

    # 人脸分析
    faces = fp.detect_faces(wm)
    report = fp.analyze(user_mask, faces) if faces else fp.FaceReport()

    # Mask 自动收紧
    # 与真实 App 完全一致的证据链：OCR 文字框 + 半透明检测 + 对比度残差
    # 第六轮起，这两个证据函数收敛到 core/pipeline.py（界面与批量共用同一实现）
    ocr_boxes = pipe.ocr_boxes_in_mask(wm, user_mask)
    evidence = pipe.translucent_evidence(wm, user_mask)
    rr = mr.refine(wm, user_mask, ocr_boxes=ocr_boxes, evidence_mask=evidence)
    refined = rr.mask
    ip.save_image(out_dir / "03_refined_mask.png", refined)

    plan = fp.make_plan(report, wm.shape[:2], refined, base_dilate=dilate,
                        base_feather=feather, base_tone=0.35, protect_faces=True)
    if faces:
        refined, protect_notes, _ = fp.enforce_protection(refined, faces)
        plan.notes = protect_notes + plan.notes
        ip.save_image(out_dir / "03_refined_mask.png", refined)
    # 第五轮：repair_mask（必须 100% 替换）与 blend_mask（含过渡带）分离
    repair_mask = mp.dilate_mask(refined, plan.dilate) if plan.dilate else refined
    repair_mask = mr.clip_to_user(repair_mask, user_mask, allow_px=0)
    blend_mask = (mp.dilate_mask(repair_mask, int(plan.feather)) if plan.feather else repair_mask)
    blend_mask = mr.clip_to_user(blend_mask, user_mask, allow_px=0)

    engine = inp.get_engine(backend="lama")
    outcome = engine.inpaint(
        wm, blend_mask, dilate=0, feather=plan.feather, color_match=plan.tone_strength,
        sharpen=min(0.15, 0.2) if report.mode != "normal" else 0.2,
        denoise=0, seamless=False, max_side=max_side, grain=plan.grain,
        grain_strength=plan.grain_strength,
        grain_exclude=plan.face_mask if report.mode != "normal" else None,
        roi=plan.roi if plan.use_roi else None,
        tile_components=plan.tile_mode, repair_mask=repair_mask,
        fallback=True, return_raw=True,
    )
    if outcome.raw_image is not None:
        ip.save_image(out_dir / "04_lama_raw.png", outcome.raw_image)
    ip.save_image(out_dir / "05_final.png", outcome.image)

    m = metrics(gt if gt_clean else None, wm, outcome.image, user_mask, refined,
                outcome.mask_used, plan.feather, faces)
    # 水印边缘残留专项（Test 2）：repair_mask 最外圈 3px 内的水印响应变化
    band = mp.ensure_binary(repair_mask) & ~mp.erode_mask(mp.ensure_binary(repair_mask), 3)
    if int(band.sum()) > 50:
        before_r = float(np.abs(mr._overlay_response(wm, 25))[band > 0].mean())
        after_r = float(np.abs(mr._overlay_response(outcome.image, 25))[band > 0].mean())
        m["edge_residual_before"] = before_r
        m["edge_residual_after"] = after_r
        m["edge_residual_cut"] = 1.0 - after_r / max(1e-6, before_r)
    m["seconds"] = outcome.seconds
    ip.save_image(out_dir / "07_repair_mask.png", repair_mask)

    # 局部放大对比：原图 | 用户涂抹 | 最终 | 真值
    boxes = mp.mask_to_boxes(user_mask, min_area=50)
    x, y, w, h = max(boxes, key=lambda b: b[2] * b[3])
    pad = 60
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(wm.shape[1], x + w + pad), min(wm.shape[0], y + h + pad)
    user_vis = mp.mask_overlay(wm, user_mask)[y0:y1, x0:x1]
    crops = [wm[y0:y1, x0:x1], user_vis, outcome.image[y0:y1, x0:x1]]
    if gt_clean and gt is not None:
        crops.append(gt[y0:y1, x0:x1])
    gap = np.full((crops[0].shape[0], 8, 3), 255, np.uint8)
    stacked = crops[0]
    for c in crops[1:]:
        stacked = np.hstack([stacked, gap, c])
    ip.save_image(out_dir / "06_zoom.png", stacked)

    print(f"\n=== {name} （{wm.shape[1]}×{wm.shape[0]}，{outcome.seconds:.1f}s，"
          f"人脸模式：{plan.mode}{'，ROI 重建' if plan.use_roi else ''}"
          f"{'，分块重建' if plan.tile_mode else ''}）===")
    print("   " + fmt("结果", m))
    print(f"   收紧依据：{rr.method}（置信度 {rr.confidence:.2f}，保留 {rr.stats.get('keep_ratio', 0) * 100:.0f}%）"
          f" ｜ {rr.note}")
    if plan.notes:
        print("   " + plan.notes[0])
    return m


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="第四轮质量测试")
    parser.add_argument("--only", default="")
    parser.add_argument("--max-side", type=int, default=1600)
    args = parser.parse_args(argv)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    print("=" * 96)
    print("第四轮质量测试：Mask 自动收紧（笔刷印记） + 人脸保护（五官不变形）")
    print("关键：范围外改动≈0 ｜ 改动/涂抹越小越好 ｜ 笔刷泄漏≈0 ｜ 五官区最大变化≈0")
    print("=" * 96)
    print("人脸检测能力：" + fp.describe_capability())

    cloth = load_cloth_photo()
    clean_face = load_clean_face()
    id_photo = load_id_photo()
    none_builder = (lambda: None)

    def cloth_scene():
        if cloth is None:
            return None
        base = np.ascontiguousarray(cloth)
        h, w = base.shape[:2]
        wm = stamp_text(base, "www.example.com", w * 0.5, h * 0.5, 30, 115)
        user = brush_stroke_mask((h, w), [(w * 0.15, h * 0.45, w * 0.85, h * 0.47),
                                          (w * 0.15, h * 0.55, w * 0.85, h * 0.53)], 38)
        return base, wm, user

    def user_real_scene():
        """真实测试样例：平铺水印人像样例（无干净真值，只验证"保留性 + 不露笔刷"）。"""
        if id_photo is None:
            return None
        base = np.ascontiguousarray(id_photo)
        h, w = base.shape[:2]
        strokes = [(8, y, w - 8, y + 4) for y in range(14, h - 14, 44)]
        return base, base, brush_stroke_mask((h, w), strokes, 34)

    cases: List[Tuple[str, object, bool]] = [
        ("01_plain_text", gradient_scene, True),
        ("02_cloth_text", cloth_scene, True),
        ("03_face_cheek", (lambda: face_scene_with_text(clean_face, "cheek")) if clean_face is not None else none_builder, True),
        ("04_face_eyes", (lambda: face_scene_with_text(clean_face, "eyes")) if clean_face is not None else none_builder, True),
        ("05_face_nose", (lambda: face_scene_with_text(clean_face, "nose")) if clean_face is not None else none_builder, True),
        ("06_face_mouth", (lambda: face_scene_with_text(clean_face, "mouth")) if clean_face is not None else none_builder, True),
        ("07_face_and_cloth", (lambda: face_and_cloth_scene(clean_face)) if clean_face is not None else none_builder, True),
        ("08_translucent_tiled", (lambda: tiled_translucent_scene(clean_face)) if clean_face is not None else none_builder, True),
        ("09_logo", (lambda: logo_scene(cloth)) if cloth is not None else none_builder, True),
        ("10_large_area", (lambda: large_area_scene(cloth)) if cloth is not None else none_builder, True),
        ("11_user_real_tiled", user_real_scene, False),
    ]

    summary: List[str] = []
    for name, builder, gt_clean in cases:
        if args.only and args.only not in name:
            continue
        try:
            built = builder()
            if built is None:
                print(f"\n=== {name} ===\n   跳过（缺少素材）")
                continue
            gt, wm, user = built
            m = run_case(name, gt, wm, user, gt_clean=gt_clean, max_side=args.max_side)
            summary.append(
                f"{name:<18} 范围外改动 {m.get('outside_change', 0) * 100:6.3f}% ｜ "
                f"改动/涂抹 {m.get('changed_ratio', 0) * 100:5.0f}% ｜ 笔刷泄漏 {m.get('brush_leak', 0) * 100:5.2f}% ｜ "
                + (f"MAE {m['mae_in']:5.2f} ｜ " if "mae_in" in m else "（无干净真值）｜ ")
                + (f"五官区最大变化 {m['face_change']:5.1f}" if "face_change" in m else "")
            )
        except Exception as exc:  # noqa: BLE001
            print(f"\n=== {name} ===\n   ❌ {exc.__class__.__name__}: {exc}")
            traceback.print_exc()

    print("\n" + "=" * 96)
    print("汇总")
    print("=" * 96)
    for line in summary:
        print("  " + line)
    print(f"\n所有阶段图已保存到：{OUT_ROOT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
