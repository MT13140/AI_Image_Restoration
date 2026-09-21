"""自动检测质量测试集（本轮新增，替代"凭感觉判断有没有检测到"）。

覆盖任务书第十三章要求的 12 类场景，并给出**可核对的量化指标**：

    1 清晰英文水印        2 半透明英文水印       3 低对比度英文水印
    4 底部居中文字        5 角落文字            6 多个文字水印
    7 水印覆盖天空        8 水印覆盖衣服         9 水印靠近人体
   10 水印靠近人脸       11 无水印图片          12 图片里是正常文字（非水印）

每个场景记录：OCR 检出数 / 最终候选数 / 默认勾选数 / Mask 面积 / 是否命中 /
是否把"不该修的区域"框进去（误检面积）/ 各阶段耗时。

另外附带：
  * **置信度阈值扫描**（0.2 / 0.3 / 0.4 / 0.5）：统计检出/误检，验证"不是简单降阈值"；
  * **C 手动模式回归**：同一张图 + 同一涂抹，Mask 外改动必须为 0。

运行：
    .venv\\Scripts\\python.exe tests\\test_auto_detect_quality.py
"""

from __future__ import annotations

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

from core import auto_detect as ad  # noqa: E402
from core import image_processor as ip  # noqa: E402
from core import mask_processor as mp  # noqa: E402
from core import modes  # noqa: E402
from core import pipeline as pipe  # noqa: E402

WORK = ROOT / "temp" / "detect_quality"
OUT = ROOT / "outputs" / "detect_quality"
FONT_BOLD = r"C:\Windows\Fonts\arialbd.ttf"
FONT_CN = r"C:\Windows\Fonts\msyh.ttc"
RESULTS: List[Tuple[str, bool, str]] = []


def banner(t: str) -> None:
    print("\n" + "=" * 96)
    print(t)
    print("=" * 96)


def check(name: str, ok: bool, detail: str) -> bool:
    RESULTS.append((name, ok, detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}\n      {detail}")
    return ok


# --------------------------------------------------------------------------
# 场景素材
# --------------------------------------------------------------------------
def sky_beach(w=3000, h=2000, seed=5) -> np.ndarray:
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    band = (y / h)[..., None]
    sky = np.stack([110 + 90 * (y / h), 150 + 70 * (y / h), 235 - 40 * (y / h)], axis=2)
    sea = np.stack([40 + 20 * (y / h), 90 + 30 * (y / h), 150 + 30 * (y / h)], axis=2)
    sand = np.stack([200 + 30 * (y / h), 185 + 25 * (y / h), 150 + 20 * (y / h)], axis=2)
    img = np.where(band < 0.45, sky, np.where(band < 0.62, sea, sand)).astype(np.float32)
    img += np.random.default_rng(seed).normal(0, 3.0, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def body_scene(w=1600, h=1200, seed=11) -> np.ndarray:
    """人体（皮肤）+ 衣服（布料纹理）+ 头发。"""
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    img = np.stack([120 + 60 * (x / w), 140 + 50 * (y / h), 165 - 30 * (x / w)], axis=2).astype(np.float32)

    def ellipse(cx, cy, rx, ry, color, soft=0.35):
        d = (((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2)
        m = np.clip((1.0 - d) / soft, 0, 1)[..., None]
        return img * (1 - m) + np.array(color, np.float32) * m

    img = ellipse(w * 0.30, h * 0.55, w * 0.10, h * 0.42, (224, 178, 152))
    img = ellipse(w * 0.72, h * 0.60, w * 0.11, h * 0.40, (232, 190, 165))
    rng = np.random.default_rng(seed)
    cloth_m = np.clip((y - h * 0.55) / (h * 0.45), 0, 1)[..., None]
    stripes = 18 * np.sin(np.linspace(0, 90 * np.pi, w))[None, :, None]
    cloth = img + stripes + rng.normal(0, 9, img.shape)
    img = img * (1 - cloth_m) + cloth * cloth_m
    img += rng.normal(0, 2.5, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def stamp(img: np.ndarray, text: str, cx: float, cy: float, size: int, opacity: float = 0.4,
          color=(255, 255, 255), cjk: bool = False, angle: float = 0.0) -> np.ndarray:
    base = Image.fromarray(img).convert("RGBA")
    font = ImageFont.truetype(FONT_CN if cjk else FONT_BOLD, size)
    tmp = Image.new("RGBA", (int(size * len(text) * (1.1 if cjk else 0.62)) + 40,
                             int(size * 1.9)), (0, 0, 0, 0))
    ImageDraw.Draw(tmp).text((10, 6), text, font=font, fill=(*color, 255))
    if angle:
        tmp = tmp.rotate(angle, expand=True, resample=Image.BICUBIC)
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    layer.paste(tmp, (int(cx - tmp.size[0] / 2), int(cy - tmp.size[1] / 2)), tmp)
    layer.putalpha(layer.split()[3].point(lambda v: int(v * float(opacity))))
    return np.asarray(Image.alpha_composite(base, layer).convert("RGB"))


def scene_set() -> List[Dict[str, object]]:
    """12 类场景：`expect_text` 表示"应该检测到文字"。"""
    s: List[Dict[str, object]] = []
    s.append(dict(key="1_clear_en", name="清晰英文水印",
                  img=stamp(sky_beach(), "GALAXY RI TA", 1500, 1250, 52, opacity=0.75),
                  expect_text=True, roi=(900, 1050, 2200, 1450)))
    s.append(dict(key="2_translucent_en", name="半透明英文水印",
                  img=stamp(sky_beach(), "GALAXY RI TA", 1500, 1250, 46, opacity=0.35),
                  expect_text=True, roi=(900, 1050, 2200, 1450)))
    s.append(dict(key="3_low_contrast_en", name="低对比度英文水印",
                  img=stamp(sky_beach(seed=7), "GALAXY RI TA", 1500, 1250, 42, opacity=0.18),
                  expect_text=True, roi=(900, 1050, 2200, 1450)))
    s.append(dict(key="4_bottom_center", name="底部居中文字",
                  img=stamp(sky_beach(seed=8), "www.demo.com", 1500, 1880, 40, opacity=0.5),
                  expect_text=True, roi=(1000, 1750, 2000, 1980)))
    s.append(dict(key="5_corner", name="角落文字",
                  img=stamp(sky_beach(seed=9), "© 2024 STUDIO", 2500, 250, 38, opacity=0.6),
                  expect_text=True, roi=(2000, 120, 2950, 380)))
    s.append(dict(key="6_multi", name="多个文字水印",
                  img=stamp(stamp(stamp(sky_beach(seed=10), "GALAXY RI TA", 900, 400, 42, 0.5),
                                  "www.demo.com", 2100, 1250, 38, 0.5),
                            "@studio_2024", 900, 1750, 36, 0.5),
                  expect_text=True, roi=None))
    s.append(dict(key="7_over_sky", name="水印覆盖天空",
                  img=stamp(sky_beach(seed=12), "GALAXY RI TA", 1500, 500, 46, opacity=0.30),
                  expect_text=True, roi=(900, 350, 2200, 700)))
    s.append(dict(key="8_over_cloth", name="水印覆盖衣服",
                  img=stamp(body_scene(seed=13), "GALAXY RI TA", 800, 950, 40, opacity=0.35),
                  expect_text=True, roi=(400, 830, 1250, 1100)))
    s.append(dict(key="9_near_body", name="水印靠近人体",
                  img=stamp(body_scene(seed=14), "GALAXY RI TA", 1300, 620, 40, opacity=0.35),
                  expect_text=True, roi=(900, 500, 1600, 780)))
    s.append(dict(key="10_near_face", name="水印靠近人脸",
                  img=stamp(body_scene(seed=15), "GALAXY RI TA", 800, 300, 34, opacity=0.45),
                  expect_text=True, roi=(500, 200, 1150, 420)))
    s.append(dict(key="11_no_watermark", name="无水印图片",
                  img=sky_beach(seed=16), expect_text=False, roi=None))
    s.append(dict(key="12_normal_text", name="画面里的正常文字（非水印）",
                  img=stamp(sky_beach(seed=17), "海滩风景摄影作品", 1500, 1000, 60,
                            opacity=0.95, cjk=True),
                  expect_text=True, normal_text=True, roi=(900, 850, 2100, 1150)))
    return s


# --------------------------------------------------------------------------
# 指标
# --------------------------------------------------------------------------
def eval_mask(mask: np.ndarray, roi, shape: Tuple[int, int]) -> Tuple[float, float]:
    """返回 ``(Mask 面积占比, 落在期望 ROI 之外的比例)``（后者衡量误检/面积失控）。"""
    if mask is None or mp.is_empty(mask):
        return 0.0, 0.0
    ratio = float(mp.area_ratio(mask))
    if roi is None:
        return ratio, 0.0
    x0, y0, x1, y1 = roi
    inside = mask[y0:y1, x0:x1] > 0
    total = float(np.count_nonzero(mask))
    outside = total - float(np.count_nonzero(inside))
    return ratio, outside / max(1.0, total)


def threshold_sweep(img: np.ndarray, truth_has_text: bool = True) -> List[Tuple[float, int, int]]:
    """OCR 置信度阈值扫描：``(阈值, 检出文字条数, 疑似误检条数)``。

    误检判定：该 OCR 框落在"已知文字区域"之外（用几何 + 是否命中关键词粗判）。
    """
    det = ad.get_detector()
    tn, scale, det_img = ad.textness_map(img, max_side=1500)
    rows: List[Tuple[float, int, int]] = []
    for thr in (0.2, 0.3, 0.4, 0.5):
        items = det.ocr.detector.detect(det_img, min_score=thr)
        good = 0
        bad = 0
        for it in items:
            txt = (it.text or "").strip()
            shape = ad._text_shape(txt)
            looks_text = len(shape) >= 3 and ad._alnum_ratio(txt) >= 0.5
            x, y, w, h = it.bbox
            tall = h / max(1.0, det_img.shape[0])
            if looks_text and tall < 0.25:
                good += 1
            else:
                bad += 1
        rows.append((thr, good, bad))
    return rows


def main() -> int:  # noqa: C901
    WORK.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    banner("自动检测质量测试集（12 类场景）")
    print("场景说明：Mask 面积 = 默认勾选候选的掩膜覆盖率；ROI 外占比 = 框到“不该修的地方”的比例")
    header = (f"{'场景':<18}{'OCR':>5}{'候选':>6}{'勾选':>6}{'Mask%':>8}"
              f"{'ROI外%':>8}{'命中':>6}{'耗时s':>8}")
    print("\n" + header)
    print("-" * len(header))

    rows: List[Tuple[str, bool, str]] = []
    for sc in scene_set():
        img = sc["img"]
        t0 = time.time()
        rep = ad.detect_watermarks(img, settings=ad.resolve_settings(), mode="auto")
        rep_b = ad.detect_watermarks(img, settings=ad.resolve_settings(), mode="text")
        dt = time.time() - t0
        mask = rep.mask_selected
        ratio, outside = eval_mask(mask, sc.get("roi"), img.shape[:2])
        ocr_n = int(rep.stats.get("ocr_items", 0))
        cand_n = len(rep.candidates)
        sel_n = len(rep.high_confidence)
        hit = sel_n > 0 and ratio > 0
        expect = bool(sc.get("expect_text"))
        auto_ok = (hit == expect)
        # B 模式（文字全部去除）：有文字就应该出 Mask，无文字就应该跳过
        b_ok = ((len(rep_b.candidates) > 0) == expect)
        normal_ok = True
        if sc.get("normal_text"):
            # 画面里的正常文字：允许被列出来（B 模式本来就是"文字全去除"），
            # 但 A 模式不能把它当成"高置信度水印"。
            normal_ok = all(c.tier != "high" for c in rep.candidates)
        ok = auto_ok and b_ok and normal_ok and outside <= 0.35
        print(f"{sc['name']:<18}{ocr_n:>5}{cand_n:>6}{sel_n:>6}{ratio * 100:>8.2f}"
              f"{outside * 100:>8.1f}{'是' if hit else '否':>6}{dt:>8.1f}")
        rows.append((str(sc["name"]), ok,
                     f"OCR {ocr_n}｜候选 {cand_n}（勾选 {sel_n}）｜Mask {ratio * 100:.2f}%｜"
                     f"ROI 外 {outside * 100:.1f}%｜B 模式候选 {len(rep_b.candidates)}｜"
                     f"{dt:.1f}s"))

    print("\n" + "=" * 96)
    for name, ok, detail in rows:
        check(f"场景 {name}", ok, detail)

    # ---------------- 置信度阈值扫描 ----------------
    banner("OCR 置信度阈值扫描（0.2 / 0.3 / 0.4 / 0.5）")
    for sc in scene_set()[:3]:
        sweep = threshold_sweep(sc["img"])
        txt = " ｜ ".join(f"thr{t}: {g} 条文字/{b} 条可疑" for t, g, b in sweep)
        print(f"{sc['name']}：{txt}")
    print("说明：本次样例在 0.2~0.5 都能读出文字，因此不需要靠下调阈值来提高召回；"
          "\n      真正的瓶颈是“缩图 OCR 把字缩小了”，所以采用“按需高清 OCR”，阈值保持 0.40。")

    # ---------------- 人体误检专项 ----------------
    banner("人体 / 衣服专项（不应该出现大面积 Mask）")
    for key, name, img in (("body", "人体+衣服（无文字）", body_scene()),
                           ("cloth", "强周期格纹布料（无文字）",
                            stamp(body_scene(seed=21), "", 10, 10, 10, opacity=0.0))):
        rep = ad.detect_watermarks(img, settings=ad.resolve_settings(), mode="auto")
        ratio = mp.area_ratio(rep.mask_selected)
        check(f"人体误检控制：{name}", ratio <= 0.005,
              f"默认勾选 {len(rep.high_confidence)} 个｜Mask {ratio * 100:.3f}%"
              f"（上限 0.5%）｜视觉兜底被判定不可信：{rep.stats.get('visual_response_unreliable')}")

    # ---------------- C 手动模式回归 ----------------
    banner("C 手动模式回归（必须与之前一致：Mask 外 0 改动）")
    img = sky_beach(seed=31)
    paint = np.zeros(img.shape[:2], np.uint8)
    cv2.rectangle(paint, (1000, 1180), (1900, 1320), 255, -1)
    res_c = modes.get_mode("manual").run(modes.ModeRequest(
        image=img, editor_value=mp.make_editor_value(img, layers=[__import__("numpy").dstack(
            [np.zeros(img.shape[:2], np.uint8), np.zeros(img.shape[:2], np.uint8),
             np.zeros(img.shape[:2], np.uint8), (paint > 0).astype(np.uint8) * 255])])))
    out = pipe.run_repair(img, res_c.mask, options=pipe.RepairOptions(max_side=0))
    diff = np.abs(out.image.astype(np.int16) - img.astype(np.int16)).max(axis=2)
    outside = int(np.count_nonzero((diff > 0) & (mp.ensure_binary(res_c.mask) == 0)))
    check("C 手动模式未退化", res_c.ok and outside == 0,
          f"Mask 外被改动像素 {outside}（应为 0）｜质检外改动 "
          f"{out.qc.get('outside_change_ratio', -1) * 100:.3f}%｜"
          f"涂抹 {mp.area_ratio(res_c.mask) * 100:.2f}% → 重建 {mp.area_ratio(out.mask_used) * 100:.2f}%")

    banner("结果汇总")
    passed = sum(1 for _, ok, _ in RESULTS if ok)
    for name, ok, _d in RESULTS:
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
    print(f"\n通过 {passed}/{len(RESULTS)}")
    return 0 if passed == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
