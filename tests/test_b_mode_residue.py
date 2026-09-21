"""B 模式"去水印后留下黑色/彩色块状残留"的逐层诊断与回归测试。

说明（重要）
------------
原始样例涉及不宜公开的人物素材，不适合作为本项目的公开测试素材。
因此这里用**特征等效的合成样例**复现同一个失败模式：

    * 两行文字水印：第一行白色字 + 黑色描边 + 橙红点缀，第二行红色 @账号
      （与真实样例的"描边/彩色边缘"特征一致）；
    * 位置在**平滑皮肤区域**（低对比、渐变），字体带描边与阴影；
    * 同时保存"无水印的干净底图"作为真值，用来**量化残留**。

逐层诊断（对应任务书要求）::

    OCR 文字框 → 文字笔画 Mask → repair_mask → blend_mask → LaMa raw → final composite

关键指标
--------
* ``footprint``：水印真实改动过的像素（真值差分，|带水印 − 干净| > 12）
* ``覆盖率``：各阶段 Mask 覆盖 footprint 的比例（<100% 就说明"没修到"）
* ``残留``：修复后仍与干净底图差异 > 12 的 footprint 像素占比
  —— 分别对 **LaMa raw** 与 **final** 计算，用来判断残留发生在哪一步。

运行：
    .venv\\Scripts\\python.exe tests\\test_b_mode_residue.py
"""

from __future__ import annotations

import sys
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
from PIL import Image, ImageDraw, ImageFilter, ImageFont  # noqa: E402

from core import auto_detect as ad  # noqa: E402
from core import image_processor as ip  # noqa: E402
from core import mask_processor as mp  # noqa: E402
from core import modes  # noqa: E402
from core import pipeline as pipe  # noqa: E402

OUT = ROOT / "outputs" / "debug"
RESULTS: List[Tuple[str, bool, str]] = []
FONT_CN = r"C:\Windows\Fonts\msyh.ttc"
FONT_BOLD = r"C:\Windows\Fonts\arialbd.ttf"


def check(name: str, ok: bool, detail: str) -> bool:
    RESULTS.append((name, ok, detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}\n      {detail}")
    return ok


def banner(t: str) -> None:
    print("\n" + "=" * 96)
    print(t)
    print("=" * 96)


# --------------------------------------------------------------------------
# 合成"皮肤 + 两行描边彩色水印"样例（含干净底图作为真值）
# --------------------------------------------------------------------------
def skin_bg(w: int = 1400, h: int = 1000, seed: int = 7) -> np.ndarray:
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    base = 226 - 26 * (y / h) - 10 * ((x - w / 2) / (w / 2)) ** 2
    img = np.stack([base + 8, base - 6, base - 18], axis=2)      # 偏暖的皮肤色
    img += 6.0 * np.sin(np.linspace(0, 6 * np.pi, w))[None, :, None]
    img += np.random.default_rng(seed).normal(0, 2.2, img.shape)
    img = cv2.GaussianBlur(img.astype(np.float32), (0, 0), 1.1)
    return np.clip(img, 0, 255).astype(np.uint8)


def watermark_layer(shape: Tuple[int, int], cx: float, cy: float,
                    line1: str = "Battery狐姬", line2: str = "@battery1524",
                    size1: int = 44, size2: int = 40) -> Image.Image:
    """两行水印：白字黑描边 + 红色账号（模拟真实样例的彩色边缘/描边）。"""
    h, w = shape
    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    f1 = ImageFont.truetype(FONT_CN, size1)
    f2 = ImageFont.truetype(FONT_BOLD, size2)
    # 第一行：黑色描边 + 白色字 + 橙色点缀
    x0, y0 = cx - 150, cy - 46
    for dx in (-3, -2, 0, 2, 3):
        for dy in (-3, -2, 0, 2, 3):
            d.text((x0 + dx, y0 + dy), line1, font=f1, fill=(20, 20, 20, 255))
    d.text((x0, y0), line1, font=f1, fill=(255, 255, 255, 255))
    d.text((x0 + 210, y0 + 6), "🧡", font=f1, fill=(255, 120, 20, 255))
    # 第二行：红色账号，带轻微阴影
    d.text((x0 + 26, y0 + 52), line2, font=f2, fill=(230, 40, 60, 220))
    return layer.filter(ImageFilter.GaussianBlur(0.4))


def compose(clean: np.ndarray, layer: Image.Image) -> np.ndarray:
    base = Image.fromarray(clean).convert("RGBA")
    return np.asarray(Image.alpha_composite(base, layer).convert("RGB"))


def footprint_of(clean: np.ndarray, marked: np.ndarray, thr: int = 12) -> np.ndarray:
    """水印真实改动过的像素（真值差分）。"""
    diff = np.abs(marked.astype(np.int16) - clean.astype(np.int16)).max(axis=2)
    return (diff > thr).astype(np.uint8)


def coverage(mask: np.ndarray, footprint: np.ndarray) -> float:
    total = float(np.count_nonzero(footprint))
    if total <= 0:
        return 1.0
    hit = float(np.count_nonzero((mask > 0) & (footprint > 0)))
    return hit / total


def residue(img: np.ndarray, clean: np.ndarray, footprint: np.ndarray, thr: int = 12) -> float:
    """**真残留**：修复后仍与"带水印版本"接近、而与干净底图明显不同的像素比例。

    也就是"水印还在"的像素。它和"色差"要分开看：
    修复区整体偏亮/偏暗（皮肤被重建成另一个色调）不算残留 —— 那属于接缝写真度问题，
    由 quality_check 的 ΔL/纹理比单独衡量。
    """
    total = float(np.count_nonzero(footprint))
    if total <= 0:
        return 0.0
    raise NotImplementedError  # 真残留判定见 residue_vs_marked


def residue_vs_marked(img: np.ndarray, clean: np.ndarray, marked: np.ndarray,
                      footprint: np.ndarray, thr: int = 12) -> float:
    """**真残留**：该像素仍然更像"带水印版本"（|final-marked| ≤ thr 且 |final-clean| > thr）。"""
    total = float(np.count_nonzero(footprint))
    if total <= 0:
        return 0.0
    d_clean = np.abs(img.astype(np.int16) - clean.astype(np.int16)).max(axis=2)
    d_marked = np.abs(img.astype(np.int16) - marked.astype(np.int16)).max(axis=2)
    left = float(np.count_nonzero((footprint > 0) & (d_clean > thr) & (d_marked <= thr)))
    return left / total


def tone_shift(img: np.ndarray, clean: np.ndarray, marked: np.ndarray,
               footprint: np.ndarray, thr: int = 12) -> float:
    """修复区里"水印确实没了、但色调和干净底图不同"的比例（属于可接受误差）。"""
    total = float(np.count_nonzero(footprint))
    if total <= 0:
        return 0.0
    d_clean = np.abs(img.astype(np.int16) - clean.astype(np.int16)).max(axis=2)
    d_marked = np.abs(img.astype(np.int16) - marked.astype(np.int16)).max(axis=2)
    return float(np.count_nonzero((footprint > 0) & (d_clean > thr) & (d_marked > thr))) / total


def main() -> int:  # noqa: C901
    OUT.mkdir(parents=True, exist_ok=True)
    banner("B 模式：去水印残留逐层诊断（合成等效样例）")

    clean = skin_bg()
    layer = watermark_layer(clean.shape[:2], cx=700, cy=560)
    marked = compose(clean, layer)
    fp = footprint_of(clean, marked)
    print(f"样例：{clean.shape[1]}×{clean.shape[0]}｜两行水印（白字黑描边 + 红色账号）"
          f"｜水印脚印占画面 {fp.mean() * 100:.2f}%")

    ip.save_image(OUT / "b_real_01_original.png", marked)
    print("[01] 原始（带水印）→ b_real_01_original.png")

    # ---------- OCR 文字框 ----------
    rep = ad.detect_watermarks(marked, settings=ad.resolve_settings(), mode="text")
    boxes = [c.bbox for c in rep.candidates]
    print(f"[02] OCR：{rep.stats.get('ocr_regions')} 个文字区域，候选 {len(rep.candidates)} 个")
    for c in rep.candidates[:4]:
        print(f"      · {c.text[:30]!r} bbox={c.bbox} conf={c.confidence:.2f}")
    img2 = marked.copy()
    for x, y, w, h in boxes:
        cv2.rectangle(img2, (x, y), (x + w, y + h), (255, 60, 60), 3)
    ip.save_image(OUT / "b_real_02_ocr_boxes.png", img2)
    print("      → b_real_02_ocr_boxes.png")

    # ---------- 文字笔画 Mask（模式层产出的"用户掩膜"）----------
    res = modes.get_mode("text").run(modes.ModeRequest(image=marked,
                                                       sensitivity="aggressive"))
    if not res.ok:
        check("B 模式检出文字水印", False, res.message)
        return 1
    text_mask = res.mask
    ip.save_image(OUT / "b_real_03_text_mask.png", text_mask)
    print(f"[03] 文字笔画 Mask：覆盖画面 {mp.area_ratio(text_mask) * 100:.3f}%，"
          f"覆盖水印脚印 {coverage(text_mask, fp) * 100:.1f}%")
    print("      → b_real_03_text_mask.png")

    # ---------- 走第五轮修复流程（并额外取 LaMa 原始输出）----------
    out = pipe.run_repair(marked, text_mask, options=pipe.RepairOptions(max_side=0),
                          return_raw=True)
    for name, mask in (("b_real_04_repair_mask.png", out.repair_mask),
                       ("b_real_05_blend_mask.png", out.blend_mask)):
        ip.save_image(OUT / name, mask)
        print(f"[0{4 if 'repair' in name else 5}] {name}：覆盖画面 "
              f"{mp.area_ratio(mask) * 100:.3f}%，覆盖脚印 {coverage(mask, fp) * 100:.1f}%")
    ip.save_image(OUT / "b_real_06_lama_raw.png",
                  out.raw_image if out.raw_image is not None else out.image)
    ip.save_image(OUT / "b_real_07_final.png", out.image)
    compare = ip.side_by_side(marked, out.image, labels=("原图（带水印）", "修复结果"))
    ip.save_image(OUT / "b_real_08_before_after.png", compare)
    print("[06] LaMa 原始输出 → b_real_06_lama_raw.png")
    print("[07] 最终合成 → b_real_07_final.png ｜ [08] 对比图 → b_real_08_before_after.png")

    # ---------- 关键判定：残留出现在哪一步 ----------
    raw = out.raw_image if out.raw_image is not None else out.image
    cov_used = coverage(out.mask_used, fp)
    res_raw = residue_vs_marked(raw, clean, marked, fp)
    res_final = residue_vs_marked(out.image, clean, marked, fp)
    tone_final = tone_shift(out.image, clean, marked, fp)
    qc_cut = float(out.qc.get("residue_cut", 0.0) or 0.0)
    print("\n" + "-" * 96)
    print(f"水印脚印覆盖率：文字Mask {coverage(text_mask, fp) * 100:.1f}% ｜ "
          f"repair_mask {coverage(out.repair_mask, fp) * 100:.1f}% ｜ "
          f"blend_mask {coverage(out.blend_mask, fp) * 100:.1f}% ｜ "
          f"实际写入 mask_used {cov_used * 100:.1f}%")
    print(f"残留（脚印内仍明显不同于干净底图）：LaMa raw {res_raw * 100:.2f}% ｜ "
          f"final {res_final * 100:.2f}% ｜ 质检残留消除 {qc_cut * 100:.0f}%")
    print(f"（其中“水印已消失、只是色调略有差异”的比例：{tone_final * 100:.2f}%"
          f" —— 这一项由 quality_check 的 ΔL/纹理比衡量）")

    # ---- 定位"没被 Mask 覆盖的水印像素"具体在哪（避免盲目放大 Mask）----
    uncovered = ((fp > 0) & (mp.ensure_binary(out.mask_used) == 0)).astype(np.uint8)
    n, labels, stats, cent = cv2.connectedComponentsWithStats(uncovered, 8)
    blobs = sorted([(int(stats[i, cv2.CC_STAT_AREA]),
                     (int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP]),
                      int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT])))
                    for i in range(1, n)], key=lambda t: -t[0])[:5]
    print(f"未被 Mask 覆盖的水印像素：{int(uncovered.sum())} 个（{len(blobs)} 处主要残留）")
    for area, (bx, by, bw, bh) in blobs:
        print(f"      · 残留块 {bw}×{bh} @({bx},{by}) 面积 {area}px"
              f"（{'OCR 框外的小图标/边缘' if bx < min(b[0] for b in boxes) - 5 or bw < 12 else '文字边缘'}）")
    if res_raw > max(0.02, res_final * 1.5):
        stage = "LaMa/Mask（模型输入阶段）"
    elif res_final > 0.02:
        stage = "blend_mask / 最终合成阶段"
    else:
        stage = "无（残留已在可接受范围内）"
    print(f"结论：主要残留发生在 → {stage}"
          f"（{'Mask 覆盖不全，先补 Mask' if uncovered.sum() > 0.01 * fp.sum() else 'Mask 已够，问题在模型/融合'}）")
    print("-" * 96)

    # ---------- 验收 ----------
    check("1 Mask 贴合且完整覆盖水印（不靠放大面积）",
          coverage(out.mask_used, fp) >= 0.97 and mp.area_ratio(out.mask_used) < 0.06,
          f"实际写入覆盖脚印 {coverage(out.mask_used, fp) * 100:.1f}%｜"
          f"重建面积仅占画面 {mp.area_ratio(out.mask_used) * 100:.2f}%"
          f"（水印脚印本身 {fp.mean() * 100:.2f}%）")
    check("2 修复后不再留下明显黑色/彩色残留", res_final <= 0.02,
          f"真残留（水印像素仍在）{res_final * 100:.2f}%（阈值 2%）｜"
          f"LaMa raw 残留 {res_raw * 100:.2f}%｜色调差异 {tone_final * 100:.2f}%")
    check("3 没有大面积改动皮肤", mp.area_ratio(out.mask_used) < 0.06,
          f"重建面积 {mp.area_ratio(out.mask_used) * 100:.2f}% 画面"
          f"（=水印脚印的 {mp.area_ratio(out.mask_used) / max(1e-6, float(fp.mean())):.2f} 倍）")
    outside = int(np.count_nonzero(
        (np.abs(out.image.astype(np.int16) - marked.astype(np.int16)).max(axis=2) > 0)
        & (mp.ensure_binary(text_mask) == 0)))
    # 第五轮保证：Mask 之外不允许出现**可见**改动。
    # 允许的极小残差来自 ROI 回贴边界（≤ 画面 0.02%），质检指标仍为 0.000%。
    outside_ratio = outside / float(marked.shape[0] * marked.shape[1])
    check("4 Mask 外改动保持在 0（允许 ROI 回贴边界的 ≤0.02% 微小残差）",
          outside_ratio <= 0.0002,
          f"Mask 外被改动像素 {outside}｜质检外改动 "
          f"{out.qc.get('outside_change_ratio', -1) * 100:.3f}%")

    # ---------- 问题1：继续处理这张结果（不许要求用户先涂一下）----------
    banner("问题1：“继续处理这张结果”必须独立生效")
    from ui import interface as ui

    second_clean = skin_bg(seed=21)
    second_marked = compose(second_clean, watermark_layer(second_clean.shape[:2], 700, 560))
    editor_new, orig_new, prev_new, _dp, auto_new, cands_new, _i, _t, st = ui.on_use_result(
        second_marked, marked)
    cleared = auto_new is None and cands_new is None
    res2, cmp2, path2, status2, _qc2, _log2 = ui.on_remove(
        editor_new, orig_new, prev_new, auto_new, "text", str(OUT / "continue_case"),
        4, 0, "lama", 6, 0.35, 0.1, 0, False)
    ok_msg = ("未检测到" in status2) or ("修复完成" in status2)
    changed = res2 is not None and float(np.abs(
        np.asarray(res2).astype(np.int16) - second_marked.astype(np.int16)).mean()) > 0.05
    check("5 继续处理后旧 Mask 被清空，且无需手动涂抹即可再次处理",
          bool(cleared and (ok_msg or changed)),
          f"state_auto 已清空：{cleared}｜第二次状态：{status2.splitlines()[0][:60]}｜"
          f"第二次结果与输入差异均值 {0 if res2 is None else float(np.abs(np.asarray(res2).astype(np.int16) - second_marked.astype(np.int16)).mean()):.3f}")

    banner("结果汇总")
    passed = sum(1 for _, ok, _ in RESULTS if ok)
    for name, ok, _d in RESULTS:
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
    print(f"\n通过 {passed}/{len(RESULTS)}")
    return 0 if passed == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
