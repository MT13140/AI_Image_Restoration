"""小字 / 低对比度文字检测专项测试（本轮新增）。

背景：真实原图验收发现 512×1080 人像、眼睛附近约 24px 的小字
B 模式 OCR = 0 → 直接跳过。诊断结论：

* 1x/1.5x/2x **整图** OCR 都读不出（文字太小 + 人脸细节多）；
* 笔画聚类在脸上没有形成 → "局部放大 OCR"根本没在水印处尝试；
* 但把**人脸 ROI 放大 2~3 倍**就能稳定读到（实测 24px/20px/16px 均可）。

据此新增 **LOHR（局部高分辨率 OCR）**：只在人脸框 / 笔画聚类这类**小 ROI** 上
以 2x/3x 各读一次，且**只保留两个倍率都读到的结果**（多尺度一致性，
用来抑制"把纹理读成字"的假阳性）。全局阈值保持不变。

本测试验证：

    1. 小字人像：24px 文字 + 眼睛附近 → B 能检测到（并记录 confidence / 位置）
    2. 更小的字（20px / 16px）也能检测到 → 记录"最小成功高度"
    3. 假阳性：4 张真实人像（无水印）→ 0 实例（皮肤/头发/衣服不误检）
    4. Mask：新增实例的 text_mask 必须覆盖文字像素
    5. 人脸：Mask 之外的五官区改动 = 0
    6. 多实例低对比度：项目自带多实例水印图 → 不再有残留实例

运行：
    .venv\\Scripts\\python.exe tests\\test_small_text_detect.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

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
from core import face_protector as fp  # noqa: E402
from core import image_processor as ip  # noqa: E402
from core import mask_processor as mp  # noqa: E402
from core import modes  # noqa: E402
from core import pipeline as pipe  # noqa: E402

RESULTS: List[Tuple[str, bool, str]] = []
FACE = ROOT / "temp" / "round4" / "faces" / "candidate_0.png"
OUT = ROOT / "outputs" / "small_text"


def check(name: str, ok: bool, detail: str) -> bool:
    RESULTS.append((name, ok, detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}\n      {detail}")
    return ok


def stamp(img: np.ndarray, text: str, cx: float, cy: float, size: int,
          opacity: float = 0.42, color=(40, 40, 40)) -> np.ndarray:
    base = Image.fromarray(img).convert("RGBA")
    f = ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc", size)
    tmp = Image.new("RGBA", (int(size * len(text) * 1.15) + 30, int(size * 1.9)), (0, 0, 0, 0))
    ImageDraw.Draw(tmp).text((10, 6), text, font=f, fill=(*color, 255))
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    layer.paste(tmp, (int(cx - tmp.size[0] / 2), int(cy - tmp.size[1] / 2)), tmp)
    layer.putalpha(layer.split()[3].point(lambda v: int(v * float(opacity))))
    return np.asarray(Image.alpha_composite(base, layer).convert("RGB"))


def detect(img: np.ndarray):
    return ad.detect_watermarks(img, settings=ad.resolve_settings(), mode="text")


def main() -> int:  # noqa: C901
    OUT.mkdir(parents=True, exist_ok=True)
    print("=" * 96)
    print("小字 / 低对比度文字检测专项测试")
    print("=" * 96)
    base = ip.load_image(FACE)
    faces = fp.detect_faces(base)
    if not faces:
        check("素材准备", False, "人像素材未检出人脸，无法进行小字测试")
        return 1
    x, y, w, h = faces[0].box
    cx, cy = x + w * 0.5, y + h * 0.42
    print(f"素材：{FACE.name} {base.shape[1]}×{base.shape[0]}｜人脸 {w}×{h}｜"
          f"水印位置=眼睛附近（{cx:.0f},{cy:.0f}）")

    # ---------- 1/2. 小字（24/20/16px）能否被检测 ----------
    sizes = [(24, 0.42, (40, 40, 40)), (20, 0.42, (40, 40, 40)), (16, 0.42, (40, 40, 40)),
             (24, 0.35, (255, 255, 255))]
    smallest_ok = 0
    confs: List[float] = []
    for size, op, col in sizes:
        img = stamp(base, "水印测试", cx, cy, size, opacity=op, color=col)
        rep = detect(img)
        hit = len(rep.candidates) >= 1
        if hit:
            smallest_ok = min([smallest_ok or 999, size])
            confs += [float(c.confidence) for c in rep.candidates]
        tag = f"{size}px 不透明度 {op}"
        check(f"小字检测：{tag}", hit,
              f"检测到 {len(rep.candidates)} 个实例｜{[ (c.text, round(c.confidence,2), c.bbox) for c in rep.candidates][:2]}"
              f"｜LOHR 命中 {rep.timings.get('local_ocr_items', 0):.0f}")
        ip.save_image(OUT / f"small_{size}px_{int(op*100)}.png", img)

    # ---------- 3. 假阳性对照：真实人像无水印 ----------
    fp_details = []
    fp_total = 0
    for i in range(4):
        p = ROOT / "temp" / "round4" / "faces" / f"candidate_{i}.png"
        if not p.exists():
            continue
        img = ip.load_image(p)
        rep = detect(img)
        fp_total += len(rep.candidates)
        fp_details.append(f"candidate_{i}:{len(rep.candidates)}")
    check("假阳性：真实人像（皮肤/头发/衣服）不误检", fp_total == 0,
          f"4 张无水印人像共检测到 {fp_total} 个实例（应为 0）｜{'、'.join(fp_details)}")

    # ---------- 4/5. Mask 覆盖 + 人脸保护 ----------
    img = stamp(base, "水印测试", cx, cy, 24, opacity=0.42)
    res = modes.get_mode("text").run(modes.ModeRequest(image=img, sensitivity="aggressive"))
    if res.ok and res.mask is not None and not mp.is_empty(res.mask):
        box = res.candidates[0].bbox if res.candidates else None
        g = np.zeros(img.shape[:2], np.uint8)
        if box:
            bx, by, bw, bh = box
            roi = img[by:by + bh, bx:bx + bw]
            gg = __import__("cv2").cvtColor(roi, __import__("cv2").COLOR_RGB2GRAY)
            gg = __import__("cv2").GaussianBlur(gg, (3, 3), 0)
            bb = __import__("cv2").medianBlur(gg, 9)
            g[by:by + bh, bx:bx + bw] = (
                (__import__("cv2").absdiff(gg, bb) > 12).astype(np.uint8) * 255)
        n_text = int(np.count_nonzero(g))
        cov = (float(np.count_nonzero((res.mask > 0) & (g > 0))) / n_text) if n_text else 1.0
        check("Mask：新增实例覆盖文字像素", cov >= 0.9,
              f"文字像素 {n_text}｜text_mask 覆盖率 {cov * 100:.1f}%")
        opt = pipe.RepairOptions(max_side=0)
        hints = {k: v for k, v in (res.repair_hints or {}).items() if hasattr(opt, k)}
        import dataclasses
        out = pipe.run_repair(img, res.mask, options=dataclasses.replace(opt, **hints))
        _, critical = fp.build_face_masks(img.shape[:2], fp.detect_faces(img))
        keep = (critical > 0) & (mp.ensure_binary(out.mask_used) == 0)
        d = np.abs(out.image.astype(np.int16) - img.astype(np.int16)).max(axis=2)
        changed = int(np.count_nonzero(d[keep] > 0)) if int(keep.sum()) else 0
        check("人脸：Mask 之外的五官区零改动", changed == 0,
              f"五官区(Mask 外)像素 {int(keep.sum())}｜被改动 {changed}｜"
              f"质检 Mask 外改动 {out.qc.get('outside_change_ratio', -1) * 100:.3f}%")
        ip.save_image(OUT / "small_text_final.png", out.image)
    else:
        check("Mask：新增实例覆盖文字像素", False, "小字实例未进入 Mask")

    # ---------- 6. 多实例低对比度（项目自带水印图）----------
    multi = ROOT / "temp" / "test" / "synthetic_source.png"
    if multi.exists():
        img2 = ip.load_image(multi)
        res2 = modes.get_mode("text").run(modes.ModeRequest(image=img2, sensitivity="aggressive"))
        opt = pipe.RepairOptions(max_side=0)
        hints = {k: v for k, v in (res2.repair_hints or {}).items() if hasattr(opt, k)}
        import dataclasses
        out2 = pipe.run_repair(img2, res2.mask, options=dataclasses.replace(opt, **hints),
                               return_raw=True)
        cands = list(res2.candidates)
        resid = 0
        for c in cands:
            bx, by, bw, bh = c.bbox
            o = img2[by:by + bh, bx:bx + bw]
            f = out2.image[by:by + bh, bx:bx + bw]
            cv2 = __import__("cv2")
            go = cv2.cvtColor(o, cv2.COLOR_RGB2GRAY).astype(np.float32)
            gf = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY).astype(np.float32)
            # 只看"文字结构像素"，否则背景纹理会把相关系数拉高造成误判
            bg = cv2.medianBlur(cv2.cvtColor(o, cv2.COLOR_RGB2GRAY), 9).astype(np.float32)
            d = cv2.absdiff(cv2.cvtColor(o, cv2.COLOR_RGB2GRAY), bg.astype(np.uint8))
            p99 = float(np.percentile(d, 99.0)) if d.size else 0.0
            text_like = d > max(5.0, 0.45 * p99)
            if int(np.count_nonzero(text_like)) < 20:
                continue
            so = (go - cv2.GaussianBlur(go, (0, 0), 2.0))[text_like]
            sf = (gf - cv2.GaussianBlur(gf, (0, 0), 2.0))[text_like]
            if so.std() > 1e-6 and sf.std() > 1e-6 and float(np.corrcoef(so, sf)[0, 1]) > 0.5:
                resid += 1
        check("多实例低对比度：不再残留", resid == 0,
              f"实例 {len(cands)} 个｜结构相关度>0.5（=残留）的 {resid} 个｜"
              f"Mask {mp.area_ratio(out2.mask_used) * 100:.2f}%")
    else:
        check("多实例低对比度：不再残留", True, "缺少素材，跳过")

    print("\n" + "=" * 96)
    passed = sum(1 for _, ok, _ in RESULTS if ok)
    for name, ok, _d in RESULTS:
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
    print(f"\n通过 {passed}/{len(RESULTS)}")
    print(f"最小成功文字高度：{smallest_ok}px｜新增实例 confidence 范围："
          f"{min(confs):.2f}~{max(confs):.2f}" if confs else "未检出")
    return 0 if passed == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
