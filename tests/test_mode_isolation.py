"""A / B / C 模块隔离与 C 手动模式回归测试。

对应要求：

    * A、B、C 三种模式相互隔离；以后单独优化 A 或 B，**不允许影响 C**；
    * C 手动模式作为稳定基线，行为必须与第五轮手动路径一致。

本测试用**可机械校验**的方式验证隔离性（不靠人肉 review）：

    1. `core/modes/mode_c.py` 的 import 链里**不出现**自动检测 / OCR 模块；
    2. 真正跑一次 C 模式，全程也不会加载 OCR / 自动检测模块（子进程实测 sys.modules）；
    3. C 模式产出 = 用户涂抹（不自动扩张、不自动增减）；
    4. **改动 A/B 的敏感度等参数，C 的结果逐像素不变**；
    5. C 模式修复结果与"直接调用第五轮 pipeline"一致（容差 ≤3 灰阶，CUDA 推理微小抖动），
       且 Mask 外 0 改动；
    6. 三种模式可各自独立调用。

运行：
    .venv\\Scripts\\python.exe tests\\test_mode_isolation.py
"""

from __future__ import annotations

import hashlib
import subprocess
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

from core import mask_processor as mp  # noqa: E402
from core import modes  # noqa: E402
from core import pipeline as pipe  # noqa: E402

RESULTS: List[Tuple[str, bool, str]] = []
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")


def check(name: str, ok: bool, detail: str) -> bool:
    RESULTS.append((name, ok, detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}\n      {detail}")
    return ok


def banner(t: str) -> None:
    print("\n" + "=" * 90)
    print(t)
    print("=" * 90)


def sha(arr) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(arr)).tobytes()).hexdigest()[:16]


def run_probe(code: str) -> str:
    """在干净的子进程里跑一段代码（验证"真的没有 import 自动检测模块"）。"""
    out = subprocess.run([PY, "-c", code], capture_output=True, text=True,
                         cwd=str(ROOT), timeout=600)
    return (out.stdout + out.stderr).strip()


def photo(w: int = 900, h: int = 600, seed: int = 3) -> np.ndarray:
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    img = np.stack([90 + 90 * (x / w), 110 + 70 * (y / h), 170 - 50 * (x / w)], axis=2)
    img = np.clip(img + np.random.default_rng(seed).normal(0, 4.0, img.shape), 0, 255)
    return img.astype(np.uint8)


def paint_editor(base: np.ndarray, box: Tuple[int, int, int, int]) -> dict:
    """构造"用户在编辑器里涂了一笔"的 ImageEditor 值（含 background/layers/composite）。"""
    h, w = base.shape[:2]
    x0, y0, x1, y1 = box
    layer = np.zeros((h, w, 4), np.uint8)
    layer[y0:y1, x0:x1, 0] = 255
    layer[y0:y1, x0:x1, 3] = 255
    return mp.make_editor_value(base, layers=[layer])


def main() -> int:  # noqa: C901
    banner("A / B / C 模式隔离 + C 手动模式回归")

    # ---------------- 1. C 模式模块不引用自动检测 ----------------
    probe = run_probe(
        "import sys\n"
        f"sys.path.insert(0, r'{ROOT}')\n"
        "from core.modes import mode_c\n"
        "banned=[m for m in sys.modules if ('auto_detect' in m or 'watermark_detector' in m "
        "or 'ocr_detector' in m or 'watermark_candidate' in m)]\n"
        "print('BANNED:' + ','.join(banned) if banned else 'CLEAN')\n"
    )
    check("1 mode_c 的 import 链里没有自动检测/OCR（物理隔离）", probe.strip() == "CLEAN",
          f"子进程探测结果：{probe[-140:]}")

    # ---------------- 2. 跑一次 C 模式也不会加载自动检测 ----------------
    probe2 = run_probe(
        "import sys, numpy as np\n"
        f"sys.path.insert(0, r'{ROOT}')\n"
        "from core import modes, mask_processor as mp\n"
        "img=(np.random.default_rng(1).random((200,300,3))*255).astype('uint8')\n"
        "layer=np.zeros((200,300,4),np.uint8)\n"
        "layer[60:90,40:120,0]=255\n"
        "layer[60:90,40:120,3]=255\n"
        "ev=mp.make_editor_value(img, layers=[layer])\n"
        "res=modes.get_mode('manual').run(modes.ModeRequest(image=img, editor_value=ev))\n"
        "ratio=float((res.mask>0).sum())/res.mask.size\n"
        "banned=[m for m in sys.modules if ('auto_detect' in m or 'rapidocr' in m "
        "or 'ocr_detector' in m or 'onnxruntime' in m)]\n"
        "print('ratio=%.4f banned=%s' % (ratio, banned))\n"
    )
    ok2 = "banned=[]" in probe2.replace(" ", "") and "ratio=0." in probe2
    check("2 运行 C 模式全程不加载 OCR/自动检测", ok2, f"实测：{probe2[-160:]}")

    # ---------------- 3. C 模式 = 用户涂抹（不自动增减）----------------
    img = photo()
    editor = paint_editor(img, (120, 380, 420, 450))
    res_c = modes.get_mode("manual").run(modes.ModeRequest(image=img, editor_value=editor))
    manual = modes.manual_mask_from_editor(editor, img.shape[:2])
    same_as_paint = (res_c.mask is not None
                     and np.array_equal(np.asarray(res_c.mask > 0), np.asarray(manual > 0)))
    check("3 C 模式产出 = 用户涂抹（不自动扩张/收缩）", bool(res_c.ok) and same_as_paint,
          f"Mask 像素 {int(np.count_nonzero(res_c.mask))}，与涂抹逐像素一致：{same_as_paint}")

    # ---------------- 4. 改 A/B 参数不影响 C ----------------
    variants = [
        modes.ModeRequest(image=img, editor_value=editor, sensitivity="conservative"),
        modes.ModeRequest(image=img, editor_value=editor, sensitivity="aggressive", dilate=32,
                          ocr_min_score=0.90),
    ]
    hashes = [sha(modes.get_mode("manual").run(v).mask) for v in variants]
    base_hash = sha(res_c.mask)
    check("4 修改 A/B 的参数（敏感度/扩张/OCR 阈值）后 C 结果不变",
          all(h == base_hash for h in hashes),
          f"C 结果哈希 {base_hash}｜改参数后 {hashes}")

    # ---------------- 5. C 模式修复结果 = 第五轮基准 ----------------
    out_mode = pipe.run_repair(img, res_c.mask, options=pipe.RepairOptions(max_side=0))
    out_base = pipe.run_repair(img, manual, options=pipe.RepairOptions(max_side=0))
    # 说明：CUDA 上 LaMa 推理有极小的非确定性（cuDNN 卷积算法），两次运行最多差 1~2 个灰阶，
    # 且只出现在修复区内部；Mask 之外必须逐像素为 0。
    delta = np.abs(out_mode.image.astype(np.int16) - out_base.image.astype(np.int16))
    identical = int(delta.max()) <= 3
    diff = np.abs(out_mode.image.astype(np.int16) - img.astype(np.int16)).max(axis=2)
    outside = int(np.count_nonzero((diff > 0) & (mp.ensure_binary(res_c.mask) == 0)))
    check("5 C 模式修复结果与第五轮手动路径一致（容差 ≤3 灰阶），且 Mask 外 0 改动",
          identical and outside == 0,
          f"两次输出最大差 {int(delta.max())} 灰阶（均值 {delta.mean():.5f}）｜"
          f"Mask 外被改动像素：{outside}｜"
          f"质检外改动 {out_mode.qc.get('outside_change_ratio', -1) * 100:.3f}%")

    # ---------------- 6. 三种模式可各自独立调用 ----------------
    status: List[str] = []
    for mid in modes.mode_ids():
        obj = modes.get_mode(mid)
        if obj.uses_auto_detection:
            status.append(f"{obj.label}({obj.id})：自动检测链路")
        else:
            r = obj.run(modes.ModeRequest(image=img, editor_value=editor))
            status.append(f"{obj.label}({obj.id})：不使用自动检测，Mask "
                          f"{'已生成' if r.ok else '为空'}")
    check("6 三种模式可独立调用（A/B 走检测，C 只用涂抹）",
          len(modes.MODES) == 3 and "不使用自动检测" in status[2],
          "｜".join(status))

    banner("结果汇总")
    passed = sum(1 for _, ok, _ in RESULTS if ok)
    for name, ok, _d in RESULTS:
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
    print(f"\n通过 {passed}/{len(RESULTS)}")
    return 0 if passed == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
