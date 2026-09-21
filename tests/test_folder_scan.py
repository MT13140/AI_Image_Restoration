"""文件夹选择 / 扫描 / 路径处理专项测试（本轮新增）。

覆盖任务书第十一章要求：
    1 JPG  2 JPEG  3 PNG  4 WEBP  5 BMP  6 大小写扩展名
    7 空文件夹  8 子文件夹与图片混合  9 输入目录≠输出目录  10 D 盘真实路径

并检查：
    * 选择文件夹后能给出"找到几张图片 + 前几个文件名"的中文摘要；
    * 选择**文件**的入口存在（能直接看到文件夹里的图片文件名）；
    * 输出目录 = 用户选择的目录，内部 restored/masks/reports 只是程序内部结构；
    * 不覆盖原图（同名自动加编号）；
    * 代码里**没有**硬编码 Desktop / C:\\Users 路径。
"""

from __future__ import annotations

import shutil
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
from PIL import Image  # noqa: E402

from core import batch_processor as bp  # noqa: E402
from core import folder_picker as fpk  # noqa: E402

WORK = ROOT / "temp" / "folder_scan"
RESULTS: List[Tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str) -> bool:
    RESULTS.append((name, ok, detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}\n      {detail}")
    return ok


def make_img(path: Path, size=(60, 40)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((size[1], size[0], 3), 128, np.uint8)).save(path)


def main() -> int:  # noqa: C901
    if WORK.exists():
        shutil.rmtree(WORK, ignore_errors=True)
    WORK.mkdir(parents=True, exist_ok=True)
    print("=" * 96)
    print("文件夹选择 / 扫描 / 路径专项测试")
    print("=" * 96)

    # ---------- 1~6. 五种格式 + 大小写扩展名 ----------
    src = WORK / "input"
    names = ["a.JPG", "b.jpeg", "c.PNG", "d.WebP", "e.BMP", "f.TIF", "ignore.txt", "note.md"]
    for n in names:
        if n.lower().endswith((".txt", ".md")):
            (src / n).parent.mkdir(parents=True, exist_ok=True)
            (src / n).write_text("x", encoding="utf-8")
        else:
            make_img(src / n)
    found = bp.collect_images([str(src)])
    exts = sorted({p.suffix.lower() for p in found})
    check("1-6 五种格式 + 大小写扩展名都能识别",
          len(found) == 6 and set(exts) == {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif"},
          f"识别到 {len(found)} 张：{[p.name for p in found]}｜扩展名（已转小写）{exts}｜"
          f"非图片文件被忽略：{not any(p.suffix.lower() in ('.txt', '.md') for p in found)}")

    # ---------- 8. 子文件夹与图片混合 ----------
    make_img(src / "sub" / "deep" / "g.jpg")
    found2 = bp.collect_images([str(src)])
    check("8 子文件夹里的图片也会被扫描到", len(found2) == 7,
          f"根目录+子目录共 {len(found2)} 张（含 sub/deep/g.jpg："
          f"{any(p.name == 'g.jpg' for p in found2)}）")

    # ---------- 7. 空文件夹 ----------
    empty = WORK / "empty"
    empty.mkdir(parents=True, exist_ok=True)
    check("7 空文件夹", bp.collect_images([str(empty)]) == []
          and fpk.summarize_images([]) == "该文件夹中未找到支持的图片文件",
          f"扫描结果为空；摘要文案：{fpk.summarize_images([])}")

    # ---------- 选择文件夹后的摘要文案 ----------
    summary = fpk.summarize_images([str(p) for p in found2])
    check("选择文件夹后自动统计数量 + 前几个文件名",
          ("已找到 7 张图片" in summary) and ("a.JPG" in summary or "b.jpeg" in summary),
          f"摘要：{summary}")

    # ---------- 选择“文件”的入口（能看到文件名）----------
    has_files_api = hasattr(fpk, "pick_files") and callable(fpk.pick_files)
    has_type_filter = any("*.jpg" in pat for pat in fpk.IMAGE_PATTERNS)
    check("提供“选择图片文件”入口（可直观看到文件夹内的图片）",
          has_files_api and has_type_filter,
          f"pick_files 可用：{has_files_api}｜文件类型过滤：{list(fpk.IMAGE_PATTERNS)[:3]}…")

    # ---------- 9. 输入目录 ≠ 输出目录 ----------
    out_dir = WORK / "output"
    dirs = bp.prepare_output_dirs(str(out_dir))
    distinct = str(dirs["root"]) != str(src)
    internal_ok = all((dirs[k]).is_dir() for k in ("restored", "masks", "reports"))
    check("9 输入目录与输出目录分离，内部结构独立",
          distinct and internal_ok and not (out_dir / "restored").samefile(src),
          f"输出根目录 = {dirs['root']}（≠ 输入 {src}）｜内部子目录：restored/masks/reports 已创建")

    # ---------- 不覆盖原图 ----------
    p1 = bp.unique_output_path(dirs["restored"], "photo", "_restored", ".jpg")
    p1.write_bytes(b"x")
    p2 = bp.unique_output_path(dirs["restored"], "photo", "_restored", ".jpg")
    check("输出不覆盖已有文件（同名自动加编号）", p2.name == "photo_restored_1.jpg",
          f"第一次：{p1.name}｜同名第二次：{p2.name}")

    # ---------- 10. D 盘真实路径（项目就在 D 盘，路径含中文/非 ASCII 也要能处理）----------
    d_path_ok = str(ROOT).lower().startswith("d:") and (ROOT / "app.py").exists()
    cn_dir = WORK / "中文 目录"
    make_img(cn_dir / "图 片.JPG")
    cn_found = bp.collect_images([str(cn_dir)])
    check("10 D 盘真实路径 + 中文/空格路径", d_path_ok and len(cn_found) == 1,
          f"项目路径 {ROOT}｜中文+空格目录扫描到 {len(cn_found)} 张："
          f"{[p.name for p in cn_found]}")

    # ---------- 没有硬编码 Desktop ----------
    bad: List[str] = []
    for f in list((ROOT / "core").glob("*.py")) + list((ROOT / "ui").glob("*.py")) + \
            [ROOT / "config.py", ROOT / "app.py"]:
        for i, line in enumerate(f.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
            # 只报"路径式"硬编码；文档里出现"桌面"这类说明词不算
            if "C:\\Users" in line or "Desktop" in line:
                bad.append(f"{f.name}:{i}")
            elif "桌面" in line and (":\\" in line or ":/" in line):
                bad.append(f"{f.name}:{i}")
    check("没有硬编码 Desktop / C:\\Users 路径", not bad,
          f"检查 core/、ui/、config.py、app.py｜命中：{bad or '无'}")

    print("\n" + "=" * 96)
    passed = sum(1 for _, ok, _ in RESULTS if ok)
    for name, ok, _d in RESULTS:
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
    print(f"\n通过 {passed}/{len(RESULTS)}")
    return 0 if passed == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
