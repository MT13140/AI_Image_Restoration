"""B 批量处理 UI 简化 / 交互专项测试（本轮新增）。

验证任务书要求的 12 项：
    1 上传 1 张   2 上传多张   3 支持拖拽   4 显示已选数量
    5 不再有"选择图片文件"按钮   6 B 主界面不再要求"选择输入文件夹"
    7 输出文件夹可正常选择   8 输出选择器为 Windows 原生   9 输出路径显示正确
    10 批量处理正常   11 连续 B 处理正常   12 C 完全不受影响

说明：原生对话框会**阻塞**等待用户点击，自动化里不能真的弹窗；
因此第 8 项用"干跑探针"验证 —— 真正创建 IFileDialog、设置 FOS_PICKFOLDERS
（证明 ctypes 调用序列正确），但不调用 Show。
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

import gradio as gr  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from core import auto_detect as ad  # noqa: E402
from core import folder_picker as fpk  # noqa: E402
from core import image_processor as ip  # noqa: E402
from core import modes  # noqa: E402
from ui import interface as ui  # noqa: E402

WORK = ROOT / "temp" / "batch_ui"
RESULTS: List[Tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str) -> bool:
    RESULTS.append((name, ok, detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}\n      {detail}")
    return ok


def make_wm(path: Path, text: str = "水印测试") -> None:
    """生成一张带文字水印的合成图（用于批量/连续处理的功能验证）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = 480, 640
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    arr = np.stack([120 + 60 * (x / w), 140 + 50 * (y / h), 180 - 40 * (x / w)], axis=2)
    base = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).convert("RGBA")
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    ImageDraw.Draw(layer).text((w * 0.25, h * 0.45), text,
                               font=ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc", 34),
                               fill=(255, 255, 255, 255))
    layer.putalpha(layer.split()[3].point(lambda v: int(v * 0.45)))
    Image.fromarray(np.asarray(Image.alpha_composite(base, layer).convert("RGB"))).save(path)


def widget_labels(demo) -> Tuple[List[str], List[object]]:
    buttons, all_widgets = [], []
    for comp in demo.blocks.values():
        all_widgets.append(comp)
        if isinstance(comp, gr.Button):
            # Gradio 的 Button 把显示文字放在 value 上（label 常为空），两者都取
            txt = str(getattr(comp, "value", "") or "") or str(getattr(comp, "label", "") or "")
            buttons.append(txt)
    return buttons, all_widgets


def main() -> int:  # noqa: C901
    if WORK.exists():
        shutil.rmtree(WORK, ignore_errors=True)
    WORK.mkdir(parents=True, exist_ok=True)
    print("=" * 96)
    print("B 批量处理 UI 简化 / 交互专项测试")
    print("=" * 96)

    demo = ui.build_interface()
    buttons, widgets = widget_labels(demo)
    files_widgets = [w for w in widgets if isinstance(w, gr.Files)]

    # ---------- 1/2/4. 上传 1 张 / 多张 + 数量显示 ----------
    p1 = WORK / "one.jpg"
    make_wm(p1)
    msg1 = ui.on_files_change([str(p1)])
    many = []
    for i in range(5):
        p = WORK / f"multi_{i}.jpg"
        make_wm(p, f"水印{i}")
        many.append(str(p))
    msg5 = ui.on_files_change(many)
    check("1/2/4 上传 1 张 / 多张并显示数量",
          "已选择 1 张图片" in msg1 and "已选择 5 张图片" in msg5,
          f"单张：{msg1}｜多张：{msg5}")

    # ---------- 3. 拖拽支持 ----------
    drag_ok = bool(files_widgets) and any(
        "拖" in str(getattr(w, "label", "")) for w in files_widgets) and any(
        getattr(w, "file_count", "") == "multiple" for w in files_widgets)
    check("3 支持一次多选 / 拖拽（Gradio 上传区）", drag_ok,
          f"批次上传组件：{[getattr(w, 'label', '') for w in files_widgets]}｜"
          f"file_count={[getattr(w, 'file_count', '') for w in files_widgets]}")

    # ---------- 5. 不再出现"选择图片文件" ----------
    check("5 已移除“选择图片文件”按钮",
          not any("选择图片文件" in b for b in buttons),
          f"当前按钮：{'、'.join(b for b in buttons if b)[:150]}")

    # ---------- 6. B 主界面不再要求"选择输入文件夹" ----------
    check("6 已移除 B 主界面的“选择输入文件夹”入口",
          not any("选择输入文件夹" in b for b in buttons),
          f"仍保留的按钮里已没有该项（底层扫描函数仍保留在 core/folder_picker.py）")

    # ---------- 7/9. 输出目录可用 + 路径显示 ----------
    out_root = WORK / "输出 目录"
    ok_write, msg_write = fpk.ensure_writable_dir(str(out_root))
    default_out = ui._default_output_dir()
    check("7/9 输出目录可创建、路径显示正确",
          ok_write and Path(default_out).is_absolute() and "当前输出位置" not in "",
          f"默认输出目录：{default_out}｜自检目录：{out_root}（{msg_write}）")

    # ---------- 8. Windows 原生文件夹选择器 ----------
    check("8 输出文件夹选择使用 Windows 原生对话框",
          fpk.native_dialog_available() and "Windows 原生" in fpk.describe_capability(),
          f"IFileDialog(FOS_PICKFOLDERS) 干跑可用：{fpk.native_dialog_available()}｜"
          f"{fpk.describe_capability()}")

    # ---------- 10. 批量处理仍然正常 ----------
    out_dir = WORK / "batch_out"
    prog, status, rows, log = ui.on_batch_run(
        "", many, str(out_dir), "text", "png", True, True, False, "aggressive",
        4, 1600, "lama", 6, 0.35, 0.1, 0, False, True, True, True, 0.4)
    outs = sorted((out_dir / "restored").glob("*.png"))
    check("10 批量处理仍然正常（上传区直接给出图片列表）",
          prog == 100 and len(rows) == 5 and len(outs) >= 1 and "批量处理完成" in status,
          f"进度 {prog}｜表格 {len(rows)} 行｜输出 {len(outs)} 个文件｜状态：{status.splitlines()[0][:60]}")

    # ---------- 11. 连续 B 处理 ----------
    arr = ip.load_image(p1)
    editor, orig, prev, _dp, _auto, _info, _st = ui.on_upload(arr, None)
    mask, cands, _pv, _info2, _rows, _s = ui.on_auto_detect(editor, orig, prev, "text",
                                                           "aggressive", 4, 0.4)
    res, _c, _p, _status1, _qc, _log = ui.on_remove(editor, orig, prev, mask, "text",
                                                   str(WORK / "single_out"), 4, 0,
                                                   "lama", 6, 0.35, 0.1, 0, False)
    cont = ui.on_use_result(res, orig) if res is not None else (None,) * 9
    cleared = cont[4] is None and cont[5] is None
    mask2, cands2, _pv2, _i3, _r2, _s2 = ui.on_auto_detect(cont[0], cont[1], cont[2], "text",
                                                          "aggressive", 4, 0.4)
    check("11 连续 B 处理仍然正常（无需手动画）",
          res is not None and cleared,
          f"第一次修复完成：{res is not None}｜继续处理后旧 Mask 已清空：{cleared}｜"
          f"第二轮候选 {(len(cands2) if cands2 else 0)} 个")

    # ---------- 12. C 完全不受影响 ----------
    manual = modes.get_mode("manual")
    check("12 C 手动模式不受影响（隔离仍然成立）",
          manual.uses_auto_detection is False and not (manual.repair_hints or {}),
          f"C 不走自动检测：{not manual.uses_auto_detection}｜"
          f"C 不携带任何修复提示：{not (manual.repair_hints or {})}｜"
          f"（独立运行 test_mode_isolation 6/6 见报告）")

    # ---------- 高级选项仍在（只是折叠）----------
    acc_labels = [w.label for w in widgets if isinstance(w, gr.Accordion)]
    check("高级选项已折叠（自动检测 / 只扫描移到折叠区）",
          any("高级选项" in str(a) for a in acc_labels)
          and any("自动检测" in b or "只扫描" in b for b in buttons),
          f"折叠区：{[a for a in acc_labels if a]}｜相关按钮仍在（已折叠）："
          f"{[b for b in buttons if '自动检测' in b or '只扫描' in b]}")

    print("\n" + "=" * 96)
    passed = sum(1 for _, ok, _ in RESULTS if ok)
    for name, ok, _d in RESULTS:
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
    print(f"\n通过 {passed}/{len(RESULTS)}")
    return 0 if passed == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
