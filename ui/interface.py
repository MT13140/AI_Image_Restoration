"""Gradio 界面（第六轮·UI 简化版）。

设计原则（对应本轮要求）：

* **默认简单**：第一屏只有「选模式 → 上传图片 → 涂抹/自动检测 → 开始修复 → 看结果」；
* **主按钮唯一**：每个模式只有一个显眼的「开始修复 / 确认修复」；
* **按需展开**：参数、检测明细、处理日志、质量报告全部收进折叠区，默认关闭；
* **模式隔离**：界面只调用 ``core/modes`` 里的模式对象产出 Mask，
  修复统一交给第五轮的 ``core/pipeline.run_repair()``，界面不碰任何修复算法。

几个必须保留的关键点（历史踩过的坑）：

1. ``ImageEditor`` 的值必须是 ``{background, layers, composite}`` 三个键；
2. 编辑器里放的是**等比预览缩略图**，修复用全分辨率原图，Mask 按比例映射回去；
3. C 手动模式只使用"用户涂抹"（``confirmed_mask=None``），
   保证"用户没涂到的地方绝对不会被修改"。
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import gradio as gr
import numpy as np

from config import (
    DEFAULT_FEATHER,
    DEFAULT_MASK_DILATE,
    DEFAULT_MAX_SIDE,
    OUTPUTS_DIR,
    TEMP_DIR,
    detect_device,
    ensure_dirs,
    get_logger,
)
from core import batch_processor as bp
from core import face_protector as fp
from core import folder_picker as fpk
from core import image_processor as ip
from core import inpainting as inp
from core import mask_processor as mp
from core import modes
from core import pipeline as pipe
from core import watermark_candidate as wc
from core.error_handler import AppError, guarded

logger = get_logger("ai_restore.ui")

DEVICE = detect_device()

#: 送入编辑器显示的预览图长边上限（只影响显示，不影响原图与最终输出分辨率）
EDITOR_MAX_SIDE = 1800
#: 编辑器画布最大显示高度（占视口比例），保证整张图完整可见且页面不会无限变长
EDITOR_MAX_VH = 42

NOTICE = (
    "请只处理你本人拍摄、创作，或已获得权利人授权编辑的图片。"
    "马赛克等已彻底破坏像素的区域，AI 只能做推测式重建。"
)


# --------------------------------------------------------------------------
# 通用工具
# --------------------------------------------------------------------------
def _device_badge() -> str:
    if DEVICE.is_gpu:
        return f"AI 加速：CUDA（{DEVICE.gpu_name}）"
    return "AI 加速：CPU（未检测到可用 NVIDIA 显卡，速度会慢一些）"


def _fmt_sec(sec: float) -> str:
    sec = float(sec)
    return f"{sec:.1f} 秒" if sec < 60 else f"{sec / 60:.1f} 分钟"


def _fmt_timings(t: Optional[Dict[str, float]]) -> str:
    if not t:
        return ""
    order = (("ocr_ms", "OCR"), ("mask_refine_ms", "Mask 收紧"), ("mask_ms", "Mask 生成"),
             ("face_ms", "人脸"), ("lama_ms", "LaMa"), ("post_ms", "后处理"),
             ("qc_ms", "质检"), ("total_ms", "总计"))
    parts = [f"{label} {float(t[key]) / 1000:.2f}s" for key, label in order
             if key in t and float(t[key]) > 0]
    return " ｜ ".join(parts)


def make_preview(image: np.ndarray, max_side: int = EDITOR_MAX_SIDE) -> np.ndarray:
    """生成只用于界面显示的缩略图（等比缩放，绝不裁切）。"""
    arr = mp.to_uint8(image)
    h, w = arr.shape[:2]
    longest = max(h, w)
    if longest <= int(max_side):
        return arr.copy()
    scale = float(max_side) / float(longest)
    size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    return cv2.resize(arr, size, interpolation=cv2.INTER_AREA)


def _same_image(a: Optional[np.ndarray], b: Optional[np.ndarray], tol: float = 8.0) -> bool:
    if a is None or b is None:
        return False
    if a.shape[:2] != b.shape[:2]:
        return False
    sa = cv2.resize(mp.to_uint8(a), (32, 32), interpolation=cv2.INTER_AREA).astype(np.int16)
    sb = cv2.resize(mp.to_uint8(b), (32, 32), interpolation=cv2.INTER_AREA).astype(np.int16)
    return float(np.abs(sa - sb).mean()) <= tol


def _editor_background(editor_value) -> Optional[np.ndarray]:
    if not isinstance(editor_value, dict):
        return None
    bg = editor_value.get("background")
    if bg is None:
        return None
    try:
        arr, _ = ip.normalize_image(bg)
        return arr
    except Exception:
        return None


def _resolve_images(editor_value, stored_original, stored_preview):
    """确定"全分辨率原图"与"编辑器预览图"。"""
    bg = _editor_background(editor_value)
    if stored_original is None:
        if bg is None:
            return None, None
        return bg, make_preview(bg)
    ref = stored_preview if stored_preview is not None else stored_original
    same = bg is not None and _same_image(bg, ref)
    if bg is not None and ref is not None and bg.shape[:2] == ref.shape[:2]:
        try:
            delta = float(np.abs(mp.to_uint8(bg).astype(np.int16)
                                 - mp.to_uint8(ref).astype(np.int16)).mean())
            logger.info("编辑器背景与预览差异=%.2f（同一张=%s）→ 使用%s", delta, same,
                        "保存的全分辨率原图" if (same or bg is None) else "编辑器背景作为新原图")
        except Exception:
            pass
    if same or bg is None:
        preview = stored_preview if stored_preview is not None else make_preview(stored_original)
        return stored_original, preview
    return bg, make_preview(bg)


def _mask_from_editor(editor_value, auto_mask, target_shape, dilate: int = 0) -> np.ndarray:
    """编辑器涂抹 + 已确认的自动 Mask → 原分辨率 Mask。

    实际解析逻辑放在 ``core/modes/base.py``（模式层统一维护，界面不再各写一份）。
    """
    manual = modes.manual_mask_from_editor(editor_value, target_shape)
    mask = modes.combine_masks(manual, auto_mask)
    return mp.dilate_mask(mask, int(dilate)) if dilate else mask


def _mask_layer(mask: np.ndarray, color: Tuple[int, int, int] = (255, 59, 48)) -> np.ndarray:
    m = mp.ensure_binary(mask)
    layer = np.zeros((m.shape[0], m.shape[1], 4), np.uint8)
    sel = m > 0
    layer[sel, 0] = color[0]
    layer[sel, 1] = color[1]
    layer[sel, 2] = color[2]
    layer[sel, 3] = 255
    return layer


def _describe_mask(mask: np.ndarray) -> str:
    stats = modes.mask_stats(mask)
    if stats["regions"] == 0:
        return "当前还没有选中区域。"
    return f"已选中 {stats['regions']} 处区域，约占画面 {stats['ratio'] * 100:.2f}%。"


def _mode_hint(mode: str) -> str:
    obj = modes.get_mode(mode)
    extra = {
        "auto": "点「自动检测水印」→ 看 Mask 预览 → 「确认修复」。",
        "text": "点「自动检测文字」→ 看 Mask 预览 → 「确认修复」；也可以直接选文件夹批量处理。",
        "manual": "直接用画笔涂掉要去除的内容，然后点「开始修复」。",
    }.get(obj.id, "")
    return f"**{obj.label}**：{obj.hint}　{extra}"


def _rows_from_report(candidates: Sequence) -> List[List[object]]:
    return [c.to_row() for c in candidates]


def _default_output_dir() -> str:
    return str(OUTPUTS_DIR)


def _status(icon: str, text: str) -> str:
    """主界面状态：短句 + 图标，不堆技术细节。"""
    return f"**{icon} {text}**"


# --------------------------------------------------------------------------
# ① 上传
# --------------------------------------------------------------------------
@guarded(
    fallback=[gr.update(), None, None, None, None, "上传失败，请重试。",
              "⚠️ 上传失败，请重新选择图片。"],
    action="上传图片",
)
def on_upload(uploaded, stored_original):
    """上传后立刻显示完整图片（编辑器里是等比缩略预览，原图分辨率不变）。"""
    original, info = ip.normalize_image(uploaded, max_side=0)
    preview = make_preview(original)
    editor_value = mp.make_editor_value(preview, layers=[])
    info_md = f"已加载：{ip.image_info_text(info)}"
    status = _status("✓", f"图片已加载（{original.shape[1]}×{original.shape[0]}），"
                          f"可以开始选择要去除的内容。")
    return editor_value, original, preview, preview, None, info_md, status


# --------------------------------------------------------------------------
# ② 涂抹状态
# --------------------------------------------------------------------------
@guarded(fallback=[None, "涂抹状态读取失败，请重新上传图片。"], action="选择区域")
def on_editor_change(editor_value, stored_original, stored_preview, auto_mask):
    original, preview = _resolve_images(editor_value, stored_original, stored_preview)
    if original is None or preview is None:
        return None, "等待上传图片…"
    mask_full = _mask_from_editor(editor_value, auto_mask, original.shape[:2])
    if mp.is_empty(mask_full):
        return preview, "还没有选中区域：可以涂抹，或点自动检测。"
    mask_preview = cv2.resize(mask_full, (preview.shape[1], preview.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
    return mp.mask_overlay(preview, mask_preview), _status("✓", _describe_mask(mask_full))


@guarded(fallback=[gr.update(), None, "清空失败，请重试。"], action="清空涂抹")
def on_clear_paint(editor_value, stored_original, stored_preview):
    original, preview = _resolve_images(editor_value, stored_original, stored_preview)
    if original is None or preview is None:
        return gr.update(), None, "请先上传图片。"
    return mp.make_editor_value(preview, layers=[]), preview, _status("✓", "已清空涂抹。")


# --------------------------------------------------------------------------
# ③ 自动检测（A / B 模式，走 core/modes）
# --------------------------------------------------------------------------
@guarded(
    fallback=[None, None, None, "检测失败，请重试。", [], "⚠️ 自动检测失败，请查看日志。"],
    action="自动检测",
)
def on_auto_detect(editor_value, stored_original, stored_preview, mode,
                   sensitivity, dilate, ocr_min):
    """按当前模式检测并生成 Mask 预览（**只出候选，不修复**）。"""
    original, preview = _resolve_images(editor_value, stored_original, stored_preview)
    if original is None or preview is None:
        raise AppError("请先上传图片。")
    obj = modes.get_mode(mode)
    if not obj.uses_auto_detection:
        raise AppError("「手动选择」模式不需要自动检测：请直接在图片上涂抹。")

    req = modes.ModeRequest(image=original, editor_value=editor_value,
                            sensitivity=sensitivity, dilate=max(2, int(dilate)),
                            ocr_min_score=float(ocr_min) if ocr_min else 0.40)
    t0 = time.time()
    res = obj.run(req)
    rows = _rows_from_report(res.candidates)
    lines = [f"### {obj.label}：检测结果", res.message]
    lines += res.lines[1:]
    for w in (res.warnings or []):
        lines.append(f"⚠️ {w}")
    lines.append("**下一步**：确认 Mask 没问题就点「确认修复」；想改 Mask 就点「编辑 Mask」"
                 "或展开「检测详情」修改表格。")
    lines.append(f"（检测用时 {_fmt_sec(time.time() - t0)}）")
    info_md = "\n\n".join(lines)

    if res.mask is None or mp.is_empty(res.mask):
        return None, res.candidates, preview, info_md, rows, _status("…", res.message)
    mask_preview = cv2.resize(res.mask, (preview.shape[1], preview.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
    return (res.mask, res.candidates, mp.mask_overlay(preview, mask_preview), info_md, rows,
            _status("✓", res.message))


def _selected_flags(table) -> List[bool]:
    flags: List[bool] = []
    if table is None:
        return flags
    try:
        rows = table.values.tolist() if hasattr(table, "values") else list(table)
    except Exception:
        return flags
    for row in rows:
        if not row:
            continue
        cell = str(row[0]).strip().lower()
        flags.append(cell in {"选中", "是", "true", "1", "yes", "y", "✓", "✅"})
    return flags


@guarded(fallback=[None, None, "应用失败，请重试。", "⚠️ 应用表格修改失败。"],
         action="应用表格修改")
def on_apply_table(table, candidates, editor_value, stored_original, stored_preview, dilate):
    """按「检测详情」表格里的勾选重建 Mask。"""
    if not candidates:
        raise AppError("还没有检测结果，请先点自动检测。")
    original, preview = _resolve_images(editor_value, stored_original, stored_preview)
    if original is None or preview is None:
        raise AppError("请先上传图片。")
    from core import auto_detect as ad
    flags = _selected_flags(table)
    if len(flags) < len(candidates):
        flags = flags + [False] * (len(candidates) - len(flags))
    mask = ad.mask_from_candidates(original, candidates, flags[:len(candidates)],
                                   dilate=max(2, int(dilate)))
    for i, c in enumerate(candidates):
        c.selected = bool(flags[i]) if i < len(flags) else c.selected
    rows = _rows_from_report(candidates)
    if mp.is_empty(mask):
        return None, preview, "已取消全部候选：当前没有待去除区域。", rows
    mask_preview = cv2.resize(mask, (preview.shape[1], preview.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
    return mask, mp.mask_overlay(preview, mask_preview), _describe_mask(mask), rows


@guarded(fallback=[gr.update(), None, None, "确认失败，请重试。"], action="编辑 Mask")
def on_confirm_detection(editor_value, stored_original, stored_preview, auto_mask):
    """把自动 Mask 放进编辑器（红色图层），用户可以继续用画笔/橡皮擦微调。"""
    original, preview = _resolve_images(editor_value, stored_original, stored_preview)
    if original is None or preview is None:
        raise AppError("请先上传图片。")
    if auto_mask is None or mp.is_empty(auto_mask):
        raise AppError("当前没有检测结果，请先点自动检测。")
    mask_preview = cv2.resize(auto_mask, (preview.shape[1], preview.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
    editor_value = mp.make_editor_value(preview, layers=[_mask_layer(mask_preview)])
    return (editor_value, auto_mask, mp.mask_overlay(preview, mask_preview),
            _status("✓", "Mask 已放入编辑区：可以直接修复，也可以用橡皮擦/画笔调整。"))


@guarded(fallback=[None, None, "尚未执行检测。", "清除失败，请重试。"], action="清除自动 Mask")
def on_clear_detection(editor_value, stored_original, stored_preview):
    original, preview = _resolve_images(editor_value, stored_original, stored_preview)
    if original is None:
        return None, None, "尚未执行检测。", "请先上传图片。"
    return None, preview, "已清除检测结果。", _status("✓", "已清除检测结果。")


# --------------------------------------------------------------------------
# ④ 开始修复（第五轮修复引擎，三模式共用）
# --------------------------------------------------------------------------
@guarded(fallback=[None, None, None, "修复失败，请重试。", "", ""], action="开始修复")
def on_remove(
    editor_value, stored_original, stored_preview, auto_mask, mode, output_dir,
    dilate, max_side, backend, feather, color_match, sharpen, denoise, seamless,
    auto_refine=True, protect_faces=True, hq_local=True, detect_only=False,
):
    """产 Mask（模式层）→ 第五轮修复引擎 → 保存。

    输出 6 项：结果图、对比图、下载文件、**主界面简短状态**、质量报告、处理日志。
    """
    original, preview = _resolve_images(editor_value, stored_original, stored_preview)
    if original is None:
        raise AppError("请先上传图片。")

    obj = modes.get_mode(mode)
    if detect_only:
        return (gr.update(), gr.update(), gr.update(),
                _status("…", "当前勾选了「仅检测不修复」：图片没有被修改。"),
                "", "仅检测模式：未执行修复。")

    req = modes.ModeRequest(
        image=original, editor_value=editor_value,
        confirmed_mask=auto_mask if obj.uses_auto_detection else None,
        sensitivity="aggressive", dilate=max(2, int(dilate)),
    )
    res = obj.run(req)
    if not res.ok or res.mask is None or mp.is_empty(res.mask):
        # 自动模式（A/B）"没有可处理的文字"属于**正常状态**，不是错误：
        # 直接返回友好提示，避免页面出现"修复失败，请重试"这种误导文案。
        # 手动模式（C）保持原行为（抛 AppError，提示去涂抹）——C 的行为不变。
        if obj.uses_auto_detection:
            msg = res.message or "当前没有检测到可继续处理的文字，无需再次修复。"
            return (gr.update(), gr.update(), gr.update(),
                    _status("…", msg), "", "本次未执行修复：" + msg)
        raise AppError(res.message or "没有可修复的区域。")

    out = pipe.run_repair(
        original, res.mask,
        options=pipe.RepairOptions(
            dilate=int(dilate), max_side=int(max_side), backend=backend,
            feather=int(feather), color_match=float(color_match), sharpen=float(sharpen),
            denoise=int(denoise), seamless=bool(seamless),
            auto_refine=bool(auto_refine), protect_faces=bool(protect_faces),
            hq_local=bool(hq_local),
            # 模式层给出的修复建议（B：把送进 LaMa 的孔洞扩开几像素，
            # 否则模型会在"零余量孔洞"里把文字抄回来；写回范围不变）
            edge_allow_px=int(res.repair_hints.get("edge_allow_px", 0) or 0),
            hole_grow=int(res.repair_hints.get("hole_grow", 0) or 0),
        ),
    )
    plan, qc = out.plan, out.qc
    timings = dict(out.timings or {})

    t_save = time.time()
    dirs = bp.prepare_output_dirs(output_dir or _default_output_dir())
    out_path = bp.unique_output_path(dirs["restored"], "result", "_restored", ".png")
    ip.save_image(out_path, out.image)
    compare = ip.side_by_side(original, out.image, labels=("原图", "修复结果"))
    compare_path = bp.unique_output_path(dirs["restored"], "result", "_compare", ".png")
    ip.save_image(compare_path, compare)
    mask_path = bp.unique_output_path(dirs["masks"], "result", "_mask", ".png")
    ip.save_image(mask_path, out.mask_used)
    user_mask_path = bp.unique_output_path(dirs["masks"], "result", "_user_mask", ".png")
    ip.save_image(user_mask_path, out.user_mask)
    ip.save_image(bp.unique_output_path(dirs["masks"], "result", "_refined_mask", ".png"),
                  out.refined_mask)
    try:
        TEMP_DIR.mkdir(parents=True, exist_ok=True)
        download_path = ip.make_temp_path("download")
        ip.save_image(download_path, out.image)
    except Exception as exc:  # noqa: BLE001
        logger.warning("生成下载副本失败：%s", exc)
        download_path = out_path
    timings["save_ms"] = round((time.time() - t_save) * 1000.0, 1)

    # ---- 主界面：一句话状态 ----
    status = _status("✓", f"修复完成（{_fmt_sec(out.seconds)}｜{obj.label}｜"
                          f"{'CUDA' if DEVICE.is_gpu else 'CPU'}）")
    status += (f"\n\n结果已保存：`{out_path.name}`　｜　"
               f"修复范围占画面 {mp.area_ratio(out.mask_used) * 100:.2f}%"
               f"（Mask 外像素保持原样）")
    for n in out.notes:
        status += f"\n\n> ⚠️ {n}"
    if mp.area_ratio(out.mask_used) > 0.20:
        status += ("\n\n> ⚠️ 修复面积较大，LaMa 只能做推测式重建，建议分成几块分次修复。")

    # ---- 折叠区：质量报告 ----
    qc_lines = ["**质量报告**（展开才显示，供核对用）", ""]
    if qc.get("available"):
        qc_lines += [
            f"- Mask 外改动：**{qc['outside_change_ratio'] * 100:.3f}%**（应为 0）",
            f"- 接缝亮度差 ΔL={qc['edge_luma_delta']:+.1f}｜色差 Δa/Δb="
            f"{qc['edge_chroma_delta'][0]:+.1f}/{qc['edge_chroma_delta'][1]:+.1f}",
            f"- 纹理比 {qc['hf_ratio']:.2f}×（1.0 = 与周围一致）｜清晰度比 {qc['sharp_ratio']:.2f}×",
            f"- 水印残留消除 {qc['residue_cut'] * 100:.0f}%",
        ]
        if qc.get("warnings"):
            qc_lines.append("")
            qc_lines.append("> ⚠️ 检测到可能的伪影：")
            qc_lines += [f"> - {w}" for w in qc["warnings"]]
    else:
        qc_lines.append("- 本次未生成质检数据。")
    qc_md = "\n".join(qc_lines)

    # ---- 折叠区：处理日志 ----
    log_lines = [
        "**处理日志**（技术细节，排查问题时看）", "",
        f"- 模式：{obj.label}",
        f"- 待去除区域：占画面 {mp.area_ratio(out.user_mask) * 100:.2f}%",
        f"- 实际重建范围：占画面 {mp.area_ratio(out.mask_used) * 100:.2f}%"
        f"（收缩到 {float(np.count_nonzero(out.mask_used)) / max(1.0, float(np.count_nonzero(out.user_mask))) * 100:.0f}%）",
        f"- Mask 收紧：{out.refine_note or '未启用'}"
        + (f"（置信度 {out.refine_confidence:.2f}）" if out.refine_note else ""),
        f"- 人脸保护：{plan.notes[0] if plan and plan.notes else '未检测到人脸/人脸不在修复范围内'}",
        f"- 修复策略：{plan.summary() if plan else '默认'}",
        f"- 后端：{out.backend_label}｜设备：{out.device}",
        f"- 后处理：{'、'.join(out.steps) or '无'}",
        f"- 耗时统计：{_fmt_timings(timings)}",
        f"- 输出分辨率：{out.image.shape[1]} × {out.image.shape[0]}（与原图一致）",
        f"- 修复结果：`{out_path}`",
        f"- 对比图：`{compare_path}`",
        f"- 实际重建范围：`{mask_path}`",
        f"- 用户涂抹范围：`{user_mask_path}`",
        *[f"- 模式说明：{line}" for line in (res.lines or [])],
    ]
    log_md = "\n".join(log_lines)
    # 说明：曾经尝试"返回缩小到 1600 的展示图以加快网页渲染"，
    # 但那会破坏"回调返回的就是原分辨率结果"这一契约（自动化测试与用户预期都依赖它），
    # 实测收益也很小（瓶颈不在这里），因此**保持返回原分辨率**。
    return out.image, compare, str(download_path), status, qc_md, log_md


@guarded(fallback=[gr.update(), None, None, None, "切换失败，请重试。"], action="用结果继续处理")
def on_use_result(result_image, stored_original):
    """把修复结果设为新的输入图，并**清空上一张图遗留的检测/Mask 状态**。

    这是一个必须的修复：早期版本只替换了图片与编辑器，没有清空 `state_auto`
    （上一张图"已确认的自动 Mask"）。而 A/B 模式一旦看到"已有确认 Mask"就会
    **跳过重新检测**，于是第二次修复实际用的是**上一张图的旧 Mask** ——
    表现就是“点了继续处理但没反应，必须手动涂一下才有反应”。
    """
    if result_image is None:
        raise AppError("还没有修复结果。")
    original, info = ip.normalize_image(result_image, max_side=0)
    preview = make_preview(original)
    return (mp.make_editor_value(preview, layers=[]), original, preview, preview,
            None, None, "尚未执行自动检测。", [],
            _status("✓", "已把修复结果设为新的原图，可以继续处理其它位置。"))


@guarded(fallback=[gr.update()], action="查看对比")
def on_show_compare(compare_image):
    if compare_image is None:
        raise AppError("还没有修复结果。")
    return gr.update(visible=True)


@guarded(fallback=["模型加载失败，请查看日志。"], action="预加载模型")
def on_preload_model(backend):
    engine = inp.get_engine(backend=backend)
    t0 = time.time()
    ok, reason = engine.ensure_loaded()
    if ok:
        return (f"✓ AI 模型已就绪：{engine.backend_label}｜"
                f"设备：{'CUDA' if DEVICE.is_gpu else 'CPU'}｜用时 {_fmt_sec(time.time() - t0)}")
    raise AppError(f"AI 修复模型加载失败：{reason}")


# --------------------------------------------------------------------------
# ⑤ 文件夹选择 / 批处理
# --------------------------------------------------------------------------
@guarded(fallback=["", "⚠️ 打开文件夹选择窗口失败，请重试。"], action="选择输入文件夹")
def on_pick_input_dir(current):
    path, err = fpk.pick_folder_safe("选择【输入】图片文件夹", current)
    if path is None:
        return current or "", ("已取消选择，仍使用原来的输入文件夹。"
                               if not err else f"无法打开文件夹选择窗口：{err}")
    files = bp.collect_images([path])
    # 选择后立刻扫描（大小写不敏感），并把前几个文件名显示出来，
    # 这样用户不用离开页面就能确认"这个文件夹里到底有没有图片"。
    return path, (f"✓ **用户选择的输入文件夹**：`{path}`\n\n"
                  f"- {fpk.summarize_images([str(f) for f in files])}")


@guarded(fallback=[gr.update(), "⚠️ 打开文件选择窗口失败，请重试。"], action="选择图片文件")
def on_pick_input_files(initial_dir):
    """选择图片**文件**（可多选）：这个对话框会直接列出文件夹里的文件名，
    解决"选目录只能看到文件夹、看不到图片"的问题。"""
    paths, err = fpk.pick_files("选择要处理的图片（可多选）", initial_dir)
    if not paths:
        return gr.update(), ("已取消选择，仍使用原来选择的图片。"
                             if not err else f"无法打开文件选择窗口：{err}")
    files = bp.collect_images(paths)
    if not files:
        return gr.update(), "选中的文件中没有支持的图片（支持 JPG / JPEG / PNG / WEBP / BMP / TIF）"
    return (gr.update(value=[str(p) for p in files]),
            f"✓ **用户选择的图片**：{fpk.summarize_images([str(p) for p in files])}")


@guarded(fallback=["", "⚠️ 打开文件夹选择窗口失败，请重试。"], action="选择输出文件夹")
def on_pick_output_dir(current):
    path, err = fpk.pick_folder_safe("选择输出文件夹", current or _default_output_dir())
    if path is None:
        return current or _default_output_dir(), ("已取消选择，仍使用原来的输出文件夹。"
                                                 if not err else f"无法打开文件夹选择窗口：{err}")
    ok, msg = fpk.ensure_writable_dir(path)
    if not ok:
        raise AppError(f"输出文件夹不可用：{msg}")
    # 主界面只显示用户需要知道的两件事：选的是哪个目录、不会覆盖原图。
    # 内部的 restored/ masks/ reports/ 结构放到「高级设置」里说明，不占主界面。
    return path, (f"**当前输出位置**：`{path}`\n\n"
                  f"- 不会覆盖原图，同名自动加编号（`_1`、`_2`…）")


@guarded(fallback=["尚未选择图片。"], action="读取已选图片")
def on_files_change(files):
    """上传区变化时显示"已选择 N 张图片（前几个文件名）"。"""
    paths = []
    for f in (files or []):
        paths.append(f if isinstance(f, str) else getattr(f, "name", str(f)))
    images = bp.collect_images(paths)
    if not images:
        return "尚未选择图片：可以一次选多张，或把多张图片一起拖进上面的上传区。"
    names = [p.name for p in images]
    head = "、".join(names[:5]) + ("…" if len(names) > 5 else "")
    return f"✓ **已选择 {len(images)} 张图片**：{head}"


@guarded(fallback=[[]], action="扫描输入文件夹")
def on_scan_batch(in_dir, files, mode, sensitivity, dilate, ocr_min):
    targets = ([in_dir] if in_dir else []) + [f if isinstance(f, str) else getattr(f, "name", str(f))
                                              for f in (files or [])]
    if not targets:
        raise AppError("请先选择输入文件夹（或选择若干图片）。")
    return bp.scan_files(targets, mode=mode if mode in ("auto", "text") else "text",
                         sensitivity=sensitivity, dilate=max(2, int(dilate)),
                         ocr_min_score=float(ocr_min) if ocr_min else 0.40)


@guarded(fallback=[0, "批量处理失败，请查看日志。", [], ""], action="批量处理")
def on_batch_run(
    in_dir, files, out_dir, mode, out_format, save_masks, save_reports, batch_detect_only,
    sensitivity, dilate, max_side, backend, feather, color_match, sharpen, denoise, seamless,
    auto_refine, protect_faces, hq_local, ocr_min,
    progress=gr.Progress(),
):
    """批量处理：逐张独立检测 → Mask → 第五轮修复 → 保存（单张失败不影响整批）。"""
    targets = ([in_dir] if in_dir else []) + [f if isinstance(f, str) else getattr(f, "name", str(f))
                                              for f in (files or [])]
    if not targets:
        raise AppError("请先选择输入文件夹（或选择若干图片）。")
    paths = bp.collect_images(targets)
    if not paths:
        raise AppError("没有找到图片（支持 JPG / JPEG / PNG / WEBP / BMP / TIF）。")
    if not out_dir:
        raise AppError("请先选择输出文件夹。")
    mode = mode if mode in ("auto", "text") else "text"

    def cb(i: int, n: int, name: str, status: str) -> None:
        try:
            progress(min(0.999, i / max(1, n)), desc=f"{status} {min(i + 1, n)}/{n}：{name}")
        except Exception:
            pass

    summary = bp.run_batch(
        [str(p) for p in paths],
        output_dir=out_dir,
        out_format=out_format,
        mode=mode,
        options=pipe.RepairOptions(
            dilate=int(dilate), max_side=int(max_side), backend=backend,
            feather=int(feather), color_match=float(color_match), sharpen=float(sharpen),
            denoise=int(denoise), seamless=bool(seamless),
            auto_refine=bool(auto_refine), protect_faces=bool(protect_faces),
            hq_local=bool(hq_local),
        ),
        save_masks=bool(save_masks),
        save_reports=bool(save_reports),
        repair=not bool(batch_detect_only),
        progress_cb=cb,
        sensitivity=sensitivity,
        dilate=max(2, int(dilate)),
        ocr_min_score=float(ocr_min) if ocr_min else 0.40,
    )
    rows = [it.to_row() for it in summary.items]
    # 主界面只给一行结论；详细清单（跳过/失败明细、批次报告路径）进「处理日志」。
    short = (f"✓ **批量处理完成**：成功 {summary.ok} ｜ 跳过 {summary.skipped} ｜ "
             f"失败 {summary.failed}（共 {summary.total} 张，总耗时 {summary.seconds:.1f}s，"
             f"平均 {summary.avg_seconds:.1f}s/张）")
    if summary.failed:
        short += f"\n\n⚠️ 有 {summary.failed} 张失败，详见下方「处理日志与批量结果」。"
    detail = summary.text()
    if summary.report_path:
        detail += f"\n- 批次汇总报告：`{summary.report_path}`"
    return 100, short, rows, detail


@guarded(fallback=["无法打开目录，请手动打开。"], action="打开输出文件夹")
def on_open_output_dir(path):
    target = Path(path) if path else Path(OUTPUTS_DIR)
    if not target.exists():
        target.mkdir(parents=True, exist_ok=True)
    try:
        os.startfile(str(target))
        return f"✓ 已在资源管理器中打开：`{target}`"
    except Exception as exc:
        raise AppError(f"无法打开目录 {target}：{exc}")


@guarded(fallback=[gr.update(), gr.update(), gr.update(), gr.update(), "切换失败。"],
         action="切换模式")
def on_mode_change(mode):
    """按模式只显示相关的操作区（A/B/C 各自独立）。"""
    obj = modes.get_mode(mode)
    return (
        gr.update(visible=obj.id == "manual"),
        gr.update(visible=obj.id == "auto"),
        gr.update(visible=obj.id == "text"),
        gr.update(visible=obj.uses_auto_detection),
        _mode_hint(obj.id),
    )


# --------------------------------------------------------------------------
# 界面
# --------------------------------------------------------------------------
def build_interface() -> gr.Blocks:
    ensure_dirs()
    css = f"""
    .gradio-container {{max-width: 1180px !important;}}
    #hero h1 {{margin-bottom: 2px; text-align:center;}}
    #subtitle {{text-align:center; color:#6b7280; margin-top:0;}}
    .badge {{display:inline-block;padding:2px 10px;border-radius:10px;background:#eef4ff;
            color:#1d4ed8;font-weight:600;font-size:12px;}}
    .badge-row {{text-align:center; margin-bottom:6px;}}
    .card {{border:1px solid #ececf1; border-radius:14px; padding:12px 16px 6px 16px;
            background:#fff; margin-bottom:10px;}}
    .card-title {{font-weight:700; font-size:15px; margin:0 0 4px 0;}}
    .muted {{color:#6b7280; font-size:13px;}}
    .mode-hint {{color:#4b5563; font-size:13px; margin:2px 0 8px 0;}}
    #mode-tabs label {{font-weight:600;}}
    /* 编辑器画布：等比完整显示，且不把页面撑得无限长 */
    #mask-editor, #mask-editor .image-container, #mask-editor .wrap {{
        height: auto !important; max-height: none !important;
        min-height: 0 !important; overflow: visible !important;
    }}
    #mask-editor canvas {{
        max-width: 100% !important; max-height: {EDITOR_MAX_VH}vh !important;
        width: auto !important; height: auto !important;
    }}
    """

    with gr.Blocks(title="AI 智能图像修复", theme=gr.themes.Soft(), css=css,
                   analytics_enabled=False) as demo:
        gr.Markdown("# AI 智能图像修复", elem_id="hero")
        gr.Markdown("OCR 文字去除 · AI 局部修复 · 批量处理", elem_id="subtitle")
        gr.Markdown(f"<div class='badge-row'><span class='badge'>{_device_badge()}</span></div>")
        gr.Markdown(f"<div class='muted' style='text-align:center'>{NOTICE}</div>")

        state_original = gr.State(None)
        state_preview = gr.State(None)
        state_auto = gr.State(None)
        state_cands = gr.State(None)
        state_in_dir = gr.State("")
        state_out_dir = gr.State(_default_output_dir())

        # ---------------- ① 选择模式 ----------------
        mode = gr.Radio(choices=modes.mode_choices(), value=modes.DEFAULT_MODE,
                        label="选择处理方式", elem_id="mode-tabs")
        mode_hint = gr.Markdown(_mode_hint(modes.DEFAULT_MODE), elem_classes=["mode-hint"])

        # ---------------- ② 图片工作区（左：涂抹 / 右：结果）----------------
        with gr.Row(elem_classes=["card"]):
            with gr.Column(scale=3, min_width=320):
                gr.Markdown("<div class='card-title'>① 上传图片并涂抹（可选）</div>")
                upload = gr.Image(label="上传图片（点击上传 / 拖入 / Ctrl+V 粘贴）", type="numpy",
                                  sources=("upload", "clipboard"), height=150)
                editor = gr.ImageEditor(
                    label="在图片上涂抹需要去除的区域", type="numpy", elem_id="mask-editor",
                    height=None, sources=(),
                    brush=gr.Brush(default_size=18, colors=["#ff3b30"], color_mode="fixed"),
                    eraser=gr.Eraser(default_size=18), show_download_button=False,
                )
                upload_info = gr.Markdown("上传后图片会自动显示在上面。", elem_classes=["muted"])
            with gr.Column(scale=3, min_width=320):
                gr.Markdown("<div class='card-title'>② 修复结果</div>")
                result_image = gr.Image(label="修复结果（原分辨率）", type="numpy",
                                        interactive=False, height=260)
                with gr.Row():
                    compare_btn = gr.Button("查看对比", size="sm")
                    use_result_btn = gr.Button("继续处理这张结果", size="sm")
                download_file = gr.File(label="保存图片（PNG）", interactive=False)

        compare_image = gr.Image(label="原图 / 修复结果 对比", type="numpy",
                                 interactive=False, visible=False, height=280)

        status_main = gr.Markdown("准备就绪：上传图片 → 选择/涂抹区域 → 开始修复。")

        # ---------------- ③ 各模式的操作区（只显示当前模式）----------------
        with gr.Group(visible=False) as panel_c:
            with gr.Row(elem_classes=["card"]):
                with gr.Column(scale=4):
                    gr.Markdown("**请在图片上涂抹需要去除的区域**，然后点「开始修复」。"
                                "涂错了用编辑器里的橡皮擦，或点「清空涂抹」重来。",
                                elem_classes=["muted"])
                with gr.Column(scale=1, min_width=140):
                    c_clear_btn = gr.Button("清空涂抹", size="sm")
                with gr.Column(scale=2, min_width=170):
                    c_run_btn = gr.Button("开始修复", variant="primary", size="lg")

        with gr.Group(visible=True) as panel_a:
            with gr.Row(elem_classes=["card"]):
                with gr.Column(scale=4):
                    gr.Markdown("**自动找出水印候选** → 看下面的 Mask 预览 → 确认后修复。",
                                elem_classes=["muted"])
                    a_info = gr.Markdown("尚未检测。", elem_classes=["muted"])
                with gr.Column(scale=2, min_width=170):
                    a_run_btn = gr.Button("确认修复", variant="primary", size="lg")
            with gr.Row():
                auto_btn = gr.Button("自动检测水印", variant="secondary")
                edit_mask_btn = gr.Button("编辑 Mask", size="sm")
                clear_detect_btn = gr.Button("清除检测结果", size="sm")

        with gr.Group(visible=False) as panel_b:
            with gr.Row(elem_classes=["card"]):
                with gr.Column(scale=3, min_width=300):
                    gr.Markdown("**① 上传要处理的图片**", elem_classes=["mode-hint"])
                    b_files = gr.Files(
                        label="上传图片（点击上传 / 拖入 / Ctrl+V 粘贴，可一次选多张）",
                        file_count="multiple", file_types=["image"],
                    )
                    files_info = gr.Markdown("尚未选择图片。", elem_classes=["muted"])
                with gr.Column(scale=2, min_width=240):
                    gr.Markdown("**② 输出位置**", elem_classes=["mode-hint"])
                    pick_out_btn = gr.Button("选择输出文件夹", variant="secondary")
                    out_dir_status = gr.Markdown(f"尚未选择，使用默认输出目录：`{_default_output_dir()}`",
                                                 elem_classes=["muted"])
                with gr.Column(scale=2, min_width=220):
                    gr.Markdown("**③ 开始处理**", elem_classes=["mode-hint"])
                    b_batch_btn = gr.Button("开始批量处理", variant="primary", size="lg")
                    batch_progress = gr.Slider(label="处理进度", minimum=0, maximum=100, value=0,
                                               interactive=False)
            b_info = gr.Markdown("尚未检测。", elem_classes=["muted"])
            # ---- 高级选项（默认折叠）：普通用户"上传 → 选输出 → 开始"即可，不需要看这些 ----
            with gr.Accordion("高级选项（一般不需要改）", open=False):
                gr.Markdown(
                    "下面这些是处理细节；默认值就是经过验证的参数。\n\n"
                    f"- 程序会在你选择的输出文件夹里自动创建 `restored/`（修复图）、"
                    f"`masks/`（掩膜）、`reports/`（报告）三个**内部子目录**，"
                    f"这是程序自己的组织结构，不需要你理解或选择。",
                    elem_classes=["muted"])
                with gr.Row():
                    b_run_btn = gr.Button("确认修复（当前单张图片）", variant="secondary")
                    b_detect_btn = gr.Button("先看单张的 Mask（自动检测文字）", variant="secondary")
                    scan_btn = gr.Button("只扫描检测（批量，不修复）", size="sm")
                with gr.Row():
                    batch_save_masks = gr.Checkbox(label="保存 Mask 图", value=True)
                    batch_save_reports = gr.Checkbox(label="保存 JSON 报告", value=True)
                    batch_detect_only = gr.Checkbox(label="仅检测不修复", value=False)
                out_format = gr.Radio(
                    choices=[("输出 PNG（推荐）", "png"), ("保持原格式", "keep")],
                    value="png", label="输出格式",
                )

        # ---------------- ④ Mask 预览（A / B 共用）----------------
        with gr.Group(visible=True) as detect_area:
            with gr.Row(elem_classes=["card"]):
                with gr.Column(scale=3):
                    gr.Markdown("<div class='card-title'>③ Mask 预览（要去除的区域）</div>")
                    detect_preview = gr.Image(label="原图 + 半透明红色 Mask", type="numpy",
                                              interactive=False, height=220)
                with gr.Column(scale=2):
                    detect_info = gr.Markdown("尚未执行自动检测。", elem_classes=["muted"])

        # ---------------- 折叠区：高级设置 ----------------
        with gr.Accordion("高级设置（参数微调 / 模型）", open=False):
            gr.Markdown("普通使用不需要修改这里；默认值就是经过验证的参数。",
                        elem_classes=["muted"])
            with gr.Row():
                sensitivity = gr.Radio(choices=wc.sensitivity_choices(), value="aggressive",
                                       label="自动检测敏感度（A/B 模式）")
            with gr.Row():
                dilate = gr.Slider(label="Mask 扩张（像素）", minimum=0, maximum=64, step=1,
                                   value=DEFAULT_MASK_DILATE)
                max_side = gr.Slider(label="处理分辨率上限（长边，0=不缩放）", minimum=0,
                                     maximum=4096, step=64, value=DEFAULT_MAX_SIDE)
                ocr_min = gr.Slider(label="OCR 置信度阈值", minimum=0.1, maximum=0.95,
                                    step=0.05, value=0.4)
            backend = gr.Radio(
                choices=[("LaMa（AI 深度学习，推荐）", "lama"),
                         ("OpenCV Telea（传统算法，备用）", "opencv")],
                value="lama", label="修复方式",
            )
            with gr.Row():
                feather = gr.Slider(label="羽化宽度", minimum=0, maximum=40, step=1,
                                    value=DEFAULT_FEATHER)
                color_match = gr.Slider(label="接缝光照校正强度", minimum=0.0, maximum=1.0,
                                        step=0.05, value=0.35)
                sharpen = gr.Slider(label="修复区锐化", minimum=0.0, maximum=2.0, step=0.05,
                                    value=0.1)
            with gr.Row():
                denoise = gr.Slider(label="修复区降噪", minimum=0, maximum=20, step=1, value=0)
                seamless = gr.Checkbox(label="无缝融合（稍慢）", value=False)
            with gr.Row():
                auto_refine = gr.Checkbox(label="Mask 自动收紧（推荐）", value=True)
                face_guard = gr.Checkbox(label="人脸保护（推荐）", value=True)
                hq_local = gr.Checkbox(label="高质量局部修复（推荐）", value=True)
            detect_only = gr.Checkbox(label="仅检测不修复（不改图）", value=False)
            preload_btn = gr.Button("预加载 AI 模型", size="sm")
            model_status = gr.Markdown(
                f"运行环境：{'CUDA' if DEVICE.is_gpu else 'CPU'}｜模型：models/big-lama.pt｜"
                f"{fp.describe_capability()}｜{fpk.describe_capability()}",
                elem_classes=["muted"])

        # ---------------- 折叠区：检测详情 ----------------
        with gr.Accordion("检测详情（候选列表 / 文字区域）", open=False):
            detect_table = gr.Dataframe(
                headers=["选中", "类型", "置信度等级", "置信度", "来源", "文字",
                         "位置", "面积", "判定依据"],
                datatype=["str"] * 9, wrap=True, interactive=True,
                label="候选明细（可改「选中」列后点「应用表格修改」）",
            )
            apply_table_btn = gr.Button("应用表格修改", size="sm")

        # ---------------- 折叠区：处理日志 / 批量结果 ----------------
        with gr.Accordion("处理日志与批量结果", open=False):
            log_md = gr.Markdown("尚无日志。", elem_classes=["muted"])
            batch_status = gr.Markdown("尚未开始批量处理。", elem_classes=["muted"])
            open_out_btn = gr.Button("打开输出文件夹", size="sm")
            batch_table = gr.Dataframe(
                headers=["#", "文件名", "状态", "耗时", "OCR文字区域", "选中候选",
                         "Mask占比", "分阶段耗时", "说明", "输出文件"],
                datatype=["number", "str", "str", "str", "number", "number", "str", "str",
                          "str", "str"],
                wrap=True, label="每张图片的结果",
            )

        # ---------------- 折叠区：质量报告 ----------------
        with gr.Accordion("质量报告（接缝 / 纹理 / 残留）", open=False):
            qc_md = gr.Markdown("修复后这里会显示质量数据。", elem_classes=["muted"])

        gr.Markdown("<div class='muted' style='text-align:center'>"
                    "AI 修复：LaMa（big-lama）｜自动检测：PP-OCRv4（RapidOCR）+ 图像分析。"
                    "自动检测结果均为候选，请以人工确认为准；极低对比度 / 纯图形水印请用手动模式。"
                    "</div>")

        # ---------------- 事件绑定 ----------------
        mode.change(on_mode_change, inputs=[mode],
                    outputs=[panel_c, panel_a, panel_b, detect_area, mode_hint])

        upload.change(
            on_upload,
            inputs=[upload, state_original],
            outputs=[editor, state_original, state_preview, detect_preview, state_auto,
                     upload_info, status_main],
        )
        editor.change(
            on_editor_change,
            inputs=[editor, state_original, state_preview, state_auto],
            outputs=[detect_preview, status_main],
        )
        c_clear_btn.click(
            on_clear_paint,
            inputs=[editor, state_original, state_preview],
            outputs=[editor, detect_preview, status_main],
        )
        auto_btn.click(
            on_auto_detect,
            inputs=[editor, state_original, state_preview, mode, sensitivity, dilate, ocr_min],
            outputs=[state_auto, state_cands, detect_preview, detect_info, detect_table, status_main],
        )
        b_detect_btn.click(
            on_auto_detect,
            inputs=[editor, state_original, state_preview, mode, sensitivity, dilate, ocr_min],
            outputs=[state_auto, state_cands, detect_preview, detect_info, detect_table, status_main],
        )
        apply_table_btn.click(
            on_apply_table,
            inputs=[detect_table, state_cands, editor, state_original, state_preview, dilate],
            outputs=[state_auto, detect_preview, status_main, detect_table],
        )
        edit_mask_btn.click(
            on_confirm_detection,
            inputs=[editor, state_original, state_preview, state_auto],
            outputs=[editor, state_auto, detect_preview, status_main],
        )
        clear_detect_btn.click(
            on_clear_detection,
            inputs=[editor, state_original, state_preview],
            outputs=[state_auto, detect_preview, detect_info, status_main],
        )
        for btn in (c_run_btn, a_run_btn, b_run_btn):
            btn.click(
                on_remove,
                inputs=[editor, state_original, state_preview, state_auto, mode, state_out_dir,
                        dilate, max_side, backend, feather, color_match, sharpen, denoise,
                        seamless, auto_refine, face_guard, hq_local, detect_only],
                outputs=[result_image, compare_image, download_file, status_main, qc_md, log_md],
            )
        compare_btn.click(on_show_compare, inputs=[compare_image], outputs=[compare_image])
        use_result_btn.click(
            on_use_result,
            inputs=[result_image, state_original],
            outputs=[editor, state_original, state_preview, detect_preview,
                     state_auto, state_cands, detect_info, detect_table, status_main],
        )
        preload_btn.click(on_preload_model, inputs=[backend], outputs=[model_status])

        # 批量输入只保留"上传区"这一条路径（不再要求先选输入文件夹）。
        # 底层的文件夹扫描函数仍在 core/folder_picker.py 与 core/batch_processor.py 保留，
        # 供将来需要时使用，但不再是 B 主界面的必经步骤。
        b_files.change(on_files_change, inputs=[b_files], outputs=[files_info])
        pick_out_btn.click(on_pick_output_dir, inputs=[state_out_dir],
                           outputs=[state_out_dir, out_dir_status])
        open_out_btn.click(on_open_output_dir, inputs=[state_out_dir], outputs=[batch_status])
        scan_btn.click(
            on_scan_batch,
            inputs=[state_in_dir, b_files, mode, sensitivity, dilate, ocr_min],
            outputs=[batch_table],
        )
        b_batch_btn.click(
            on_batch_run,
            inputs=[state_in_dir, b_files, state_out_dir, mode, out_format,
                    batch_save_masks, batch_save_reports, batch_detect_only,
                    sensitivity, dilate, max_side, backend, feather, color_match, sharpen,
                    denoise, seamless, auto_refine, face_guard, hq_local, ocr_min],
            outputs=[batch_progress, batch_status, batch_table, log_md],
        )

    return demo
