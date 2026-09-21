"""批量处理（第六轮）。

设计要点（对应任务书第十一 ~ 十四章）：

* **三种模式**：``auto``（A 智能自动识别）/ ``text``（B 文字全部去除）/
  ``manual``（C 手动，批量时不做自动检测）；
* **每张图片独立 OCR / 独立生成 Mask**：绝不把第一张的 Mask 复制给其它图片；
* **OCR 无结果时明确跳过**：状态写"未检测到文字（跳过）"，绝不假装成功；
* **不覆盖原图**：输出名 ``原名_restored.<ext>``，同名自动加 ``_1``、``_2``；
* **单张失败不影响整批**：记录原因后继续下一张；
* **分阶段耗时**：OCR / Mask / LaMa / 后处理 / 总计，逐张记录并汇总。

修复阶段**完全复用** ``core/pipeline.py``（第五轮验证过的 LaMa + 双 Mask +
人脸保护 + ROI 优先 + 高频纹理匹配 + quality_check），本模块只做调度与落盘。
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from config import OUTPUTS_DIR, ensure_dirs, get_logger
from . import image_processor as ip
from . import mask_processor as mp
from . import modes
from .pipeline import RepairOptions, run_repair

logger = get_logger("ai_restore.batch")

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
ProgressCB = Optional[Callable[[int, int, str, str], None]]
LogCB = Optional[Callable[[str], None]]

MODE_LABELS = {
    "auto": "A 智能自动识别",
    "text": "B 文字全部去除",
    "manual": "C 手动选择",
}


@dataclass
class BatchItemResult:
    index: int
    name: str
    path: str
    status: str = "pending"          # ok / failed / skipped / detected
    message: str = ""
    seconds: float = 0.0
    out_path: str = ""
    mask_path: str = ""
    report_path: str = ""
    candidates: int = 0
    selected: int = 0
    ocr_regions: int = 0
    mask_ratio: float = 0.0
    timings: Dict[str, float] = field(default_factory=dict)
    qc: Dict[str, object] = field(default_factory=dict)
    log: List[str] = field(default_factory=list)

    def to_row(self) -> List[object]:
        icon = {"ok": "成功", "failed": "失败", "skipped": "跳过",
                "detected": "仅检测"}.get(self.status, self.status)
        t = self.timings
        timing = ""
        if t:
            timing = (f"OCR {t.get('ocr_ms', 0) / 1000:.2f}s / Mask {t.get('mask_ms', 0) / 1000:.2f}s"
                      f" / LaMa {t.get('lama_ms', 0) / 1000:.2f}s")
        return [self.index + 1, self.name, icon, f"{self.seconds:.1f}s",
                self.ocr_regions, self.selected, f"{self.mask_ratio * 100:.2f}%",
                timing, self.message[:70],
                Path(self.out_path).name if self.out_path else ""]


@dataclass
class BatchSummary:
    items: List[BatchItemResult] = field(default_factory=list)
    output_dir: str = ""
    mode: str = "auto"
    total: int = 0
    ok: int = 0
    failed: int = 0
    skipped: int = 0
    detected_only: int = 0
    seconds: float = 0.0
    detected_total: int = 0
    ocr_total: int = 0
    report_path: str = ""

    @property
    def avg_seconds(self) -> float:
        done = [i.seconds for i in self.items if i.status in ("ok", "detected")]
        return float(sum(done) / len(done)) if done else 0.0

    def avg_timings(self) -> Dict[str, float]:
        items = [i for i in self.items if i.status in ("ok", "detected") and i.timings]
        out: Dict[str, float] = {}
        for key in ("ocr_ms", "mask_refine_ms", "mask_ms", "lama_ms", "post_ms", "total_ms"):
            vals = [float(i.timings.get(key, 0.0)) for i in items]
            if any(v > 0 for v in vals):
                out[key] = float(sum(vals) / len(vals))
        return out

    def text(self) -> str:
        lines = [
            f"**批量处理完成（{MODE_LABELS.get(self.mode, self.mode)}）："
            f"{self.ok} 成功 / {self.skipped} 跳过 / {self.failed} 失败**（共 {self.total} 张）",
            f"- 总耗时：{self.seconds:.1f}s ｜ 平均每张：{self.avg_seconds:.1f}s",
            f"- OCR 检出的文字区域总数：{self.ocr_total} ｜ 完成修复：{self.detected_total} 张",
            f"- 输出目录：`{self.output_dir}`",
        ]
        if self.detected_only:
            lines.insert(1, f"- 仅检测（未修复）：{self.detected_only} 张")
        if self.skipped:
            skipped = [i for i in self.items if i.status == "skipped"]
            lines.append(f"- 跳过 {len(skipped)} 张：" +
                         "、".join(f"{i.name}（{i.message}）" for i in skipped[:6]) +
                         ("…" if len(skipped) > 6 else ""))
        if self.failed:
            lines.append("- **失败明细**：")
            for i in self.items:
                if i.status == "failed":
                    lines.append(f"    - `{i.name}`：{i.message}")
        avg = self.avg_timings()
        if avg:
            labels = (("ocr_ms", "OCR"), ("mask_refine_ms", "Mask 收紧"), ("mask_ms", "Mask 生成"),
                      ("lama_ms", "LaMa"), ("post_ms", "后处理"), ("total_ms", "总计"))
            lines.append("- 平均阶段耗时：" + " ｜ ".join(
                f"{label} {avg[key] / 1000:.2f}s" for key, label in labels if key in avg))
        return "\n".join(lines)


def collect_images(paths: Sequence[str]) -> List[Path]:
    """把用户选择的一组路径展开成图片文件列表（支持传入文件夹）。"""
    out: List[Path] = []
    for raw in paths:
        if raw is None:
            continue
        p = Path(str(raw))
        try:
            if p.is_dir():
                out.extend(sorted(q for q in p.rglob("*") if q.suffix.lower() in IMAGE_SUFFIXES))
            elif p.suffix.lower() in IMAGE_SUFFIXES:
                out.append(p)
        except Exception:
            continue
    seen, uniq = set(), []
    for p in out:
        try:
            key = str(p.resolve()).lower()
        except Exception:
            key = str(p).lower()
        if key not in seen:
            seen.add(key)
            uniq.append(p)
    return uniq


def prepare_output_dirs(output_dir: Optional[str]) -> Dict[str, Path]:
    """创建（并返回）输出目录结构：根目录 + restored / masks / reports。"""
    ensure_dirs()
    root = Path(output_dir) if output_dir else Path(OUTPUTS_DIR)
    root.mkdir(parents=True, exist_ok=True)
    dirs = {
        "root": root,
        "restored": root / "restored",
        "masks": root / "masks",
        "reports": root / "reports",
    }
    for d in (dirs["restored"], dirs["masks"], dirs["reports"]):
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def unique_output_path(directory: Path, stem: str, suffix: str, ext: str,
                       max_tries: int = 999) -> Path:
    """**绝不覆盖已存在的文件**：``name.png`` → ``name_1.png`` → ``name_2.png`` …"""
    directory = Path(directory)
    ext = ext if str(ext).startswith(".") else f".{ext}"
    candidate = directory / f"{stem}{suffix}{ext}"
    if not candidate.exists():
        return candidate
    for i in range(1, int(max_tries) + 1):
        alt = directory / f"{stem}{suffix}_{i}{ext}"
        if not alt.exists():
            return alt
    return directory / f"{stem}{suffix}_{int(time.time())}{ext}"


def _out_suffix(original_path: Path, out_format: str) -> str:
    if out_format == "keep":
        s = original_path.suffix.lower()
        return s if s in IMAGE_SUFFIXES else ".png"
    return ".png"


def detect_for_image(img: np.ndarray, mode: str, detect_kwargs: Dict[str, object]):
    """按模式（走 ``core/modes``）产出 Mask，返回 ``(结果, 耗时毫秒)``。

    **每张图片独立调用**：绝不使用上一张图的 Mask。
    """
    t0 = time.time()
    kwargs = dict(detect_kwargs or {})
    obj = modes.get_mode(mode)
    try:
        req = modes.ModeRequest(
            image=img,
            sensitivity=str(kwargs.get("sensitivity", "aggressive")),
            dilate=int(kwargs.get("dilate", 3) or 3),
            ocr_min_score=float(kwargs.get("ocr_min_score", 0.40) or 0.40),
            protect_faces=bool(kwargs.get("protect_faces", True)),
        )
        res = obj.run(req)
    except Exception as exc:
        logger.exception("自动检测失败：%s", exc)
        res = modes.ModeResult(mode=obj.id, mask=None, ok=False,
                               message=f"自动检测失败：{exc.__class__.__name__}: {exc}",
                               warnings=["自动检测失败，已跳过该图片。"])
    return res, (time.time() - t0) * 1000.0


def process_one(
    path: Path,
    dirs: Dict[str, Path],
    mode: str = "auto",
    out_format: str = "png",
    options: Optional[RepairOptions] = None,
    save_masks: bool = True,
    save_reports: bool = True,
    detect_kwargs: Optional[Dict[str, object]] = None,
    repair: bool = True,
) -> BatchItemResult:
    """处理单张图片：读取 → 独立检测 → Mask → 第五轮修复 → 保存。"""
    item = BatchItemResult(index=0, name=path.name, path=str(path))
    t0 = time.time()
    try:
        img = ip.load_image(path)                       # 全分辨率 RGB
        item.log.append(f"[{MODE_LABELS.get(mode, mode)}] 图片：{path.name}"
                        f"（{img.shape[1]}×{img.shape[0]}）")

        mask = None
        res = None
        if mode in ("auto", "text"):
            res, detect_ms = detect_for_image(img, mode, dict(detect_kwargs or {}))
            item.timings["ocr_ms"] = round(float(detect_ms), 1)
            item.candidates = len(res.candidates)
            item.selected = int(res.details.get("high", 0)) or len(res.candidates)
            item.ocr_regions = int(res.details.get("ocr_regions", len(res.candidates)) or 0)
            item.qc["detect_warnings"] = list(res.warnings)
            item.log.append(f"  OCR 检测：发现 {item.ocr_regions} 个文字区域"
                            f"（候选 {item.candidates} 个，默认选中 {item.selected} 个）")
            mask = res.mask if res.ok else None

        if mask is None or mp.is_empty(mask):
            item.status = "skipped"
            item.message = ("未检测到文字（OCR 文字区域 0 个），跳过修复" if mode == "text"
                            else "未检测到水印候选，跳过修复")
            item.log.append(f"  Result：跳过（{item.message}）")
            item.seconds = time.time() - t0
            item.timings["total_ms"] = round(item.seconds * 1000.0, 1)
            return item

        item.mask_ratio = float(mp.area_ratio(mask))
        item.log.append(f"  Mask：{len(mp.mask_to_boxes(mask, min_area=8))} 个区域，"
                        f"覆盖面积 {item.mask_ratio * 100:.2f}%")

        if not repair:
            item.status = "detected"
            item.message = f"仅检测：{item.ocr_regions} 个文字区域 / Mask {item.mask_ratio * 100:.2f}%"
            if save_masks:
                mask_path = unique_output_path(dirs["masks"], path.stem, "_mask", ".png")
                ip.save_image(mask_path, mask)
                item.mask_path = str(mask_path)
            item.seconds = time.time() - t0
            item.timings["total_ms"] = round(item.seconds * 1000.0, 1)
            item.log.append("  Result：仅检测（未修复）")
            return item

        # 模式层给出的修复建议（例如 B 的 hole_pad）在这里生效：
        # 批处理与单张走**同一套**模式参数，行为一致。
        opt = options or RepairOptions()
        hints = dict(getattr(modes.get_mode(mode), "repair_hints", {}) or {})
        if hints:
            import dataclasses
            opt = dataclasses.replace(opt, **{k: v for k, v in hints.items()
                                              if hasattr(opt, k)})
        out = run_repair(img, mask, options=opt)
        item.timings.update({k: float(v) for k, v in (out.timings or {}).items()})
        item.qc.update({k: v for k, v in (out.qc or {}).items() if k != "warnings"})
        item.qc["warnings"] = (out.qc or {}).get("warnings", [])
        face_mode = getattr(out.plan, "mode", "normal")
        item.log.append(f"  Face Protection：{face_mode}")
        item.log.append(f"  Inpainting：{out.backend_label}"
                        f" ｜ LaMa {(out.timings or {}).get('lama_ms', 0) / 1000:.2f}s"
                        f" ｜ 后处理 {(out.timings or {}).get('post_ms', 0) / 1000:.2f}s")

        ext = _out_suffix(path, out_format)
        out_path = unique_output_path(dirs["restored"], path.stem, "_restored", ext)
        ip.save_image(out_path, out.image)
        item.out_path = str(out_path)
        if save_masks:
            mask_path = unique_output_path(dirs["masks"], path.stem, "_mask", ".png")
            ip.save_image(mask_path, out.mask_used)
            ip.save_image(unique_output_path(dirs["masks"], path.stem, "_user_mask", ".png"),
                          out.user_mask)
            item.mask_path = str(mask_path)
        item.status = "ok"
        item.message = (f"重建 {mp.area_ratio(out.mask_used) * 100:.2f}% 画面"
                        + (f"；质检警告 {len(item.qc.get('warnings', []))} 条"
                           if item.qc.get("warnings") else ""))
        item.seconds = time.time() - t0
        item.timings["total_ms"] = round(item.seconds * 1000.0, 1)
        item.log.append(f"  Result：修复完成 → {Path(out_path).name}")

        if save_reports:
            report_path = unique_output_path(dirs["reports"], path.stem, "_report", ".json")
            payload = {
                "file": str(path),
                "output": str(out_path),
                "mode": mode,
                "resolution": f"{img.shape[1]}x{img.shape[0]}",
                "seconds": round(item.seconds, 3),
                "timings_ms": dict(item.timings),
                "backend": out.backend_label,
                "device": out.device,
                "ocr_regions": item.ocr_regions,
                "candidates": item.candidates,
                "selected": item.selected,
                "mask_ratio_user": round(mp.area_ratio(out.user_mask), 5),
                "mask_ratio_refined": round(mp.area_ratio(out.refined_mask), 5),
                "mask_ratio_rebuilt": round(mp.area_ratio(out.mask_used), 5),
                "face_mode": face_mode,
                "plan": getattr(out.plan, "summary", lambda: "")(),
                "roi": out.used_roi,
                "tiles": out.used_tiles,
                "refine": out.refine_note,
                "steps": out.steps,
                "notes": out.notes,
                "quality_check": {k: v for k, v in (out.qc or {}).items()},
                "detect_summary": getattr(res, "message", ""),
                "detect_warnings": list(getattr(res, "warnings", []) or []),
                "mode_label": modes.mode_label(mode),
            }
            report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                                   encoding="utf-8")
            item.report_path = str(report_path)
        return item
    except Exception as exc:                       # 单张失败不影响整批
        item.status = "failed"
        item.message = f"{exc.__class__.__name__}: {exc}"
        item.seconds = time.time() - t0
        item.timings["total_ms"] = round(item.seconds * 1000.0, 1)
        item.log.append(f"  Result：失败（{item.message}）")
        logger.exception("批量处理失败：%s", path)
        return item


def run_batch(
    files: Sequence[str],
    output_dir: Optional[str] = None,
    out_format: str = "png",
    mode: str = "auto",
    auto_detect: bool = True,
    options: Optional[RepairOptions] = None,
    save_masks: bool = True,
    save_reports: bool = True,
    progress_cb: ProgressCB = None,
    log_cb: LogCB = None,
    max_files: int = 500,
    detect_kwargs: Optional[Dict[str, object]] = None,
    sensitivity: str = "aggressive",
    dilate: int = 3,
    ocr_min_score: float = 0.40,
    repair: bool = True,
) -> BatchSummary:
    """批量修复。返回汇总结果（含每张图片的状态、耗时与失败原因）。"""
    t_start = time.time()
    paths = collect_images(files)[: int(max_files)]
    dirs = prepare_output_dirs(output_dir)
    mode = mode if mode in MODE_LABELS else "auto"
    if mode == "manual":
        auto_detect = False                      # C 模式：不做自动检测
    summary = BatchSummary(output_dir=str(dirs["root"]), mode=mode)
    summary.total = len(paths)
    merged_kwargs: Dict[str, object] = {
        "sensitivity": sensitivity, "dilate": int(dilate),
        "ocr_min_score": float(ocr_min_score),
    }
    merged_kwargs.update(dict(detect_kwargs or {}))

    if not paths:
        summary.seconds = time.time() - t_start
        return summary

    def emit(line: str) -> None:
        logger.info(line)
        if log_cb:
            try:
                log_cb(line)
            except Exception:
                pass

    for i, path in enumerate(paths):
        if progress_cb:
            progress_cb(i, len(paths), path.name, "读取图片")
        item = process_one(
            path, dirs, mode=mode, out_format=out_format, options=options,
            save_masks=save_masks, save_reports=save_reports,
            detect_kwargs=merged_kwargs, repair=bool(repair),
        )
        item.index = i
        summary.items.append(item)
        for line in item.log:
            emit(line)
        t = item.timings
        emit(f"  耗时：总计 {t.get('total_ms', 0) / 1000:.2f}s（OCR {t.get('ocr_ms', 0) / 1000:.2f}s / "
             f"Mask {(t.get('mask_refine_ms', 0) + t.get('mask_ms', 0)) / 1000:.2f}s / "
             f"LaMa {t.get('lama_ms', 0) / 1000:.2f}s / 后处理 {t.get('post_ms', 0) / 1000:.2f}s）")
        if item.status == "ok":
            summary.ok += 1
            summary.detected_total += 1
        elif item.status == "detected":
            summary.detected_only += 1
        elif item.status == "skipped":
            summary.skipped += 1
        else:
            summary.failed += 1
        summary.ocr_total += int(item.ocr_regions)
        if progress_cb:
            progress_cb(i + 1, len(paths), path.name, item.status)

    summary.seconds = time.time() - t_start
    try:
        summary_path = dirs["reports"] / f"batch_summary_{time.strftime('%Y%m%d_%H%M%S')}.json"
        summary_path.write_text(json.dumps({
            "mode": mode, "total": summary.total, "ok": summary.ok,
            "failed": summary.failed, "skipped": summary.skipped,
            "detected_only": summary.detected_only,
            "seconds": round(summary.seconds, 2),
            "avg_seconds": round(summary.avg_seconds, 2),
            "avg_timings_ms": summary.avg_timings(),
            "output_dir": summary.output_dir,
            "items": [asdict(it) for it in summary.items],
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        summary.report_path = str(summary_path)
    except Exception as exc:
        logger.warning("批量汇总报告写入失败：%s", exc)
    return summary


def scan_files(
    files: Sequence[str],
    mode: str = "text",
    detect_kwargs: Optional[Dict[str, object]] = None,
    max_files: int = 60,
    progress_cb: ProgressCB = None,
    sensitivity: str = "aggressive",
    dilate: int = 3,
    ocr_min_score: float = 0.40,
) -> List[List[object]]:
    """**只检测不修复**：逐张扫描，返回"每张图检测到什么"的表格（批量预览用）。"""
    paths = collect_images(files)[: int(max_files)]
    rows: List[List[object]] = []
    merged_kwargs: Dict[str, object] = {
        "sensitivity": sensitivity, "dilate": int(dilate),
        "ocr_min_score": float(ocr_min_score),
    }
    merged_kwargs.update(dict(detect_kwargs or {}))
    for i, path in enumerate(paths):
        if progress_cb:
            progress_cb(i, len(paths), path.name, "检测中")
        size, ocr_n, sel_n, ratio, status, note = "", 0, 0, 0.0, "待处理", ""
        try:
            img = ip.load_image(path)
            size = f"{img.shape[1]}×{img.shape[0]}"
            res, _ms = detect_for_image(img, mode if mode in ("auto", "text") else "text",
                                        merged_kwargs)
            mask = res.mask if res.ok else None
            ocr_n = int(res.details.get("ocr_regions", len(res.candidates)) or 0)
            sel_n = int(res.details.get("high", 0)) or len(res.candidates)
            ratio = float(mp.area_ratio(mask)) if mask is not None else 0.0
            if mask is None or mp.is_empty(mask):
                status, note = "无文字/无候选", "处理时会跳过"
            else:
                status, note = "可处理", f"Mask {ratio * 100:.2f}%"
        except Exception as exc:
            status, note = "读取失败", f"{exc.__class__.__name__}: {exc}"
        rows.append([path.name, size, ocr_n, sel_n, f"{ratio * 100:.2f}%", status, note])
    if progress_cb:
        progress_cb(len(paths), len(paths), "", "完成")
    return rows
