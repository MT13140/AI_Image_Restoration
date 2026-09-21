"""修复流程共享模块（第六轮新增）。

**这里没有新算法**：只是把第五轮已经验证有效的流程（Mask 收紧 → 人脸保护 →
repair_mask/blend_mask 分离 → ROI 优先 → LaMa → 高频匹配融合 → 自动质检）
抽成一个函数，让「单张修复」和「批量修复」走**完全相同**的实现，
避免两套代码产生行为差异。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from config import (
    DEFAULT_FEATHER,
    DEFAULT_MASK_DILATE,
    DEFAULT_MAX_SIDE,
    detect_device,
    get_logger,
)
from . import face_protector as fp
from . import image_processor as ip
from . import inpainting as inp
from . import mask_processor as mp
from . import mask_refiner as mr
from . import watermark_detector as wd

logger = get_logger("ai_restore.pipeline")

_DETECTOR = wd.WatermarkDetector()


@dataclass
class RepairOptions:
    """修复参数（与界面滑条一一对应）。"""

    dilate: int = DEFAULT_MASK_DILATE
    max_side: int = DEFAULT_MAX_SIDE
    backend: str = "lama"
    feather: int = DEFAULT_FEATHER
    color_match: float = 0.35
    sharpen: float = 0.1
    denoise: int = 0
    seamless: bool = False
    auto_refine: bool = True
    protect_faces: bool = True
    hq_local: bool = True
    #: **允许"融合环"超出用户掩膜的像素数**（0=严格不超出，第五轮 C 的默认行为）。
    #: 为什么需要：B 模式的"用户掩膜"就是紧贴文字的自动掩膜，于是
    #: ``clip_to_user(..., allow_px=0)`` 会把计划里的扩张**整段裁掉**，
    #: 结果是"送进 LaMa 的孔洞 = 文字掩膜本身、没有任何余量"——实测这时
    #: LaMa 会顺着孔洞两侧的像素把原文字抄回来（红色水印变成金色块、残留 21%）。
    #: 允许一个**很小的**环（默认 B: dilate+feather ≤10px）之后，残留降到 0.2%。
    #: 注意：核心区(repair_mask)不变，超出的只是羽化融合环，不会扩大到背景/人体。
    edge_allow_px: int = 0
    #: **只扩大"送进模型的孔洞"**（像素，0=关闭）。
    #: 这是比 edge_allow_px 更精确的做法：模型看到更宽的孔洞（不会抄回文字），
    #: 而实际写回/融合范围**完全不变**，因此 Mask 面积一点都不会变大。
    #: B（文字全部去除）使用；A/C 默认 0，行为不变。
    hole_grow: int = 0
    #: **保留检测到的水印实例**（B 用，A/C 默认 False）。
    #: 为什么需要：Mask 收紧（mask_refiner）以"OCR 文字框 + 对比度证据"为依据，
    #: 而重复/平铺水印里由"模板匹配 / 点阵展开"补出来的实例没有对应 OCR 框，
    #: 会被当成噪声内缩甚至整块删除（实测：某个实例的掩膜覆盖率从 100% 掉到 0%，
    #: 于是那一处水印完全没被修 —— 这正是"有些实例成功、有些实例失败"的原因）。
    #: 打开后：某个实例若被收紧掉超过 25%，就**按原样保留该实例**（面积不会超过
    #: 检测阶段给出的掩膜），其余实例仍正常收紧。
    keep_user_instances: bool = False


@dataclass
class RepairOutput:
    """一次修复的全部中间结果（供界面展示、批量保存与报告）。"""

    image: np.ndarray
    mask_used: np.ndarray
    repair_mask: np.ndarray
    blend_mask: np.ndarray
    user_mask: np.ndarray
    refined_mask: np.ndarray
    qc: Dict[str, object] = field(default_factory=dict)
    plan: Optional[fp.RepairPlan] = None
    face_report: Optional[fp.FaceReport] = None
    seconds: float = 0.0
    backend_label: str = ""
    refine_note: str = ""
    refine_confidence: float = 1.0
    notes: List[str] = field(default_factory=list)
    steps: List[str] = field(default_factory=list)
    device: str = ""
    used_roi: bool = False
    used_tiles: bool = False
    #: 分阶段耗时（毫秒）：mask_refine / face / inpaint / lama / post / qc / total
    timings: Dict[str, float] = field(default_factory=dict)
    #: LaMa（或兜底后端）**未经后处理/融合**的原始输出，仅诊断时返回（默认 None）
    raw_image: Optional[np.ndarray] = None


def ocr_boxes_in_mask(original: np.ndarray, user_mask: np.ndarray, max_side: int = 1280) -> List[List[int]]:
    """用 OCR 在"用户涂抹范围"内找文字框（Mask 收紧的最可靠证据）。"""
    try:
        h, w = original.shape[:2]
        scale = min(1.0, float(max_side) / float(max(h, w)))
        small = (cv2.resize(original, (max(1, int(w * scale)), max(1, int(h * scale))),
                            interpolation=cv2.INTER_AREA) if scale < 1 else original)
        items = _DETECTOR.ocr.detector.detect(small, min_score=0.35)
        if not items:
            return []
        small_mask = cv2.resize(user_mask, (small.shape[1], small.shape[0]),
                                interpolation=cv2.INTER_NEAREST) > 0
        inv = 1.0 / scale if scale > 0 else 1.0
        boxes: List[List[int]] = []
        for it in items:
            x, y, bw, bh = it.bbox
            cx, cy = int(x + bw / 2), int(y + bh / 2)
            if 0 <= cy < small_mask.shape[0] and 0 <= cx < small_mask.shape[1] and small_mask[cy, cx]:
                boxes.append([int(round(x * inv)), int(round(y * inv)),
                              int(round(max(2, bw * inv))), int(round(max(2, bh * inv)))])
        return boxes
    except Exception as exc:
        logger.warning("OCR 辅助收紧失败：%s", exc)
        return []


def translucent_evidence(original: np.ndarray, user_mask: np.ndarray) -> Optional[np.ndarray]:
    """在涂抹范围附近做半透明水印检测，作为 Mask 收紧的第二类证据。"""
    try:
        boxes = mp.mask_to_boxes(user_mask, min_area=16)
        if not boxes:
            return None
        x, y, w, h = max(boxes, key=lambda b: b[2] * b[3])
        pad = 24
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(original.shape[1], x + w + pad), min(original.shape[0], y + h + pad)
        crop = np.ascontiguousarray(original[y0:y1, x0:x1])
        res = _DETECTOR.detect_translucent(crop, sensitivity=1.0, dilate=0)
        if not res.found:
            return None
        full = np.zeros(original.shape[:2], np.uint8)
        full[y0:y1, x0:x1] = res.mask
        return full
    except Exception as exc:
        logger.warning("半透明证据检测失败：%s", exc)
        return None


def _hole_pad_px(mask: np.ndarray, min_width: int, max_pad: int = 4) -> int:
    """根据掩膜里"笔画的中位宽度"，算出需要补多少像素才能让孔洞达到 ``min_width``。

    用距离变换估计宽度：掩膜内每个像素到边界的距离 ×2 ≈ 该处的局部宽度。
    取中位数作为"笔画宽度"的代表值（避免被个别粗块带偏）。
    返回 0~``max_pad``：**只在掩膜确实太细时才补，且补的量有上限**。
    """
    m = mp.ensure_binary(mask)
    if mp.is_empty(m):
        return 0
    dist = cv2.distanceTransform((m > 0).astype(np.uint8), cv2.DIST_L2, 3)
    vals = dist[dist > 0]
    if vals.size == 0:
        return 0
    width = 2.0 * float(np.median(vals))
    need = int(np.ceil((float(min_width) - width) / 2.0))
    return int(np.clip(need, 0, int(max_pad)))


def _restore_detected_instances(user: np.ndarray, refined: np.ndarray,
                                keep_ratio: float = 0.75, min_area: int = 12
                                ) -> Tuple[np.ndarray, int]:
    """把"被收紧掉的检测实例"按原样恢复（B 模式专用）。

    逐连通域比较"用户/检测掩膜"与"收紧后的掩膜"：
    若某个实例被保留得不足 ``keep_ratio``，认为它被误删（重复水印里由模板匹配
    补出来的实例没有 OCR 框，最容易被删），于是**按检测到的形状**整块恢复。
    恢复的面积不会超过检测阶段给出的掩膜，因此不是"扩大 Mask"。
    """
    u = mp.ensure_binary(user)
    m = mp.ensure_binary(refined, u.shape[:2])
    if mp.is_empty(u):
        return m, 0
    n, labels, stats, _ = cv2.connectedComponentsWithStats((u > 0).astype(np.uint8), 8)
    out = m.copy()
    restored = 0
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < int(min_area):
            continue
        comp = labels == i
        keep = int(np.count_nonzero(comp & (m > 0)))
        if keep / float(area) < float(keep_ratio):
            out[comp] = 255
            restored += 1
    return out, restored


def run_repair(
    original: np.ndarray,
    user_mask: np.ndarray,
    options: Optional[RepairOptions] = None,
    auto_evidence: bool = True,
    return_raw: bool = False,
) -> RepairOutput:
    """执行第五轮验证过的完整修复流程。

    ``user_mask``：安全边界（用户涂抹 ∪ 自动检测结果）。所有产出都**不会超出**它。
    ``return_raw=True``：额外返回"LaMa 原始输出"（不做后处理/融合），
    仅用于诊断"残留到底来自模型还是来自融合"，不影响正常流程。
    """
    opt = options or RepairOptions()
    t0 = time.time()
    timings: Dict[str, float] = {}
    img = mp.to_uint8(original)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    user = mp.clean_mask(mp.ensure_binary(user_mask, img.shape[:2]), min_area=4, close_px=1)
    if mp.is_empty(user):
        raise ValueError("Mask 为空：请先涂抹水印区域，或先执行自动检测并确认")
    timings["mask_prepare_ms"] = round((time.time() - t0) * 1000.0, 1)

    # ① Mask 自动收紧
    t_refine = time.time()
    refined = user
    refine_note, refine_conf = "", 1.0
    if opt.auto_refine:
        boxes = ocr_boxes_in_mask(img, user) if auto_evidence else []
        evid = translucent_evidence(img, user) if auto_evidence else None
        rr = mr.refine(img, user, ocr_boxes=boxes, evidence_mask=evid)
        refined, refine_note, refine_conf = rr.mask, rr.note, rr.confidence
    if opt.keep_user_instances:
        refined, restored = _restore_detected_instances(user, refined)
        if restored:
            plan_note = (f"保留了 {restored} 个被收紧掉的重复水印实例"
                         f"（模板匹配/点阵展开补出的实例没有 OCR 框，收紧时会误删）")
            refine_note = (refine_note + "；" + plan_note) if refine_note else plan_note
    timings["mask_refine_ms"] = round((time.time() - t_refine) * 1000.0, 1)

    # ② 人脸分析 → 五官硬保护 → 修复计划
    t_face = time.time()
    faces = fp.detect_faces(img) if opt.protect_faces else []
    report = fp.analyze(refined, faces) if faces else fp.FaceReport()
    protect_notes: List[str] = []
    if faces and opt.protect_faces:
        refined, protect_notes, _stats = fp.enforce_protection(refined, faces)
    plan = fp.make_plan(
        report, img.shape[:2], refined,
        base_dilate=int(opt.dilate), base_feather=int(opt.feather),
        base_tone=float(opt.color_match), base_grain=True,
        protect_faces=bool(opt.protect_faces), prefer_local=bool(opt.hq_local),
    )
    plan.notes = protect_notes + plan.notes
    if not opt.hq_local:
        plan.use_roi = False
    timings["face_ms"] = round((time.time() - t_face) * 1000.0, 1)

    # ③ repair_mask / blend_mask（都不允许超出用户涂抹范围）
    repair_mask = mp.dilate_mask(refined, plan.dilate) if plan.dilate else refined
    repair_mask = mr.clip_to_user(repair_mask, user, allow_px=0)
    if mp.is_empty(repair_mask):
        repair_mask = refined
    blend_mask = mp.dilate_mask(repair_mask, int(plan.feather)) if plan.feather else repair_mask
    # B 模式允许"融合环"超出用户掩膜一个小余量（见 RepairOptions.edge_allow_px）；
    # C/A 默认 0 → 与第五轮行为完全一致。
    blend_mask = mr.clip_to_user(blend_mask, user, allow_px=int(opt.edge_allow_px))
    if mp.is_empty(blend_mask):
        blend_mask = repair_mask

    # ④ 修复
    t_inpaint = time.time()
    engine = inp.get_engine(backend=opt.backend)
    sharpen_eff = min(float(opt.sharpen), 0.15) if report.mode != "normal" else float(opt.sharpen)
    outcome = engine.inpaint(
        img, blend_mask, dilate=0, feather=plan.feather, color_match=plan.tone_strength,
        sharpen=sharpen_eff, denoise=int(opt.denoise), seamless=bool(opt.seamless),
        max_side=int(opt.max_side) if opt.max_side else 0,
        grain=plan.grain, grain_strength=plan.grain_strength,
        grain_exclude=plan.face_mask if report.mode != "normal" else None,
        roi=plan.roi if plan.use_roi else None,
        tile_components=plan.tile_mode, repair_mask=repair_mask, fallback=True,
        return_raw=bool(return_raw),
        hole_grow=int(opt.hole_grow),
    )
    timings["inpaint_ms"] = round((time.time() - t_inpaint) * 1000.0, 1)
    inner_timings = (outcome.info or {}).get("timings") or {}
    if isinstance(inner_timings, dict):
        for key in ("infer_ms", "post_ms"):
            if key in inner_timings:
                timings["lama_ms" if key == "infer_ms" else "post_ms"] = float(inner_timings[key])

    # ⑤ 自动质检
    t_qc = time.time()
    qc = ip.quality_check(
        img, outcome.image, outcome.mask_used, core_mask=repair_mask,
        face_mask=plan.face_mask if report.mode != "normal" else None,
    )
    timings["qc_ms"] = round((time.time() - t_qc) * 1000.0, 1)
    timings["total_ms"] = round((time.time() - t0) * 1000.0, 1)

    return RepairOutput(
        image=outcome.image,
        mask_used=outcome.mask_used,
        repair_mask=repair_mask,
        blend_mask=blend_mask,
        user_mask=user,
        refined_mask=refined,
        qc=qc,
        plan=plan,
        face_report=report,
        seconds=time.time() - t0,
        backend_label=outcome.backend_label,
        refine_note=refine_note,
        refine_confidence=refine_conf,
        notes=list(outcome.info.get("notes", [])),
        steps=list(outcome.info.get("post", {}).get("steps", [])),
        device=str(outcome.info.get("device", detect_device().device)),
        used_roi=bool(plan.use_roi),
        used_tiles=bool(plan.tile_mode),
        timings=timings,
        raw_image=outcome.raw_image if return_raw else None,
    )
