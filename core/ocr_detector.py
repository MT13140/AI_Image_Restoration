"""OCR 文字检测模块（PaddleOCR 的 PP-OCR 模型 + ONNXRuntime 推理）。

为什么默认用 RapidOCR：
* RapidOCR 直接复用 PaddleOCR 的 PP-OCRv4 检测/识别/方向分类模型，并转换为 ONNX；
* 模型文件随 wheel 一起分发，Windows 下无需额外编译、无需联网下载；
* 相比完整 PaddlePaddle 安装，兼容性与安装成功率在 Windows 上更高。

如果环境中安装了 PaddleOCR（可选），可以设置环境变量
``AI_RESTORE_OCR=paddle`` 切换到 PaddleOCR 后端。
"""

from __future__ import annotations

import os
import re
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from . import mask_processor as mp

# --------------------------------------------------------------------------
# 文本规则
# --------------------------------------------------------------------------
RE_URL = re.compile(
    r"(https?://|www\.)[^\s]+"
    r"|[\w\-]{2,}\.(?:com|cn|net|org|io|top|xyz|vip|cc|me|info|biz|tv|shop|site|online|app|art|club|link|live|fun|pro|tech|store|wang|xin|ren|group|work|space|website|press)(?:/[^\s]*)?",
    re.IGNORECASE,
)
RE_DATE = re.compile(
    r"\d{2,4}\s*[-/.年]\s*\d{1,2}\s*[-/.月]\s*\d{1,2}\s*日?"
    r"|\d{1,2}\s*[-/.]\s*\d{1,2}\s*[-/.]\s*\d{2,4}"
)
RE_TIME = re.compile(r"\b\d{1,2}\s*[:：]\s*\d{2}(?:\s*[:：]\s*\d{2})?(?:\s*(?:AM|PM|am|pm))?\b")
RE_DIGITS = re.compile(r"^[\d\s\-–—:：/.,+()#*·]+$")
RE_AT = re.compile(r"^@[\w\-.]+$")
RE_COPYRIGHT = re.compile(r"(©|\(c\)|版权|版权所有|copyright|all rights reserved|盗图必究|禁止转载)", re.IGNORECASE)
RE_PLATFORM = re.compile(
    r"(抖音|快手|小红书|微博|微信|公众号|知乎|B站|bilibili|taobao|天猫|京东|拼多多|"
    r"watermark|shutterstock|gettyimages|dreamstime|视觉中国|东方IC|昵图网|千图网|花瓣)",
    re.IGNORECASE,
)

#: 文字类型 -> 默认"是否更像水印"的先验权重
TYPE_WATERMARK_PRIOR = {
    "url": 0.95,
    "copyright": 0.95,
    "platform": 0.9,
    "at": 0.75,
    "date": 0.8,
    "time": 0.8,
    "digits": 0.7,
    "text": 0.45,
}


@dataclass
class TextItem:
    """一条 OCR 结果。"""

    box: np.ndarray                 # (4, 2) float32 四角坐标
    text: str
    score: float
    kind: str = "text"              # url/date/time/digits/at/copyright/platform/text
    watermark_score: float = 0.0    # 0~1，越大越像"叠加水印"
    reasons: List[str] = field(default_factory=list)

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        xs = self.box[:, 0]
        ys = self.box[:, 1]
        x0, y0 = int(np.floor(xs.min())), int(np.floor(ys.min()))
        x1, y1 = int(np.ceil(xs.max())), int(np.ceil(ys.max()))
        return x0, y0, max(1, x1 - x0), max(1, y1 - y0)

    @property
    def height(self) -> int:
        return self.bbox[3]

    @property
    def area(self) -> int:
        return self.bbox[2] * self.bbox[3]

    def to_dict(self) -> Dict[str, object]:
        x, y, w, h = self.bbox
        return {
            "text": self.text,
            "score": round(float(self.score), 4),
            "kind": self.kind,
            "watermark_score": round(float(self.watermark_score), 4),
            "bbox": [x, y, w, h],
            "center": [x + w / 2.0, y + h / 2.0],
            "reasons": list(self.reasons),
        }


def classify_text(text: str) -> Tuple[str, float, List[str]]:
    """根据文本内容判断类型，并给出"像水印"的置信度。"""
    t = (text or "").strip()
    reasons: List[str] = []
    kind = "text"

    if RE_URL.search(t):
        kind = "url"
        reasons.append("包含网址/域名")
    elif RE_COPYRIGHT.search(t):
        kind = "copyright"
        reasons.append("包含版权标记")
    elif RE_PLATFORM.search(t):
        kind = "platform"
        reasons.append("包含平台水印关键词")
    elif RE_AT.search(t):
        kind = "at"
        reasons.append("包含 @ 账号名")
    elif RE_DATE.search(t):
        kind = "date"
        reasons.append("符合日期格式")
    elif RE_TIME.search(t):
        kind = "time"
        reasons.append("符合时间格式")
    elif RE_DIGITS.match(t) and len(re.sub(r"\s", "", t)) >= 3:
        kind = "digits"
        reasons.append("纯数字/符号")

    score = TYPE_WATERMARK_PRIOR.get(kind, 0.45)
    if len(t) <= 2 and kind == "text":
        score -= 0.15
        reasons.append("文字过短，更可能是画面内容")
    if len(t) >= 30 and kind == "text":
        score -= 0.05
    return kind, float(np.clip(score, 0.0, 1.0)), reasons


# --------------------------------------------------------------------------
# OCR 引擎
# --------------------------------------------------------------------------
class OCRDetector:
    """OCR 检测器（懒加载，线程安全）。"""

    def __init__(self, backend: Optional[str] = None, min_score: float = 0.5):
        self.backend = (backend or os.environ.get("AI_RESTORE_OCR") or "rapidocr").lower()
        self.min_score = float(min_score)
        self._engine = None
        self._lock = threading.Lock()
        self._load_error: Optional[str] = None
        self.load_seconds: float = 0.0
        self.last_infer_seconds: float = 0.0

    # ---------------- 加载 ----------------
    @property
    def available(self) -> bool:
        return self._engine is not None

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    def ensure_loaded(self) -> bool:
        """加载模型，返回是否成功。"""
        if self._engine is not None:
            return True
        with self._lock:
            if self._engine is not None:
                return True
            import time

            t0 = time.time()
            try:
                if self.backend == "paddle":
                    self._engine = self._load_paddle()
                    self.backend = "paddle"
                else:
                    self._engine = self._load_rapidocr()
                    self.backend = "rapidocr"
                self._load_error = None
            except Exception as exc:  # pragma: no cover - 环境相关
                self._load_error = f"{exc.__class__.__name__}: {exc}"
                self._engine = None
            self.load_seconds = time.time() - t0
        return self._engine is not None

    def _load_rapidocr(self):
        from rapidocr_onnxruntime import RapidOCR

        return RapidOCR()

    def _load_paddle(self):
        from paddleocr import PaddleOCR  # type: ignore

        return PaddleOCR(use_angle_cls=True, lang="ch", show_log=False)

    # ---------------- 推理 ----------------
    def detect(self, image_rgb: np.ndarray, min_score: Optional[float] = None) -> List[TextItem]:
        """对 RGB/灰度图做文字检测，返回结构化结果。"""
        if not self.ensure_loaded():
            raise RuntimeError(f"OCR 引擎不可用：{self._load_error}")

        img = mp.to_uint8(image_rgb)
        if img.ndim == 2:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        else:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        import time

        t0 = time.time()
        if self.backend == "paddle":
            raw = self._engine.ocr(img_bgr, cls=True)
        else:
            raw = self._engine(img_bgr)
        self.last_infer_seconds = time.time() - t0

        items = self._parse(raw)
        thr = self.min_score if min_score is None else float(min_score)
        items = [it for it in items if it.score >= thr and it.text.strip()]
        for it in items:
            it.kind, it.watermark_score, it.reasons = classify_text(it.text)
        items.sort(key=lambda i: -i.watermark_score)
        return items

    @staticmethod
    def _to_box(box) -> Optional[np.ndarray]:
        arr = np.asarray(box, dtype=np.float32).reshape(-1, 2)
        if arr.shape[0] < 4:
            return None
        return arr[:4]

    def _parse(self, raw) -> List[TextItem]:
        """兼容 RapidOCR / PaddleOCR 的不同返回结构。"""
        items: List[TextItem] = []
        if raw is None:
            return items

        # RapidOCR: (result, elapse) ；PaddleOCR: [[ [box, (text, score)], ... ]]
        payload = raw
        if isinstance(raw, tuple) and len(raw) == 2 and isinstance(raw[0], (list, type(None))):
            payload = raw[0]

        def push(box, text, score):
            b = self._to_box(box)
            if b is None:
                return
            try:
                s = float(score)
            except Exception:
                s = 0.0
            items.append(TextItem(box=b, text=str(text).strip(), score=s))

        if payload is None:
            return items

        for entry in payload:
            if entry is None:
                continue
            # 兼容 paddle 的页面级嵌套
            if isinstance(entry, (list, tuple)) and len(entry) > 0 and isinstance(entry[0], (list, tuple, np.ndarray)):
                first = entry[0]
                if isinstance(first, (list, tuple)) and len(first) == 2 and not isinstance(first[1], (list, tuple)) \
                        and not isinstance(first[0], (list, tuple, np.ndarray)):
                    push(entry[0], entry[1], 1.0 if len(entry) < 3 else entry[2])
                    continue
            if isinstance(entry, dict):
                push(entry.get("box") or entry.get("points"), entry.get("text", ""), entry.get("score", 0.0))
                continue
            if isinstance(entry, (list, tuple)):
                if len(entry) >= 3:
                    push(entry[0], entry[1], entry[2])
                elif len(entry) == 2:
                    second = entry[1]
                    if isinstance(second, (list, tuple)) and len(second) == 2:
                        push(entry[0], second[0], second[1])
                    else:
                        push(entry[0], second, 1.0)
                else:
                    for sub in entry:
                        if isinstance(sub, (list, tuple)) and len(sub) >= 2:
                            box = sub[0]
                            info = sub[1] if len(sub) > 1 else None
                            if isinstance(info, (list, tuple)) and len(info) == 2:
                                push(box, info[0], info[1])
        return items

    # ---------------- Mask 生成 ----------------
    def build_mask(
        self,
        shape: Tuple[int, int],
        items: Sequence[TextItem],
        dilate: int = 4,
        min_score: Optional[float] = None,
        kinds: Optional[Sequence[str]] = None,
    ) -> np.ndarray:
        """根据 OCR 结果生成 Mask。"""
        thr = self.min_score if min_score is None else float(min_score)
        mask = np.zeros((int(shape[0]), int(shape[1])), np.uint8)
        for it in items:
            if it.score < thr:
                continue
            if kinds is not None and it.kind not in set(kinds):
                continue
            poly = np.round(it.box).astype(np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(mask, [poly], 255)
        return mp.dilate_mask(mask, dilate)


_DEFAULT: Optional[OCRDetector] = None


def get_default_detector() -> OCRDetector:
    """进程内共享的 OCR 检测器（避免重复加载模型）。"""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = OCRDetector()
    return _DEFAULT
