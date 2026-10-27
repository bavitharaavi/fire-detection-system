"""Vision detectors for fire and smoke.

YoloDetector     — deep-learning detector (Ultralytics YOLOv8/YOLO11) trained on fire/smoke.
ColorMotionDetector — classical OpenCV fallback: flame-colour segmentation in
                   YCrCb + HSV combined with temporal flicker. Runs anywhere with no
                   weights; also used as a second opinion to reduce YOLO false positives.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Protocol

import cv2
import numpy as np


@dataclass
class Detection:
    label: str            # "fire" | "smoke"
    confidence: float     # 0..1
    box: tuple            # (x1, y1, x2, y2) pixels
    source: str = "yolo"

    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.box
        return max(0, x2 - x1) * max(0, y2 - y1)


class Detector(Protocol):
    def detect(self, frame: np.ndarray) -> List[Detection]: ...


class YoloDetector:
    """Wraps an Ultralytics YOLO model whose classes include 'fire' and/or 'smoke'."""

    def __init__(self, weights: str = "weights/fire_smoke_yolo.pt", conf: float = 0.35,
                 iou: float = 0.45, imgsz: int = 640, device: Optional[str] = None):
        from ultralytics import YOLO  # heavy import, only when used

        self.model = YOLO(weights)
        self.conf, self.iou, self.imgsz, self.device = conf, iou, imgsz, device
        self.names = {int(k): v.lower() for k, v in self.model.names.items()}

    def detect(self, frame: np.ndarray) -> List[Detection]:
        res = self.model.predict(frame, conf=self.conf, iou=self.iou, imgsz=self.imgsz,
                                 device=self.device, verbose=False)[0]
        out = []
        for box, conf, cls in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy(),
                                  res.boxes.cls.cpu().numpy()):
            label = self.names.get(int(cls), str(int(cls)))
            if label in ("fire", "flame", "smoke"):
                out.append(Detection("fire" if label == "flame" else label, float(conf),
                                     tuple(int(v) for v in box), "yolo"))
        return out


class ColorMotionDetector:
    """Rule-based flame detector.

    A pixel is flame-coloured when (Celik & Demirel, 2009 YCbCr rules):
        Y > Cb,  Cr > Cb,  Y > mean(Y),  Cr > mean(Cr),  |Cb - Cr| >= tau
    intersected with an HSV red-orange-yellow band. Real flames flicker, so
    candidate regions must also show frame-to-frame change; static orange
    objects (signs, clothing) are suppressed after the first few frames.
    """

    def __init__(self, min_area: int = 400, tau: int = 40, flicker_thresh: float = 0.04,
                 require_motion: bool = True):
        self.min_area = min_area
        self.tau = tau
        self.flicker_thresh = flicker_thresh
        self.require_motion = require_motion
        self._prev_mask: Optional[np.ndarray] = None
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def flame_mask(self, frame: np.ndarray) -> np.ndarray:
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb).astype(np.int16)
        y, cr, cb = ycrcb[..., 0], ycrcb[..., 1], ycrcb[..., 2]
        rule = (y > cb) & (cr > cb) & (y > y.mean()) & (cr > cr.mean()) & (np.abs(cb - cr) >= self.tau)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv_rule = cv2.inRange(hsv, (0, 80, 150), (35, 255, 255)) > 0
        mask = (rule & hsv_rule).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel, iterations=2)

    def detect(self, frame: np.ndarray) -> List[Detection]:
        mask = self.flame_mask(frame)
        prev, self._prev_mask = self._prev_mask, mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            flicker = 1.0
            if prev is not None:
                roi_now, roi_prev = mask[y:y + h, x:x + w], prev[y:y + h, x:x + w]
                flicker = float(np.count_nonzero(roi_now != roi_prev)) / max(1, w * h)
                if self.require_motion and flicker < self.flicker_thresh:
                    continue
            fill = area / float(w * h)
            conf = float(np.clip(0.35 + 0.35 * fill + 2.0 * min(flicker, 0.15), 0, 0.95))
            out.append(Detection("fire", conf, (x, y, x + w, y + h), "color"))
        return out


def build_detector(kind: str, **kw) -> Detector:
    if kind == "yolo":
        return YoloDetector(**kw)
    if kind == "color":
        return ColorMotionDetector(**kw)
    raise ValueError(f"unknown detector '{kind}' (use yolo | color)")


def draw(frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
    colors = {"fire": (0, 0, 255), "smoke": (160, 160, 160)}
    for d in detections:
        x1, y1, x2, y2 = d.box
        c = colors.get(d.label, (0, 255, 255))
        cv2.rectangle(frame, (x1, y1), (x2, y2), c, 2)
        text = f"{d.label} {d.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), c, -1)
        cv2.putText(frame, text, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return frame
