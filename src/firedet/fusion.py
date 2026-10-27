"""Fuse vision detections and sensor readings into a single risk level.

False alarms are the main failure mode of camera-only fire detection
(sunsets, orange lights, reflections). Two safeguards:
  1. Temporal persistence — vision confidence is averaged over a sliding
     window, so a single-frame false positive can't trigger an alarm.
  2. Sensor corroboration — temperature, smoke concentration and the IR
     flame sensor raise or confirm the vision score.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import IntEnum
from typing import List, Optional

from .detection import Detection
from .sensors import SensorReading


class Level(IntEnum):
    NORMAL = 0
    WARNING = 1
    ALARM = 2


@dataclass
class FusionConfig:
    window: int = 15                 # frames (~0.5 s at 30 fps)
    warning_score: float = 0.35
    alarm_score: float = 0.6
    temp_warn_c: float = 45.0
    temp_alarm_c: float = 57.0       # typical fixed-temperature heat-detector rating
    smoke_warn_ppm: float = 300.0
    smoke_alarm_ppm: float = 600.0
    vision_weight: float = 0.6
    sensor_weight: float = 0.4


@dataclass
class Assessment:
    level: Level
    score: float
    vision_score: float
    sensor_score: float
    reasons: List[str]


class RiskEngine:
    def __init__(self, cfg: FusionConfig | None = None):
        self.cfg = cfg or FusionConfig()
        self._history: deque = deque(maxlen=self.cfg.window)

    def _vision_frame_score(self, detections: List[Detection]) -> float:
        fire = [d.confidence for d in detections if d.label == "fire"]
        smoke = [d.confidence for d in detections if d.label == "smoke"]
        return max(max(fire, default=0.0), 0.8 * max(smoke, default=0.0))

    @staticmethod
    def _ramp(x: Optional[float], lo: float, hi: float) -> float:
        if x is None:
            return 0.0
        return min(1.0, max(0.0, (x - lo) / (hi - lo) * 0.5 + 0.5)) if x >= lo else 0.0

    def _sensor_score(self, r: Optional[SensorReading], reasons: List[str]) -> Optional[float]:
        if r is None:
            return None
        c = self.cfg
        t = self._ramp(r.temp_c, c.temp_warn_c, c.temp_alarm_c)
        s = self._ramp(r.smoke_ppm, c.smoke_warn_ppm, c.smoke_alarm_ppm)
        f = 1.0 if r.flame else 0.0
        if t:
            reasons.append(f"temperature {r.temp_c:.1f}°C")
        if s:
            reasons.append(f"smoke {r.smoke_ppm:.0f} ppm")
        if f:
            reasons.append("IR flame sensor triggered")
        return max(t, s, f)

    def update(self, detections: List[Detection], reading: Optional[SensorReading] = None) -> Assessment:
        self._history.append(self._vision_frame_score(detections))
        # Persistence: mean over a full window (missing frames count as 0).
        vision = sum(self._history) / self.cfg.window
        reasons: List[str] = []
        if vision > 0:
            reasons.append(f"vision {vision:.2f} over {len(self._history)} frames")
        sensor = self._sensor_score(reading, reasons)

        if sensor is None:  # camera-only mode
            score = vision
        else:
            score = self.cfg.vision_weight * vision + self.cfg.sensor_weight * sensor
            # Strong independent evidence from either modality escalates directly.
            score = max(score, sensor * 0.9 if sensor >= 1.0 else 0.0, vision if vision >= 0.75 else 0.0)

        level = Level.NORMAL
        if score >= self.cfg.alarm_score:
            level = Level.ALARM
        elif score >= self.cfg.warning_score:
            level = Level.WARNING
        return Assessment(level, round(score, 3), round(vision, 3),
                          round(sensor or 0.0, 3), reasons)

    def reset(self) -> None:
        self._history.clear()
