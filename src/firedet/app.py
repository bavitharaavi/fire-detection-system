"""Real-time fire & smoke monitoring.

Examples
    python -m firedet.app --source 0                                  # webcam, YOLO
    python -m firedet.app --source demo.mp4 --detector color          # no weights needed
    python -m firedet.app --source rtsp://cam/stream --serial /dev/ttyUSB0 --email ops@example.com
    python -m firedet.app --source demo.mp4 --detector color --simulate-sensors --headless
"""
from __future__ import annotations

import argparse
import logging
import time
from typing import Optional

import cv2
import yaml

from .alerts import AlertManager, ConsoleNotifier, EmailNotifier, TelegramNotifier
from .detection import build_detector, draw
from .fusion import FusionConfig, Level, RiskEngine
from .sensors import SerialSensor, SimulatedSensor

BANNER = {Level.NORMAL: ((0, 160, 0), "NORMAL"), Level.WARNING: ((0, 165, 255), "WARNING"),
          Level.ALARM: ((0, 0, 255), "FIRE ALARM")}


def overlay(frame, assessment, fps: float, reading=None):
    color, text = BANNER[assessment.level]
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 34), color, -1)
    cv2.putText(frame, f"{text}  score={assessment.score:.2f}  {fps:4.1f} fps", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    if reading is not None:
        t = f"{reading.temp_c:.1f}C" if reading.temp_c is not None else "n/a"
        sm = f"{reading.smoke_ppm:.0f}ppm" if reading.smoke_ppm is not None else "n/a"
        s = f"T={t}  smoke={sm}  flame={int(reading.flame)}"
        cv2.putText(frame, s, (10, frame.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    return frame


def run(source, detector_kind="yolo", weights=None, conf=0.35, serial_port: Optional[str] = None,
        simulate_sensors=False, emails=None, telegram=False, headless=False, location="Camera 1",
        cooldown=60.0, fusion_cfg: Optional[FusionConfig] = None, save_video: Optional[str] = None,
        max_frames: Optional[int] = None):
    det_kwargs = {"conf": conf, "weights": weights} if detector_kind == "yolo" else {}
    if detector_kind == "yolo" and weights is None:
        det_kwargs["weights"] = "weights/fire_smoke_yolo.pt"
    detector = build_detector(detector_kind, **det_kwargs)

    cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video source: {source}")
    is_file = not str(source).isdigit() and "://" not in str(source)
    video_clock = (lambda: cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0) if is_file else None

    if serial_port:
        sensor = SerialSensor(serial_port)
    elif simulate_sensors:
        sensor = SimulatedSensor(fire_after_s=3, clock=video_clock)
    else:
        sensor = None
    notifiers = [ConsoleNotifier()]
    if emails:
        notifiers.append(EmailNotifier(emails))
    if telegram:
        notifiers.append(TelegramNotifier())

    engine = RiskEngine(fusion_cfg)
    alerts = AlertManager(notifiers, cooldown_s=cooldown, location=location, sensor=sensor,
                          clock=video_clock or time.time)
    writer = None
    fps_ema, t_prev, n = 0.0, time.time(), 0
    stats = {"frames": 0, "max_level": Level.NORMAL, "alerts": 0}
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            detections = detector.detect(frame)
            reading = sensor.read() if sensor else None
            a = engine.update(detections, reading)
            draw(frame, detections)

            now = time.time()
            fps_ema = 0.9 * fps_ema + 0.1 * (1.0 / max(now - t_prev, 1e-6))
            t_prev = now
            overlay(frame, a, fps_ema, reading)
            alerts.handle(a, frame if a.level > Level.NORMAL else None)

            stats["frames"] += 1
            stats["max_level"] = max(stats["max_level"], a.level)
            if save_video:
                if writer is None:
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(save_video, cv2.VideoWriter_fourcc(*"mp4v"), 25, (w, h))
                writer.write(frame)
            if not headless:
                cv2.imshow("Fire Detection", frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
            n += 1
            if max_frames and n >= max_frames:
                break
    finally:
        cap.release()
        if writer:
            writer.release()
        if sensor:
            sensor.close()
        if not headless:
            cv2.destroyAllWindows()
    stats["alerts"] = alerts.sent_count
    stats["max_level"] = stats["max_level"].name
    return stats


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default="0", help="webcam index, video file or RTSP URL")
    p.add_argument("--detector", choices=["yolo", "color"], default="yolo")
    p.add_argument("--weights", default=None)
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--config", default="configs/system.yaml", help="fusion thresholds")
    p.add_argument("--serial", default=None, help="Arduino port, e.g. /dev/ttyUSB0 or COM3")
    p.add_argument("--simulate-sensors", action="store_true")
    p.add_argument("--email", nargs="*", default=None)
    p.add_argument("--telegram", action="store_true")
    p.add_argument("--location", default="Camera 1")
    p.add_argument("--cooldown", type=float, default=60.0)
    p.add_argument("--save-video", default=None)
    p.add_argument("--headless", action="store_true")
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO)

    fusion = FusionConfig()
    try:
        with open(a.config) as f:
            fusion = FusionConfig(**(yaml.safe_load(f) or {}).get("fusion", {}))
    except FileNotFoundError:
        pass
    stats = run(a.source, a.detector, a.weights, a.conf, a.serial, a.simulate_sensors, a.email,
                a.telegram, a.headless, a.location, a.cooldown, fusion, a.save_video)
    print(stats)


if __name__ == "__main__":
    main()
