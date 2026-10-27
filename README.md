# Advanced Fire Detection System

Real-time **fire and smoke detection** from a camera feed using **YOLO** and
**OpenCV**, fused with **environmental sensors** (smoke, temperature, IR flame)
on an Arduino to raise fast alerts while suppressing false alarms.

**Skills:** Python · OpenCV · YOLO (Ultralytics) · Computer Vision · Sensor Fusion · Arduino / IoT

---

## Architecture

```
 Camera / RTSP / video ─► Detector ──────────────┐
   • YoloDetector (fire, smoke classes)          │
   • ColorMotionDetector (YCrCb+HSV + flicker)   ├─► RiskEngine ─► NORMAL / WARNING / ALARM
                                                 │    • temporal persistence window
 Arduino (MQ-2, DHT22, IR flame) ─ serial JSON ──┘    • sensor corroboration
                                                              │
                                     AlertManager ◄───────────┘
                                     • console / e-mail (SMTP) / Telegram, snapshot attached
                                     • per-level cooldown, buzzer on ALARM
```

| Module | Purpose |
|---|---|
| `src/firedet/detection.py` | YOLO detector + classical OpenCV flame detector (no weights needed) |
| `src/firedet/sensors.py` | Serial reader for the Arduino node, plus a simulator |
| `src/firedet/fusion.py` | Sliding-window vision score + sensor thresholds → risk level |
| `src/firedet/alerts.py` | Notifications with cooldown and snapshot capture |
| `src/firedet/app.py` | Real-time loop, overlay, optional recording |
| `src/firedet/train.py` | Fine-tune YOLOv8/YOLO11 on a fire/smoke dataset |
| `firmware/fire_sensors/` | Arduino sketch (MQ-2, DHT22, KY-026, buzzer) |

### Reducing false alarms

- A single-frame detection can't trigger anything: vision confidence is averaged over a
  15-frame window.
- The OpenCV detector requires **flicker** (frame-to-frame shape change), so static orange
  objects like signs, lamps and clothing are ignored.
- When sensors are connected, camera-only evidence produces a WARNING; ALARM needs either
  sustained strong vision or sensor corroboration (heat, smoke, IR flame).

## Setup

```bash
git clone https://github.com/<your-username>/fire-detection-system.git
cd fire-detection-system
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Run

```bash
# Try it immediately — no model weights, no hardware:
python scripts/make_demo_video.py --out demo.mp4
python -m firedet.app --source demo.mp4 --detector color --simulate-sensors

# Webcam with a trained YOLO model:
python -m firedet.app --source 0 --weights weights/fire_smoke_yolo.pt

# IP camera + Arduino sensors + alerts:
export FIREDET_SMTP_USER=you@gmail.com FIREDET_SMTP_PASSWORD=<app-password>
python -m firedet.app --source rtsp://<camera>/stream --serial /dev/ttyUSB0 \
       --email security@example.com --location "Warehouse A"
```

Press `q` to quit. `--headless` disables the window, `--save-video out.mp4` records the annotated feed.

Telegram alerts: create a bot with @BotFather, then set `FIREDET_TG_TOKEN` and
`FIREDET_TG_CHAT_ID` and pass `--telegram`.

## Train the YOLO model

1. Download a fire/smoke dataset in YOLO format (see [`data/README.md`](data/README.md)) into `data/fire_smoke/`.
2. Train:

```bash
python -m firedet.train --data configs/fire_smoke_data.yaml --model yolov8n.pt --epochs 100
```

Best weights are copied to `weights/fire_smoke_yolo.pt`. Hue augmentation is kept low on
purpose — colour is a key cue for fire.

## Hardware

| Component | Pin | Notes |
|---|---|---|
| MQ-2 smoke/gas sensor | A0 | Burn in 24–48 h, then send `CALIBRATE` over serial to get R0 |
| KY-026 IR flame sensor | D2 | Active LOW |
| DHT22 temperature | D4 | Adafruit DHT library |
| Active buzzer | D8 | Turned on by the app on ALARM |

Flash `firmware/fire_sensors/fire_sensors.ino` with the Arduino IDE. The node prints
`{"temp_c":24.6,"smoke_ppm":132,"flame":0}` twice a second.

Thresholds live in `configs/system.yaml`.

## Results

Fill in after training/testing:

| Metric | Value |
|---|---|
| YOLO mAP@50 (fire / smoke) | _fill in_ |
| Inference FPS (device) | _fill in_ |
| Time from ignition to ALARM | _fill in_ |
| False alarms over N hours of normal footage | _fill in_ |

## Tests

```bash
pytest      # detector, fusion logic, sensor parsing, alert cooldown
```

## Safety note

This is a research/prototype system and is **not** a certified fire alarm. It should supplement,
never replace, code-compliant smoke and heat detectors.

## License

MIT
