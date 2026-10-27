"""Fine-tune a YOLO model on a fire/smoke dataset (YOLO format).

    python -m firedet.train --data configs/fire_smoke_data.yaml --model yolov8n.pt --epochs 100

The best weights are copied to weights/fire_smoke_yolo.pt for the app.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def train(data: str, model: str = "yolov8n.pt", epochs: int = 100, imgsz: int = 640, batch: int = 16,
          device: str | None = None, project: str = "runs", name: str = "fire_smoke"):
    from ultralytics import YOLO

    yolo = YOLO(model)
    yolo.train(
        data=data, epochs=epochs, imgsz=imgsz, batch=batch, device=device, project=project, name=name,
        patience=20, cos_lr=True, close_mosaic=10,
        # Fire is colour-defined: keep hue augmentation small so orange stays orange.
        hsv_h=0.005, hsv_s=0.5, hsv_v=0.4, fliplr=0.5, mosaic=1.0, mixup=0.1,
    )
    metrics = yolo.val(data=data, imgsz=imgsz)
    print(f"mAP50={metrics.box.map50:.4f}  mAP50-95={metrics.box.map:.4f}")

    best = Path(yolo.trainer.save_dir) / "weights" / "best.pt"
    Path("weights").mkdir(exist_ok=True)
    shutil.copy(best, "weights/fire_smoke_yolo.pt")
    print("Saved weights/fire_smoke_yolo.pt")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="configs/fire_smoke_data.yaml")
    p.add_argument("--model", default="yolov8n.pt", help="yolov8n/s/m.pt or yolo11n.pt")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default=None)
    a = p.parse_args()
    train(a.data, a.model, a.epochs, a.imgsz, a.batch, a.device)


if __name__ == "__main__":
    main()
