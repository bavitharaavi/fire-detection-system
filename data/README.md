# Dataset

Images and labels are not committed. Expected YOLO layout:

```
data/fire_smoke/
├── images/{train,val,test}/*.jpg
└── labels/{train,val,test}/*.txt     # class x_center y_center width height (normalized)
```

Classes: `0 = fire`, `1 = smoke` (see `configs/fire_smoke_data.yaml`).

## Public datasets

- **D-Fire** — ~21k images with fire and smoke boxes, YOLO format.
- **Roboflow Universe** — several "fire and smoke" projects exportable as "YOLOv8".
- **FASDD** (Flame and Smoke Detection Dataset).

Add **hard negatives** (sunsets, orange lights, red vehicles, fog, steam) with empty label files —
this is the most effective way to reduce false alarms.
