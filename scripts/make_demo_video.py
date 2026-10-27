"""Render a synthetic test video: a static orange 'sign' (should NOT alarm) and a
flickering flame that appears after a few seconds (should alarm).

    python scripts/make_demo_video.py --out demo.mp4
    python -m firedet.app --source demo.mp4 --detector color
"""
import argparse

import cv2
import numpy as np


def flame(frame, cx, base_y, t, rng, scale=1.0):
    for layer, (color, w, h) in enumerate([((0, 80, 230), 70, 150), ((0, 160, 255), 48, 110),
                                            ((120, 230, 255), 26, 65)]):
        pts = []
        for k in range(24):
            ang = np.pi * k / 23
            jitter = 1 + 0.18 * np.sin(7 * t + 3 * k + layer) + 0.08 * rng.standard_normal()
            x = cx + scale * w * np.cos(ang)
            y = base_y - scale * h * np.sin(ang) ** 0.7 * jitter * (1.4 if 8 < k < 16 else 1.0)
            pts.append((int(x), int(y)))
        cv2.fillPoly(frame, [np.array(pts, np.int32)], color)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="demo.mp4")
    p.add_argument("--seconds", type=float, default=8)
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--fire-at", type=float, default=3.0)
    a = p.parse_args()
    w, h = 640, 480
    rng = np.random.default_rng(0)
    vw = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (w, h))
    for i in range(int(a.seconds * a.fps)):
        t = i / a.fps
        frame = np.full((h, w, 3), (70, 60, 55), np.uint8)                 # dark room
        cv2.rectangle(frame, (0, 360), (w, h), (40, 45, 50), -1)            # floor
        cv2.rectangle(frame, (470, 80), (590, 140), (0, 140, 255), -1)      # static orange sign
        frame = cv2.add(frame, rng.integers(0, 6, frame.shape, dtype=np.uint8))
        if t >= a.fire_at:
            flame(frame, 220, 370, t, rng, scale=min(1.0, 0.4 + (t - a.fire_at) * 0.3))
        vw.write(frame)
    vw.release()
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
