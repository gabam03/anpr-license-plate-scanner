#!/usr/bin/env python3
"""
Generate a synthetic test image of a car-like scene with a plate,
optionally skewed, for testing the pipeline without real photos.

  python make_test_image.py --text LND123AB --out samples/synthetic.jpg
  python make_test_image.py --text KJA456CD --skew 12 --out samples/synthetic_skewed.jpg
"""

import argparse

import cv2
import numpy as np


def make_plate(text, w=520, h=120):
    plate = np.full((h, w, 3), 245, np.uint8)
    cv2.rectangle(plate, (4, 4), (w - 5, h - 5), (20, 20, 20), 4)
    label = f"{text[:3]}-{text[3:]}"
    font, scale, thick = cv2.FONT_HERSHEY_DUPLEX, 2.6, 6
    (tw, th), _ = cv2.getTextSize(label, font, scale, thick)
    cv2.putText(plate, label, ((w - tw) // 2, (h + th) // 2), font, scale, (15, 15, 15), thick, cv2.LINE_AA)
    return plate


def make_scene(text, skew=0, seed=1):
    rng = np.random.default_rng(seed)
    scene = rng.integers(70, 110, (720, 1280, 3), dtype=np.uint8)          # road / background
    cv2.rectangle(scene, (300, 180), (980, 640), (60, 45, 120), -1)          # car body
    cv2.rectangle(scene, (380, 220), (900, 380), (40, 40, 40), -1)           # windscreen
    plate = make_plate(text)
    ph, pw = plate.shape[:2]
    cx, cy = 640, 520
    src = np.float32([[0, 0], [pw, 0], [pw, ph], [0, ph]])
    dx, dy = pw * 0.6, ph * 0.6
    sk = skew
    dst = np.float32([[cx - dx, cy - dy + sk], [cx + dx, cy - dy - sk], [cx + dx, cy + dy - sk], [cx - dx, cy + dy + sk]])
    m = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(plate, m, (scene.shape[1], scene.shape[0]))
    mask = cv2.warpPerspective(np.full((ph, pw), 255, np.uint8), m, (scene.shape[1], scene.shape[0]))
    scene[mask > 0] = warped[mask > 0]
    noise = rng.normal(0, 4, scene.shape)
    return np.clip(scene + noise, 0, 255).astype(np.uint8)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--text", default="LND123AB", help="8-character plate, e.g. LND123AB")
    p.add_argument("--skew", type=int, default=0, help="perspective skew in pixels")
    p.add_argument("--out", default="samples/synthetic.jpg")
    a = p.parse_args()
    cv2.imwrite(a.out, make_scene(a.text, a.skew))
    print(f"Saved {a.out}")
