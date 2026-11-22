"""
Licence plate detection and reading.

Pipeline:
  1. Resize and convert to greyscale
  2. Bilateral filter: removes noise but keeps edges sharp
  3. Canny edge detection
  4. Find contours, keep 4-sided shapes with a plate-like aspect ratio
  5. Perspective-correct ("warp") each candidate to a flat rectangle
  6. Threshold and OCR with Tesseract, restricted to A-Z and 0-9
  7. Clean up the text using the Nigerian plate format
"""

from dataclasses import dataclass

import cv2
import numpy as np
import pytesseract

from .plate_format import normalise

PLATE_ASPECT_MIN = 2.0   # width / height
PLATE_ASPECT_MAX = 6.0
MIN_AREA_FRACTION = 0.002
TESS_CONFIG = "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


@dataclass
class PlateResult:
    text: str
    valid: bool
    raw_ocr: str
    box: np.ndarray  # 4 corner points in original image coordinates


def order_corners(pts):
    """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
    pts = pts.reshape(4, 2).astype("float32")
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]], dtype="float32")


def warp_plate(image, corners, out_w=520, out_h=120):
    """Flatten a skewed plate into a straight rectangle."""
    dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype="float32")
    m = cv2.getPerspectiveTransform(order_corners(corners), dst)
    return cv2.warpPerspective(image, m, (out_w, out_h))


def find_candidates(image, max_candidates=5):
    """Return up to max_candidates 4-point contours that look like plates, largest first."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 11, 17, 17)
    edges = cv2.Canny(blur, 30, 200)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:40]

    min_area = MIN_AREA_FRACTION * image.shape[0] * image.shape[1]
    found = []
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        if len(approx) != 4:
            continue
        _, _, w, h = cv2.boundingRect(approx)
        if h == 0 or not (PLATE_ASPECT_MIN <= w / h <= PLATE_ASPECT_MAX):
            continue
        found.append(approx)
        if len(found) >= max_candidates:
            break
    return found


def read_plate(plate_img):
    """OCR a flattened plate image. Returns raw OCR text."""
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    gray = gray[int(h * 0.08):int(h * 0.92), int(w * 0.03):int(w * 0.97)]  # trim the border
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return pytesseract.image_to_string(thresh, config=TESS_CONFIG).strip()


def detect(image, max_width=1000):
    """
    Detect and read plates in a BGR image.
    Returns a list of PlateResult, valid plates first.
    """
    scale = 1.0
    if image.shape[1] > max_width:
        scale = max_width / image.shape[1]
        work = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        work = image

    results = []
    for corners in find_candidates(work):
        raw = read_plate(warp_plate(work, corners))
        if not raw:
            continue
        text, valid = normalise(raw)
        if any(r.text == text for r in results):
            continue  # the plate's inner and outer border can both be found
        box = (corners.reshape(4, 2) / scale).astype(int)
        results.append(PlateResult(text, valid, raw, box))

    results.sort(key=lambda r: not r.valid)
    return results


def annotate(image, results):
    """Draw boxes and plate text on a copy of the image."""
    out = image.copy()
    for r in results:
        color = (0, 170, 0) if r.valid else (0, 140, 255)
        cv2.polylines(out, [r.box.reshape(-1, 1, 2)], True, color, 3)
        x, y = r.box[:, 0].min(), r.box[:, 1].min()
        cv2.putText(out, r.text, (int(x), max(30, int(y) - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    return out
