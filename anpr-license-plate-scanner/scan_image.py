#!/usr/bin/env python3
"""
Read licence plates from an image.

  python scan_image.py car.jpg
  python scan_image.py car.jpg --save result.jpg
"""

import argparse
import sys

import cv2

from anpr import annotate, detect


def main():
    p = argparse.ArgumentParser(description="Detect and read licence plates in an image.")
    p.add_argument("image")
    p.add_argument("--save", help="save an annotated copy of the image")
    args = p.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        print(f"Could not read image: {args.image}")
        return 2

    results = detect(image)
    if not results:
        print("No plate found.")
    for r in results:
        status = "valid" if r.valid else "unverified"
        print(f"Plate: {r.text:12s} ({status}; raw OCR: '{r.raw_ocr}')")

    if args.save:
        cv2.imwrite(args.save, annotate(image, results))
        print(f"Annotated image saved to {args.save}")
    return 0 if any(r.valid for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
