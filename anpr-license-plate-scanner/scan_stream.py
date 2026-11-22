#!/usr/bin/env python3
"""
Read plates from a live IP camera (RTSP) or webcam and log them to CSV.

  python scan_stream.py --source rtsp://user:pass@192.168.1.64:554/stream1
  python scan_stream.py --source 0            # first webcam
  python scan_stream.py --source 0 --show     # show the video with boxes

Designed for a Raspberry Pi: only every Nth frame is processed, and the same
plate isn't logged again within a cooldown period.
"""

import argparse
import csv
import os
import time
from datetime import datetime

import cv2

from anpr import annotate, detect


def open_source(source):
    return cv2.VideoCapture(int(source) if source.isdigit() else source)


def main():
    p = argparse.ArgumentParser(description="Live ANPR from an IP camera or webcam.")
    p.add_argument("--source", required=True, help="RTSP URL, video file, or webcam index")
    p.add_argument("--every", type=int, default=5, help="process every Nth frame (default 5)")
    p.add_argument("--cooldown", type=float, default=30, help="seconds before logging the same plate again")
    p.add_argument("--log", default="plates_log.csv", help="CSV log file")
    p.add_argument("--snapshots", default="snapshots", help="folder for snapshot images of each read")
    p.add_argument("--show", action="store_true", help="display the video (needs a screen)")
    args = p.parse_args()

    os.makedirs(args.snapshots, exist_ok=True)
    new_file = not os.path.exists(args.log)
    log_file = open(args.log, "a", newline="")
    writer = csv.writer(log_file)
    if new_file:
        writer.writerow(["timestamp", "plate", "raw_ocr", "snapshot"])

    cap = open_source(args.source)
    last_seen = {}
    frame_no = 0
    print(f"Reading from {args.source}. Press Ctrl+C to stop.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Stream lost, reconnecting in 3 s...")
                cap.release()
                time.sleep(3)
                cap = open_source(args.source)
                continue

            frame_no += 1
            if frame_no % args.every:
                continue

            results = [r for r in detect(frame) if r.valid]
            now = time.time()
            for r in results:
                if now - last_seen.get(r.text, 0) < args.cooldown:
                    continue
                last_seen[r.text] = now
                stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                snap = os.path.join(args.snapshots, f"{r.text}_{int(now)}.jpg")
                cv2.imwrite(snap, annotate(frame, [r]))
                writer.writerow([stamp, r.text, r.raw_ocr, snap])
                log_file.flush()
                print(f"[{stamp}] {r.text}")

            if args.show:
                cv2.imshow("ANPR", annotate(frame, results))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        log_file.close()
        cv2.destroyAllWindows()
        print(f"Log saved to {args.log}")


if __name__ == "__main__":
    main()
