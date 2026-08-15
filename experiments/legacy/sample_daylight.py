#!/usr/bin/env python3
"""Sample frames from each raw clip, compute brightness, and report day/night segments."""
import cv2
import numpy as np
import os
import glob

RAW_DIR = "data/raw_clips"
SAMPLE_EVERY_SEC = 30
DAY_THRESHOLD = 90.0  # mean pixel brightness below this = night

def brightness_of_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(gray.mean())

def main():
    files = sorted(glob.glob(os.path.join(RAW_DIR, "*.mp4")))
    for path in files:
        vid = os.path.splitext(os.path.basename(path))[0]
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"\n=== {vid} ({duration/60:.0f} min) ===")
        samples = []
        t = 0.0
        while t < duration:
            frame_num = int(t * fps)
            if frame_num >= total:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ok, frame = cap.read()
            if ok:
                samples.append((t, brightness_of_frame(frame)))
            t += SAMPLE_EVERY_SEC

        # find contiguous daytime runs
        runs = []
        start = None
        for (t, b) in samples:
            is_day = b >= DAY_THRESHOLD
            if is_day and start is None:
                start = t
            elif not is_day and start is not None:
                runs.append((start, t))
                start = None
        if start is not None:
            runs.append((start, duration))

        # filter runs >= 3 min
        long_runs = [(s, e) for (s, e) in runs if e - s >= 180]
        if long_runs:
            for (s, e) in long_runs:
                print(f"  DAYTIME {s/60:.0f}-{e/60:.0f} min ({e-s:.0f}s)")
        else:
            print("  No sustained daytime segments found")
        # show brightness extremes
        if samples:
            bmin = min(samples, key=lambda x: x[1])
            bmax = max(samples, key=lambda x: x[1])
            print(f"  Brightness range: {bmin[1]:.0f} @ {bmin[0]/60:.0f}m .. {bmax[1]:.0f} @ {bmax[0]/60:.0f}m")
        cap.release()

if __name__ == "__main__":
    main()
