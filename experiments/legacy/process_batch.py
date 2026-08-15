#!/usr/bin/env python3
"""Batch process all raw clips: extract a daytime segment from each and run
the jaywalking pipeline on GPU. Produces annotated video + CSV per clip."""
import os
import subprocess
import sys
import time

RAW_DIR = "data/raw_clips"
OUT_DIR = "output"
SEGMENT_SECONDS = 180  # 3 minutes of daytime footage per video

# Daytime segments found by sample_daylight.py (start_sec, end_sec)
# First sustained daytime run per video
DAYTIME = {
    "-TPJot7-HTs.mp4": (480, 720),     # 8-12 min
    "3ai7SUaPoHM.mp4": (0, 180),
    "AxQcSoA9vGQ.mp4": (0, 180),
    "G1I_PlmL_YA.mp4": (0, 180),
    "gmDBzijaIAA.mp4": (0, 180),
    "JY-Xyiept88.mp4": (0, 180),
    "MAj6y23vNuU.mp4": (0, 180),
    "oDejyTLYUTE.mp4": (0, 180),
    "qOx5CwCrN9k.mp4": (0, 180),
    "qzimFzMh6lA.mp4": (0, 180),
    "wCKLtcGQnWc.mp4": (0, 180),
    "wMu6Va5PhGY.mp4": None,           # no sustained daytime
    "z3Gx2hp3Vo8.mp4": (0, 180),
    "ZByZSqoqzaI.mp4": (0, 180),
    "ZruaEnhtYLA.mp4": (0, 180),
}


def get_fps(path):
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0", path],
        capture_output=True, text=True)
    num, den = r.stdout.strip().split("/")
    return round(int(num) / int(den))


def extract_segment(src, dst, start, length):
    subprocess.run(["ffmpeg", "-y", "-ss", str(start), "-t", str(length),
                    "-i", src, "-c:v", "libx264", "-preset", "fast",
                    "-crf", "23", "-an", dst],
                   capture_output=True, check=True)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    videos = sorted(DAYTIME.keys())
    if len(sys.argv) > 1:
        videos = [v for v in videos if sys.argv[1] in v]

    for video in videos:
        seg = DAYTIME.get(video)
        src = os.path.join(RAW_DIR, video)
        if not os.path.exists(src):
            print(f"[SKIP] {video}: not found")
            continue
        if seg is None:
            print(f"[SKIP] {video}: no daytime segment")
            continue

        start, end = seg
        length = min(SEGMENT_SECONDS, end - start)
        vid_id = os.path.splitext(video)[0]
        clip = os.path.join("/tmp", f"clip_{vid_id}.mp4")
        out_video = os.path.join(OUT_DIR, f"{vid_id}_annotated.mp4")

        print(f"\n=== {vid_id} ===")
        print(f"  Extracting {length}s from {start}s...")
        try:
            extract_segment(src, clip, start, length)
        except subprocess.CalledProcessError as e:
            print(f"  FAILED extraction: {e}")
            continue

        fps = get_fps(clip)
        print(f"  Clip: {length}s @ {fps} fps ({length * fps} frames)")
        t0 = time.time()
        r = subprocess.run(
            [sys.executable, "run_demo.py", "--input", clip,
             "--output", out_video, "--fps", str(fps), f"--title={vid_id}"],
            capture_output=True, text=True)
        elapsed = time.time() - t0
        if r.returncode != 0:
            print(f"  FAILED pipeline: {r.stderr[-500:]}")
            continue
        print(f"  Done in {elapsed/60:.1f} min -> {out_video}")
        os.remove(clip)

    print("\n=== Batch complete ===")


if __name__ == "__main__":
    main()
