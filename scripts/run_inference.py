#!/usr/bin/env python3
"""
CLI entry point to run jaywalking detection on a single video file or a directory of videos.
Usage:
    python scripts/run_inference.py --video data/raw_clips/video_0014.mp4
    python scripts/run_inference.py --video data/raw_clips/video_0014.mp4 --mode ensemble
    python scripts/run_inference.py --dir data/raw_clips/ --limit 5
"""
import argparse
import json
import sys
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.pipeline import get_pipeline


def main():
    parser = argparse.ArgumentParser(description="Run Jaywalking Detection Inference")
    parser.add_argument("--video", type=str, help="Path to a single video file")
    parser.add_argument("--dir", type=str, help="Path to a directory containing video files")
    parser.add_argument("--mode", choices=["alpamayo", "vlm", "cv", "ensemble", "full_video"], default="alpamayo", help="Pipeline mode (default: alpamayo)")
    parser.add_argument("--limit", type=int, default=None, help="Max videos to process if --dir is used")
    parser.add_argument("--prompt", choices=["canonical", "right_of_way"], default="canonical", help="VLM prompt preset")
    args = parser.parse_args()

    if not args.video and not args.dir:
        parser.error("Must specify either --video or --dir")

    pipeline = get_pipeline(args.mode)

    if args.video:
        video_path = Path(args.video)
        if not video_path.exists():
            print(f"Error: Video file not found: {video_path}")
            sys.exit(1)

        print(f"\nRunning {args.mode.upper()} inference on {video_path.name}...")
        result = pipeline.predict(video_path)
        print(json.dumps(result, indent=2))
        print(f"\n>>> Final Decision: {result['prediction'].upper()} (Confidence: {result['confidence']})")

    elif args.dir:
        video_dir = Path(args.dir)
        if not video_dir.exists():
            print(f"Error: Directory not found: {video_dir}")
            sys.exit(1)

        videos = sorted(list(video_dir.glob("*.mp4")))
        if args.limit:
            videos = videos[:args.limit]

        print(f"\nRunning {args.mode.upper()} inference on {len(videos)} videos in {video_dir}...\n")
        print(f"{'Video Name':<20} | {'Prediction':<12} | {'Confidence':<10} | {'Time':<6} | Reason")
        print("-" * 75)

        for v in videos:
            res = pipeline.predict(v)
            print(f"{v.name:<20} | {res['prediction']:<12} | {res['confidence']:<10} | {res['elapsed_seconds']:>4.1f}s | {res['reason']}")


if __name__ == "__main__":
    main()
