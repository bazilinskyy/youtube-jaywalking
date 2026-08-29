#!/usr/bin/env python3
"""
Single-video inference CLI using the frozen production architecture.

Usage:
  python3 scripts/run_inference.py --video path/to/video.mp4
"""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.pipeline.jaywalking_pipeline import JaywalkingPipeline


def main():
    parser = argparse.ArgumentParser(description="Run Jaywalking Detection on a single video clip.")
    parser.add_argument("--video", type=str, required=True, help="Path to input video file (.mp4).")
    parser.add_argument("--output", type=str, default=None, help="Optional path to save JSON results.")
    args = parser.parse_args()

    if not os.path.isfile(args.video):
        print(f"Error: Video file not found at {args.video}", file=sys.stderr)
        sys.exit(1)

    print(f"Initializing Jaywalking Detection Pipeline (Exp57 Frozen Production)...")
    pipeline = JaywalkingPipeline()

    print(f"Processing: {args.video}...")
    result = pipeline.process_video(args.video)

    print("\n" + "=" * 60)
    print(f"VIDEO:          {args.video}")
    print(f"PREDICTION:     {result['prediction']}")
    print(f"DECISION PATH:  {result['decision_path']}")
    print(f"VLM VOTES:      {result['votes']}")
    print(f"CROSSWALK:      {result['crosswalk_status']}")
    print(f"ROAD STRUCTURE: {result['road_structure_status']}")
    print(f"LATERAL DISP:   {result['lateral_displacement']}")
    print(f"ROAD OVERLAP:   {result['road_overlap']}")
    print(f"LATENCY:        {result['latency_sec']}s")
    print("=" * 60)

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as fp:
            json.dump(result, fp, indent=2)
        print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
