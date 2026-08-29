#!/usr/bin/env python3
"""Unified benchmark evaluation runner for the frozen Jaywalking detection pipeline.

Evaluates the frozen production architecture across canonical, development, or locked test splits.

Usage:
  uv run python scripts/evaluate.py --split development
  uv run python scripts/evaluate.py --split locked_test
  uv run python scripts/evaluate.py --split canonical
  uv run python scripts/evaluate.py --manifest path/to/custom_manifest.csv --video-dir path/to/videos
"""

import argparse
import hashlib
import os
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.pipeline.jaywalking_pipeline import JaywalkingPipeline  # noqa: E402
from src.utils.metrics import calculate_classification_metrics  # noqa: E402


SPLIT_DEFAULTS = {
    "development": {
        "manifest": "datasets/manifests/development_manifest.csv",
        "video_dir": "jaad_pedestrian_100/videos",
        "output_dir": "results/development",
        "description": "DEVELOPMENT BENCHMARK (69 VIDEOS — EXP57 FROZEN PIPELINE)",
    },
    "locked_test": {
        "manifest": "datasets/manifests/locked_test_manifest.csv",
        "video_dir": "jaad_pedestrian_100/videos",
        "output_dir": "results/locked_test",
        "description": "LOCKED TEST BENCHMARK (30 UNSEEN VIDEOS — EXP58 EVALUATION)",
    },
    "canonical": {
        "manifest": "experiments/legacy/mapping.csv",
        "video_dir": "videos",
        "output_dir": "results/canonical",
        "description": "CANONICAL BENCHMARK (39 VIDEOS — EXP57 FROZEN PIPELINE)",
    },
}


def evaluate_split(
    manifest_path: str,
    video_dir: str,
    output_dir: Optional[str] = None,
    split_name: Optional[str] = None,
) -> dict:
    """Executes evaluation over a benchmark manifest using the frozen production pipeline.

    Args:
        manifest_path: Path to CSV dataset manifest containing video/clip mappings and ground truth.
        video_dir: Directory containing input video MP4 files.
        output_dir: Directory where per-video results and summary CSVs will be saved.
        split_name: Optional label for the evaluation run.

    Returns:
        Dictionary of calculated evaluation metrics.

    Raises:
        FileNotFoundError: If the manifest file is not found.
    """
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"Manifest not found at {manifest_path}")

    with open(manifest_path, "rb") as fp:
        manifest_hash = hashlib.sha256(fp.read()).hexdigest()

    df = pd.read_csv(manifest_path)
    total_clips = len(df)

    # Normalize ground truth column
    if "ground_truth" not in df.columns:
        if "is_jaywalking" in df.columns:
            df["ground_truth"] = df["is_jaywalking"].apply(
                lambda x: "JAYWALKING" if int(x) == 1 else "COMPLIANT"
            )
        else:
            raise KeyError("Manifest must contain 'ground_truth' or 'is_jaywalking' column.")

    print("=" * 80)
    print(f"EVALUATION: {split_name or manifest_path}")
    print(f"Manifest:   {manifest_path}")
    print(f"SHA-256:    {manifest_hash}")
    print(f"Videos:     {total_clips}")
    print("=" * 80)

    pipeline = JaywalkingPipeline()
    records = []
    y_true = []
    y_pred = []

    for idx, (_, row) in enumerate(df.iterrows(), start=1):
        if "clip_name" in row:
            cname = str(row["clip_name"])
        elif "video_id" in row:
            vid_id = str(row["video_id"])
            cname = f"{vid_id}.mp4" if not vid_id.endswith(".mp4") else vid_id
        else:
            cname = f"video_{idx:04d}.mp4"

        gt = str(row["ground_truth"]).upper()
        vpath = os.path.join(video_dir, cname)

        if not os.path.isfile(vpath):
            print(f"[{idx:02d}/{total_clips:02d}] {cname:<16} WARNING: File not found at {vpath}. Skipping.")
            continue

        res = pipeline.process_video(vpath)
        pred = res["prediction"]
        is_corr = (pred == gt)
        sym = "✓" if is_corr else "✗"

        y_true.append(gt)
        y_pred.append(pred)

        print(
            f"[{idx:02d}/{total_clips:02d}] {cname:<16} GT={gt:<10} Pred={pred:<10} {sym:<2} "
            f"({res['latency_sec']}s) Path: {res['decision_path']}"
        )

        records.append({
            "video_id": cname,
            "ground_truth": gt,
            "prediction": pred,
            "is_correct": is_corr,
            "votes": str(res["votes"]),
            "crosswalk_status": res["crosswalk_status"],
            "road_structure_status": res["road_structure_status"],
            "junction_status": res["junction_status"],
            "lateral_displacement": res["lateral_displacement"],
            "road_overlap": res["road_overlap"],
            "decision_path": res["decision_path"],
            "latency_sec": res["latency_sec"],
        })

    metrics = calculate_classification_metrics(y_true, y_pred)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        pd.DataFrame(records).to_csv(os.path.join(output_dir, "per_video_results.csv"), index=False)
        pd.DataFrame([metrics]).to_csv(os.path.join(output_dir, "results_summary.csv"), index=False)

    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print(f"Accuracy:    {metrics['accuracy']}% ({metrics['tp'] + metrics['tn']}/{metrics['total']})")
    print(f"Precision:   {metrics['precision']}%")
    print(f"Recall:      {metrics['recall']}%")
    print(f"Specificity: {metrics['specificity']}%")
    print(f"F1 Score:    {metrics['f1_score']}%")
    print(f"Confusion:   TP={metrics['tp']}, TN={metrics['tn']}, FP={metrics['fp']}, FN={metrics['fn']}")
    print("=" * 80)

    return metrics


def main():
    """CLI entry point for running benchmark evaluations."""
    parser = argparse.ArgumentParser(
        description="Unified Jaywalking Detection Benchmark Evaluator."
    )
    parser.add_argument(
        "--split",
        choices=["development", "locked_test", "canonical"],
        default=None,
        help="Preconfigured evaluation split.",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Custom manifest CSV path.",
    )
    parser.add_argument(
        "--video-dir",
        type=str,
        default=None,
        help="Directory containing video MP4 files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save evaluation results.",
    )
    args = parser.parse_args()

    if args.split:
        cfg = SPLIT_DEFAULTS[args.split]
        manifest_path = args.manifest or cfg["manifest"]
        video_dir = args.video_dir or cfg["video_dir"]
        output_dir = args.output_dir or cfg["output_dir"]
        split_name = cfg["description"]
    elif args.manifest and args.video_dir:
        manifest_path = args.manifest
        video_dir = args.video_dir
        output_dir = args.output_dir
        split_name = f"Custom Split ({manifest_path})"
    else:
        parser.error("Either --split or both --manifest and --video-dir must be specified.")

    evaluate_split(
        manifest_path=manifest_path,
        video_dir=video_dir,
        output_dir=output_dir,
        split_name=split_name,
    )


if __name__ == "__main__":
    main()
