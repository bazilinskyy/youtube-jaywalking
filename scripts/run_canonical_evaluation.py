#!/usr/bin/env python3
"""
Runs evaluation on the canonical 39-video development benchmark.

Usage:
  python3 scripts/run_canonical_evaluation.py
"""

import os
import sys
from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.pipeline.jaywalking_pipeline import JaywalkingPipeline  # noqa: E402
from src.utils.metrics import calculate_classification_metrics  # noqa: E402


def main():
    manifest_path = "experiments/legacy/mapping.csv"
    video_dir = "videos"
    out_dir = "results/canonical"
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 80)
    print("CANONICAL 39-VIDEO BENCHMARK EVALUATION (FROZEN EXP57 PIPELINE)")
    print("=" * 80)

    # Load canonical mapping
    df = pd.read_csv(manifest_path)
    if "is_jaywalking" in df.columns:
        df["ground_truth"] = df["is_jaywalking"].apply(lambda x: "JAYWALKING" if int(x) == 1 else "COMPLIANT")

    pipeline = JaywalkingPipeline()
    records = []

    y_true = []
    y_pred = []

    for idx, (_, row) in enumerate(df.iterrows(), start=1):
        vid_id = str(row["video_id"])
        cname = f"{vid_id}.mp4" if not vid_id.endswith(".mp4") else vid_id
        vpath = os.path.join(video_dir, cname)
        gt = str(row["ground_truth"]).upper()

        if not os.path.isfile(vpath):
            print(f"Warning: {vpath} not found. Skipping.")
            continue

        res = pipeline.process_video(vpath)
        pred = res["prediction"]
        is_corr = (pred == gt)
        sym = "✓" if is_corr else "✗"

        y_true.append(gt)
        y_pred.append(pred)

        print(f"[{idx:02d}/{len(df):02d}] {cname:<16} GT={gt:<10} Pred={pred:<10} {sym:<2} ({res['latency_sec']}s)")

        records.append({
            "video_id": cname,
            "ground_truth": gt,
            "prediction": pred,
            "is_correct": is_corr,
            "votes": str(res["votes"]),
            "decision_path": res["decision_path"],
            "latency_sec": res["latency_sec"],
        })

    metrics = calculate_classification_metrics(y_true, y_pred)

    # Save results
    pd.DataFrame(records).to_csv(os.path.join(out_dir, "per_video_results.csv"), index=False)
    pd.DataFrame([metrics]).to_csv(os.path.join(out_dir, "results_summary.csv"), index=False)

    print("\n" + "=" * 80)
    print(f"CANONICAL EVALUATION COMPLETE (N={len(y_true)})")
    print(
        f"Accuracy: {metrics['accuracy']}% | Recall: {metrics['recall']}% | "
        f"Specificity: {metrics['specificity']}% | F1: {metrics['f1_score']}%"
    )
    print(f"Confusion: TP={metrics['tp']}, TN={metrics['tn']}, FP={metrics['fp']}, FN={metrics['fn']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
