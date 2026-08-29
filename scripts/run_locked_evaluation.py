#!/usr/bin/env python3
"""
Executes reproducible evaluation of the frozen production architecture on the locked 30-video test benchmark.

Usage:
  python3 scripts/run_locked_evaluation.py
"""

import hashlib
import os
import sys
from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.pipeline.jaywalking_pipeline import JaywalkingPipeline  # noqa: E402
from src.utils.metrics import calculate_classification_metrics  # noqa: E402


def main():
    manifest_path = "datasets/manifests/locked_test_manifest.csv"
    video_dir = "jaad_pedestrian_100/videos"
    out_dir = "results/locked_test"
    os.makedirs(out_dir, exist_ok=True)

    with open(manifest_path, "rb") as fp:
        manifest_hash = hashlib.sha256(fp.read()).hexdigest()

    df = pd.read_csv(manifest_path)
    n_test = len(df)

    print("=" * 80)
    print("LOCKED TEST BENCHMARK EVALUATION (FROZEN PRODUCTION PIPELINE)")
    print(f"Manifest: {manifest_path}")
    print(f"SHA-256:  {manifest_hash}")
    print(f"Videos:   {n_test}")
    print("=" * 80)

    pipeline = JaywalkingPipeline()
    records = []
    y_true = []
    y_pred = []

    for idx, (_, row) in enumerate(df.iterrows(), start=1):
        cname = str(row["clip_name"])
        gt = str(row["ground_truth"]).upper()
        vpath = os.path.join(video_dir, cname)

        res = pipeline.process_video(vpath)
        pred = res["prediction"]
        is_corr = (pred == gt)
        sym = "✓" if is_corr else "✗"

        y_true.append(gt)
        y_pred.append(pred)

        print(
            f"[{idx:02d}/{n_test:02d}] {cname:<16} GT={gt:<10} Pred={pred:<10} {sym:<2} "
            f"({res['latency_sec']}s) | {res['decision_path']}"
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
            "decision_path": res["decision_path"],
            "latency_sec": res["latency_sec"],
        })

    metrics = calculate_classification_metrics(y_true, y_pred)

    # Save CSV deliverables
    pd.DataFrame(records).to_csv(os.path.join(out_dir, "per_video_results.csv"), index=False)
    pd.DataFrame([metrics]).to_csv(os.path.join(out_dir, "results_summary.csv"), index=False)

    print("\n" + "=" * 80)
    print("LOCKED TEST EVALUATION COMPLETE")
    print(
        f"Accuracy: {metrics['accuracy']}% | Precision: {metrics['precision']}% | "
        f"Recall: {metrics['recall']}% | Specificity: {metrics['specificity']}% | F1: {metrics['f1_score']}%"
    )
    print(f"Confusion: TP={metrics['tp']}, TN={metrics['tn']}, FP={metrics['fp']}, FN={metrics['fn']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
