#!/usr/bin/env python3
"""
Source-Segment Temporal Boundary Diagnostic Experiment (39 Clips)

Evaluates the VLM baseline (qwen2.5vl:7b via FullVideoVLMDetector) using 5-frame uniform sampling
across the exact source temporal segment corresponding to each of the 39 benchmark clips in data/raw_clips/.

Usage:
    python experiments/run_source_segment_diagnostic.py
"""

import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.vlm.alpamayo_detector import FullVideoVLMDetector
from src.vlm.client import encode_frame_to_base64


def run_source_segment_diagnostic():
    gt_path = "data/ground_truth.csv"
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"Ground truth missing: {gt_path}")

    df_gt = pd.read_csv(gt_path)
    eval_df = df_gt[df_gt["is_evaluated"] == True].copy()
    total_videos = len(eval_df)

    print("=" * 80)
    print("STARTING SOURCE-SEGMENT TEMPORAL BOUNDARY DIAGNOSTIC EXPERIMENT (39 CLIPS)")
    print("Sampling: 5 uniform frames across exact source segment [1, N_total_frames]")
    print(f"Total Videos: {total_videos}")
    print("=" * 80)

    detector = FullVideoVLMDetector()
    results = []

    t_start = time.time()

    # PHASE 1: INFERENCE (STRICTLY NO GROUND TRUTH ACCESS DURING INFERENCE)
    print("\n[PHASE 1: EXECUTING INFERENCE OVER SOURCE TEMPORAL SEGMENTS]")
    for idx, (_, row) in enumerate(eval_df.iterrows(), start=1):
        video_path = str(row["video_path"])
        clip_name = str(row["clip_name"])

        print(f"[{idx}/{total_videos}] Processing {clip_name}... ", end="", flush=True)

        if not os.path.exists(video_path):
            print(f"ERROR: File missing {video_path}")
            continue

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()

        # Sample 5 uniform frames across the exact source temporal segment
        sample_indices = np.linspace(1, total_frames, num=5, dtype=int).tolist()

        cap = cv2.VideoCapture(video_path)
        frames = []
        for f_idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx - 1)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
        cap.release()

        t0 = time.time()
        b64_list = [encode_frame_to_base64(f, quality=85) for f in frames]
        raw_response = detector.client.generate_chat(prompt=detector.coc_prompt, base64_images=b64_list)
        parsed = detector.parse_coc_response(raw_response)
        elapsed = round(time.time() - t0, 3)

        verdict = parsed["prediction"]
        print(f"Frames=[1..{total_frames}] -> Verdict={verdict:<10} ({elapsed}s)")

        results.append({
            "clip_name": clip_name,
            "video_path": video_path,
            "total_frames": total_frames,
            "fps": fps,
            "duration_seconds": round(total_frames / fps, 2),
            "sampled_indices": sample_indices,
            "prediction": verdict,
            "reasoning": parsed["chain_of_causation"],
            "raw_response": raw_response,
            "inference_time": elapsed,
        })

    total_inference_time = round(time.time() - t_start, 2)

    # PHASE 2: METRICS & COMPARISON (AFTER INFERENCE COMPLETION)
    print("\n[PHASE 2: EVALUATING METRICS & DIRECT COMPARISON]")
    gt_map = {row["clip_name"]: str(row["ground_truth"]).lower() for _, row in eval_df.iterrows()}

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    per_video_table = []

    for r in results:
        clip_name = r["clip_name"]
        gt_label = gt_map[clip_name]
        pred_label = r["prediction"].lower()

        is_correct = (pred_label == gt_label)
        if gt_label == "jaywalking" and pred_label == "jaywalking":
            tp += 1
        elif gt_label == "compliant" and pred_label == "compliant":
            tn += 1
        elif gt_label == "compliant" and pred_label == "jaywalking":
            fp += 1
        elif gt_label == "jaywalking" and pred_label == "compliant":
            fn += 1

        per_video_table.append({
            "clip_name": clip_name,
            "ground_truth": gt_label,
            "prediction": pred_label,
            "correct": is_correct,
            "elapsed": r["inference_time"],
        })

    acc = round((tp + tn) / total_videos * 100, 2)
    prec = round(tp / (tp + fp) * 100, 2) if (tp + fp) > 0 else 0.0
    rec = round(tp / (tp + fn) * 100, 2) if (tp + fn) > 0 else 0.0
    spec = round(tn / (tn + fp) * 100, 2) if (tn + fp) > 0 else 0.0
    f1 = round(2 * prec * rec / (prec + rec), 2) if (prec + rec) > 0 else 0.0
    avg_time = round(total_inference_time / total_videos, 2)

    # Print Table
    print("\n" + "=" * 75)
    print("SOURCE-SEGMENT DIAGNOSTIC RESULTS TABLE")
    print("=" * 75)
    print(f"{'Clip Name':<18} | {'GT':<11} | {'Prediction':<12} | {'Correct':<8} | {'Latency':<7}")
    print("-" * 75)
    for r in per_video_table:
        corr_str = "YES" if r["correct"] else "NO"
        print(f"{r['clip_name']:<18} | {r['ground_truth']:<11} | {r['prediction']:<12} | {corr_str:<8} | {r['elapsed']:>5.2f}s")
    print("=" * 75)
    print(f"Accuracy:        {acc}% ({tp+tn}/{total_videos})")
    print(f"Precision:       {prec}%")
    print(f"Recall:          {rec}%")
    print(f"Specificity:     {spec}%")
    print(f"F1 Score:        {f1}%")
    print(f"Confusion Matrix: TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    print(f"Total Time:      {total_inference_time}s (avg {avg_time}s/clip)")
    print("=" * 75)

    # Print Direct 3-Way Baseline Comparison Table
    print("\n" + "=" * 80)
    print("DIRECT 3-WAY COMPARISON: TEMPORAL BOUNDARY IMPACT DIAGNOSIS")
    print("=" * 80)
    print(f"{'Evaluation Setup':<45} | {'Accuracy':<10} | {'F1 Score':<8} | {'Recall':<8} | {'Specificity':<11}")
    print("-" * 80)
    print(f"{'1. Pre-cut short-clip VLM baseline':<45} | {'97.44%':<10} | {'96.77%':<8} | {'100.0%':<8} | {'95.83%':<11}")
    print(f"{'2. Auto-detected long video event (Exp 14)':<45} | {'61.54%':<10} | {'51.61%':<8} | {'53.33%':<8} | {'66.67%':<11}")
    print(f"{'3. Exact source-segment + 5-frame uniform':<45} | {f'{acc}%':<10} | {f'{f1}%':<8} | {f'{rec}%':<8} | {f'{spec}%':<11}")
    print("=" * 80)

    # Save JSON Output
    out_json_path = "outputs/source_segment_diagnostic_results.json"
    os.makedirs("outputs", exist_ok=True)
    with open(out_json_path, "w") as f:
        json.dump({
            "experiment": "Source-Segment Temporal Boundary Diagnostic Experiment",
            "model": "qwen2.5vl:7b",
            "total_videos": total_videos,
            "metrics": {
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "specificity": spec,
                "f1": f1,
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "total_inference_time_seconds": total_inference_time,
                "avg_inference_time_seconds": avg_time,
            },
            "video_results": results,
        }, f, indent=2)

    print(f"Saved machine-readable diagnostic results to: {out_json_path}")

    # Append Experiment 16 to RESEARCH_LOG.md
    log_entry = f"""

## Experiment 16 — Source-Segment Temporal Boundary Diagnostic Experiment (39 Clips)
* **Date:** 2026-08-18
* **Model:** `qwen2.5vl:7b` via `FullVideoVLMDetector` (VLM Baseline)
* **Research Question:** Are temporal boundaries (source segment vs automatically detected event interval) responsible for the performance gap between 97.44% and 61.54%?
* **Experimental Setup:** Extracted 5 uniform frames across the exact source temporal segment $[1, N_{{\\text{{total\\_frames}}}}]$ for all 39 clips with zero ground-truth leakage during inference.
* **Empirical Results:**
  - **Accuracy:** **{acc}%** ({tp+tn}/{total_videos})
  - **Precision:** **{prec}%**
  - **Recall:** **{rec}%**
  - **Specificity:** **{spec}%**
  - **F1 Score:** **{f1}%**
  - **Confusion Matrix:** $\\text{{TP}}={tp}, \\text{{TN}}={tn}, \\text{{FP}}={fp}, \\text{{FN}}={fn}$
  - **Total Inference Time:** {total_inference_time}s (avg {avg_time}s/clip)

* **Direct 3-Way Baseline Comparison:**

| Evaluation Setup / Pipeline | Accuracy | F1 Score | Recall | Specificity | Precision | TP | TN | FP | FN |
|---|---|---|---|---|---|---|---|---|---|
| **1. Pre-cut short-clip VLM baseline** | **97.44%** | **96.77%** | **100.0%** | **95.83%** | **93.75%** | 15 | 23 | 1 | 0 |
| **2. Auto-detected long video event (Exp 14)** | **61.54%** | **51.61%** | **53.33%** | **66.67%** | **50.00%** | 8 | 16 | 8 | 7 |
| **3. Exact source-segment + 5-frame uniform** | **{acc}%** | **{f1}%** | **{rec}%** | **{spec}%** | **{prec}%** | {tp} | {tn} | {fp} | {fn} |

* **Diagnostic Conclusion:**
  - This experiment conclusively establishes whether restoring exact source temporal boundaries recovers the 97.44% baseline performance or if other factor changes contribute to the difference.
"""
    with open("RESEARCH_LOG.md", "a") as f:
        f.write(log_entry)

    print("Updated RESEARCH_LOG.md with Experiment 16 results.")


if __name__ == "__main__":
    run_source_segment_diagnostic()
