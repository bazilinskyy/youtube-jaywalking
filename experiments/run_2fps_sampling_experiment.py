#!/usr/bin/env python3
"""
Controlled Fixed 2 FPS Temporal Sampling Density Experiment (39 Original JAAD Videos)

Samples each single-call event envelope [min(F_start), max(F_end)] at a FIXED 2 FPS rate:
  sample_times = start_time, start_time + 0.5s, start_time + 1.0s, ... until end_time

To comply with Ollama API's maximum vision payload capacity (<= 12 frames per request),
longer 2 FPS grids are capped at 12 uniformly sampled frames.

Executes ONE VLM INFERENCE CALL for the complete envelope.

Usage:
    python experiments/run_2fps_sampling_experiment.py
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
from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.vlm.alpamayo_detector import FullVideoVLMDetector
from src.vlm.client import encode_frame_to_base64
from scripts.run_long_video_vlm_experiment import extract_candidate_events, merge_overlapping_events


def run_2fps_experiment():
    gt_path = "data/ground_truth.csv"
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"Ground truth missing: {gt_path}")

    df_gt = pd.read_csv(gt_path)
    eval_df = df_gt[df_gt["is_evaluated"] == True].copy()
    total_videos = len(eval_df)

    print("=" * 80)
    print("STARTING CONTROLLED FIXED 2 FPS TEMPORAL DENSITY EXPERIMENT (39 CLIPS)")
    print("Sampling Grid: 2 FPS (0.5s intervals), capped at max 12 frames per payload for Ollama stability")
    print(f"Total Videos: {total_videos}")
    print("=" * 80)

    detector = FullVideoVLMDetector()
    results = []

    t_start = time.time()

    # PHASE 1: INFERENCE (STRICTLY NO GROUND TRUTH ACCESS DURING INFERENCE)
    print("\n[PHASE 1: EXECUTING INFERENCE OVER 2 FPS TEMPORAL SAMPLING GRID]")
    for idx, (_, row) in enumerate(eval_df.iterrows(), start=1):
        video_path = str(row["video_path"])
        clip_name = str(row["clip_name"])

        total_frames, fps, duration, cands = extract_candidate_events(video_path)
        merged = merge_overlapping_events(cands, fps=fps)

        if merged:
            env_s = min(m["start_frame"] for m in merged)
            env_e = max(m["end_frame"] for m in merged)
        else:
            env_s = 1
            env_e = total_frames

        t_s = (env_s - 1) / fps
        t_e = (env_e - 1) / fps

        t_grid = np.arange(t_s, t_e + 1e-5, 0.5)
        raw_indices = [min(total_frames, max(1, int(round(t * fps)) + 1)) for t in t_grid]
        sample_indices = list(dict.fromkeys(raw_indices))  # Preserves order, removes duplicates

        # Cap at max 12 frames for Ollama API payload capacity limit
        if len(sample_indices) > 12:
            sub_idx = np.linspace(0, len(sample_indices) - 1, num=12, dtype=int)
            sample_indices = [sample_indices[i] for i in sub_idx]

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

        verdict = parsed["prediction"].upper()

        print(f"[{idx}/{total_videos}] {clip_name}: Envelope=[{env_s}..{env_e}] | Frames Sent={len(frames):>2} | Verdict={verdict:<10} ({elapsed}s)")

        results.append({
            "clip_name": clip_name,
            "video_path": video_path,
            "total_frames": total_frames,
            "fps": fps,
            "duration_seconds": duration,
            "envelope_bounds": [env_s, env_e],
            "frames_sent_count": len(frames),
            "sampled_indices": sample_indices,
            "prediction": verdict,
            "reasoning": parsed["chain_of_causation"],
            "inference_time": elapsed,
        })

    total_benchmark_time = round(time.time() - t_start, 2)

    # PHASE 2: METRICS & COMPARATIVE EVALUATION (POST-INFERENCE ONLY)
    print("\n[PHASE 2: EVALUATING METRICS & DIRECT COMPARISON]")
    gt_map = {row["clip_name"]: str(row["ground_truth"]).lower() for _, row in eval_df.iterrows()}

    tp = tn = fp = fn = 0
    per_video_table = []
    tot_frames_sent = sum(r["frames_sent_count"] for r in results)

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
            "frames_sent": r["frames_sent_count"],
            "elapsed": r["inference_time"],
        })

    acc = round((tp + tn) / total_videos * 100, 2)
    prec = round(tp / (tp + fp) * 100, 2) if (tp + fp) > 0 else 0.0
    rec = round(tp / (tp + fn) * 100, 2) if (tp + fn) > 0 else 0.0
    spec = round(tn / (tn + fp) * 100, 2) if (tn + fp) > 0 else 0.0
    f1 = round(2 * prec * rec / (prec + rec), 2) if (prec + rec) > 0 else 0.0
    avg_time = round(total_benchmark_time / total_videos, 2)
    avg_frames = round(tot_frames_sent / total_videos, 2)

    # Print Table
    print("\n" + "=" * 80)
    print("FIXED 2 FPS TEMPORAL SAMPLING RESULTS TABLE")
    print("=" * 80)
    print(f"{'Clip Name':<18} | {'GT':<11} | {'Prediction':<12} | {'Correct':<8} | {'Frames':<6} | {'Latency':<7}")
    print("-" * 80)
    for r in per_video_table:
        corr_str = "YES" if r["correct"] else "NO"
        print(f"{r['clip_name']:<18} | {r['ground_truth']:<11} | {r['prediction']:<12} | {corr_str:<8} | {r['frames_sent']:>6} | {r['elapsed']:>5.2f}s")
    print("=" * 80)
    print(f"Accuracy:                  {acc}% ({tp+tn}/{total_videos})")
    print(f"Precision:                 {prec}%")
    print(f"Recall:                    {rec}%")
    print(f"Specificity:               {spec}%")
    print(f"F1 Score:                  {f1}%")
    print(f"Confusion Matrix:          TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    print(f"Total Benchmark Time:      {total_benchmark_time}s (avg {avg_time}s/clip)")
    print(f"Average Frames Sent/Video: {avg_frames} frames")
    print("=" * 80)

    # Direct 6-Way Comparison Table
    print("\n" + "=" * 85)
    print("DIRECT 6-WAY TEMPORAL SAMPLING COMPARISON TABLE")
    print("=" * 85)
    print(f"{'Evaluation Setup / Pipeline':<42} | {'Accuracy':<9} | {'F1 Score':<8} | {'Recall':<8} | {'Specificity':<11}")
    print("-" * 85)
    print(f"{'1. Historical short-clip baseline':<42} | {'97.44%':<9} | {'96.77%':<8} | {'100.0%':<8} | {'95.83%':<11}")
    print(f"{'2. 5-frame uniform (Exp 14)':<42} | {'61.54%':<9} | {'51.61%':<8} | {'53.33%':<8} | {'66.67%':<11}")
    print(f"{'3. 10-frame uniform (Exp 15)':<42} | {'64.10%':<9} | {'50.00%':<8} | {'46.67%':<8} | {'75.00%':<11}")
    print(f"{'4. Single-call envelope (Arch B - Exp 17)':<42} | {'71.79%':<9} | {'56.00%':<8} | {'46.67%':<8} | {'87.50%':<11}")
    print(f"{'5. ±1.5s expansion (Exp 18)':<42} | {'51.28%':<9} | {'34.48%':<8} | {'33.33%':<8} | {'62.50%':<11}")
    print(f"{'6. Fixed 2 FPS temporal sampling (Exp 19)':<42} | {f'{acc}%':<9} | {f'{f1}%':<8} | {f'{rec}%':<8} | {f'{spec}%':<11}")
    print("=" * 85)

    # Save JSON Output
    out_json_path = "outputs/fixed_2fps_sampling_experiment_results.json"
    os.makedirs("outputs", exist_ok=True)
    with open(out_json_path, "w") as f:
        json.dump({
            "experiment": "Controlled Fixed 2 FPS Temporal Sampling Density Experiment",
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
                "total_benchmark_time_seconds": total_benchmark_time,
                "avg_inference_time_seconds": avg_time,
                "avg_frames_sent_per_video": avg_frames,
            },
            "video_results": results,
        }, f, indent=2)

    print(f"\nSaved machine-readable 2 FPS sampling results to: {out_json_path}")

    # Append Experiment 19 to RESEARCH_LOG.md
    log_entry = f"""

## Experiment 19 — Controlled Fixed 2 FPS Temporal Density Experiment (39 Clips)
* **Date:** 2026-08-18
* **Model:** `qwen2.5vl:7b` via `FullVideoVLMDetector` (VLM Baseline)
* **Research Question:** Does sampling each single-call event envelope at a fixed 2 FPS rate ($t_0, t_0+0.5s, t_0+1.0s, \\dots$, capped at max 12 frames) provide sufficient frame density across extended event durations to improve zero-shot VLM accuracy?
* **Experimental Protocol:**
  - Sampled each event envelope $[\\min(F_{{\\text{{start}}}}), \\max(F_{{\\text{{end}}}})]$ at a fixed 2 FPS temporal grid (average {avg_frames} frames/video, capped at max 12 frames for Ollama API payload capacity).
  - Executed exactly 1 VLM inference call per video with the complete 2 FPS frame sequence (no OR logic).
* **Empirical Results:**
  - **Accuracy:** **{acc}%** ({tp+tn}/{total_videos})
  - **Precision:** **{prec}%**
  - **Recall:** **{rec}%**
  - **Specificity:** **{spec}%**
  - **F1 Score:** **{f1}%**
  - **Confusion Matrix:** $\\text{{TP}}={tp}, \\text{{TN}}={tn}, \\text{{FP}}={fp}, \\text{{FN}}={fn}$
  - **Average Frames Sent per Video:** {avg_frames} frames
  - **Total Latency:** {total_benchmark_time}s (avg {avg_time}s/clip)

* **Direct 6-Way Baseline Comparison Table:**

| Evaluation Setup / Pipeline | Accuracy | F1 Score | Recall | Specificity | Precision | TP | TN | FP | FN | Avg Frames |
|---|---|---|---|---|---|---|---|---|---|---|
| **1. Historical short-clip baseline** | **97.44%** | **96.77%** | **100.0%** | **95.83%** | **93.75%** | 15 | 23 | 1 | 0 | 5.0 |
| **2. 5-frame uniform (Exp 14)** | **61.54%** | **51.61%** | **53.33%** | **66.67%** | **50.00%** | 8 | 16 | 8 | 7 | 5.0 |
| **3. 10-frame uniform (Exp 15)** | **64.10%** | **50.00%** | **46.67%** | **75.00%** | **53.85%** | 7 | 18 | 6 | 8 | 10.0 |
| **4. Single-call envelope (Arch B - Exp 17)** | **71.79%** | **56.00%** | **46.67%** | **87.50%** | **70.00%** | 7 | 21 | 3 | 8 | 5.0 |
| **5. ±1.5s expansion (Exp 18)** | **51.28%** | **34.48%** | **33.33%** | **62.50%** | **35.71%** | 5 | 15 | 9 | 10 | 5.0 |
| **6. Fixed 2 FPS temporal sampling (Exp 19)** | **{acc}%** | **{f1}%** | **{rec}%** | **{spec}%** | **{prec}%** | {tp} | {tn} | {fp} | {fn} | {avg_frames} |
"""
    with open("RESEARCH_LOG.md", "a") as f:
        f.write(log_entry)

    print("Updated RESEARCH_LOG.md with Experiment 19 results.")


if __name__ == "__main__":
    run_2fps_experiment()
