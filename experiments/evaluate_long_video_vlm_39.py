#!/usr/bin/env python3
"""
Long Video Multi-Event VLM Baseline Benchmark Evaluation (39 Clips)

Evaluates the long-video pipeline (YOLO11x + ByteTrack -> Temporal Event Merging -> VLM Baseline)
across the canonical 39 development clips.

Usage:
    python experiments/evaluate_long_video_vlm_39.py
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


def run_long_video_benchmark():
    gt_path = "data/ground_truth.csv"
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"Ground truth file missing: {gt_path}")

    df_gt = pd.read_csv(gt_path)
    eval_df = df_gt[df_gt["is_evaluated"] == True].copy()
    total_videos = len(eval_df)

    print("=" * 75)
    print("STARTING SCALED LONG-VIDEO VLM BASELINE BENCHMARK (39 CLIPS)")
    print(f"Pipeline: YOLO11x + ByteTrack -> Event Merging -> VLM Baseline (qwen2.5vl:7b)")
    print(f"Total Videos: {total_videos}")
    print("=" * 75)

    detector = FullVideoVLMDetector()
    video_results = []
    
    t_start_benchmark = time.time()

    # PHASE 1: INFERENCE (STRICTLY NO GROUND TRUTH ACCESS)
    print("\n[PHASE 1: EXECUTING LONG-VIDEO INFERENCE PIPELINE]")
    for idx, (_, row) in enumerate(eval_df.iterrows(), start=1):
        video_path = str(row["video_path"])
        clip_name = str(row["clip_name"])
        video_id = Path(clip_name).stem

        print(f"[{idx}/{total_videos}] Processing {clip_name}... ", end="", flush=True)

        if not os.path.exists(video_path):
            print(f"ERROR: Missing file {video_path}")
            continue

        t0_video = time.time()

        # Step A: Candidate Extraction & Event Merging
        total_frames, fps, duration, raw_candidates = extract_candidate_events(video_path)
        merged_events = merge_overlapping_events(raw_candidates, fps=fps)

        event_details = []
        cap = cv2.VideoCapture(video_path)
        vlm_time_video = 0.0

        # Step B: VLM Baseline Inference Per Merged Event
        if merged_events:
            for m in merged_events:
                s_f = m["start_frame"]
                e_f = m["end_frame"]
                sample_indices = np.linspace(s_f, e_f, num=5, dtype=int).tolist()

                frames = []
                for f_idx in sample_indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx - 1)
                    ret, frame = cap.read()
                    if ret:
                        frames.append(frame)

                t0_vlm = time.time()
                b64_list = [encode_frame_to_base64(f, quality=85) for f in frames]
                raw_response = detector.client.generate_chat(prompt=detector.coc_prompt, base64_images=b64_list)
                parsed = detector.parse_coc_response(raw_response)
                elapsed_vlm = round(time.time() - t0_vlm, 3)
                vlm_time_video += elapsed_vlm

                event_details.append({
                    "event_id": m["event_id"],
                    "track_ids": m["track_ids"],
                    "start_frame": s_f,
                    "end_frame": e_f,
                    "start_timestamp": m["start_timestamp"],
                    "end_timestamp": m["end_timestamp"],
                    "duration": m["duration"],
                    "verdict": parsed["prediction"],
                    "reasoning": parsed["chain_of_causation"],
                    "raw_response": raw_response,
                    "inference_time": elapsed_vlm,
                })
        cap.release()

        # Step C: Fixed Video Aggregation
        has_jaywalking = any(e["verdict"].upper() == "JAYWALKING" for e in event_details)
        video_verdict = "JAYWALKING" if has_jaywalking else "COMPLIANT"
        video_elapsed = round(time.time() - t0_video, 2)

        print(f"Candidates={len(raw_candidates)}, Merged={len(merged_events)} -> Verdict={video_verdict} ({video_elapsed}s)")

        video_results.append({
            "clip_name": clip_name,
            "video_id": video_id,
            "video_path": video_path,
            "duration_seconds": duration,
            "total_frames": total_frames,
            "fps": fps,
            "raw_candidate_count": len(raw_candidates),
            "merged_event_count": len(merged_events),
            "raw_candidates": raw_candidates,
            "merged_events": merged_events,
            "event_details": event_details,
            "video_verdict": video_verdict,
            "video_elapsed_seconds": video_elapsed,
            "total_vlm_latency_seconds": round(vlm_time_video, 2),
        })

    total_benchmark_time = round(time.time() - t_start_benchmark, 2)

    # PHASE 2: METRICS & GROUND TRUTH EVALUATION (POST-INFERENCE ONLY)
    print("\n[PHASE 2: EVALUATING METRICS & FAILURE ANALYSIS]")
    gt_map = {row["clip_name"]: str(row["ground_truth"]).lower() for _, row in eval_df.iterrows()}

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    failure_cases = {
        "localization_failure": [],  # e.g., 0 events detected or missed crossing
        "vlm_classification_failure": [],  # event detected correctly but VLM misclassified
        "event_merging_failure": [],
        "other": [],
    }

    per_video_table = []

    for r in video_results:
        clip_name = r["clip_name"]
        gt_label = gt_map.get(clip_name, "unknown")
        pred_label = r["video_verdict"].lower()

        is_correct = (pred_label == gt_label)
        
        if gt_label == "jaywalking" and pred_label == "jaywalking":
            tp += 1
        elif gt_label == "compliant" and pred_label == "compliant":
            tn += 1
        elif gt_label == "compliant" and pred_label == "jaywalking":
            fp += 1
        elif gt_label == "jaywalking" and pred_label == "compliant":
            fn += 1

        if not is_correct:
            if r["merged_event_count"] == 0:
                failure_cases["localization_failure"].append({
                    "clip": clip_name, "gt": gt_label, "pred": pred_label, "reason": "Zero events detected by CV stage."
                })
            else:
                failure_cases["vlm_classification_failure"].append({
                    "clip": clip_name, "gt": gt_label, "pred": pred_label, "reason": f"VLM classified event as {pred_label.upper()}."
                })

        per_video_table.append({
            "clip_name": clip_name,
            "ground_truth": gt_label,
            "pred_verdict": pred_label,
            "correct": is_correct,
            "raw_candidates": r["raw_candidate_count"],
            "merged_events": r["merged_event_count"],
            "elapsed_s": r["video_elapsed_seconds"],
        })

    # Calculations
    acc = round((tp + tn) / total_videos * 100, 2) if total_videos > 0 else 0.0
    prec = round(tp / (tp + fp) * 100, 2) if (tp + fp) > 0 else 0.0
    rec = round(tp / (tp + fn) * 100, 2) if (tp + fn) > 0 else 0.0
    spec = round(tn / (tn + fp) * 100, 2) if (tn + fp) > 0 else 0.0
    f1 = round(2 * prec * rec / (prec + rec), 2) if (prec + rec) > 0 else 0.0

    total_raw_cands = sum(r["raw_candidate_count"] for r in video_results)
    total_merged_events = sum(r["merged_event_count"] for r in video_results)
    zero_event_videos = [r["clip_name"] for r in video_results if r["merged_event_count"] == 0]
    multi_event_videos = [r["clip_name"] for r in video_results if r["merged_event_count"] > 1]

    # Print Full Table
    print("\n" + "=" * 80)
    print("LONG-VIDEO MULTI-EVENT VLM BASELINE PERFORMANCE TABLE")
    print("=" * 80)
    print(f"{'Clip Name':<18} | {'GT':<11} | {'Pipeline Pred':<14} | {'Correct':<8} | {'Raw/Merged':<10} | {'Latency':<7}")
    print("-" * 80)
    for row in per_video_table:
        corr_str = "YES" if row["correct"] else "NO"
        cands_str = f"{row['raw_candidates']}/{row['merged_events']}"
        print(f"{row['clip_name']:<18} | {row['ground_truth']:<11} | {row['pred_verdict']:<14} | {corr_str:<8} | {cands_str:<10} | {row['elapsed_s']:>5.2f}s")
    print("=" * 80)

    # Print Summary Metrics
    print(f"LONG-VIDEO PIPELINE ACCURACY: {acc}% ({tp+tn}/{total_videos})")
    print(f"Precision:   {prec}%")
    print(f"Recall:      {rec}%")
    print(f"Specificity: {spec}%")
    print(f"F1 Score:    {f1}%")
    print(f"Confusion Matrix: TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    print("-" * 80)
    print(f"EVENT & LOCALIZATION STATS:")
    print(f"  * Total Raw Candidates Extracted:  {total_raw_cands}")
    print(f"  * Total Merged Events Evaluated:  {total_merged_events}")
    print(f"  * Videos with 0 Events Detected:   {len(zero_event_videos)} ({zero_event_videos})")
    print(f"  * Videos with Multiple Events:     {len(multi_event_videos)} ({multi_event_videos})")
    print("-" * 80)
    print(f"FAILURE BREAKDOWN:")
    print(f"  * Crossing Detector / Localization Failures: {len(failure_cases['localization_failure'])}")
    for f in failure_cases['localization_failure']:
        print(f"     - {f['clip']}: GT={f['gt']}, Pred={f['pred']} ({f['reason']})")
    print(f"  * VLM Classification Failures:             {len(failure_cases['vlm_classification_failure'])}")
    for f in failure_cases['vlm_classification_failure']:
        print(f"     - {f['clip']}: GT={f['gt']}, Pred={f['pred']} ({f['reason']})")
    print(f"  * Total Benchmark Execution Time:          {total_benchmark_time}s (avg {total_benchmark_time/total_videos:.2f}s/video)")
    print("=" * 80)

    # Save JSON Results
    out_json_path = "outputs/long_video_vlm_results.json"
    os.makedirs("outputs", exist_ok=True)
    with open(out_json_path, "w") as f:
        json.dump({
            "pipeline": "Long-Video CV Event Detector + Temporal IoU Merging + VLM Baseline (qwen2.5vl:7b)",
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
                "total_raw_candidates": total_raw_cands,
                "total_merged_events": total_merged_events,
                "zero_event_videos_count": len(zero_event_videos),
                "multi_event_videos_count": len(multi_event_videos),
                "total_benchmark_time_seconds": total_benchmark_time,
            },
            "failure_cases": failure_cases,
            "video_results": video_results,
        }, f, indent=2)

    print(f"Saved machine-readable long-video results to: {out_json_path}")

    # Append Experiment 14 to RESEARCH_LOG.md
    log_entry = f"""

## Experiment 14 — Long-Video Multi-Event VLM Baseline Evaluation (39 Clips)
* **Date:** 2026-08-18
* **Model:** `qwen2.5vl:7b` via `FullVideoVLMDetector` (VLM Baseline)
* **Crossing Event Localization:** YOLO11x + ByteTrack tracking with generic Temporal IoU event merging (`tIoU >= 0.35` / `relative overlap >= 0.50`)
* **Pipeline Architecture:**
  $$\\text{{Original Long Video}} \\longrightarrow \\text{{CV Crossing Detector}} \\longrightarrow \\text{{Temporal Event Merging}} \\longrightarrow \\text{{VLM Baseline (1 per Event)}} \\longrightarrow \\text{{Video Aggregation}}$$
* **Dataset:** 39 JAAD Original Video Clips (`data/ground_truth.csv`)
* **Empirical Results:**
  - **Accuracy:** **{acc}%** ({tp+tn}/{total_videos})
  - **Precision:** **{prec}%**
  - **Recall:** **{rec}%**
  - **Specificity:** **{spec}%**
  - **F1 Score:** **{f1}%**
  - **Confusion Matrix:** $\\text{{TP}}={tp}, \\text{{TN}}={tn}, \\text{{FP}}={fp}, \\text{{FN}}={fn}$
* **Localization & Event Statistics:**
  - Total Raw Candidates Extracted: {total_raw_cands}
  - Total Merged Events Evaluated: {total_merged_events}
  - Videos with 0 Events Detected: {len(zero_event_videos)}
  - Videos with Multiple Merged Events: {len(multi_event_videos)}
* **Failure Analysis Breakdown:**
  - Crossing Detector / Localization Failures: {len(failure_cases['localization_failure'])} clips (e.g., 0 events detected)
  - VLM Classification Failures: {len(failure_cases['vlm_classification_failure'])} clips (event localized cleanly, but VLM misclassified right-of-way)
* **Comparison against Baselines:**
  - V1 Keyframe Majority Vote (39 pre-cut short clips): 69.23% Accuracy
  - Standalone 5-Frame VLM Baseline (39 pre-cut short clips): 97.44% Accuracy
  - **Full Long-Video End-to-End Pipeline (39 long clips):** **{acc}%** Accuracy
  - *Evaluation Setup Note:* The 97.44% baseline evaluated pre-cut short clips with pre-localized frame boundaries. The long-video pipeline operates on full un-cut videos, requiring automatic temporal localization of pedestrian crossing events before VLM inference.
"""
    with open("RESEARCH_LOG.md", "a") as f:
        f.write(log_entry)

    print("Updated RESEARCH_LOG.md with Experiment 14 results.")


if __name__ == "__main__":
    run_long_video_benchmark()
