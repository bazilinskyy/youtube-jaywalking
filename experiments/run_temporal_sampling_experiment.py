#!/usr/bin/env python3
"""
Controlled Temporal-Sampling Experiment across 39 Original JAAD Videos

Evaluates three fixed sampling strategies over automatically detected event intervals [F_start, F_end]:
  - Strategy A: 5-frame uniform (baseline)
  - Strategy B: 10-frame uniform (denser)
  - Strategy C: 5-frame center-focused (middle 50% event interval)

Usage:
    python experiments/run_temporal_sampling_experiment.py
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


def get_sampled_frame_indices(start_frame: int, end_frame: int, strategy: str) -> list[int]:
    """Generates deterministic frame sampling indices for a given event interval."""
    if strategy == "strategy_a_5_uniform":
        return np.linspace(start_frame, end_frame, num=5, dtype=int).tolist()
    elif strategy == "strategy_b_10_uniform":
        return np.linspace(start_frame, end_frame, num=10, dtype=int).tolist()
    elif strategy == "strategy_c_5_center":
        # Fixed Rule: Sample 5 frames across the middle 50% interval [F_start + 0.25*range, F_start + 0.75*range]
        frame_range = end_frame - start_frame
        if frame_range <= 4:
            return np.linspace(start_frame, end_frame, num=5, dtype=int).tolist()
        c_start = int(start_frame + 0.25 * frame_range)
        c_end = int(start_frame + 0.75 * frame_range)
        return np.linspace(c_start, c_end, num=5, dtype=int).tolist()
    else:
        raise ValueError(f"Unknown sampling strategy: {strategy}")


def run_sampling_experiment():
    gt_path = "data/ground_truth.csv"
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"Ground truth missing: {gt_path}")

    df_gt = pd.read_csv(gt_path)
    eval_df = df_gt[df_gt["is_evaluated"] == True].copy()
    total_videos = len(eval_df)

    print("=" * 80)
    print("STARTING CONTROLLED TEMPORAL-SAMPLING EXPERIMENT (39 LONG VIDEOS)")
    print("Evaluating Strategies:")
    print("  1. Strategy A: 5-frame uniform across [F_start, F_end]")
    print("  2. Strategy B: 10-frame uniform across [F_start, F_end]")
    print("  3. Strategy C: 5-frame center-focused across middle 50% [F_start, F_end]")
    print(f"Total Videos: {total_videos}")
    print("=" * 80)

    detector = FullVideoVLMDetector()
    strategies = [
        ("strategy_a_5_uniform", "5-frame uniform"),
        ("strategy_b_10_uniform", "10-frame uniform"),
        ("strategy_c_5_center", "5-frame center-focused"),
    ]

    # Pre-extract CV events per video to guarantee identical localization across all strategies
    print("\n[PHASE 1: CV EVENT LOCALIZATION & TEMPORAL MERGING]")
    video_events_map = {}
    for idx, (_, row) in enumerate(eval_df.iterrows(), start=1):
        video_path = str(row["video_path"])
        clip_name = str(row["clip_name"])
        total_frames, fps, duration, cands = extract_candidate_events(video_path)
        merged = merge_overlapping_events(cands, fps=fps)
        video_events_map[clip_name] = {
            "video_path": video_path,
            "total_frames": total_frames,
            "fps": fps,
            "duration": duration,
            "candidates": cands,
            "merged": merged,
        }
        print(f"  [{idx}/{total_videos}] {clip_name}: Candidates={len(cands)}, Merged={len(merged)}")

    all_strategy_results = {}

    # PHASE 2: INFERENCE PER STRATEGY (STRICTLY NO GROUND TRUTH ACCESS)
    print("\n[PHASE 2: EXECUTING INFERENCE FOR ALL THREE SAMPLING STRATEGIES]")
    for strat_key, strat_label in strategies:
        print("\n" + "-" * 75)
        print(f"RUNNING STRATEGY: {strat_label.upper()} ({strat_key})")
        print("-" * 75)

        t_start_strat = time.time()
        strat_video_results = []

        for idx, (_, row) in enumerate(eval_df.iterrows(), start=1):
            clip_name = str(row["clip_name"])
            v_data = video_events_map[clip_name]
            video_path = v_data["video_path"]
            merged_events = v_data["merged"]

            t0_v = time.time()
            cap = cv2.VideoCapture(video_path)
            event_details = []
            vlm_time_v = 0.0

            if merged_events:
                for m in merged_events:
                    s_f = m["start_frame"]
                    e_f = m["end_frame"]
                    sample_indices = get_sampled_frame_indices(s_f, e_f, strat_key)

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
                    vlm_time_v += elapsed_vlm

                    event_details.append({
                        "event_id": m["event_id"],
                        "track_ids": m["track_ids"],
                        "start_frame": s_f,
                        "end_frame": e_f,
                        "start_timestamp": m["start_timestamp"],
                        "end_timestamp": m["end_timestamp"],
                        "duration": m["duration"],
                        "sampled_indices": sample_indices,
                        "verdict": parsed["prediction"],
                        "reasoning": parsed["chain_of_causation"],
                        "inference_time": elapsed_vlm,
                    })
            cap.release()

            # Aggregation: ANY event == JAYWALKING -> video = JAYWALKING, else COMPLIANT
            has_jaywalking = any(e["verdict"].upper() == "JAYWALKING" for e in event_details)
            video_verdict = "JAYWALKING" if has_jaywalking else "COMPLIANT"
            v_elapsed = round(time.time() - t0_v, 2)

            print(f"  [{idx}/{total_videos}] {clip_name}: Verdict={video_verdict:<10} (VLM={vlm_time_v:.2f}s, Total={v_elapsed}s)")

            strat_video_results.append({
                "clip_name": clip_name,
                "video_path": video_path,
                "duration_seconds": v_data["duration"],
                "raw_candidate_count": len(v_data["candidates"]),
                "merged_event_count": len(merged_events),
                "event_details": event_details,
                "video_verdict": video_verdict,
                "vlm_time_seconds": round(vlm_time_v, 2),
                "total_elapsed_seconds": v_elapsed,
            })

        total_strat_time = round(time.time() - t_start_strat, 2)
        all_strategy_results[strat_key] = {
            "label": strat_label,
            "total_strat_time": total_strat_time,
            "video_results": strat_video_results,
        }

    # PHASE 3: EVALUATION & METRICS CALCULATION (AFTER INFERENCE)
    print("\n[PHASE 3: EVALUATING METRICS & COMPARATIVE ANALYSIS]")
    gt_map = {row["clip_name"]: str(row["ground_truth"]).lower() for _, row in eval_df.iterrows()}

    summary_metrics = {}

    for strat_key, strat_label in strategies:
        res_data = all_strategy_results[strat_key]
        video_results = res_data["video_results"]

        tp = 0
        tn = 0
        fp = 0
        fn = 0
        misclassified = []

        for r in video_results:
            clip_name = r["clip_name"]
            gt_label = gt_map[clip_name]
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
                misclassified.append({
                    "clip_name": clip_name,
                    "ground_truth": gt_label,
                    "prediction": pred_label,
                    "merged_event_count": r["merged_event_count"],
                    "events": [
                        {"event_id": e["event_id"], "bounds": [e["start_frame"], e["end_frame"]], "verdict": e["verdict"]}
                        for e in r["event_details"]
                    ],
                })

        acc = round((tp + tn) / total_videos * 100, 2)
        prec = round(tp / (tp + fp) * 100, 2) if (tp + fp) > 0 else 0.0
        rec = round(tp / (tp + fn) * 100, 2) if (tp + fn) > 0 else 0.0
        spec = round(tn / (tn + fp) * 100, 2) if (tn + fp) > 0 else 0.0
        f1 = round(2 * prec * rec / (prec + rec), 2) if (prec + rec) > 0 else 0.0
        tot_time = res_data["total_strat_time"]
        avg_time = round(tot_time / total_videos, 2)

        summary_metrics[strat_key] = {
            "label": strat_label,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "specificity": spec,
            "f1": f1,
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "total_time_seconds": tot_time,
            "avg_time_per_video_seconds": avg_time,
            "misclassified_count": len(misclassified),
            "misclassified": misclassified,
        }

    # Print Direct Comparison Table
    print("\n" + "=" * 85)
    print("CONTROLLED TEMPORAL-SAMPLING COMPARISON TABLE (39 ORIGINAL JAAD VIDEOS)")
    print("=" * 85)
    print(f"{'Strategy':<25} | {'Accuracy':<9} | {'F1':<8} | {'Recall':<8} | {'Specificity':<11} | {'Total Time':<10} | {'Avg/Vid':<7}")
    print("-" * 85)
    for strat_key, strat_label in strategies:
        m = summary_metrics[strat_key]
        print(f"{m['label']:<25} | {m['accuracy']:>7.2f}% | {m['f1']:>6.2f}% | {m['recall']:>6.2f}% | {m['specificity']:>9.2f}% | {m['total_time_seconds']:>8.2f}s | {m['avg_time_per_video_seconds']:>5.2f}s")
    print("=" * 85)

    # Detailed Confusion Matrix breakdown
    print("\nCONFUSION MATRIX BREAKDOWN:")
    for strat_key, strat_label in strategies:
        m = summary_metrics[strat_key]
        print(f"  * {m['label']:<24}: TP={m['tp']:<2}, TN={m['tn']:<2}, FP={m['fp']:<2}, FN={m['fn']:<2} | Acc={m['accuracy']}%")

    # Localization Stats
    zero_event_vids = [k for k, v in video_events_map.items() if len(v["merged"]) == 0]
    one_event_vids = [k for k, v in video_events_map.items() if len(v["merged"]) == 1]
    multi_event_vids = [k for k, v in video_events_map.items() if len(v["merged"]) > 1]

    print("\nLOCALIZATION & EVENT DISTRIBUTION:")
    print(f"  * Videos with 0 detected events: {len(zero_event_vids)} ({zero_event_vids})")
    print(f"  * Videos with 1 merged event:    {len(one_event_vids)}")
    print(f"  * Videos with multiple events:   {len(multi_event_vids)} ({multi_event_vids})")

    # Save JSON Output
    out_json_path = "outputs/temporal_sampling_experiment_results.json"
    os.makedirs("outputs", exist_ok=True)
    with open(out_json_path, "w") as f:
        json.dump({
            "experiment": "Controlled Temporal-Sampling Experiment across 39 Original JAAD Videos",
            "model": "qwen2.5vl:7b",
            "total_videos": total_videos,
            "event_distribution": {
                "zero_event_videos": zero_event_vids,
                "one_event_videos_count": len(one_event_vids),
                "multi_event_videos": multi_event_vids,
            },
            "summary_metrics": summary_metrics,
            "all_strategy_results": all_strategy_results,
        }, f, indent=2)

    print(f"\nSaved machine-readable sampling results to: {out_json_path}")

    # Append Experiment 15 to RESEARCH_LOG.md
    m_a = summary_metrics["strategy_a_5_uniform"]
    m_b = summary_metrics["strategy_b_10_uniform"]
    m_c = summary_metrics["strategy_c_5_center"]

    log_entry = f"""

## Experiment 15 — Controlled Temporal-Sampling Experiment (39 Original JAAD Videos)
* **Date:** 2026-08-18
* **Model:** `qwen2.5vl:7b` via `FullVideoVLMDetector` (VLM Baseline)
* **Research Question:** Is temporal frame sampling the cause of the VLM performance drop from 97.44% (pre-cut short clips) to 61.54% (long videos)?
* **Experimental Protocol:** Evaluated three fixed, zero-leakage sampling strategies across the automatically detected event intervals $[F_{{\\text{{start}}}}, F_{{\\text{{end}}}}]$ for all 39 original JAAD videos:
  1. **Strategy A (5-frame uniform):** 5 frames spaced evenly across $[F_{{\\text{{start}}}}, F_{{\\text{{end}}}}]$.
  2. **Strategy B (10-frame uniform):** 10 frames spaced evenly across $[F_{{\\text{{start}}}}, F_{{\\text{{end}}}}]$.
  3. **Strategy C (5-frame center-focused):** 5 frames centered around the middle 50% interval of $[F_{{\\text{{start}}}}, F_{{\\text{{end}}}}]$.
* **Empirical Results Comparison:**

| Strategy | Accuracy | F1 Score | Recall | Specificity | Precision | TP | TN | FP | FN | Total Time | Avg/Video |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **5-frame uniform** | **{m_a['accuracy']}%** | **{m_a['f1']}%** | {m_a['recall']}% | {m_a['specificity']}% | {m_a['precision']}% | {m_a['tp']} | {m_a['tn']} | {m_a['fp']} | {m_a['fn']} | {m_a['total_time_seconds']}s | {m_a['avg_time_per_video_seconds']}s |
| **10-frame uniform** | **{m_b['accuracy']}%** | **{m_b['f1']}%** | {m_b['recall']}% | {m_b['specificity']}% | {m_b['precision']}% | {m_b['tp']} | {m_b['tn']} | {m_b['fp']} | {m_b['fn']} | {m_b['total_time_seconds']}s | {m_b['avg_time_per_video_seconds']}s |
| **5-frame center-focused** | **{m_c['accuracy']}%** | **{m_c['f1']}%** | {m_c['recall']}% | {m_c['specificity']}% | {m_c['precision']}% | {m_c['tp']} | {m_c['tn']} | {m_c['fp']} | {m_c['fn']} | {m_c['total_time_seconds']}s | {m_c['avg_time_per_video_seconds']}s |

* **Key Findings & Diagnosis:**
  - Comparing 5-frame uniform vs 10-frame uniform vs 5-frame center-focused empirically isolates the impact of frame density and temporal centering during long-video inference.
  - The results demonstrate whether temporal frame selection accounts for the performance gap between short pre-cut clips (97.44%) and long original videos.
"""
    with open("RESEARCH_LOG.md", "a") as f:
        f.write(log_entry)

    print("Updated RESEARCH_LOG.md with Experiment 15 results.")


if __name__ == "__main__":
    run_sampling_experiment()
