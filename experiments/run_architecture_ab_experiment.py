#!/usr/bin/env python3
"""
Controlled Architecture A vs Architecture B Experiment (39 Original JAAD Videos)

Isolates the effect of Multi-Event OR Aggregation vs Single-Call Event Envelope:
  - Architecture A: Current Event-Based Pipeline (1 VLM call per merged event, ANY event == JAYWALKING)
  - Architecture B: Single-Call Event Envelope (1 VLM call across [min(F_start), max(F_end)])

Usage:
    python experiments/run_architecture_ab_experiment.py
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


def run_architecture_ab_experiment():
    gt_path = "data/ground_truth.csv"
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"Ground truth missing: {gt_path}")

    df_gt = pd.read_csv(gt_path)
    eval_df = df_gt[df_gt["is_evaluated"] == True].copy()
    total_videos = len(eval_df)

    print("=" * 80)
    print("CONTROLLED EXPERIMENT: ARCHITECTURE A vs ARCHITECTURE B (39 CLIPS)")
    print("  * Architecture A: Multi-Event VLM Calls + ANY JAYWALKING Aggregation")
    print("  * Architecture B: Single-Call Event Envelope [min(F_start), max(F_end)]")
    print(f"Total Videos: {total_videos}")
    print("=" * 80)

    detector = FullVideoVLMDetector()

    # Pre-extract CV events per video to guarantee identical localization
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

    # PHASE 2: INFERENCE FOR ARCHITECTURE A & B (STRICTLY NO GROUND TRUTH ACCESS)
    print("\n[PHASE 2: EXECUTING INFERENCE FOR ARCHITECTURE A & B]")

    results_arch_a = []
    results_arch_b = []

    t_start_all = time.time()

    for idx, (_, row) in enumerate(eval_df.iterrows(), start=1):
        clip_name = str(row["clip_name"])
        v_data = video_events_map[clip_name]
        video_path = v_data["video_path"]
        total_frames = v_data["total_frames"]
        merged_events = v_data["merged"]

        cap = cv2.VideoCapture(video_path)

        # ----------------------------------------------------
        # ARCHITECTURE A: 1 VLM call per merged event + OR logic
        # ----------------------------------------------------
        t0_a = time.time()
        event_details_a = []
        vlm_time_a = 0.0

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
                vlm_time_a += elapsed_vlm

                event_details_a.append({
                    "event_id": m["event_id"],
                    "bounds": [s_f, e_f],
                    "verdict": parsed["prediction"],
                    "reasoning": parsed["chain_of_causation"],
                    "vlm_time": elapsed_vlm,
                })

        has_jw_a = any(e["verdict"].upper() == "JAYWALKING" for e in event_details_a)
        verdict_a = "JAYWALKING" if has_jw_a else "COMPLIANT"
        elapsed_a = round(time.time() - t0_a, 2)

        # ----------------------------------------------------
        # ARCHITECTURE B: Single-Call Event Envelope
        # ----------------------------------------------------
        t0_b = time.time()
        if merged_events:
            env_s = min(m["start_frame"] for m in merged_events)
            env_e = max(m["end_frame"] for m in merged_events)
        else:
            env_s = 1
            env_e = total_frames

        sample_indices_b = np.linspace(env_s, env_e, num=5, dtype=int).tolist()
        frames_b = []
        for f_idx in sample_indices_b:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx - 1)
            ret, frame = cap.read()
            if ret:
                frames_b.append(frame)

        t0_vlm_b = time.time()
        b64_list_b = [encode_frame_to_base64(f, quality=85) for f in frames_b]
        raw_response_b = detector.client.generate_chat(prompt=detector.coc_prompt, base64_images=b64_list_b)
        parsed_b = detector.parse_coc_response(raw_response_b)
        vlm_time_b = round(time.time() - t0_vlm_b, 3)
        verdict_b = parsed_b["prediction"].upper()
        elapsed_b = round(time.time() - t0_b, 2)

        cap.release()

        print(f"  [{idx}/{total_videos}] {clip_name}: Arch A={verdict_a:<10} | Arch B={verdict_b:<10} (Merged Events={len(merged_events)})")

        results_arch_a.append({
            "clip_name": clip_name,
            "verdict": verdict_a,
            "event_details": event_details_a,
            "vlm_time_seconds": round(vlm_time_a, 2),
            "total_elapsed_seconds": elapsed_a,
        })

        results_arch_b.append({
            "clip_name": clip_name,
            "verdict": verdict_b,
            "envelope_bounds": [env_s, env_e],
            "reasoning": parsed_b["chain_of_causation"],
            "vlm_time_seconds": vlm_time_b,
            "total_elapsed_seconds": elapsed_b,
        })

    # PHASE 3: EVALUATION & COMPARATIVE ANALYSIS (POST-INFERENCE ONLY)
    print("\n[PHASE 3: EVALUATING METRICS & DISAGREEMENT ANALYSIS]")
    gt_map = {row["clip_name"]: str(row["ground_truth"]).lower() for _, row in eval_df.iterrows()}

    def compute_arch_metrics(results):
        tp = tn = fp = fn = 0
        for r in results:
            clip = r["clip_name"]
            gt = gt_map[clip]
            pred = r["verdict"].lower()
            if gt == "jaywalking" and pred == "jaywalking":
                tp += 1
            elif gt == "compliant" and pred == "compliant":
                tn += 1
            elif gt == "compliant" and pred == "jaywalking":
                fp += 1
            elif gt == "jaywalking" and pred == "compliant":
                fn += 1

        acc = round((tp + tn) / total_videos * 100, 2)
        prec = round(tp / (tp + fp) * 100, 2) if (tp + fp) > 0 else 0.0
        rec = round(tp / (tp + fn) * 100, 2) if (tp + fn) > 0 else 0.0
        spec = round(tn / (tn + fp) * 100, 2) if (tn + fp) > 0 else 0.0
        f1 = round(2 * prec * rec / (prec + rec), 2) if (prec + rec) > 0 else 0.0
        tot_t = round(sum(r["total_elapsed_seconds"] for r in results), 2)
        avg_t = round(tot_t / total_videos, 2)
        return {
            "accuracy": acc, "precision": prec, "recall": rec, "specificity": spec, "f1": f1,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn, "total_time": tot_t, "avg_time": avg_t,
        }

    m_a = compute_arch_metrics(results_arch_a)
    m_b = compute_arch_metrics(results_arch_b)

    # Disagreements Analysis
    disagreements = []
    for r_a, r_b in zip(results_arch_a, results_arch_b):
        clip = r_a["clip_name"]
        gt = gt_map[clip]
        v_a = r_a["verdict"]
        v_b = r_b["verdict"]
        if v_a != v_b:
            disagreements.append({
                "clip_name": clip,
                "ground_truth": gt,
                "merged_event_count": len(video_events_map[clip]["merged"]),
                "merged_event_bounds": [[m["start_frame"], m["end_frame"]] for m in video_events_map[clip]["merged"]],
                "event_predictions": [e["verdict"] for e in r_a["event_details"]],
                "arch_a_verdict": v_a,
                "arch_b_verdict": v_b,
            })

    # Print Direct Comparison Table
    print("\n" + "=" * 85)
    print("DIRECT ARCHITECTURE COMPARISON TABLE (39 ORIGINAL JAAD VIDEOS)")
    print("=" * 85)
    print(f"{'Architecture':<40} | {'Accuracy':<9} | {'F1':<8} | {'Recall':<8} | {'Specificity':<11} | {'Total Time':<10} | {'Avg/Vid':<7}")
    print("-" * 85)
    print(f"{'Arch A (Multi-Event VLM + OR Logic)':<40} | {m_a['accuracy']:>7.2f}% | {m_a['f1']:>6.2f}% | {m_a['recall']:>6.2f}% | {m_a['specificity']:>9.2f}% | {m_a['total_time']:>8.2f}s | {m_a['avg_time']:>5.2f}s")
    print(f"{'Arch B (Single-Call Event Envelope)':<40} | {m_b['accuracy']:>7.2f}% | {m_b['f1']:>6.2f}% | {m_b['recall']:>6.2f}% | {m_b['specificity']:>9.2f}% | {m_b['total_time']:>8.2f}s | {m_b['avg_time']:>5.2f}s")
    print("=" * 85)

    print("\nCONFUSION MATRICES:")
    print(f"  * Architecture A: TP={m_a['tp']}, TN={m_a['tn']}, FP={m_a['fp']}, FN={m_a['fn']} | Accuracy={m_a['accuracy']}%")
    print(f"  * Architecture B: TP={m_b['tp']}, TN={m_b['tn']}, FP={m_b['fp']}, FN={m_b['fn']} | Accuracy={m_b['accuracy']}%")

    print(f"\nDISAGREEMENTS BETWEEN ARCHITECTURE A AND ARCHITECTURE B ({len(disagreements)} clips):")
    print("-" * 85)
    for d in disagreements:
        print(f"  * {d['clip_name']}: GT={d['ground_truth']} | Arch A={d['arch_a_verdict']} vs Arch B={d['arch_b_verdict']} | Events={d['merged_event_count']} (Preds: {d['event_predictions']})")

    # Save JSON Output
    out_json_path = "outputs/architecture_ab_experiment_results.json"
    os.makedirs("outputs", exist_ok=True)
    with open(out_json_path, "w") as f:
        json.dump({
            "experiment": "Controlled Architecture A vs Architecture B Experiment across 39 Original JAAD Videos",
            "total_videos": total_videos,
            "metrics": {
                "architecture_a": m_a,
                "architecture_b": m_b,
            },
            "disagreements": disagreements,
            "results_arch_a": results_arch_a,
            "results_arch_b": results_arch_b,
        }, f, indent=2)

    print(f"\nSaved machine-readable architecture results to: {out_json_path}")

    # Append Experiment 17 to RESEARCH_LOG.md
    log_entry = f"""

## Experiment 17 — Controlled Architecture A vs Architecture B Experiment (39 Clips)
* **Date:** 2026-08-18
* **Model:** `qwen2.5vl:7b` via `FullVideoVLMDetector` (VLM Baseline)
* **Research Question:** Are multiple VLM calls + OR aggregation responsible for the accuracy degradation from 97.44%?
* **Experimental Comparison:**
  - **Architecture A (Current Event-Based Pipeline):** 1 VLM call per merged event, `ANY event == JAYWALKING` OR aggregation.
  - **Architecture B (Single-Call Event Envelope):** 1 VLM call over the entire event envelope $[\min(F_{{\\text{{start}}}}), \max(F_{{\\text{{end}}}})]$ with 5 uniform frames.

| Architecture | Accuracy | F1 Score | Recall | Specificity | Precision | TP | TN | FP | FN | Total Time | Avg/Video |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Arch A (Multi-Event VLM + OR Logic)** | **{m_a['accuracy']}%** | **{m_a['f1']}%** | {m_a['recall']}% | {m_a['specificity']}% | {m_a['precision']}% | {m_a['tp']} | {m_a['tn']} | {m_a['fp']} | {m_a['fn']} | {m_a['total_time']}s | {m_a['avg_time']}s |
| **Arch B (Single-Call Event Envelope)** | **{m_b['accuracy']}%** | **{m_b['f1']}%** | {m_b['recall']}% | {m_b['specificity']}% | {m_b['precision']}% | {m_b['tp']} | {m_b['tn']} | {m_b['fp']} | {m_b['fn']} | {m_b['total_time']}s | {m_b['avg_time']}s |

* **Disagreements:** {len(disagreements)} clips out of 39 differed between Architecture A and Architecture B.
* **Empirical Conclusion:**
  - Isolating Architecture A vs Architecture B reveals whether OR aggregation across multi-events introduces False Positive noise compared to a single unified envelope call.
"""
    with open("RESEARCH_LOG.md", "a") as f:
        f.write(log_entry)

    print("Updated RESEARCH_LOG.md with Experiment 17 results.")


if __name__ == "__main__":
    run_architecture_ab_experiment()
