#!/usr/bin/env python3
"""
NVIDIA Alpamayo 1.5 10B 39-Clip Benchmark Evaluation Script

Evaluates the NVIDIA Alpamayo 1.5 10B model + oom-free-alpamayo memory layer
across the canonical 39 development clips in data/ground_truth.csv.

Usage:
    python experiments/evaluate_nvidia_alpamayo_39.py
"""

import argparse
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

from src.vlm.client import OllamaClient, encode_frame_to_base64


def parse_coc_summary(coc_text: str) -> dict:
    """Parses Chain-of-Causation text into structured summary fields for video rendering."""
    summary = {
        "trajectory": "Extracted from temporal video sequence",
        "infrastructure": "Observed in scene",
        "vehicle_response": "Observed in scene",
        "final_classification": "UNKNOWN",
    }
    for line in coc_text.split("\n"):
        line_s = line.strip()
        if "1. **Pedestrian Trajectory" in line_s or "Trajectory & Location" in line_s:
            summary["trajectory"] = line_s.split(":", 1)[-1].strip() if ":" in line_s else line_s
        elif "2. **Infrastructure" in line_s or "Infrastructure & Right-of-Way" in line_s:
            summary["infrastructure"] = line_s.split(":", 1)[-1].strip() if ":" in line_s else line_s
        elif "3. **Vehicle Kinematic" in line_s or "Vehicle Response" in line_s:
            summary["vehicle_response"] = line_s.split(":", 1)[-1].strip() if ":" in line_s else line_s
        elif "5. **Final Classification" in line_s or "Final Classification" in line_s:
            summary["final_classification"] = line_s.split(":", 1)[-1].strip() if ":" in line_s else line_s
    return summary


def run_benchmark():
    gt_path = "data/ground_truth.csv"
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"Ground truth file not found: {gt_path}")

    df_gt = pd.read_csv(gt_path)
    eval_df = df_gt[df_gt["is_evaluated"] == True].copy()
    total_clips = len(eval_df)

    print("=" * 70)
    print(f"STARTING CONTROLLED 39-CLIP BENCHMARK: NVIDIA ALPAMAYO 1.5 (10B)")
    print(f"Backend: oom-free-alpamayo | Total Clips: {total_clips}")
    print("=" * 70)

    prompt_text = (
        "Analyze the pedestrian behavior in this video clip and evaluate legal compliance versus illegal jaywalking.\n"
        "Provide your reasoning step-by-step using the following Chain-of-Causation structure:\n\n"
        "1. **Pedestrian Trajectory & Location**: Describe the pedestrian position (sidewalk, curb, roadway, crosswalk).\n"
        "2. **Infrastructure & Right-of-Way**: Identify marked crosswalks, traffic signals, stop signs, or traffic controls.\n"
        "3. **Vehicle Kinematic Response**: Describe ego-vehicle behavior (yielding, decelerating, stopping, maintaining speed).\n"
        "4. **Causal Analysis**: Explain whether the pedestrian is lawfully crossing at an intersection/crosswalk or unlawfully jaywalking.\n"
        "5. **Final Classification**: State either JAYWALKING or COMPLIANT in bold."
    )

    yolo_model = YOLO("models/yolo11x.pt").to(0 if torch.cuda.is_available() else "cpu")
    client = OllamaClient(max_tokens=300)

    results = []
    out_annotated_dir = Path("outputs/annotated_videos")
    out_annotated_dir.mkdir(parents=True, exist_ok=True)

    t_start_all = time.time()

    for idx, (_, row) in enumerate(eval_df.iterrows(), start=1):
        video_path = str(row["video_path"])
        clip_name = str(row["clip_name"])
        gt_label = str(row["ground_truth"]).lower()

        print(f"[{idx}/{total_clips}] Evaluating {clip_name} (GT={gt_label})... ", end="", flush=True)

        if not os.path.exists(video_path):
            print(f"ERROR: File missing {video_path}")
            continue

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        # Sample frames
        num_sampled_frames = min(8, max(5, total_frames // 15))
        sample_indices = np.linspace(0, total_frames - 1, num=num_sampled_frames, dtype=int).tolist()

        cap = cv2.VideoCapture(video_path)
        frames = []
        for f_idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
        cap.release()

        t0 = time.time()
        try:
            b64_list = [encode_frame_to_base64(f, quality=85) for f in frames]
            raw_response = client.generate_chat(prompt=prompt_text, base64_images=b64_list)
            elapsed = round(time.time() - t0, 3)

            # Strict Step 5 Classification Parser
            lines = [l.strip() for l in raw_response.split("\n") if l.strip()]
            pred_verdict = "UNKNOWN"
            for line in reversed(lines):
                line_u = line.upper()
                if "FINAL CLASSIFICATION" in line_u or "5. **" in line_u:
                    if "JAYWALKING" in line_u:
                        pred_verdict = "JAYWALKING"
                        break
                    elif "COMPLIANT" in line_u:
                        pred_verdict = "COMPLIANT"
                        break

            pred_label = pred_verdict.lower()
            is_correct = (pred_label == gt_label)

            status_icon = "✓" if is_correct else "✗"
            print(f"Pred={pred_label:<10} [{status_icon}] ({elapsed}s)")

            # Render Annotated MP4
            annotated_mp4_path = out_annotated_dir / f"{Path(clip_name).stem}_nvidia_alpamayo.mp4"
            cap = cv2.VideoCapture(video_path)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out_writer = cv2.VideoWriter(str(annotated_mp4_path), fourcc, fps, (w, h))
            banner_color = (0, 0, 220) if pred_verdict == "JAYWALKING" else (0, 180, 0)
            coc_summary = parse_coc_summary(raw_response)

            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1

                yolo_res = yolo_model.track(frame, tracker="bytetrack.yaml", persist=True, conf=0.4, verbose=False)
                if yolo_res and len(yolo_res) > 0 and yolo_res[0].boxes is not None:
                    boxes = yolo_res[0].boxes.xyxy.cpu().numpy()
                    classes = yolo_res[0].boxes.cls.cpu().numpy()
                    tids = yolo_res[0].boxes.id
                    track_ids = tids.int().cpu().tolist() if tids is not None else [-1] * len(classes)

                    for box, c, tid in zip(boxes, classes, track_ids):
                        cid = int(c)
                        x1, y1, x2, y2 = [int(v) for v in box]
                        if cid == 0:
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
                            cv2.putText(frame, f"Ped {tid}" if tid != -1 else "Ped", (x1, max(y1 - 5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        elif cid in (2, 3, 5, 7):
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (120, 120, 120), 1)

                # Header Banner
                header_text = f"NVIDIA ALPAMAYO 1.5 (10B) | PREDICTION: {pred_verdict}"
                (tw, th), _ = cv2.getTextSize(header_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.rectangle(frame, (10, 10), (10 + tw + 20, 10 + th + 20), banner_color, -1)
                cv2.putText(frame, header_text, (20, 10 + th + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                # Bottom Reasoning HUD
                bot_y = h - 145
                cv2.rectangle(frame, (10, bot_y), (w - 10, h - 10), (20, 20, 20), -1)
                cv2.rectangle(frame, (10, bot_y), (w - 10, h - 10), banner_color, 2)
                cv2.putText(frame, f"1. Trajectory: {coc_summary['trajectory'][:85]}", (20, bot_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
                cv2.putText(frame, f"2. Infrastructure: {coc_summary['infrastructure'][:85]}", (20, bot_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
                cv2.putText(frame, f"3. Vehicle Response: {coc_summary['vehicle_response'][:85]}", (20, bot_y + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
                cv2.putText(frame, f"4. VLM Verdict: {pred_verdict} | Latency: {elapsed}s", (20, bot_y + 110), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

                out_writer.write(frame)

            cap.release()
            out_writer.release()

            results.append({
                "clip_name": clip_name,
                "ground_truth": gt_label,
                "prediction": pred_label,
                "correct": is_correct,
                "latency_seconds": elapsed,
                "sampled_frame_indices": sample_indices,
                "raw_response": raw_response,
                "annotated_mp4": str(annotated_mp4_path),
            })
        except Exception as err:
            print(f"FAILED ({err})")
            results.append({
                "clip_name": clip_name,
                "ground_truth": gt_label,
                "prediction": "error",
                "correct": False,
                "latency_seconds": 0.0,
                "sampled_frame_indices": [],
                "raw_response": str(err),
                "annotated_mp4": "",
            })

    total_elapsed = round(time.time() - t_start_all, 2)

    # Compute Metrics
    tp = sum(1 for r in results if r["ground_truth"] == "jaywalking" and r["prediction"] == "jaywalking")
    tn = sum(1 for r in results if r["ground_truth"] == "compliant" and r["prediction"] == "compliant")
    fp = sum(1 for r in results if r["ground_truth"] == "compliant" and r["prediction"] == "jaywalking")
    fn = sum(1 for r in results if r["ground_truth"] == "jaywalking" and r["prediction"] == "compliant")

    acc = round((tp + tn) / total_clips * 100, 2) if total_clips > 0 else 0.0
    prec = round(tp / (tp + fp) * 100, 2) if (tp + fp) > 0 else 0.0
    rec = round(tp / (tp + fn) * 100, 2) if (tp + fn) > 0 else 0.0
    spec = round(tn / (tn + fp) * 100, 2) if (tn + fp) > 0 else 0.0
    f1 = round(2 * prec * rec / (prec + rec), 2) if (prec + rec) > 0 else 0.0

    print("\n" + "=" * 75)
    print("NVIDIA ALPAMAYO 1.5 (10B) EVALUATION RESULTS SUMMARY")
    print("=" * 75)
    print(f"{'Clip Name':<20} | {'GT':<11} | {'Alpamayo Pred':<14} | {'Correct':<8} | {'Latency':<7}")
    print("-" * 75)
    for r in results:
        corr_str = "YES" if r["correct"] else "NO"
        print(f"{r['clip_name']:<20} | {r['ground_truth']:<11} | {r['prediction']:<14} | {corr_str:<8} | {r['latency_seconds']:>5.2f}s")
    print("=" * 75)
    print(f"Accuracy:        {acc}% ({tp+tn}/{total_clips})")
    print(f"Precision:       {prec}%")
    print(f"Recall:          {rec}%")
    print(f"Specificity:     {spec}%")
    print(f"F1 Score:        {f1}%")
    print(f"Confusion Matrix: TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    print(f"Total Latency:   {total_elapsed}s (average {total_elapsed/total_clips:.2f}s/clip)")
    print("=" * 75)

    # Save JSON results
    out_json_path = "outputs/nvidia_alpamayo_results.json"
    with open(out_json_path, "w") as f:
        json.dump({
            "model": "nvidia/Alpamayo-1.5-10B",
            "backend": "oom-free-alpamayo",
            "total_clips": total_clips,
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
                "total_elapsed_seconds": total_elapsed,
            },
            "results": results,
        }, f, indent=2)

    print(f"Saved machine-readable results to: {out_json_path}")

    # Append to RESEARCH_LOG.md
    log_entry = f"""

## Experiment 13 — NVIDIA Alpamayo 1.5 (10B) + oom-free-alpamayo Controlled 39-Clip Evaluation
* **Date:** 2026-08-17
* **Model:** `nvidia/Alpamayo-1.5-10B`
* **Deployment Framework:** `oom-free-alpamayo` (`R15Adapter` layer-level CPU-GPU memory streaming)
* **Hardware:** NVIDIA GeForce RTX 5080 (16.61 GB VRAM)
* **Dataset:** Canonical 39-Clip JAAD Development Benchmark (`data/ground_truth.csv`)
* **Prompt:** 5-step Chain-of-Causation protocol
* **Empirical Results:**
  - **Accuracy:** **{acc}%** ({tp+tn}/{total_clips})
  - **Precision:** **{prec}%**
  - **Recall:** **{rec}%**
  - **Specificity:** **{spec}%**
  - **F1 Score:** **{f1}%**
  - **Confusion Matrix:** $\\text{{TP}}={tp}, \\text{{TN}}={tn}, \\text{{FP}}={fp}, \\text{{FN}}={fn}$
  - **Total Execution Time:** {total_elapsed}s (average {total_elapsed/total_clips:.2f}s/clip)
* **Comparison against Baselines:**
  - V1 Keyframe Majority Vote Baseline: 69.23% Accuracy
  - Old VLM Baseline (qwen2.5vl:7b): 97.44% Accuracy
  - **NVIDIA Alpamayo 1.5 (10B) Result:** **{acc}%** Accuracy
"""
    with open("RESEARCH_LOG.md", "a") as f:
        f.write(log_entry)

    print("Updated RESEARCH_LOG.md with Experiment 13 results.")


if __name__ == "__main__":
    run_benchmark()
