#!/usr/bin/env python3
"""
Experiment 20: Classical CV Crosswalk Context + Architecture B

Integrates classical CrosswalkDetector (HSV + morphology + stripe pattern)
into long-video Architecture B pipeline.

Workflow:
  Raw Video -> YOLO11x + ByteTrack -> Merged Event Envelope -> 5 Uniform Frames ->
  Classical CV CrosswalkDetector on 5 Frames -> Injected CV Context into VLM Prompt ->
  Single VLM Call (qwen2.5vl:7b) -> Verdict.

Usage:
    # Phase 6 Validation Test (Single Video)
    python experiments/run_exp20_cv_crosswalk_experiment.py --validate-single video_0028.mp4

    # Phase 7 Full 39-Video Experiment
    python experiments/run_exp20_cv_crosswalk_experiment.py
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

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.cv.crosswalk_detector import CrosswalkDetector
from src.vlm.alpamayo_detector import FullVideoVLMDetector
from src.vlm.client import encode_frame_to_base64
from scripts.run_long_video_vlm_experiment import extract_candidate_events, merge_overlapping_events


def build_cv_crosswalk_context_prompt(frame_regions_list: list[list]) -> str:
    """Build the explicit classical CV crosswalk context block for VLM prompt."""
    lines = ["\nCLASSICAL CV CROSSWALK CONTEXT:"]
    for idx, regions in enumerate(frame_regions_list, start=1):
        if regions:
            reg_strs = [f"[{r.x1:.4f}, {r.y1:.4f}, {r.x2:.4f}, {r.y2:.4f}] (conf: {r.confidence:.2f})" for r in regions]
            lines.append(f"Frame {idx}:\nDetected 2D Crosswalk Regions:\n" + "\n".join(reg_strs))
        else:
            lines.append(f"Frame {idx}:\nNo 2D crosswalk regions detected by classical CV.")

    lines.append("\nIMPORTANT:")
    lines.append("Treat this as supporting computer-vision evidence, NOT ground truth.")
    lines.append("You must independently inspect the visual frames and determine whether")
    lines.append("the detected regions are actually relevant to the pedestrian's path.\n")
    return "\n".join(lines)


def run_single_video_validation(video_path: str):
    """Phase 6: Validate single video integration end-to-end without ground truth."""
    print("=" * 80)
    print(f"[PHASE 6 VALIDATION] Testing single video end-to-end: {video_path}")
    print("=" * 80)

    if not os.path.exists(video_path):
        # Fallback to local raw clips directory if necessary
        clip_base = os.path.basename(video_path)
        video_path = os.path.join("data/raw_clips", clip_base)

    cw_detector = CrosswalkDetector()
    vlm_detector = FullVideoVLMDetector()

    # 1. Event Detection & Merging
    t_cv0 = time.time()
    total_frames, fps, duration, cands = extract_candidate_events(video_path)
    merged = merge_overlapping_events(cands, fps=fps)

    if merged:
        env_s = min(m["start_frame"] for m in merged)
        env_e = max(m["end_frame"] for m in merged)
    else:
        env_s = 1
        env_e = total_frames

    # 2. 5 Uniform Frames Sampling across Envelope
    raw_indices = np.linspace(env_s, env_e, num=5, dtype=int)
    sample_indices = [min(total_frames, max(1, idx)) for idx in raw_indices]

    cap = cv2.VideoCapture(video_path)
    frames = []
    for f_idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx - 1)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()

    # 3. Classical Crosswalk Detection on 5 Frames
    frame_cw_regions = []
    for f in frames:
        regions = cw_detector.detect(f)
        frame_cw_regions.append(regions)
    cv_time = round(time.time() - t_cv0, 3)

    # 4. Injected CV Context Prompt Construction
    cv_prompt_text = build_cv_crosswalk_context_prompt(frame_cw_regions)
    full_prompt = vlm_detector.coc_prompt + "\n" + cv_prompt_text

    # 5. VLM Inference
    t_vlm0 = time.time()
    b64_list = [encode_frame_to_base64(f, quality=85) for f in frames]
    raw_response = vlm_detector.client.generate_chat(prompt=full_prompt, base64_images=b64_list)
    parsed = vlm_detector.parse_coc_response(raw_response)
    vlm_time = round(time.time() - t_vlm0, 3)

    print("\n1. Detected Event Envelope:")
    print(f"   Frames [{env_s} .. {env_e}] (Total Video Frames: {total_frames}, FPS: {fps})")

    print("\n2. Five Sampled Frame Indices:")
    print(f"   Indices: {sample_indices}")

    print("\n3. Classical CV Crosswalk Detections for Each Sampled Frame:")
    for idx, regs in enumerate(frame_cw_regions, start=1):
        print(f"   Frame {idx} (Index {sample_indices[idx-1]}): {regs if regs else 'No crosswalk detected'}")

    print("\n4. Exact Additional Crosswalk Context Injected into VLM Prompt:")
    print("-" * 60)
    print(cv_prompt_text.strip())
    print("-" * 60)

    print("\n5. Raw VLM Response:")
    print("-" * 60)
    print(raw_response.strip())
    print("-" * 60)

    print("\n6. Parsed VLM Verdict:")
    print(f"   Prediction: {parsed['prediction']}")

    print("\n7. Inference Latency:")
    print(f"   CV Runtime:  {cv_time}s")
    print(f"   VLM Runtime: {vlm_time}s")
    print(f"   Total:       {cv_time + vlm_time:.3f}s")
    print("=" * 80)
    print("Phase 6 Single Video Validation Succeeded!\n")


def run_full_39_experiment():
    gt_path = "data/ground_truth.csv"
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"Ground truth missing: {gt_path}")

    df_gt = pd.read_csv(gt_path)
    eval_df = df_gt[df_gt["is_evaluated"] == True].copy()
    total_videos = len(eval_df)

    print("=" * 80)
    print("EXPERIMENT 20: CLASSICAL CV CROSSWALK CONTEXT + ARCHITECTURE B (39 VIDEOS)")
    print("Zero Ground Truth Access During Inference")
    print(f"Total Videos to Evaluate: {total_videos}")
    print("=" * 80)

    cw_detector = CrosswalkDetector()
    vlm_detector = FullVideoVLMDetector()

    results = []
    tot_cv_time = 0.0
    tot_vlm_time = 0.0

    t_bench_start = time.time()

    # -------------------------------------------------------------------------
    # PHASE 7: INFERENCE FOR ALL 39 VIDEOS (NO GROUND TRUTH ACCESSED HERE)
    # -------------------------------------------------------------------------
    print("\n[PHASE 7: EXECUTING INFERENCE OVER 39 VIDEOS]")
    for idx, (_, row) in enumerate(eval_df.iterrows(), start=1):
        video_path = str(row["video_path"])
        clip_name = str(row["clip_name"])

        # 1. Event Detection & Merging
        t0_cv = time.time()
        total_frames, fps, duration, cands = extract_candidate_events(video_path)
        merged = merge_overlapping_events(cands, fps=fps)

        if merged:
            env_s = min(m["start_frame"] for m in merged)
            env_e = max(m["end_frame"] for m in merged)
        else:
            env_s = 1
            env_e = total_frames

        # 2. 5 Uniform Frames Sampling
        raw_indices = np.linspace(env_s, env_e, num=5, dtype=int)
        sample_indices = [min(total_frames, max(1, f_idx)) for f_idx in raw_indices]

        cap = cv2.VideoCapture(video_path)
        frames = []
        for f_idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx - 1)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
        cap.release()

        # 3. Classical CV Crosswalk Detection
        frame_cw_regions = []
        for f in frames:
            regions = cw_detector.detect(f)
            frame_cw_regions.append(regions)
        cv_elapsed = time.time() - t0_cv
        tot_cv_time += cv_elapsed

        # 4. Injected CV Context Prompt Construction
        cv_prompt_text = build_cv_crosswalk_context_prompt(frame_cw_regions)
        full_prompt = vlm_detector.coc_prompt + "\n" + cv_prompt_text

        # 5. VLM Inference
        t0_vlm = time.time()
        b64_list = [encode_frame_to_base64(f, quality=85) for f in frames]
        raw_response = vlm_detector.client.generate_chat(prompt=full_prompt, base64_images=b64_list)
        parsed = vlm_detector.parse_coc_response(raw_response)
        vlm_elapsed = time.time() - t0_vlm
        tot_vlm_time += vlm_elapsed

        verdict = parsed["prediction"].upper()
        total_elapsed = round(cv_elapsed + vlm_elapsed, 3)

        # Crosswalk counts for this video
        frames_with_cw = sum(1 for regs in frame_cw_regions if len(regs) > 0)
        tot_cw_count = sum(len(regs) for regs in frame_cw_regions)

        print(f"[{idx}/{total_videos}] {clip_name}: Envelope=[{env_s}..{env_e}] | CW Frames={frames_with_cw}/5 | Verdict={verdict:<10} ({total_elapsed}s)")

        results.append({
            "clip_name": clip_name,
            "video_path": video_path,
            "total_frames": total_frames,
            "fps": fps,
            "duration_seconds": duration,
            "envelope_bounds": [env_s, env_e],
            "sampled_indices": sample_indices,
            "crosswalk_regions_per_frame": [
                [{"x1": r.x1, "y1": r.y1, "x2": r.x2, "y2": r.y2, "conf": r.confidence} for r in regs]
                for regs in frame_cw_regions
            ],
            "frames_with_crosswalk_count": frames_with_cw,
            "total_crosswalk_regions_count": tot_cw_count,
            "prediction": verdict,
            "reasoning": parsed["chain_of_causation"],
            "cv_runtime": round(cv_elapsed, 3),
            "vlm_runtime": round(vlm_elapsed, 3),
            "total_runtime": total_elapsed,
        })

    tot_benchmark_time = round(time.time() - t_bench_start, 2)

    # -------------------------------------------------------------------------
    # PHASE 8 & 9: POST-INFERENCE EVALUATION (GROUND TRUTH LOADED HERE ONLY)
    # -------------------------------------------------------------------------
    print("\n[PHASE 8: EVALUATING METRICS AFTER INFERENCE COMPLETION]")
    gt_map = {row["clip_name"]: str(row["ground_truth"]).lower() for _, row in eval_df.iterrows()}

    # Load Architecture B predictions for direct failure transition analysis
    arch_b_json_path = "outputs/architecture_ab_experiment_results.json"
    arch_b_map = {}
    if os.path.exists(arch_b_json_path):
        with open(arch_b_json_path, "r") as f:
            arch_b_data = json.load(f)
            arch_b_map = {r["clip_name"]: r["verdict"].lower() for r in arch_b_data["results_arch_b"]}

    tp = tn = fp = fn = 0
    per_video_eval = []

    vids_with_cw = 0
    vids_without_cw = 0
    tot_detected_cw_regions = 0

    for r in results:
        clip_name = r["clip_name"]
        gt_label = gt_map[clip_name]
        pred_label = r["prediction"].lower()
        arch_b_pred = arch_b_map.get(clip_name, "unknown")

        is_correct = (pred_label == gt_label)

        if gt_label == "jaywalking" and pred_label == "jaywalking":
            tp += 1
        elif gt_label == "compliant" and pred_label == "compliant":
            tn += 1
        elif gt_label == "compliant" and pred_label == "jaywalking":
            fp += 1
        elif gt_label == "jaywalking" and pred_label == "compliant":
            fn += 1

        if r["total_crosswalk_regions_count"] > 0:
            vids_with_cw += 1
        else:
            vids_without_cw += 1

        tot_detected_cw_regions += r["total_crosswalk_regions_count"]

        per_video_eval.append({
            "clip_name": clip_name,
            "ground_truth": gt_label,
            "arch_b_pred": arch_b_pred,
            "exp20_pred": pred_label,
            "correct": is_correct,
            "cw_frames": f"{r['frames_with_crosswalk_count']}/5",
            "total_cw_regions": r["total_crosswalk_regions_count"],
            "cv_time": r["cv_runtime"],
            "vlm_time": r["vlm_runtime"],
            "total_time": r["total_runtime"],
        })

    acc = round((tp + tn) / total_videos * 100, 2)
    prec = round(tp / (tp + fp) * 100, 2) if (tp + fp) > 0 else 0.0
    rec = round(tp / (tp + fn) * 100, 2) if (tp + fn) > 0 else 0.0
    spec = round(tn / (tn + fp) * 100, 2) if (tn + fp) > 0 else 0.0
    f1 = round(2 * prec * rec / (prec + rec), 2) if (prec + rec) > 0 else 0.0
    avg_latency = round(tot_benchmark_time / total_videos, 2)
    avg_cw_per_frame = round(tot_detected_cw_regions / (total_videos * 5), 3)

    # -------------------------------------------------------------------------
    # PHASE 10: FAILURE & TRANSITION ANALYSIS VS ARCHITECTURE B
    # -------------------------------------------------------------------------
    arch_b_corrected = []
    arch_b_degraded = []
    new_fps = []
    new_fns = []
    cv_detected_vlm_rejected = []

    for r in per_video_eval:
        cn = r["clip_name"]
        gt = r["ground_truth"]
        b_p = r["arch_b_pred"]
        e_p = r["exp20_pred"]

        b_corr = (b_p == gt)
        e_corr = (e_p == gt)

        if not b_corr and e_corr:
            arch_b_corrected.append(cn)
        elif b_corr and not e_corr:
            arch_b_degraded.append(cn)

        if gt == "compliant" and e_p == "jaywalking" and b_p == "compliant":
            new_fps.append(cn)
        elif gt == "jaywalking" and e_p == "compliant" and b_p == "jaywalking":
            new_fns.append(cn)

        # Cases where classical CV detected crosswalk regions, but VLM predicted JAYWALKING anyway or vice versa
        if r["total_cw_regions"] > 0 and e_p == "jaywalking":
            cv_detected_vlm_rejected.append(cn)

    # Calculate Deltas against Architecture B (71.79% Acc, 70.00% Prec, 46.67% Rec, 87.50% Spec, 56.00% F1, 2.60s latency)
    b_acc, b_prec, b_rec, b_spec, b_f1, b_lat = 71.79, 70.00, 46.67, 87.50, 56.00, 2.60
    d_acc = round(acc - b_acc, 2)
    d_prec = round(prec - b_prec, 2)
    d_rec = round(rec - b_rec, 2)
    d_spec = round(spec - b_spec, 2)
    d_f1 = round(f1 - b_f1, 2)
    d_lat = round(avg_latency - b_lat, 2)

    # Print Detailed Results
    print("\n" + "=" * 90)
    print("EXPERIMENT 20 RESULTS TABLE: CLASSICAL CV CROSSWALK CONTEXT + ARCHITECTURE B")
    print("=" * 90)
    print(f"{'Clip Name':<16} | {'GT':<10} | {'Arch B':<10} | {'Exp 20':<10} | {'Correct':<7} | {'CW Regions':<10} | {'CV Time':<7} | {'VLM Time':<8}")
    print("-" * 90)
    for r in per_video_eval:
        corr_str = "YES" if r["correct"] else "NO"
        print(f"{r['clip_name']:<16} | {r['ground_truth']:<10} | {r['arch_b_pred']:<10} | {r['exp20_pred']:<10} | {corr_str:<7} | {r['total_cw_regions']:^10} | {r['cv_time']:>6.2f}s | {r['vlm_time']:>7.2f}s")
    print("=" * 90)
    print(f"Accuracy:                             {acc}% ({tp+tn}/{total_videos})")
    print(f"Precision:                            {prec}%")
    print(f"Recall:                               {rec}%")
    print(f"Specificity:                          {spec}%")
    print(f"F1 Score:                             {f1}%")
    print(f"Confusion Matrix:                     TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    print("-" * 90)
    print(f"Videos with Detected Crosswalks:      {vids_with_cw} / {total_videos}")
    print(f"Videos without Detected Crosswalks:   {vids_without_cw} / {total_videos}")
    print(f"Total Crosswalk Regions Detected:     {tot_detected_cw_regions}")
    print(f"Avg Crosswalk Regions / Sampled Frame:{avg_cw_per_frame}")
    print(f"Total Classical CV Runtime:           {round(tot_cv_time, 2)}s (avg {round(tot_cv_time/total_videos, 3)}s/video)")
    print(f"Total VLM Runtime:                    {round(tot_vlm_time, 2)}s (avg {round(tot_vlm_time/total_videos, 2)}s/video)")
    print(f"Total Benchmark Runtime:              {tot_benchmark_time}s (avg {avg_latency}s/video)")
    print("=" * 90)

    # Print Required Phase 9 Comparison Table
    print("\n" + "=" * 115)
    print("PHASE 9 REQUIRED THREE-WAY PIPELINE COMPARISON TABLE")
    print("=" * 115)
    print(f"{'Pipeline':<46} | {'Accuracy':<9} | {'Precision':<9} | {'Recall':<8} | {'Specificity':<11} | {'F1':<7} | {'TP':<3} | {'TN':<3} | {'FP':<3} | {'FN':<3} | {'Avg/video':<9}")
    print("-" * 115)
    print(f"{'Historical short-clip baseline':<46} | {'97.44%':<9} | {'93.75%':<9} | {'100.0%':<8} | {'95.83%':<11} | {'96.77%':<7} | {'15':<3} | {'23':<3} | {'1':<3} | {'0':<3} | {'5.45s':<9}")
    print(f"{'Architecture B, no CV context':<46} | {'71.79%':<9} | {'70.00%':<9} | {'46.67%':<8} | {'87.50%':<11} | {'56.00%':<7} | {'7':<3} | {'21':<3} | {'3':<3} | {'8':<3} | {'2.60s':<9}")
    print(f"{'Architecture B + classical CV crosswalk context':<46} | {f'{acc}%':<9} | {f'{prec}%':<9} | {f'{rec}%':<8} | {f'{spec}%':<11} | {f'{f1}%':<7} | {f'{tp}':<3} | {f'{tn}':<3} | {f'{fp}':<3} | {f'{fn}':<3} | {f'{avg_latency}s':<9}")
    print("=" * 115)
    print(f"ABSOLUTE CHANGE FROM ARCHITECTURE B:")
    print(f"  Δ Accuracy:    {d_acc:+} percentage points ({b_acc}% -> {acc}%)")
    print(f"  Δ Precision:   {d_prec:+} percentage points ({b_prec}% -> {prec}%)")
    print(f"  Δ Recall:      {d_rec:+} percentage points ({b_rec}% -> {rec}%)")
    print(f"  Δ Specificity: {d_spec:+} percentage points ({b_spec}% -> {spec}%)")
    print(f"  Δ F1 Score:    {d_f1:+} percentage points ({b_f1}% -> {f1}%)")
    print(f"  Δ Latency:     {d_lat:+}s/video ({b_lat}s -> {avg_latency}s)")
    print("=" * 115)

    # Save Machine-Readable JSON
    out_json_path = "outputs/exp20_cv_crosswalk_experiment_results.json"
    os.makedirs("outputs", exist_ok=True)
    with open(out_json_path, "w") as f:
        json.dump({
            "experiment": "Experiment 20: Classical CV Crosswalk Context + Architecture B",
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
                "delta_from_arch_b": {
                    "delta_accuracy": d_acc,
                    "delta_precision": d_prec,
                    "delta_recall": d_rec,
                    "delta_specificity": d_spec,
                    "delta_f1": d_f1,
                    "delta_latency_seconds": d_lat,
                },
                "total_cv_runtime_seconds": round(tot_cv_time, 2),
                "total_vlm_runtime_seconds": round(tot_vlm_time, 2),
                "total_benchmark_time_seconds": tot_benchmark_time,
                "avg_latency_per_video_seconds": avg_latency,
            },
            "crosswalk_statistics": {
                "videos_with_detected_crosswalks": vids_with_cw,
                "videos_without_detected_crosswalks": vids_without_cw,
                "total_crosswalk_regions_detected": tot_detected_cw_regions,
                "avg_crosswalk_regions_per_sampled_frame": avg_cw_per_frame,
            },
            "transition_analysis": {
                "arch_b_errors_corrected": arch_b_corrected,
                "arch_b_correct_degraded": arch_b_degraded,
                "new_false_positives": new_fps,
                "new_false_negatives": new_fns,
                "cv_detected_vlm_rejected_jaywalking": cv_detected_vlm_rejected,
            },
            "video_results": results,
        }, f, indent=2, default=lambda o: int(o) if isinstance(o, (np.int64, np.int32)) else float(o) if isinstance(o, (np.float64, np.float32)) else str(o))

    print(f"\nSaved machine-readable Exp 20 results to: {out_json_path}")

    # Append Experiment 20 to RESEARCH_LOG.md
    log_entry = f"""

## Experiment 20 — Classical CV Crosswalk Context + Architecture B (39 Clips)
* **Date:** 2026-08-18
* **Model:** `qwen2.5vl:7b` via `FullVideoVLMDetector` (VLM Baseline)
* **Hypothesis:** Does injecting explicit 2D spatial crosswalk bounding regions detected by classical computer vision (HSV + morphology + stripe pattern) directly into the VLM prompt improve zero-shot right-of-way reasoning and classification accuracy?
* **Experimental Protocol:**
  - Integrated classical `CrosswalkDetector` (`src/cv/crosswalk_detector.py` & `src/cv/crosswalk_utils.py`) into Architecture B.
  - For each video's 5 sampled frames over the unified event envelope, ran classical CV crosswalk detection.
  - Injected explicit `CLASSICAL CV CROSSWALK CONTEXT` into the Chain-of-Causation prompt (`"Detected 2D Crosswalk Regions: [x1, y1, x2, y2] (conf: 0.XX)"` or `"No 2D crosswalk regions detected by classical CV."`).
  - Executed exactly 1 VLM call per video. Zero ground-truth access during inference. Loaded `ground_truth.csv` post-inference.
* **Empirical Results:**
  - **Accuracy:** **{acc}%** ({tp+tn}/{total_videos}) (Δ vs Arch B: **{d_acc:+}** percentage points)
  - **Precision:** **{prec}%** (Δ vs Arch B: **{d_prec:+}** percentage points)
  - **Recall:** **{rec}%** (Δ vs Arch B: **{d_rec:+}** percentage points)
  - **Specificity:** **{spec}%** (Δ vs Arch B: **{d_spec:+}** percentage points)
  - **F1 Score:** **{f1}%** (Δ vs Arch B: **{d_f1:+}** percentage points)
  - **Confusion Matrix:** $\\text{{TP}}={tp}, \\text{{TN}}={tn}, \\text{{FP}}={fp}, \\text{{FN}}={fn}$
  - **Total Latency:** {tot_benchmark_time}s (avg {avg_latency}s/clip; CV avg {round(tot_cv_time/total_videos, 3)}s/clip, VLM avg {round(tot_vlm_time/total_videos, 2)}s/clip)

* **Crosswalk Detection Statistics:**
  - Videos with detected crosswalk regions: {vids_with_cw} / {total_videos}
  - Videos without detected crosswalk regions: {vids_without_cw} / {total_videos}
  - Total crosswalk regions detected across all frames: {tot_detected_cw_regions}
  - Average crosswalk regions per sampled frame: {avg_cw_per_frame}

* **Three-Way Pipeline Comparison Table:**

| Pipeline | Accuracy | Precision | Recall | Specificity | F1 Score | TP | TN | FP | FN | Avg Latency/video |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **1. Historical short-clip baseline** | **97.44%** | **93.75%** | **100.0%** | **95.83%** | **96.77%** | 15 | 23 | 1 | 0 | 5.45s |
| **2. Architecture B, no CV context** | **71.79%** | **70.00%** | **46.67%** | **87.50%** | **56.00%** | 7 | 21 | 3 | 8 | **2.60s** |
| **3. Architecture B + classical CV context (Exp 20)** | **{acc}%** | **{prec}%** | **{rec}%** | **{spec}%** | **{f1}%** | {tp} | {tn} | {fp} | {fn} | {avg_latency}s |

* **Transition & Error Analysis vs Architecture B:**
  - Architecture B errors corrected by crosswalk context: {arch_b_corrected if arch_b_corrected else 'None'}
  - Architecture B correct predictions degraded: {arch_b_degraded if arch_b_degraded else 'None'}
  - New False Positives: {new_fps if new_fps else 'None'}
  - New False Negatives: {new_fns if new_fns else 'None'}
"""
    with open("RESEARCH_LOG.md", "a") as f:
        f.write(log_entry)

    print("Updated RESEARCH_LOG.md with Experiment 20 results.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment 20: Classical CV Crosswalk Context + Architecture B")
    parser.add_argument("--validate-single", type=str, default=None, help="Run single video validation test (Phase 6)")
    args = parser.parse_args()

    if args.validate_single:
        run_single_video_validation(args.validate_single)
    else:
        run_full_39_experiment()
