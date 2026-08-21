#!/usr/bin/env python3
"""
Experiment 22: Improved Event Localization + Responsible Pedestrian Attribution

Refines pedestrian crossing event boundaries by active motion trimming,
preserves constituent pedestrian Track IDs during event merging, and assesses
responsible Track ID attribution following single-call VLM inference.

Workflow:
  Raw Video -> YOLO11x + ByteTrack -> Active Motion Trimming ->
  Track-Preserving Temporal Merging -> Single Event Envelope per Video ->
  5 Uniform Frames -> VLM Baseline (qwen2.5vl:7b) -> Verdict + Track Attribution.

Usage:
    # Phase 6 Validation on 5 Diagnostic Videos
    python experiments/run_exp22_event_localization.py --validate-5

    # Phase 9 Full 39-Video Experiment
    python experiments/run_exp22_event_localization.py
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

from src.vlm.alpamayo_detector import FullVideoVLMDetector
from src.vlm.client import encode_frame_to_base64


def extract_refined_candidate_events(video_path: str, conf_thresh: float = 0.25, stride: int = 2):
    """
    Extracts pedestrian crossing candidate events with active motion trimming and full track history.
    
    Trims leading/trailing stationary curb frames so start_frame corresponds to actual lateral roadway entry.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0, 30.0, 0.0, []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = round(total_frames / fps, 2)

    model = YOLO("models/yolo11x.pt" if os.path.exists("models/yolo11x.pt") else "yolo11x.pt")
    device = 0 if torch.cuda.is_available() else "cpu"

    tracks_x = {}      # tid -> list of normalized cx
    tracks_boxes = {}  # tid -> list of (cx, cy, bw, bh)
    tracks_frames = {} # tid -> list of frame indices

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % stride != 0:
            continue

        results = model.track(
            frame,
            tracker="bytetrack.yaml",
            persist=True,
            classes=[0],  # Person class
            conf=conf_thresh,
            verbose=False,
            device=device,
        )

        if results and len(results) > 0 and results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xywhn.cpu().numpy()
            tids = results[0].boxes.id.int().cpu().tolist()
            for box, tid in zip(boxes, tids):
                cx, cy, bw, bh = box
                if tid not in tracks_x:
                    tracks_x[tid] = []
                    tracks_boxes[tid] = []
                    tracks_frames[tid] = []
                tracks_x[tid].append(cx)
                tracks_boxes[tid].append((cx, cy, bw, bh))
                tracks_frames[tid].append(frame_idx)

    cap.release()

    candidates = []
    cand_count = 0

    for tid, xs in tracks_x.items():
        frames_list = tracks_frames[tid]
        if len(xs) < 4:
            continue

        tot_disp = abs(xs[-1] - xs[0])
        if tot_disp < 0.08:
            continue

        # Active Motion Trimming: trim leading/trailing frames where dx per frame < 0.001
        start_i = 0
        while start_i < len(xs) - 2 and abs(xs[start_i + 1] - xs[start_i]) < 0.001:
            start_i += 1

        end_i = len(xs) - 1
        while end_i > start_i + 1 and abs(xs[end_i] - xs[end_i - 1]) < 0.001:
            end_i -= 1

        trimmed_frames = frames_list[start_i:end_i + 1]
        trimmed_xs = xs[start_i:end_i + 1]

        if len(trimmed_frames) < 3:
            trimmed_frames = frames_list
            trimmed_xs = xs

        cand_count += 1
        s_frame = trimmed_frames[0]
        e_frame = trimmed_frames[-1]
        cand_disp = round(abs(trimmed_xs[-1] - trimmed_xs[0]), 3)

        candidates.append({
            "candidate_id": f"cand_{cand_count:03d}",
            "track_id": tid,
            "start_frame": s_frame,
            "end_frame": e_frame,
            "start_timestamp": round((s_frame - 1) / fps, 2),
            "end_timestamp": round((e_frame - 1) / fps, 2),
            "duration": round((e_frame - s_frame + 1) / fps, 2),
            "displacement": cand_disp,
            "all_frames": trimmed_frames,
            "all_xs": trimmed_xs,
        })

    return total_frames, fps, duration, candidates


def merge_events_preserving_tracks(candidates: list[dict], fps: float = 30.0, gap_thresh_sec: float = 0.5) -> list[dict]:
    """
    Merges overlapping or nearby candidate events while explicitly preserving all constituent Track IDs.
    """
    if not candidates:
        return []

    sorted_cands = sorted(candidates, key=lambda c: c["start_frame"])
    merged_events = []

    current_event = {
        "event_id": "event_001",
        "track_ids": [sorted_cands[0]["track_id"]],
        "start_frame": sorted_cands[0]["start_frame"],
        "end_frame": sorted_cands[0]["end_frame"],
        "candidates": [sorted_cands[0]],
    }

    gap_thresh_frames = int(gap_thresh_sec * fps)

    for next_cand in sorted_cands[1:]:
        # Check temporal overlap or small frame gap
        if next_cand["start_frame"] <= current_event["end_frame"] + gap_thresh_frames:
            current_event["end_frame"] = max(current_event["end_frame"], next_cand["end_frame"])
            if next_cand["track_id"] not in current_event["track_ids"]:
                current_event["track_ids"].append(next_cand["track_id"])
            current_event["candidates"].append(next_cand)
        else:
            merged_events.append(current_event)
            current_event = {
                "event_id": f"event_{len(merged_events)+1:03d}",
                "track_ids": [next_cand["track_id"]],
                "start_frame": next_cand["start_frame"],
                "end_frame": next_cand["end_frame"],
                "candidates": [next_cand],
            }

    merged_events.append(current_event)

    # Format timestamps & durations
    for idx, e in enumerate(merged_events, start=1):
        e["event_id"] = f"event_{idx:03d}"
        e["start_timestamp"] = round((e["start_frame"] - 1) / fps, 2)
        e["end_timestamp"] = round((e["end_frame"] - 1) / fps, 2)
        e["duration"] = round((e["end_frame"] - e["start_frame"] + 1) / fps, 2)

    return merged_events


def perform_track_attribution(vlm_verdict: str, merged_events: list[dict]) -> tuple[list[int], str]:
    """
    Deterministically attributes the VLM decision to the primary responsible pedestrian Track ID(s).
    """
    if not merged_events:
        return [], "no_tracks_detected"

    all_tids = []
    for e in merged_events:
        for tid in e["track_ids"]:
            if tid not in all_tids:
                all_tids.append(tid)

    if vlm_verdict.upper() != "JAYWALKING":
        return all_tids, "compliant_all_tracks_attributed"

    # For JAYWALKING: select candidate track with highest lateral displacement
    all_cands = [c for e in merged_events for c in e["candidates"]]
    if not all_cands:
        return all_tids, "unresolved"

    primary_cand = max(all_cands, key=lambda c: c["displacement"])
    responsible_tid = primary_cand["track_id"]

    return [responsible_tid], "attributed_to_primary_displaced_track"


def render_diagnostic_mp4(
    video_path: str,
    out_mp4_path: str,
    merged_events: list[dict],
    sample_indices: list[int],
    vlm_verdict: str,
    responsible_tids: list[int],
):
    """Generates visual debug MP4 showing bounding boxes, Track IDs, event bounds, and VLM sampled frame highlights."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_mp4_path, fourcc, fps, (width, height))

    model = YOLO("models/yolo11x.pt" if os.path.exists("models/yolo11x.pt") else "yolo11x.pt")
    device = 0 if torch.cuda.is_available() else "cpu"

    sample_indices_set = set(sample_indices)
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        is_vlm_sampled = frame_idx in sample_indices_set

        # Run detection to draw current boxes
        results = model.track(
            frame,
            tracker="bytetrack.yaml",
            persist=True,
            classes=[0],
            conf=0.25,
            verbose=False,
            device=device,
        )[0]

        if results.boxes is not None and results.boxes.id is not None:
            boxes_xyxy = results.boxes.xyxy.cpu().numpy()
            tids = results.boxes.id.int().cpu().tolist()

            for box, tid in zip(boxes_xyxy, tids):
                x1, y1, x2, y2 = map(int, box)

                is_responsible = (tid in responsible_tids)
                color = (0, 0, 255) if (is_responsible and vlm_verdict.upper() == "JAYWALKING") else (255, 165, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                lbl = f"ID:{tid} {'[RESPONSIBLE]' if is_responsible else ''}"
                cv2.putText(frame, lbl, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

        # Draw Top Banner Overlay
        banner_color = (0, 0, 255) if is_vlm_sampled else (50, 50, 50)
        cv2.rectangle(frame, (0, 0), (width, 35), banner_color, -1)

        vlm_badge = f" [VLM SAMPLED FRAME {sample_indices.index(frame_idx)+1}/5]" if is_vlm_sampled else ""
        top_txt = f"Frame {frame_idx} | Verdict: {vlm_verdict}{vlm_badge}"
        cv2.putText(frame, top_txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        writer.write(frame)

    cap.release()
    writer.release()


def run_5_video_diagnostic():
    """Phase 6: Run 5 diagnostic videos end-to-end with zero GT access."""
    target_videos = [
        "data/raw_clips/video_0028.mp4",
        "data/raw_clips/video_0073.mp4",
        "data/raw_clips/video_0082.mp4",
        "data/raw_clips/video_0110.mp4",
        "data/raw_clips/video_0003.mp4",
    ]

    out_diag_dir = "outputs/exp22_localization_diagnostics"
    os.makedirs(out_diag_dir, exist_ok=True)

    print("=" * 80)
    print("EXPERIMENT 22: 5-VIDEO DIAGNOSTIC VALIDATION")
    print("Zero Ground Truth Access During Inference")
    print("=" * 80)

    vlm_detector = FullVideoVLMDetector()
    diag_summaries = []

    for v_idx, vpath in enumerate(target_videos, start=1):
        clip_name = os.path.basename(vpath)
        print(f"\n--------------------------------------------------")
        print(f"[{v_idx}/5] DIAGNOSTIC VIDEO: {clip_name}")
        print(f"--------------------------------------------------")

        # 1. Refined Candidate Extraction
        t0_loc = time.time()
        total_frames, fps, duration, cands = extract_refined_candidate_events(vpath)

        print("\nDetected candidate tracks:")
        print(f"{'Track ID':<10} | {'start':<8} | {'end':<8} | {'duration':<10} | {'displacement':<12}")
        print("-" * 55)
        for c in cands:
            print(f"{c['track_id']:<10} | {c['start_frame']:<8} | {c['end_frame']:<8} | {c['duration']:<10.2f}s | {c['displacement']:<12.3f}")

        # 2. Track-Preserving Event Merging
        merged_events = merge_events_preserving_tracks(cands, fps=fps)

        print("\nMerged events:")
        print(f"{'Event ID':<10} | {'Track IDs':<16} | {'start':<8} | {'end':<8} | {'duration':<10}")
        print("-" * 60)
        for e in merged_events:
            print(f"{e['event_id']:<10} | {str(e['track_ids']):<16} | {e['start_frame']:<8} | {e['end_frame']:<8} | {e['duration']:<10.2f}s")

        # 3. Build Unified Event Envelope
        if merged_events:
            env_s = min(e["start_frame"] for e in merged_events)
            env_e = max(e["end_frame"] for e in merged_events)
        else:
            env_s = 1
            env_e = total_frames

        # 4. 5 Uniform Frames Sampling
        raw_indices = np.linspace(env_s, env_e, num=5, dtype=int)
        sample_indices = [min(total_frames, max(1, idx)) for idx in raw_indices]

        print("\nVLM sampled frames:")
        print(f"   [{env_s} ... {env_e}] -> Sampled Indices: {sample_indices}")

        # 5. Extract Sampled Frames & Run VLM
        cap = cv2.VideoCapture(vpath)
        frames = []
        for f_idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx - 1)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
        cap.release()

        t0_vlm = time.time()
        b64_list = [encode_frame_to_base64(f, quality=85) for f in frames]
        raw_response = vlm_detector.client.generate_chat(prompt=vlm_detector.coc_prompt, base64_images=b64_list)
        parsed = vlm_detector.parse_coc_response(raw_response)
        vlm_time = time.time() - t0_vlm

        verdict = parsed["prediction"].upper()
        print(f"\nVLM verdict: {verdict} ({vlm_time:.2f}s)")

        # 6. Responsible Pedestrian Attribution
        resp_tids, attr_status = perform_track_attribution(verdict, merged_events)
        print(f"Responsible Pedestrian Track Attribution: {resp_tids} (Status: {attr_status})")

        # 7. Render Visual Debug MP4
        out_mp4_path = os.path.join(out_diag_dir, f"{os.path.splitext(clip_name)[0]}_exp22_diag.mp4")
        render_diagnostic_mp4(vpath, out_mp4_path, merged_events, sample_indices, verdict, resp_tids)
        print(f"Saved visual debug MP4 to: {out_mp4_path}")

        diag_summaries.append({
            "clip_name": clip_name,
            "total_frames": total_frames,
            "fps": fps,
            "candidates_count": len(cands),
            "merged_events_count": len(merged_events),
            "envelope_bounds": [env_s, env_e],
            "sample_indices": sample_indices,
            "prediction": verdict,
            "responsible_track_ids": resp_tids,
            "attribution_status": attr_status,
            "vlm_time": round(vlm_time, 2),
            "output_video": out_mp4_path,
        })

    # Post-Inference GT Loading for Verification
    gt_path = "data/ground_truth.csv"
    gt_map = {}
    if os.path.exists(gt_path):
        df_gt = pd.read_csv(gt_path)
        gt_map = {row["clip_name"]: str(row["ground_truth"]).lower() for _, row in df_gt.iterrows()}

    print("\n" + "=" * 80)
    print("5-VIDEO DIAGNOSTIC POST-INFERENCE EVALUATION")
    print("=" * 80)
    for ds in diag_summaries:
        cn = ds["clip_name"]
        gt = gt_map.get(cn, "unknown")
        pred = ds["prediction"].lower()
        corr = "YES" if pred == gt else "NO"
        print(f"{cn:<16} | GT: {gt:<10} | Pred: {pred:<10} | Correct: {corr:<5} | Resp Tracks: {ds['responsible_track_ids']}")
    print("=" * 80)


def run_full_39_experiment():
    gt_path = "data/ground_truth.csv"
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"Ground truth missing: {gt_path}")

    df_gt = pd.read_csv(gt_path)
    eval_df = df_gt[df_gt["is_evaluated"] == True].copy()
    total_videos = len(eval_df)

    print("=" * 80)
    print("EXPERIMENT 22: REFINED EVENT LOCALIZATION + TRACK ATTRIBUTION (39 VIDEOS)")
    print("Zero Ground Truth Access During Inference")
    print(f"Total Videos: {total_videos}")
    print("=" * 80)

    vlm_detector = FullVideoVLMDetector()
    results = []

    tot_cands_count = 0
    tot_merged_count = 0
    tot_event_durations = []
    vids_multi_tracks = 0
    vids_unresolved_attr = 0

    t_bench_start = time.time()

    # -------------------------------------------------------------------------
    # PHASE 9: INFERENCE FOR ALL 39 VIDEOS (NO GT ACCESSED HERE)
    # -------------------------------------------------------------------------
    print("\n[PHASE 9: EXECUTING INFERENCE OVER 39 VIDEOS]")
    for idx, (_, row) in enumerate(eval_df.iterrows(), start=1):
        video_path = str(row["video_path"])
        clip_name = str(row["clip_name"])

        # 1. Refined Candidate Extraction
        t0 = time.time()
        total_frames, fps, duration, cands = extract_refined_candidate_events(video_path)

        # 2. Track-Preserving Event Merging
        merged_events = merge_events_preserving_tracks(cands, fps=fps)

        tot_cands_count += len(cands)
        tot_merged_count += len(merged_events)
        for e in merged_events:
            tot_event_durations.append(e["duration"])

        all_tids_in_vid = set(t for e in merged_events for t in e["track_ids"])
        if len(all_tids_in_vid) > 1:
            vids_multi_tracks += 1

        # 3. Build Unified Event Envelope
        if merged_events:
            env_s = min(e["start_frame"] for e in merged_events)
            env_e = max(e["end_frame"] for e in merged_events)
        else:
            env_s = 1
            env_e = total_frames

        # 4. 5 Uniform Frames Sampling
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

        # 5. VLM Inference
        b64_list = [encode_frame_to_base64(f, quality=85) for f in frames]
        raw_response = vlm_detector.client.generate_chat(prompt=vlm_detector.coc_prompt, base64_images=b64_list)
        parsed = vlm_detector.parse_coc_response(raw_response)
        elapsed = round(time.time() - t0, 3)

        verdict = parsed["prediction"].upper()

        # 6. Track Attribution
        resp_tids, attr_status = perform_track_attribution(verdict, merged_events)
        if attr_status == "unresolved":
            vids_unresolved_attr += 1

        print(f"[{idx}/{total_videos}] {clip_name}: Envelope=[{env_s}..{env_e}] | Merged Events={len(merged_events)} | Verdict={verdict:<10} ({elapsed}s)")

        results.append({
            "clip_name": clip_name,
            "video_path": video_path,
            "total_frames": total_frames,
            "fps": fps,
            "duration_seconds": duration,
            "envelope_bounds": [env_s, env_e],
            "sample_indices": sample_indices,
            "candidates_count": len(cands),
            "merged_events": merged_events,
            "prediction": verdict,
            "responsible_track_ids": resp_tids,
            "attribution_status": attr_status,
            "reasoning": parsed["chain_of_causation"],
            "inference_time": elapsed,
        })

    tot_benchmark_time = round(time.time() - t_bench_start, 2)

    # -------------------------------------------------------------------------
    # PHASE 10: POST-INFERENCE EVALUATION (GT LOADED NOW ONLY)
    # -------------------------------------------------------------------------
    print("\n[PHASE 10: EVALUATING METRICS AFTER INFERENCE COMPLETION]")
    gt_map = {row["clip_name"]: str(row["ground_truth"]).lower() for _, row in eval_df.iterrows()}

    # Load Architecture B predictions for failure analysis
    arch_b_json_path = "outputs/architecture_ab_experiment_results.json"
    arch_b_map = {}
    if os.path.exists(arch_b_json_path):
        with open(arch_b_json_path, "r") as f:
            arch_b_data = json.load(f)
            arch_b_map = {r["clip_name"]: r["verdict"].lower() for r in arch_b_data["results_arch_b"]}

    tp = tn = fp = fn = 0
    per_video_eval = []

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

        per_video_eval.append({
            "clip_name": clip_name,
            "ground_truth": gt_label,
            "arch_b_pred": arch_b_pred,
            "exp22_pred": pred_label,
            "correct": is_correct,
            "envelope": r["envelope_bounds"],
            "resp_tracks": r["responsible_track_ids"],
            "latency": r["inference_time"],
        })

    acc = round((tp + tn) / total_videos * 100, 2)
    prec = round(tp / (tp + fp) * 100, 2) if (tp + fp) > 0 else 0.0
    rec = round(tp / (tp + fn) * 100, 2) if (tp + fn) > 0 else 0.0
    spec = round(tn / (tn + fp) * 100, 2) if (tn + fp) > 0 else 0.0
    f1 = round(2 * prec * rec / (prec + rec), 2) if (prec + rec) > 0 else 0.0
    avg_latency = round(tot_benchmark_time / total_videos, 2)
    avg_evt_dur = round(float(np.mean(tot_event_durations)), 2) if tot_event_durations else 0.0

    # Calculate Deltas against Architecture B (71.79% Acc, 70.00% Prec, 46.67% Rec, 87.50% Spec, 56.00% F1, 2.60s latency)
    b_acc, b_prec, b_rec, b_spec, b_f1, b_lat = 71.79, 70.00, 46.67, 87.50, 56.00, 2.60
    d_acc = round(acc - b_acc, 2)
    d_prec = round(prec - b_prec, 2)
    d_rec = round(rec - b_rec, 2)
    d_spec = round(spec - b_spec, 2)
    d_f1 = round(f1 - b_f1, 2)
    d_lat = round(avg_latency - b_lat, 2)

    # Failure Analysis vs Arch B
    arch_b_corrected = [r["clip_name"] for r in per_video_eval if r["arch_b_pred"] != r["ground_truth"] and r["correct"]]
    arch_b_degraded = [r["clip_name"] for r in per_video_eval if r["arch_b_pred"] == r["ground_truth"] and not r["correct"]]

    print("\n" + "=" * 80)
    print("EXPERIMENT 22 RESULTS TABLE: REFINED EVENT LOCALIZATION")
    print("=" * 80)
    print(f"{'Clip Name':<16} | {'GT':<10} | {'Arch B':<10} | {'Exp 22':<10} | {'Correct':<7} | {'Resp Tracks':<15} | {'Latency':<7}")
    print("-" * 80)
    for r in per_video_eval:
        corr_str = "YES" if r["correct"] else "NO"
        print(f"{r['clip_name']:<16} | {r['ground_truth']:<10} | {r['arch_b_pred']:<10} | {r['exp22_pred']:<10} | {corr_str:<7} | {str(r['resp_tracks']):<15} | {r['latency']:>5.2f}s")
    print("=" * 80)
    print(f"Accuracy:                             {acc}% ({tp+tn}/{total_videos})")
    print(f"Precision:                            {prec}%")
    print(f"Recall:                               {rec}%")
    print(f"Specificity:                          {spec}%")
    print(f"F1 Score:                             {f1}%")
    print(f"Confusion Matrix:                     TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    print("-" * 80)
    print(f"Total Candidate Events Extracted:     {tot_cands_count}")
    print(f"Total Track-Preserving Merged Events: {tot_merged_count}")
    print(f"Average Merged Event Duration:        {avg_evt_dur}s")
    print(f"Videos with Multiple Track IDs:       {vids_multi_tracks} / {total_videos}")
    print(f"Videos with Unresolved Attribution:   {vids_unresolved_attr} / {total_videos}")
    print(f"Total Benchmark Runtime:              {tot_benchmark_time}s (avg {avg_latency}s/video)")
    print("=" * 80)

    # Print Required Three-Way Comparison Table
    print("\n" + "=" * 115)
    print("REQUIRED THREE-WAY PIPELINE COMPARISON TABLE")
    print("=" * 115)
    print(f"{'Pipeline':<46} | {'Accuracy':<9} | {'Precision':<9} | {'Recall':<8} | {'Specificity':<11} | {'F1':<7} | {'TP':<3} | {'TN':<3} | {'FP':<3} | {'FN':<3} | {'Avg/video':<9}")
    print("-" * 115)
    print(f"{'Historical short-clip baseline':<46} | {'97.44%':<9} | {'93.75%':<9} | {'100.0%':<8} | {'95.83%':<11} | {'96.77%':<7} | {'15':<3} | {'23':<3} | {'1':<3} | {'0':<3} | {'5.45s':<9}")
    print(f"{'Architecture B (baseline envelope)':<46} | {'71.79%':<9} | {'70.00%':<9} | {'46.67%':<8} | {'87.50%':<11} | {'56.00%':<7} | {'7':<3} | {'21':<3} | {'3':<3} | {'8':<3} | {'2.60s':<9}")
    print(f"{'Experiment 22 (refined event localization)':<46} | {f'{acc}%':<9} | {f'{prec}%':<9} | {f'{rec}%':<8} | {f'{spec}%':<11} | {f'{f1}%':<7} | {f'{tp}':<3} | {f'{tn}':<3} | {f'{fp}':<3} | {f'{fn}':<3} | {f'{avg_latency}s':<9}")
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
    out_json_path = "outputs/exp22_event_localization_results.json"
    os.makedirs("outputs", exist_ok=True)
    with open(out_json_path, "w") as f:
        json.dump({
            "experiment": "Experiment 22: Refined Event Localization + Responsible Pedestrian Attribution",
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
                "total_benchmark_time_seconds": tot_benchmark_time,
                "avg_latency_per_video_seconds": avg_latency,
            },
            "localization_statistics": {
                "total_candidates_extracted": tot_cands_count,
                "total_merged_events": tot_merged_count,
                "avg_merged_event_duration_seconds": avg_evt_dur,
                "videos_with_multiple_track_ids": vids_multi_tracks,
                "videos_with_unresolved_attribution": vids_unresolved_attr,
            },
            "transition_analysis": {
                "arch_b_errors_corrected": arch_b_corrected,
                "arch_b_correct_degraded": arch_b_degraded,
            },
            "video_results": results,
        }, f, indent=2, default=lambda o: int(o) if isinstance(o, (np.int64, np.int32)) else float(o) if isinstance(o, (np.float64, np.float32)) else str(o))

    print(f"\nSaved machine-readable Exp 22 results to: {out_json_path}")

    # Append Experiment 22 to RESEARCH_LOG.md
    log_entry = f"""

## Experiment 22 — Refined Event Localization + Responsible Pedestrian Attribution (39 Clips)
* **Date:** 2026-08-19
* **Model:** `qwen2.5vl:7b` via `FullVideoVLMDetector` (VLM Baseline)
* **Research Question:** Does refining pedestrian crossing event boundaries via active motion trimming (trimming leading/trailing stationary curb frames) and preserving constituent Track IDs during event merging improve long-video VLM classification accuracy while enabling responsible Track ID attribution?
* **Experimental Protocol:**
  - Active Motion Trimming: Trimmed stationary leading/trailing frames ($\Delta x < 0.001$/frame) to tighten event boundaries to exact lateral roadway entry.
  - Track-Preserving Merging: Preserved all constituent `track_ids` during temporal event merging.
  - Single-call VLM inference over 5 uniform frames per video event envelope. Zero ground-truth access during inference.
* **Empirical Results:**
  - **Accuracy:** **{acc}%** ({tp+tn}/{total_videos}) (Δ vs Arch B: **{d_acc:+}** percentage points)
  - **Precision:** **{prec}%** (Δ vs Arch B: **{d_prec:+}** percentage points)
  - **Recall:** **{rec}%** (Δ vs Arch B: **{d_rec:+}** percentage points)
  - **Specificity:** **{spec}%** (Δ vs Arch B: **{d_spec:+}** percentage points)
  - **F1 Score:** **{f1}%** (Δ vs Arch B: **{d_f1:+}** percentage points)
  - **Confusion Matrix:** $\\text{{TP}}={tp}, \\text{{TN}}={tn}, \\text{{FP}}={fp}, \\text{{FN}}={fn}$
  - **Total Latency:** {tot_benchmark_time}s (avg {avg_latency}s/clip)

* **Event Localization & Attribution Statistics:**
  - Total Candidates Extracted: {tot_cands_count}
  - Total Merged Events: {tot_merged_count} (avg duration {avg_evt_dur}s)
  - Videos with Multiple Track IDs: {vids_multi_tracks} / {total_videos}
  - Videos with Unresolved Attribution: {vids_unresolved_attr} / {total_videos}

* **Three-Way Pipeline Comparison Table:**

| Pipeline | Accuracy | Precision | Recall | Specificity | F1 Score | TP | TN | FP | FN | Avg Latency/video |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **1. Historical short-clip baseline** | **97.44%** | **93.75%** | **100.0%** | **95.83%** | **96.77%** | 15 | 23 | 1 | 0 | 5.45s |
| **2. Architecture B (baseline envelope)** | **71.79%** | **70.00%** | **46.67%** | **87.50%** | **56.00%** | 7 | 21 | 3 | 8 | **2.60s** |
| **3. Experiment 22 (refined localization)** | **{acc}%** | **{prec}%** | **{rec}%** | **{spec}%** | **{f1}%** | {tp} | {tn} | {fp} | {fn} | {avg_latency}s |

* **Failure Transition Analysis vs Architecture B:**
  - Architecture B errors corrected: {arch_b_corrected if arch_b_corrected else 'None'}
  - Architecture B correct predictions degraded: {arch_b_degraded if arch_b_degraded else 'None'}
"""
    with open("RESEARCH_LOG.md", "a") as f:
        f.write(log_entry)

    print("Updated RESEARCH_LOG.md with Experiment 22 results.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment 22: Improved Event Localization + Track Attribution")
    parser.add_argument("--validate-5", action="store_true", help="Run 5-video diagnostic validation test (Phase 6)")
    args = parser.parse_args()

    if args.validate_5:
        run_5_video_diagnostic()
    else:
        run_full_39_experiment()
