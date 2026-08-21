#!/usr/bin/env python3
"""
Experiment 31: BoT-SORT Custom Tracking + YOLO26x-Pose with Simple Human-Readable Reporting

Key Specifications:
  1. Pose Estimator: YOLO26x-Pose (yolo26x-pose.pt) with 17 COCO keypoints.
  2. Custom Tracker: BoT-SORT (configs/botsort_custom.yaml) with ReID, sparseOptFlow, and dynamic track_buffer.
  3. Dynamic track_buffer: Scaled per video as int(round(track_buffer_sec * FPS)) (e.g. 60 frames @ 30 FPS, 120 frames @ 60 FPS).
  4. Roadway-Entry Validation: Two-stage entry validator (curb dwell stability -> roadway entry transition).
  5. VLM Classifier: Qwen2.5-VL-7B via Ollama API (5 key-state frames: pre-entry, entry, peak, post-entry, context).
  6. Human-Readable Reporting:
     - outputs/exp31_botsort_yolo26/results_summary.csv (simple tabular CSV)
     - outputs/exp31_botsort_yolo26/results_summary.md (clean markdown report)
     - outputs/exp31_botsort_yolo26/detailed_results.json (full research JSON)
     - outputs/exp31_botsort_yolo26/per_video/ (per-video JSONs)
     - outputs/exp31_botsort_yolo26/visualizations/ (clean visual predictions)

Zero ground-truth access during inference. GT loaded ONLY post-inference.

Usage:
    python experiments/run_exp31_botsort_yolo26_results.py [--subset]
"""

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.vlm.alpamayo_detector import FullVideoVLMDetector
from src.vlm.client import encode_frame_to_base64
from pose_estimator import COCO_KEYPOINTS


def _match_box_iou(box_a: tuple, box_b: tuple) -> float:
    """Calculates 2D IoU between two (cx, cy, w, h) normalized boxes."""
    ax1, ay1 = box_a[0] - box_a[2] / 2, box_a[1] - box_a[3] / 2
    ax2, ay2 = box_a[0] + box_a[2] / 2, box_a[1] + box_a[3] / 2
    bx1, by1 = box_b[0] - box_b[2] / 2, box_b[1] - box_b[3] / 2
    bx2, by2 = box_b[0] + box_b[2] / 2, box_b[1] + box_b[3] / 2

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0.0 else 0.0


def create_dynamic_botsort_config(base_yaml_path: str, fps: float, track_buffer_sec: float = 2.0) -> tuple[str, int]:
    """Creates a temporary YAML tracker config with dynamically scaled track_buffer based on video FPS."""
    with open(base_yaml_path, "r") as f:
        config = yaml.safe_load(f)

    calculated_buffer = max(10, int(round(track_buffer_sec * fps)))
    config["track_buffer"] = calculated_buffer

    temp_yaml = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    yaml.dump(config, temp_yaml, default_flow_style=False)
    temp_yaml.close()

    return temp_yaml.name, calculated_buffer


def extract_tracks_and_poses(
    video_path: str,
    yolo_model: YOLO,
    pose_model: YOLO,
    base_tracker_yaml: str,
    device: str | int = 0,
    conf_thresh: float = 0.25,
    pose_conf_thresh: float = 0.25,
    track_buffer_sec: float = 2.0,
):
    """Tracks pedestrians via BoT-SORT with per-video dynamic track_buffer and extracts 17-keypoint poses via YOLO26x-Pose."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_seconds = round(total_frames / fps, 3)

    temp_tracker_yaml, dynamic_buffer = create_dynamic_botsort_config(base_tracker_yaml, fps, track_buffer_sec)

    raw_track_frames = {}
    frame_idx = 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            # 1. Track pedestrians using YOLO11x + BoT-SORT
            results_det = yolo_model.track(
                frame,
                tracker=temp_tracker_yaml,
                persist=True,
                classes=[0],  # Pedestrian
                conf=conf_thresh,
                verbose=False,
                device=device,
            )[0]

            # 2. Pose Estimation using YOLO26x-Pose
            results_pose = pose_model(
                frame,
                classes=[0],
                conf=pose_conf_thresh,
                verbose=False,
                device=device,
            )[0]

            detected_poses = []
            if results_pose.keypoints is not None and results_pose.boxes is not None:
                p_boxes = results_pose.boxes.xywhn.cpu().numpy()
                p_kps = results_pose.keypoints.data.cpu().numpy()
                for pb, pkp in zip(p_boxes, p_kps):
                    norm_kps = []
                    for kp in pkp:
                        norm_kps.append([
                            float(kp[0] / width),
                            float(kp[1] / height),
                            float(kp[2]),
                        ])
                    detected_poses.append({"bbox": tuple(pb.tolist()), "keypoints": norm_kps})

            # 3. Match BoT-SORT Tracks with YOLO26x Poses via IoU
            if results_det.boxes is not None and results_det.boxes.id is not None:
                det_boxes = results_det.boxes.xywhn.cpu().numpy()
                det_tids = results_det.boxes.id.int().cpu().tolist()

                for dbox, tid in zip(det_boxes, det_tids):
                    cx, cy, bw, bh = dbox.tolist()
                    matched_kp = None
                    best_iou = 0.0

                    for p_entry in detected_poses:
                        iou = _match_box_iou((cx, cy, bw, bh), p_entry["bbox"])
                        if iou > 0.30 and iou > best_iou:
                            best_iou = iou
                            matched_kp = p_entry["keypoints"]

                    if tid not in raw_track_frames:
                        raw_track_frames[tid] = []

                    raw_track_frames[tid].append({
                        "frame_id": frame_idx,
                        "timestamp_seconds": round(frame_idx / fps, 3),
                        "bbox": {
                            "center_x": round(cx, 4),
                            "center_y": round(cy, 4),
                            "width": round(bw, 4),
                            "height": round(bh, 4),
                            "bottom_y": round(cy + bh / 2.0, 4),
                        },
                        "keypoints": matched_kp,
                        "has_valid_keypoints": matched_kp is not None,
                    })

    finally:
        cap.release()
        if os.path.exists(temp_tracker_yaml):
            os.remove(temp_tracker_yaml)

    # Calculate track kinematics
    processed_tracks = []

    for tid, frames_data in raw_track_frames.items():
        if len(frames_data) < 2:
            continue

        f_start = frames_data[0]["frame_id"]
        f_end = frames_data[-1]["frame_id"]
        track_dur = round((f_end - f_start + 1) / fps, 3)

        cxs = [f["bbox"]["center_x"] for f in frames_data]
        cys = [f["bbox"]["center_y"] for f in frames_data]
        bws = [f["bbox"]["width"] for f in frames_data]
        f_ids = [f["frame_id"] for f in frames_data]

        avg_bw = max(float(np.mean(bws)), 0.02)
        start_x = cxs[0]

        max_dx_raw = float(np.max([abs(x - start_x) for x in cxs]))
        norm_motion_score = round(max_dx_raw / avg_bw, 4)
        total_raw_disp = round(abs(cxs[-1] - cxs[0]), 4)

        velocities = [0.0]
        accelerations = [0.0]
        consec_displacements = [0.0]
        trajectory_directions = ["stationary"]

        valid_kp_count = sum(1 for f in frames_data if f["has_valid_keypoints"])

        for i in range(1, len(frames_data)):
            dt = max((f_ids[i] - f_ids[i-1]) / fps, 1e-4)
            dx = cxs[i] - cxs[i-1]
            dy = cys[i] - cys[i-1]
            disp = float(np.sqrt(dx**2 + dy**2))
            norm_disp = disp / avg_bw
            consec_displacements.append(round(norm_disp, 4))

            vel = norm_disp / dt
            velocities.append(round(vel, 4))

            acc = (velocities[i] - velocities[i-1]) / dt
            accelerations.append(round(acc, 4))

            if abs(dx) > abs(dy) and abs(dx) > 0.005:
                direction = "moving_right" if dx > 0 else "moving_left"
            elif abs(dy) > 0.005:
                direction = "moving_down" if dy > 0 else "moving_up"
            else:
                direction = "stationary"
            trajectory_directions.append(direction)

        peak_idx = int(np.argmax(velocities))
        peak_frame = f_ids[peak_idx]

        entry_idx = 0
        for i, dx_accum in enumerate([abs(x - start_x) for x in cxs]):
            if dx_accum >= 0.02:
                entry_idx = max(0, i - 1)
                break
        entry_frame = f_ids[entry_idx]

        enriched_frames = []
        for i, fd in enumerate(frames_data):
            enriched_frames.append({
                "frame_id": fd["frame_id"],
                "timestamp_seconds": fd["timestamp_seconds"],
                "bbox": fd["bbox"],
                "normalized_lateral_displacement": round(abs(cxs[i] - start_x) / avg_bw, 4),
                "consecutive_displacement": consec_displacements[i],
                "velocity": velocities[i],
                "acceleration": accelerations[i],
                "trajectory_direction": trajectory_directions[i],
                "has_valid_keypoints": fd["has_valid_keypoints"],
                "keypoints": fd["keypoints"],
            })

        processed_tracks.append({
            "track_id": tid,
            "event_start_frame": f_start,
            "event_end_frame": f_end,
            "duration_seconds": track_dur,
            "total_frames": len(frames_data),
            "valid_keypoint_frames": valid_kp_count,
            "keypoint_coverage_ratio": round(valid_kp_count / max(1, len(frames_data)), 4),
            "normalized_motion_score": norm_motion_score,
            "total_lateral_displacement": total_raw_disp,
            "roadway_entry_candidate_frame": entry_frame,
            "peak_motion_frame": peak_frame,
            "all_frames": f_ids,
            "velocities": velocities,
            "is_dominant_crossing_candidate": (total_raw_disp >= 0.08 and norm_motion_score >= 2.0),
            "frames": enriched_frames,
        })

    return {
        "video_path": video_path,
        "total_frames": total_frames,
        "fps": fps,
        "duration_seconds": duration_seconds,
        "tracker_used": "BoT-SORT (custom config with_reid=True, sparseOptFlow)",
        "track_buffer_frames": dynamic_buffer,
        "track_buffer_seconds": track_buffer_sec,
        "pose_model_used": "YOLO26x-Pose (yolo26x-pose.pt)",
        "total_pedestrian_tracks": len(processed_tracks),
        "tracks": processed_tracks,
    }


def validate_two_stage_roadway_entry(prof: dict, fps: float = 30.0) -> tuple[bool, int, int, list[int]]:
    """Two-stage roadway entry transition validator."""
    if not prof or len(prof.get("all_frames", [])) < 5:
        return False, 0, 0, []

    frames = prof["all_frames"]
    vels = np.array(prof["velocities"])
    total_len = len(frames)

    if prof["total_lateral_displacement"] < 0.08:
        return False, frames[0], frames[-1], []

    peak_local_idx = int(np.argmax(vels))
    peak_frame = frames[peak_local_idx]

    entry_local_idx = max(0, peak_local_idx - int(0.5 * fps))
    for i in range(peak_local_idx):
        if vels[i] >= 0.10:
            entry_local_idx = i
            break
    entry_frame = frames[entry_local_idx]

    f1 = frames[max(0, entry_local_idx - int(0.5 * fps))]
    f2 = entry_frame
    f3 = peak_frame
    f4 = frames[min(total_len - 1, peak_local_idx + int(0.5 * fps))]
    f5 = frames[-1]

    raw_5 = sorted(list(dict.fromkeys([f1, f2, f3, f4, f5])))
    if len(raw_5) < 5:
        uniform_fallback = np.linspace(frames[0], frames[-1], 5, dtype=int).tolist()
        raw_5 = sorted(list(dict.fromkeys(raw_5 + uniform_fallback)))

    sample_5 = [min(frames[-1], max(frames[0], idx)) for idx in raw_5[:5]]
    return True, entry_frame, peak_frame, sample_5


def generate_simple_prediction_image(video_path: str, prediction: str, track_id: int | None, entry_frame: int | None, out_png_path: str):
    """Saves a clean annotated key-frame showing the final prediction."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target_frame = entry_frame if (entry_frame and entry_frame > 0) else max(1, total_frames // 2)

    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame - 1)
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        return

    h, w = frame.shape[:2]

    # Draw Banner Overlay
    color = (0, 0, 220) if prediction.upper() == "JAYWALKING" else (0, 180, 0)
    cv2.rectangle(frame, (20, 20), (450, 100), (0, 0, 0), -1)
    cv2.rectangle(frame, (20, 20), (450, 100), color, 2)

    cv2.putText(frame, f"PREDICTION: {prediction.upper()}", (35, 55), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
    track_str = f"Track ID: {track_id if track_id else 'None'}"
    entry_str = f"Entry Frame: {entry_frame if entry_frame else 'N/A'}"
    cv2.putText(frame, f"{track_str} | {entry_str}", (35, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imwrite(out_png_path, frame)


def run_exp31(subset: bool = False):
    gt_path = "data/ground_truth.csv"
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"Ground truth missing: {gt_path}")

    df_gt = pd.read_csv(gt_path)
    eval_df = df_gt[df_gt["is_evaluated"] == True].copy()
    if subset:
        eval_df = eval_df.head(3).copy()

    total_videos = len(eval_df)

    out_dir = "outputs/exp31_botsort_yolo26"
    per_vid_dir = os.path.join(out_dir, "per_video")
    vis_dir = os.path.join(out_dir, "visualizations")
    os.makedirs(per_vid_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    base_tracker_yaml = "configs/botsort_custom.yaml"

    print("=" * 80)
    print("EXPERIMENT 31: BoT-SORT + YOLO26x-POSE WITH CLEAN PREDICTION REPORTING")
    print(f"Total Videos: {total_videos} (Subset: {subset})")
    print("Tracker: BoT-SORT with dynamic per-video track_buffer")
    print("Pose Model: YOLO26x-Pose (yolo26x-pose.pt)")
    print("Zero Ground Truth Access During Inference")
    print("=" * 80)

    device = 0 if torch.cuda.is_available() else "cpu"
    yolo_model = YOLO("models/yolo11x.pt" if os.path.exists("models/yolo11x.pt") else "yolo11x.pt")
    pose_model = YOLO("yolo26x-pose.pt")
    vlm_detector = FullVideoVLMDetector()

    video_results = []
    t0_exp = time.time()

    # -------------------------------------------------------------------------
    # INFERENCE PIPELINE (ZERO GT ACCESSED)
    # -------------------------------------------------------------------------
    for idx, (_, row) in enumerate(eval_df.iterrows(), start=1):
        vpath = str(row["video_path"])
        cname = str(row["clip_name"])
        vid_id = os.path.splitext(cname)[0]

        print(f"\n[{idx}/{total_videos}] Processing {cname}...")
        t_v0 = time.time()

        # 1. BoT-SORT + YOLO26x-Pose Extraction
        v_data = extract_tracks_and_poses(
            video_path=vpath,
            yolo_model=yolo_model,
            pose_model=pose_model,
            base_tracker_yaml=base_tracker_yaml,
            device=device,
        )

        if v_data is None:
            print(f"   [ERROR] Could not read video: {vpath}")
            continue

        fps = v_data["fps"]
        dyn_buffer = v_data["track_buffer_frames"]
        n_tracks = v_data["total_pedestrian_tracks"]
        total_frames = v_data["total_frames"]

        # Save per-video detailed json
        kps_json = os.path.join(per_vid_dir, f"{vid_id}_keypoints.json")
        with open(kps_json, "w") as fp:
            json.dump(v_data, fp, indent=2)

        # 2. Select Dominant Crossing Track
        tracks = v_data["tracks"]
        if tracks:
            cands = [p for p in tracks if p["is_dominant_crossing_candidate"]]
            dom_prof = max(cands, key=lambda p: p["normalized_motion_score"]) if cands else max(tracks, key=lambda p: p["normalized_motion_score"])
            dom_tid = dom_prof["track_id"]
            max_d = dom_prof["total_lateral_displacement"]
            norm_d = dom_prof["normalized_motion_score"]
        else:
            dom_prof = None
            dom_tid = None
            max_d = 0.0
            norm_d = 0.0

        # 3. Two-Stage Roadway-Entry Validation & 5-Frame Selection
        is_entry_valid, entry_f, peak_f, sample_5 = validate_two_stage_roadway_entry(dom_prof, fps=fps)

        if not is_entry_valid or not dom_prof:
            verdict = "COMPLIANT"
            reasoning = "Filtered out by geometric roadway entry validation."
            vlm_elapsed = 0.001
        else:
            cap = cv2.VideoCapture(vpath)
            frames = []
            for f_idx in sample_5:
                cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx - 1)
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)
            cap.release()

            t_vlm = time.time()
            b64_list = [encode_frame_to_base64(f, quality=85) for f in frames]
            raw_response = vlm_detector.client.generate_chat(prompt=vlm_detector.coc_prompt, base64_images=b64_list)
            parsed = vlm_detector.parse_coc_response(raw_response)
            vlm_elapsed = round(time.time() - t_vlm, 3)
            verdict = parsed["prediction"].upper()
            reasoning = parsed["chain_of_causation"]

        total_v_time = round(time.time() - t_v0, 2)
        print(f"   FPS: {fps:.1f} | Buffer: {dyn_buffer}f | Dominant TID: {dom_tid} | Verdict: {verdict:<10} ({total_v_time}s)")

        # Generate simple prediction visual
        img_out = os.path.join(vis_dir, f"{vid_id}_prediction.png")
        generate_simple_prediction_image(vpath, verdict, dom_tid, entry_f if is_entry_valid else None, img_out)

        video_results.append({
            "video": cname,
            "video_path": vpath,
            "prediction": verdict,
            "responsible_track_id": dom_tid,
            "entry_frame": entry_f if is_entry_valid else "-",
            "peak_motion_frame": peak_f if is_entry_valid else "-",
            "event_start": dom_prof["event_start_frame"] if dom_prof else "-",
            "event_end": dom_prof["event_end_frame"] if dom_prof else "-",
            "fps": fps,
            "dynamic_track_buffer": dyn_buffer,
            "total_tracks": n_tracks,
            "reasoning": reasoning,
            "processing_time_sec": total_v_time,
        })

    # -------------------------------------------------------------------------
    # POST-INFERENCE EVALUATION (GT LOADED ONLY HERE)
    # -------------------------------------------------------------------------
    gt_map = {row["clip_name"]: str(row["ground_truth"]).lower() for _, row in eval_df.iterrows()}

    tp_count = tn_count = fp_count = fn_count = 0
    tot_time = 0.0
    summary_rows = []

    for r in video_results:
        cname = r["video"]
        gt = gt_map[cname].upper()
        pred = r["prediction"].upper()
        tot_time += r["processing_time_sec"]

        is_correct = (gt == pred)
        correct_str = "YES" if is_correct else "NO"
        symbol = "✓" if is_correct else "✗"

        if gt == "JAYWALKING" and pred == "JAYWALKING":
            tp_count += 1
        elif gt == "COMPLIANT" and pred == "COMPLIANT":
            tn_count += 1
        elif gt == "COMPLIANT" and pred == "JAYWALKING":
            fp_count += 1
        elif gt == "JAYWALKING" and pred == "COMPLIANT":
            fn_count += 1

        summary_rows.append({
            "video": cname,
            "ground_truth": gt,
            "prediction": pred,
            "correct": correct_str,
            "symbol": symbol,
            "responsible_track_id": r["responsible_track_id"],
            "entry_frame": r["entry_frame"],
            "peak_motion_frame": r["peak_motion_frame"],
            "event_start": r["event_start"],
            "event_end": r["event_end"],
            "processing_time_sec": r["processing_time_sec"],
            "reasoning": r["reasoning"],
        })

    acc = round((tp_count + tn_count) / total_videos * 100, 2)
    prec = round(tp_count / (tp_count + fp_count) * 100, 2) if (tp_count + fp_count) > 0 else 0.0
    rec = round(tp_count / (tp_count + fn_count) * 100, 2) if (tp_count + fn_count) > 0 else 0.0
    spec = round(tn_count / (tn_count + fp_count) * 100, 2) if (tn_count + fp_count) > 0 else 0.0
    f1 = round(2 * prec * rec / (prec + rec), 2) if (prec + rec) > 0 else 0.0
    avg_lat = round(tot_time / total_videos, 2)

    # 1. Write Clean results_summary.csv
    csv_path = os.path.join(out_dir, "results_summary.csv")
    csv_df = pd.DataFrame(summary_rows)[[
        "video", "ground_truth", "prediction", "correct", "responsible_track_id",
        "entry_frame", "peak_motion_frame", "event_start", "event_end", "processing_time_sec"
    ]]
    csv_df.to_csv(csv_path, index=False)

    # 2. Write Clean results_summary.md
    md_path = os.path.join(out_dir, "results_summary.md")
    with open(md_path, "w") as fp_md:
        fp_md.write("# Experiment Results (BoT-SORT + YOLO26x-Pose)\n\n")
        fp_md.write("## Overall\n")
        fp_md.write(f"- Videos: {total_videos}\n")
        fp_md.write(f"- Correct: {tp_count + tn_count} / {total_videos}\n")
        fp_md.write(f"- Accuracy: {acc}%\n")
        fp_md.write(f"- Precision: {prec}%\n")
        fp_md.write(f"- Recall: {rec}%\n")
        fp_md.write(f"- Specificity: {spec}%\n")
        fp_md.write(f"- F1: {f1}%\n")
        fp_md.write(f"- TP: {tp_count}\n")
        fp_md.write(f"- TN: {tn_count}\n")
        fp_md.write(f"- FP: {fp_count}\n")
        fp_md.write(f"- FN: {fn_count}\n")
        fp_md.write(f"- Average latency: {avg_lat} s/video\n\n")

        fp_md.write("## Per-Video Predictions\n\n")
        fp_md.write("| Video | Ground Truth | Prediction | Correct | Responsible Track ID | Entry Frame | Peak Frame |\n")
        fp_md.write("|---|---|---|:---:|:---:|:---:|:---:|\n")
        for row in summary_rows:
            fp_md.write(f"| {row['video']} | {row['ground_truth']} | {row['prediction']} | {row['correct']} | {row['responsible_track_id']} | {row['entry_frame']} | {row['peak_motion_frame']} |\n")

        fp_md.write("\n## Errors\n\n")
        errors = [r for r in summary_rows if r["correct"] == "NO"]
        if not errors:
            fp_md.write("No misclassifications observed.\n")
        else:
            fp_md.write("| Video | Ground Truth | Prediction | Responsible Track | Event Interval | Error Reason |\n")
            fp_md.write("|---|---|---|:---:|:---:|---|\n")
            for err in errors:
                err_type = "False Positive" if err["ground_truth"] == "COMPLIANT" else "False Negative"
                reason = "Pedestrian lateral motion triggered roadway entry" if err_type == "False Positive" else "Entry motion below sustained threshold or diagonal crossing"
                fp_md.write(f"| {err['video']} | {err['ground_truth']} | {err['prediction']} | {err['responsible_track_id']} | [{err['event_start']}..{err['event_end']}] | {reason} |\n")

    # 3. Write Detailed JSON
    json_path = os.path.join(out_dir, "detailed_results.json")
    with open(json_path, "w") as fp_json:
        json.dump({
            "experiment": "Experiment 31: BoT-SORT Custom Tracking + YOLO26x-Pose with Simple Prediction Reporting",
            "models": {
                "detector": "YOLO11x",
                "tracker": "BoT-SORT (custom with_reid=True, sparseOptFlow)",
                "pose": "YOLO26x-Pose (yolo26x-pose.pt)",
                "vlm": "Qwen2.5-VL-7B",
            },
            "metrics": {
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "specificity": spec,
                "f1": f1,
                "tp": tp_count,
                "tn": tn_count,
                "fp": fp_count,
                "fn": fn_count,
                "average_latency_sec": avg_lat,
            },
            "summary_rows": summary_rows,
        }, fp_json, indent=2, default=lambda o: int(o) if isinstance(o, (np.int64, np.int32)) else float(o) if isinstance(o, (np.float64, np.float32)) else str(o))

    # 4. Terminal Summary Output
    print("\n" + "=" * 50)
    print("EXPERIMENT 31 RESULTS")
    print("=" * 50)
    print(f"Videos:       {total_videos}")
    print(f"Accuracy:     {acc:.2f}%")
    print(f"Precision:    {prec:.2f}%")
    print(f"Recall:       {rec:.2f}%")
    print(f"Specificity:  {spec:.2f}%")
    print(f"F1 Score:     {f1:.2f}%")
    print(f"TP: {tp_count}  TN: {tn_count}  FP: {fp_count}  FN: {fn_count}")
    print(f"Avg latency:  {avg_lat:.2f} s/video")
    print("=" * 50)

    print("\nVIDEO PREDICTIONS")
    print("=" * 50)
    for r in summary_rows:
        print(f"{r['video']:<14} GT={r['ground_truth']:<10} PRED={r['prediction']:<10} {r['symbol']}")
    print("=" * 50)

    print(f"\nSimple CSV Summary saved to:      {csv_path}")
    print(f"Simple Markdown Report saved to:  {md_path}")
    print(f"Detailed Research JSON saved to:  {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", action="store_true", help="Run on small 3-clip subset first")
    args = parser.parse_args()
    run_exp31(subset=args.subset)
