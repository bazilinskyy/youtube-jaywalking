#!/usr/bin/env python3
"""
Offline Pedestrian Keypoint and Body-Movement Extraction Pipeline

Extracts frame-by-frame 2D bounding boxes, ByteTrack IDs, 17-keypoint COCO poses,
kinematic trajectory metrics (velocities, accelerations, normalized displacements, trajectory changes),
and roadway-entry candidate events across all 39 development videos.

ZERO ground-truth class labels are accessed or used during extraction.

Outputs:
    outputs/keypoint_analysis/
        per_video/<video_id>_keypoints.json
        summaries/dataset_summary.json
        visualizations/<video_id>_trajectory_diagnostic.png
        visualizations/<video_id>_keypoint_diagnostic.mp4 (for representative clips)

Usage:
    python scripts/extract_pedestrian_keypoints.py
"""

from pose_estimator import COCO_KEYPOINTS
import json
import os
import sys
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))


SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),           # Head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),   # Torso & Arms
    (5, 11), (6, 12), (11, 12),               # Hips
    (11, 13), (13, 15), (12, 14), (14, 16),   # Legs
]


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


def extract_video_keypoints_and_kinematics(
    video_path: str,
    yolo_model: YOLO,
    pose_model: YOLO,
    device: str | int = 0,
    conf_thresh: float = 0.25,
    pose_conf_thresh: float = 0.25,
    stride: int = 1,
):
    """
    Extracts all pedestrian tracks, 17-keypoint skeletons, and kinematic trajectories.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_seconds = round(total_frames / fps, 3)

    raw_track_frames = {}  # tid -> list of frame observations
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # Track pedestrians via YOLO11x + ByteTrack
        results_det = yolo_model.track(
            frame,
            tracker="bytetrack.yaml",
            persist=True,
            classes=[0],  # Pedestrian class
            conf=conf_thresh,
            verbose=False,
            device=device,
        )[0]

        # Pose Estimation via YOLO11x-Pose
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
            p_kps = results_pose.keypoints.data.cpu().numpy()  # (N, 17, 3) (x_px, y_px, conf)
            for pb, pkp in zip(p_boxes, p_kps):
                norm_kps = []
                for kp in pkp:
                    norm_kps.append([
                        float(kp[0] / width),
                        float(kp[1] / height),
                        float(kp[2]),
                    ])
                detected_poses.append({"bbox": tuple(pb.tolist()), "keypoints": norm_kps})

        # Match ByteTrack Pedestrians with Detected Poses via IoU
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
                    "keypoints": matched_kp,  # (17, 3) or None
                    "has_valid_keypoints": matched_kp is not None,
                })

    cap.release()

    # -------------------------------------------------------------------------
    # COMPUTE TRAJECTORY KINEMATICS & TRANSITION DERIVATIVES PER TRACK
    # -------------------------------------------------------------------------
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

        # Frame-by-frame derivatives: displacement, velocity, acceleration, direction
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

        # Detect Roadway Entry Candidate & Peak Motion Frame
        peak_idx = int(np.argmax(velocities))
        peak_frame = f_ids[peak_idx]

        entry_idx = 0
        for i, dx_accum in enumerate([abs(x - start_x) for x in cxs]):
            if dx_accum >= 0.02:
                entry_idx = max(0, i - 1)
                break
        entry_frame = f_ids[entry_idx]

        # Attach frame-level computed kinematics
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
            "is_dominant_crossing_candidate": (total_raw_disp >= 0.08 and norm_motion_score >= 2.0),
            "frames": enriched_frames,
        })

    return {
        "video_path": video_path,
        "width": width,
        "height": height,
        "resolution_px": f"{width}x{height}",
        "frame_count": total_frames,
        "total_frames": total_frames,
        "fps": fps,
        "duration_sec": duration_seconds,
        "duration_seconds": duration_seconds,
        "total_pedestrian_tracks": len(processed_tracks),
        "tracks": processed_tracks,
    }


def generate_trajectory_plot(video_data: dict, out_png_path: str):
    """Plots lateral pedestrian trajectories and velocity profiles over time."""
    tracks = video_data.get("tracks", [])
    if not tracks:
        return

    plt.figure(figsize=(10, 6))

    # Subplot 1: Lateral Position vs Time
    plt.subplot(2, 1, 1)
    for t in tracks:
        times = [f["timestamp_seconds"] for f in t["frames"]]
        cxs = [f["bbox"]["center_x"] for f in t["frames"]]
        lbl = f"Track {t['track_id']} ({'Dominant' if t['is_dominant_crossing_candidate'] else 'Peripheral'})"
        style = "-" if t["is_dominant_crossing_candidate"] else "--"
        width = 2.5 if t["is_dominant_crossing_candidate"] else 1.0
        plt.plot(times, cxs, style, linewidth=width, label=lbl)

    plt.title(f"Pedestrian Lateral Trajectory — {os.path.basename(video_data['video_path'])}")
    plt.ylabel("Norm Center X")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper right", fontsize=8)

    # Subplot 2: Velocity vs Time
    plt.subplot(2, 1, 2)
    for t in tracks:
        if t["is_dominant_crossing_candidate"] or len(tracks) <= 3:
            times = [f["timestamp_seconds"] for f in t["frames"]]
            vels = [f["velocity"] for f in t["frames"]]
            plt.plot(times, vels, label=f"Track {t['track_id']} Velocity")
            plt.axvline(t["roadway_entry_candidate_frame"] / video_data["fps"], color="g",
                        linestyle=":", alpha=0.7, label=f"T{t['track_id']} Entry Frame")
            plt.axvline(t["peak_motion_frame"] / video_data["fps"], color="r",
                        linestyle=":", alpha=0.7, label=f"T{t['track_id']} Peak Frame")

    plt.xlabel("Time (seconds)")
    plt.ylabel("Norm Velocity (bw/s)")
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(out_png_path, dpi=150)
    plt.close()


def generate_diagnostic_video_overlay(video_path: str, video_data: dict, out_mp4_path: str):
    """Generates an annotated MP4 overlay with bounding boxes, 17-keypoint skeletons, track IDs, and entry frames."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return

    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_mp4_path, fourcc, fps, (w, h))

    # Index frame annotations by frame_id
    frame_lut = {}
    for t in video_data.get("tracks", []):
        tid = t["track_id"]
        is_dom = t["is_dominant_crossing_candidate"]
        e_f = t["roadway_entry_candidate_frame"]
        p_f = t["peak_motion_frame"]

        for fd in t["frames"]:
            f_id = fd["frame_id"]
            if f_id not in frame_lut:
                frame_lut[f_id] = []
            frame_lut[f_id].append({
                "track_id": tid,
                "is_dom": is_dom,
                "entry_f": e_f,
                "peak_f": p_f,
                "bbox": fd["bbox"],
                "vel": fd["velocity"],
                "dir": fd["trajectory_direction"],
                "kps": fd["keypoints"],
            })

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        if frame_idx in frame_lut:
            for obs in frame_lut[frame_idx]:
                tid = obs["track_id"]
                is_dom = obs["is_dom"]
                bbox = obs["bbox"]
                bx = int((bbox["center_x"] - bbox["width"] / 2.0) * w)
                by = int((bbox["center_y"] - bbox["height"] / 2.0) * h)
                bw = int(bbox["width"] * w)
                bh = int(bbox["height"] * h)

                color = (0, 255, 0) if is_dom else (200, 200, 200)
                thickness = 2 if is_dom else 1
                cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), color, thickness)

                # Draw 17-Keypoint Skeleton
                kps = obs["kps"]
                if kps is not None:
                    kp_pts = []
                    for kp in kps:
                        kx, ky, conf = int(kp[0] * w), int(kp[1] * h), kp[2]
                        kp_pts.append((kx, ky, conf))
                        if conf > 0.25:
                            cv2.circle(frame, (kx, ky), 3, (0, 255, 255), -1)

                    for p1_idx, p2_idx in SKELETON_CONNECTIONS:
                        if p1_idx < len(kp_pts) and p2_idx < len(kp_pts):
                            x1, y1, c1 = kp_pts[p1_idx]
                            x2, y2, c2 = kp_pts[p2_idx]
                            if c1 > 0.25 and c2 > 0.25:
                                cv2.line(frame, (x1, y1), (x2, y2), (255, 128, 0), 2)

                # Overlay Track Metadata
                tag = f"T{tid} {'[DOMINANT]' if is_dom else ''} V={obs['vel']:.1f}"
                if frame_idx == obs["entry_f"]:
                    tag += " [ROADWAY ENTRY]"
                elif frame_idx == obs["peak_f"]:
                    tag += " [PEAK VELOCITY]"

                cv2.putText(frame, tag, (bx, max(15, by - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        out.write(frame)

    cap.release()
    out.release()


def run_pipeline():
    gt_path = "data/ground_truth.csv"
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"Ground truth missing: {gt_path}")

    df_gt = pd.read_csv(gt_path)
    eval_df = df_gt[df_gt["is_evaluated"]].copy()
    total_videos = len(eval_df)

    base_out = "outputs/keypoint_analysis"
    per_vid_dir = os.path.join(base_out, "per_video")
    summary_dir = os.path.join(base_out, "summaries")
    vis_dir = os.path.join(base_out, "visualizations")

    os.makedirs(per_vid_dir, exist_ok=True)
    os.makedirs(summary_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    print("=" * 80)
    print("OFFLINE PEDESTRIAN KEYPOINT & BODY-MOVEMENT EXTRACTION PIPELINE")
    print(f"Processing all {total_videos} Development Videos")
    print("Zero Ground Truth Class Access During Extraction")
    print("=" * 80)

    # Initialize Models
    device = 0 if torch.cuda.is_available() else "cpu"
    yolo_model = YOLO("models/yolo11x.pt" if os.path.exists("models/yolo11x.pt") else "yolo11x.pt")
    pose_model = YOLO("models/yolo11x-pose.pt" if os.path.exists("models/yolo11x-pose.pt") else "yolo11x-pose.pt")

    successful_videos = 0
    failed_videos = 0
    total_tracks_dataset = 0
    total_valid_kps = 0
    total_track_frames = 0
    durations = []

    representative_clips = ["video_0028.mp4", "video_0073.mp4", "video_0003.mp4", "video_0227.mp4", "video_0336.mp4"]

    t0 = time.time()

    for idx, (_, row) in enumerate(eval_df.iterrows(), start=1):
        vpath = str(row["video_path"])
        cname = str(row["clip_name"])
        vid_id = os.path.splitext(cname)[0]

        print(f"\n[{idx}/{total_videos}] Extracting Keypoints & Kinematics for {cname}...")

        try:
            v_res = extract_video_keypoints_and_kinematics(
                video_path=vpath,
                yolo_model=yolo_model,
                pose_model=pose_model,
                device=device,
            )

            if v_res is None:
                print(f"   [FAILED] Unable to open video: {vpath}")
                failed_videos += 1
                continue

            successful_videos += 1
            n_tracks = v_res["total_pedestrian_tracks"]
            total_tracks_dataset += n_tracks

            for t in v_res["tracks"]:
                durations.append(t["duration_seconds"])
                total_track_frames += t["total_frames"]
                total_valid_kps += t["valid_keypoint_frames"]

            # Save per-video JSON
            json_out = os.path.join(per_vid_dir, f"{vid_id}_keypoints.json")
            with open(json_out, "w") as f:
                json.dump(v_res, f, indent=2)

            # Generate trajectory diagnostic plot
            plot_out = os.path.join(vis_dir, f"{vid_id}_trajectory_diagnostic.png")
            generate_trajectory_plot(v_res, plot_out)

            # Generate representative video overlay
            if cname in representative_clips:
                mp4_out = os.path.join(vis_dir, f"{vid_id}_keypoint_diagnostic.mp4")
                generate_diagnostic_video_overlay(vpath, v_res, mp4_out)
                print(f"   Rendered representative diagnostic MP4: {mp4_out}")

            print(
                f"   [SUCCESS] Tracks={n_tracks} | "
                f"Valid KP Ratio={round(total_valid_kps / max(1, total_track_frames), 2)} -> Saved: {json_out}"
            )

        except Exception as e:
            print(f"   [ERROR] Extraction failed for {cname}: {e}")
            failed_videos += 1

    total_time = round(time.time() - t0, 2)
    kp_success_rate = round((total_valid_kps / max(1, total_track_frames)) * 100, 2)
    avg_tracks = round(total_tracks_dataset / max(1, successful_videos), 2)
    avg_dur = round(float(np.mean(durations)), 2) if durations else 0.0

    summary_data = {
        "pipeline": "Offline Pedestrian Keypoint and Body-Movement Extraction",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_videos_processed": total_videos,
        "successful_videos": successful_videos,
        "failed_videos": failed_videos,
        "total_pedestrian_tracks": total_tracks_dataset,
        "average_tracks_per_video": avg_tracks,
        "total_frames_evaluated": total_track_frames,
        "valid_keypoint_frames": total_valid_kps,
        "keypoint_extraction_success_rate_percent": kp_success_rate,
        "average_track_duration_seconds": avg_dur,
        "total_pipeline_time_seconds": total_time,
        "coco_keypoints_extracted": COCO_KEYPOINTS,
        "output_directory": base_out,
    }

    summary_file = os.path.join(summary_dir, "dataset_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary_data, f, indent=2)

    print("\n" + "=" * 80)
    print("EXTRACTION SUMMARY COMPLETE")
    print("=" * 80)
    print(f"Successful Videos: {successful_videos}/{total_videos}")
    print(f"Total Pedestrian Tracks: {total_tracks_dataset} (Avg {avg_tracks} tracks/video)")
    print(f"Keypoint Extraction Success Rate: {kp_success_rate}%")
    print(f"Dataset Summary Saved To: {summary_file}")
    print("=" * 80)


if __name__ == "__main__":
    run_pipeline()
