#!/usr/bin/env python3
"""
Run the jaywalking detection pipeline on specific clips and save CSV output.
Used to compare algorithm output against ground truth.
"""
import os
import sys
import csv
import cv2
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
from ultralytics import YOLO

ROOT = Path(__file__).parent
EVAL_DIR = ROOT / "evaluation"
CLIPS_POS = EVAL_DIR / "jaad_positive"
CLIPS_NEG = EVAL_DIR / "jaad_negative"
REVIEW_CSV = EVAL_DIR / "review_results.csv"
OUTPUT_CSV = EVAL_DIR / "pipeline_results.csv"

# Model config
MODEL_PATH = str(ROOT / "yolo11x.pt")
CONFIDENCE = 0.7


def get_reviewed_clips():
    """Get list of clips that have ground truth labels."""
    clips = []
    with open(REVIEW_CSV) as f:
        for row in csv.DictReader(f):
            if row["reviewer_verdict"] in ("correct", "wrong", "ambiguous"):
                # Find the clip path
                for clip_dir in [CLIPS_POS, CLIPS_NEG]:
                    path = clip_dir / row["clip_name"]
                    if path.exists():
                        clips.append({
                            "path": str(path),
                            "name": row["clip_name"],
                            "gt_label": row["auto_label"],
                            "gt_verdict": row["reviewer_verdict"],
                        })
                        break
    return clips


def run_pipeline_on_clip(clip_path, clip_name):
    """Run YOLO tracking + pose on a single clip. Returns per-frame CSV rows."""
    # Import project modules
    sys.path.insert(0, str(ROOT))
    from utils.crossing.pose import PoseEstimator
    from utils.crossing.zebra import ZebraDetector
    from utils.crossing.traffic_light import TrafficLightClassifier

    model = YOLO(MODEL_PATH).to(0 if torch.cuda.is_available() else "cpu")
    pose_estimator = PoseEstimator(str(ROOT / "yolo11x-pose.pt"))
    ZebraDetector.reset_cache()

    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        print(f"    Cannot open {clip_name}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_count = 0
    rows = []

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        frame_count += 1

        # YOLO tracking
        results = model.track(frame, tracker="bytetrack.yaml", persist=True,
                              conf=CONFIDENCE, verbose=False,
                              device=0 if torch.cuda.is_available() else "cpu")

        # Zebra detection
        zebra_polygon = ZebraDetector.get_zebra_polygon(frame)
        zebra_present = zebra_polygon is not None and len(zebra_polygon) > 0

        # Extract detections
        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy() if results[0].boxes.conf is not None else [1.0] * len(classes)
            track_ids = results[0].boxes.id.int().cpu().tolist() if results[0].boxes.id is not None else [-1] * len(classes)
            h, w = frame.shape[:2]

            for i, (box, c, conf, tid) in enumerate(zip(boxes, classes, confs, track_ids)):
                x1, y1, x2, y2 = box
                cx = ((x1 + x2) / 2) / w
                cy = ((y1 + y2) / 2) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                yolo_id = int(c)

                traffic_state = "UNKNOWN"
                facing = "UNKNOWN"
                stride = 0.0

                # Traffic light classification for TL detections
                if yolo_id == 9:
                    tl_crop = frame[max(0,int(y1)):min(h,int(y2)),
                                    max(0,int(x1)):min(w,int(x2))]
                    if tl_crop.size > 0:
                        tl_state = TrafficLightClassifier.classify_state(tl_crop, tid)
                        traffic_state = tl_state

                # Pose estimation for pedestrians
                if yolo_id == 0:
                    person_crop = frame[max(0,int(y1)):min(h,int(y2)),
                                        max(0,int(x1)):min(w,int(x2))]
                    if person_crop.size > 0:
                        _, facing, stride = pose_estimator.estimate_pose(person_crop)

                rows.append({
                    "YOLO_id": yolo_id,
                    "X-center": round(cx, 6),
                    "Y-center": round(cy, 6),
                    "Width": round(bw, 6),
                    "Height": round(bh, 6),
                    "Unique Id": tid,
                    "Frame Count": frame_count,
                    "zebra_crossing": zebra_present,
                    "traffic_light_state": traffic_state,
                    "facing_direction": facing,
                    "stride_ratio": round(stride, 6),
                })

    cap.release()
    ZebraDetector.reset_cache()
    return rows


def classify_violation(rows, fps=30):
    """Classify crossing events from pipeline output."""
    persons = [r for r in rows if r["YOLO_id"] == 0]
    if not persons:
        return "no_person", {}

    # Group by track ID
    tracks = defaultdict(list)
    for r in persons:
        tracks[r["Unique Id"]].append(r)

    violations = []
    for tid, frames in tracks.items():
        if len(frames) < 5:
            continue
        x_vals = [f["X-center"] for f in frames]
        x_range = max(x_vals) - min(x_vals)
        if x_range < 0.15:
            continue

        road_frames = sum(1 for x in x_vals if 0.2 < x < 0.8)
        if road_frames < 3:
            continue

        # Check traffic light
        tl_frames = [r for r in rows if r["YOLO_id"] == 9 and r["traffic_light_state"] != "UNKNOWN"]
        light_states = set(r["traffic_light_state"] for r in tl_frames)
        has_red = "RED" in light_states
        has_green = "GREEN" in light_states

        # Check zebra
        zebra_frames = [r for r in frames if r["zebra_crossing"]]
        has_zebra = len(zebra_frames) > 0

        # Classify
        if has_red:
            violations.append("SIGNAL_VIOLATION")
        elif not has_zebra:
            violations.append("NO_CROSSWALK")
        else:
            violations.append("compliant")

    if not violations:
        return "no_crossing", {}

    # Most severe violation
    if "SIGNAL_VIOLATION" in violations:
        return "jaywalking", {"violation_type": "SIGNAL_VIOLATION", "details": violations}
    elif "NO_CROSSWALK" in violations:
        return "jaywalking", {"violation_type": "NO_CROSSWALK", "details": violations}
    else:
        return "compliant", {"details": violations}


def main():
    clips = get_reviewed_clips()
    if not clips:
        print("No reviewed clips found. Run review_clips.py first.")
        return

    print(f"Running pipeline on {len(clips)} clips...\n")

    results = []
    for clip in clips:
        print(f"  {clip['name']} (GT: {clip['gt_label']})...", end=" ", flush=True)
        rows = run_pipeline_on_clip(clip["path"], clip["name"])

        # Get FPS
        cap = cv2.VideoCapture(clip["path"])
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        cap.release()

        pred_label, details = classify_violation(rows, fps)

        # Compare with ground truth
        match = pred_label == clip["gt_label"]
        # Also check if GT was wrong (user marked as wrong)
        if clip["gt_verdict"] == "wrong":
            match = pred_label != clip["gt_label"]  # If GT was wrong, algorithm being different is correct

        results.append({
            "clip_name": clip["name"],
            "gt_label": clip["gt_label"],
            "gt_verdict": clip["gt_verdict"],
            "pred_label": pred_label,
            "violation_type": details.get("violation_type", ""),
            "match": match,
            "num_frames": len(rows),
            "num_persons": len(set(r["Unique Id"] for r in rows if r["YOLO_id"] == 0)),
        })
        print(f"pred={pred_label} {'✓' if match else '✗'}")

    # Save results
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    # Summary
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")

    # Simple accuracy: how often pred matches GT
    matches = sum(1 for r in results if r["match"])
    print(f"Agreement with GT: {matches}/{len(results)} ({matches/len(results)*100:.0f}%)")

    # If GT was marked wrong, algorithm disagrees = correct
    gt_wrong = [r for r in results if r["gt_verdict"] == "wrong"]
    if gt_wrong:
        print(f"\nGT clips marked WRONG by you:")
        for r in gt_wrong:
            print(f"  {r['clip_name']}: GT={r['gt_label']}, algo={r['pred_label']}")

    # Confusion-like breakdown
    print(f"\nBreakdown:")
    for r in results:
        status = "TP" if r["gt_label"] == "jaywalking" and r["pred_label"] == "jaywalking" else \
                 "TN" if r["gt_label"] == "compliant" and r["pred_label"] == "compliant" else \
                 "FP" if r["gt_label"] == "compliant" and r["pred_label"] == "jaywalking" else "FN"
        print(f"  {r['clip_name']:<25} GT={r['gt_label']:<12} pred={r['pred_label']:<12} {status}")

    tp = sum(1 for r in results if r["gt_label"] == "jaywalking" and r["pred_label"] == "jaywalking")
    tn = sum(1 for r in results if r["gt_label"] == "compliant" and r["pred_label"] == "compliant")
    fp = sum(1 for r in results if r["gt_label"] == "compliant" and r["pred_label"] == "jaywalking")
    fn = sum(1 for r in results if r["gt_label"] == "jaywalking" and r["pred_label"] == "compliant")

    print(f"\n  TP={tp}  TN={tn}  FP={fp}  FN={fn}")
    if tp + fp > 0:
        print(f"  Precision: {tp/(tp+fp)*100:.0f}%")
    if tp + fn > 0:
        print(f"  Recall:    {tp/(tp+fn)*100:.0f}%")
    if tp + fp > 0 and tp + fn > 0:
        prec = tp/(tp+fp)
        rec = tp/(tp+fn)
        if prec + rec > 0:
            print(f"  F1:        {2*prec*rec/(prec+rec)*100:.0f}%")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
