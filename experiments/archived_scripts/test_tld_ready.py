#!/usr/bin/env python3
"""
TLD-READY Integration Test — 3 clips:
  1. SIGNAL_VIOLATION: red light + crossing (video_0012)
  2. NO_CROSSWALK: no zebra + crossing (video_0006)
  3. COMPLIANT: green light + crossing on zebra (video_0039)
"""
import sys, cv2, torch, numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from ultralytics import YOLO

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from utils.crossing.pose import PoseEstimator
from utils.crossing.zebra import ZebraDetector
from utils.crossing.traffic_light import TrafficLightClassifier, _get_model

CLIPS = [
    {"path": str(ROOT/"data/JAAD_clips/JAAD_clips/video_0137.mp4"), "name": "video_0137.mp4", "gt": "jaywalking", "gt_type": "SIGNAL_VIOLATION"},
    {"path": str(ROOT/"evaluation/jaad_positive/video_0006.mp4"),    "name": "video_0006.mp4",  "gt": "jaywalking", "gt_type": "NO_CROSSWALK"},
    {"path": str(ROOT/"evaluation/jaad_negative/video_0039.mp4"),    "name": "video_0039.mp4",  "gt": "compliant",  "gt_type": "green_light"},
]

YOLO_MODEL = str(ROOT / "yolo11x.pt")
CONFIDENCE = 0.7


def run_clip(clip):
    print(f"\n{'='*55}")
    print(f"  {clip['name']}  GT: {clip['gt']} ({clip['gt_type']})")
    print(f"{'='*55}")

    yolo = YOLO(YOLO_MODEL).to(0 if torch.cuda.is_available() else "cpu")
    pose_est = PoseEstimator(str(ROOT / "yolo11x-pose.pt"))
    tl_model = _get_model()
    ZebraDetector.reset_cache()
    TrafficLightClassifier.reset()

    cap = cv2.VideoCapture(clip["path"])
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_idx = 0
    person_tracks = defaultdict(list)  # tid -> [(frame, x_center)]
    tl_states_per_frame = []          # (frame, state)
    zebra_per_frame = []              # (frame, bool)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        frame_idx += 1

        results = yolo.track(frame, tracker="bytetrack.yaml", persist=True,
                             conf=CONFIDENCE, verbose=False,
                             device=0 if torch.cuda.is_available() else "cpu")

        zebra_poly = ZebraDetector.get_zebra_polygon(frame)
        zebra_present = zebra_poly is not None and len(zebra_poly) > 0
        zebra_per_frame.append(zebra_present)

        # Run TLD-READY on full frame (independent of main YOLO)
        tl_state = TrafficLightClassifier.detect_from_frame(frame)
        tl_states_per_frame.append((frame_idx, tl_state))
        if tl_state != "UNKNOWN":
            print(f"    Frame {frame_idx}: TL = {tl_state}")

        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()
            tids = results[0].boxes.id
            tids = tids.int().cpu().tolist() if tids is not None else [-1]*len(classes)

            h, w = frame.shape[:2]
            for box, c, tid in zip(boxes, classes, tids):
                x1, y1, x2, y2 = box
                cx = ((x1+x2)/2) / w
                yolo_id = int(c)

                if yolo_id == 0:  # pedestrian
                    person_tracks[tid].append((frame_idx, cx))

    cap.release()
    ZebraDetector.reset_cache()

    # --- Classification ---
    # Check pedestrian crossing: any track with wide x-range on road
    has_crossing = False
    for tid, frames in person_tracks.items():
        x_vals = [x for _, x in frames]
        if len(x_vals) < 3:
            continue
        x_range = max(x_vals) - min(x_vals)
        road_frames = sum(1 for x in x_vals if 0.15 < x < 0.85)
        if x_range > 0.12 and road_frames >= 3:
            has_crossing = True
            break

    # Traffic light states
    tl_counts = Counter(state for _, state in tl_states_per_frame)
    has_red = tl_counts.get("RED", 0) > 0
    has_green = tl_counts.get("GREEN", 0) > 0

    # Zebra
    zebra_count = sum(zebra_per_frame)
    has_zebra = zebra_count > len(zebra_per_frame) * 0.3  # present in >30% frames

    print(f"\n  --- Summary ---")
    print(f"  Frames: {frame_idx}")
    print(f"  Person tracks: {len(person_tracks)}")
    print(f"  Has crossing: {has_crossing}")
    print(f"  TL states: {dict(tl_counts)}")
    print(f"  Has red: {has_red}, Has green: {has_green}")
    print(f"  Has zebra: {has_zebra} ({zebra_count}/{len(zebra_per_frame)} frames)")

    # Classify
    if not has_crossing:
        pred = "compliant"
        reason = "no crossing detected"
    elif has_red:
        pred = "jaywalking"
        reason = "SIGNAL_VIOLATION: red light + crossing"
    elif has_green:
        pred = "compliant"
        reason = "green light — legal crossing"
    elif not has_zebra:
        pred = "jaywalking"
        reason = "NO_CROSSWALK: no zebra + crossing"
    else:
        pred = "compliant"
        reason = "crossing on zebra, no red light"

    match = pred == clip["gt"]
    print(f"\n  Prediction: {pred}")
    print(f"  Reason: {reason}")
    print(f"  GT: {clip['gt']}")
    print(f"  Match: {'✓ CORRECT' if match else '✗ WRONG'}")
    return match


if __name__ == "__main__":
    print("TLD-READY Integration Test — 3 clips")
    print("=" * 55)
    matches = sum(run_clip(c) for c in CLIPS)
    print(f"\n{'='*55}")
    print(f"RESULT: {matches}/3 correct")
    print(f"{'='*55}")
