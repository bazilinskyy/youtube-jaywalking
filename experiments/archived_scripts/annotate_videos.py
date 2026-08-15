#!/usr/bin/env python3
"""Generate annotated videos for the 3 test clips with TLD-READY detections."""
import sys, cv2, torch, numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from ultralytics import YOLO

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from utils.crossing.pose import PoseEstimator
from utils.crossing.zebra import ZebraDetector
from utils.crossing.traffic_light import TrafficLightClassifier, _get_model, _map_class

CLIPS = [
    {"path": str(ROOT/"data/JAAD_clips/JAAD_clips/video_0137.mp4"), "name": "video_0137", "gt": "jaywalking (SIGNAL_VIOLATION)"},
    {"path": str(ROOT/"evaluation/jaad_positive/video_0006.mp4"),    "name": "video_0006",  "gt": "jaywalking (NO_CROSSWALK)"},
    {"path": str(ROOT/"evaluation/jaad_negative/video_0039.mp4"),    "name": "video_0039",  "gt": "compliant (green light)"},
]

YOLO_MODEL = str(ROOT / "yolo11x.pt")
CONFIDENCE = 0.5
OUTPUT_DIR = ROOT / "evaluation" / "annotated_videos"
OUTPUT_DIR.mkdir(exist_ok=True)

# Colors
COLOR_RED = (0, 0, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_YELLOW = (0, 200, 255)
COLOR_UNKNOWN = (128, 128, 128)
COLOR_ZEBRA = (255, 200, 0)
COLOR_PERSON = (255, 255, 0)

TL_COLORS = {"RED": COLOR_RED, "GREEN": COLOR_GREEN, "YELLOW": COLOR_YELLOW, "UNKNOWN": COLOR_UNKNOWN, "OFF": (80, 80, 80)}


def draw_zebra_mask(frame, zebra_polygon):
    """Overlay semi-transparent zebra detection on frame."""
    if zebra_polygon is None or len(zebra_polygon) == 0:
        return frame
    overlay = frame.copy()
    pts = np.array(zebra_polygon, dtype=np.int32)
    cv2.fillPoly(overlay, [pts], (255, 200, 0))
    return cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)


def run_clip(clip):
    print(f"\nProcessing {clip['name']}...")
    out_path = OUTPUT_DIR / f"{clip['name']}_annotated.mp4"

    yolo = YOLO(YOLO_MODEL).to(0 if torch.cuda.is_available() else "cpu")
    tl_model = _get_model()
    ZebraDetector.reset_cache()
    TrafficLightClassifier.reset()

    cap = cv2.VideoCapture(clip["path"])
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    frame_idx = 0
    tl_states = []

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        frame_idx += 1
        vis = frame.copy()

        # --- Zebra detection ---
        zebra_poly = ZebraDetector.get_zebra_polygon(frame)
        zebra_present = zebra_poly is not None and len(zebra_poly) > 0
        if zebra_present:
            vis = draw_zebra_mask(vis, zebra_poly)
            cv2.putText(vis, "ZEBRA", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_ZEBRA, 2)

        # --- TLD-READY full-frame detection ---
        tl_results = tl_model.predict(frame, verbose=False, conf=0.15, imgsz=640)
        if tl_results and tl_results[0].boxes is not None:
            for box in tl_results[0].boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                state = _map_class(cls_id)
                if state in ("OFF", "UNKNOWN"):
                    continue
                tl_states.append(state)
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                color = TL_COLORS.get(state, COLOR_UNKNOWN)
                cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                label = f"{state} {conf:.2f}"
                cv2.putText(vis, label, (int(x1), int(y1) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # --- YOLO tracking (persons + other objects) ---
        results = yolo.track(frame, tracker="bytetrack.yaml", persist=True,
                             conf=CONFIDENCE, verbose=False,
                             device=0 if torch.cuda.is_available() else "cpu")

        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()
            tids = results[0].boxes.id
            tids = tids.int().cpu().tolist() if tids is not None else [-1]*len(classes)

            for box, c, tid in zip(boxes, classes, tids):
                x1, y1, x2, y2 = [int(v) for v in box]
                yolo_id = int(c)

                if yolo_id == 0:  # pedestrian
                    cv2.rectangle(vis, (x1, y1), (x2, y2), COLOR_PERSON, 2)
                    cv2.putText(vis, f"ID:{tid}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_PERSON, 1)
                elif yolo_id == 2:  # car
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (200, 200, 200), 1)

        # --- HUD overlay ---
        # Top-left: clip info
        cv2.putText(vis, f"Frame: {frame_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(vis, f"GT: {clip['gt']}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Top-right: TL state summary
        tl_counts = Counter(tl_states)
        if tl_counts:
            state_str = " ".join(f"{s}:{c}" for s, c in tl_counts.most_common())
            cv2.putText(vis, f"TL: {state_str}", (w - 400, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Current frame TL state
        if tl_states:
            cur_state = tl_states[-1]
            color = TL_COLORS.get(cur_state, COLOR_UNKNOWN)
            cv2.putText(vis, f"NOW: {cur_state}", (w - 300, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 3)

        writer.write(vis)

        if frame_idx % 50 == 0:
            print(f"  Frame {frame_idx}...")

    cap.release()
    writer.release()
    ZebraDetector.reset_cache()
    print(f"  Saved: {out_path}")
    return str(out_path)


if __name__ == "__main__":
    print("Generating annotated videos with TLD-READY detections")
    print("=" * 55)
    paths = [run_clip(c) for c in CLIPS]
    print(f"\n{'='*55}")
    print("Done. Videos saved to:")
    for p in paths:
        print(f"  {p}")
