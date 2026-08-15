#!/usr/bin/env python3
"""
Annotate all 13 clips with v3 pipeline logic (no zebra, phantom GREEN fix).
Replaces old annotated videos.
"""
import sys, cv2, torch, csv, time
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from ultralytics import YOLO

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from utils.crossing.traffic_light import TrafficLightClassifier, _get_model, _map_class

REVIEW_CSV = ROOT / "evaluation" / "review_results.csv"
EVENTS_CSV = ROOT / "evaluation" / "jaad_events_summary.csv"
YOLO_MODEL = str(ROOT / "yolo11x.pt")
CONFIDENCE = 0.5
OUTPUT_DIR = ROOT / "evaluation" / "annotated_all"
OUTPUT_DIR.mkdir(exist_ok=True)

C_RED = (0, 0, 255)
C_GREEN = (0, 255, 0)
C_YELLOW = (0, 200, 255)
C_WHITE = (255, 255, 255)
C_CYAN = (255, 255, 0)
C_JAYWALK = (0, 0, 255)
C_COMPLIANT = (0, 180, 0)
C_UNKNOWN = (128, 128, 128)
C_GRAY = (100, 100, 100)
C_PED = (255, 255, 255)      # WHITE for pedestrians
C_CAR = (150, 150, 150)       # GRAY for cars
C_TL_RED = (0, 0, 255)        # RED for TL
C_TL_GREEN = (0, 255, 0)      # GREEN for TL
C_TL_YELLOW = (0, 200, 255)   # YELLOW for TL


def get_clips():
    clips = []
    with open(REVIEW_CSV) as f:
        for row in csv.DictReader(f):
            if row["reviewer_verdict"] not in ("correct", "wrong", "ambiguous"):
                continue
            for d in [ROOT/"evaluation/jaad_positive", ROOT/"evaluation/jaad_negative"]:
                p = d / row["clip_name"]
                if p.exists():
                    clips.append({"path": str(p), "name": row["clip_name"],
                                  "gt": row["auto_label"], "verdict": row["reviewer_verdict"]})
                    break
    return clips


def get_jaad_meta(clip_name):
    with open(EVENTS_CSV) as f:
        for row in csv.DictReader(f):
            if row["clip_name"] == clip_name:
                return {
                    "is_signalized": row["is_signalized"] == "True",
                    "has_zebra_gt": row["has_zebra"] == "True",
                    "violation_type": row["violation_type"],
                }
    return {}


def draw_banner(frame, text, color, y_pos, font_scale=1.3, thickness=3):
    h, w = frame.shape[:2]
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    cv2.rectangle(frame, (5, y_pos-th-12), (tw+20, y_pos+10), color, -1)
    cv2.putText(frame, text, (12, y_pos), cv2.FONT_HERSHEY_SIMPLEX, font_scale, C_WHITE, thickness)


def run_clip(clip):
    print(f"  {clip['name']}...", end=" ", flush=True)
    out_path = OUTPUT_DIR / f"{clip['name']}_annotated.mp4"

    yolo = YOLO(YOLO_MODEL).to(0 if torch.cuda.is_available() else "cpu")
    tl_model = _get_model()
    TrafficLightClassifier.reset()

    cap = cv2.VideoCapture(clip["path"])
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (W, H))

    frame_idx = 0
    person_tracks = defaultdict(list)
    tl_states = []
    meta = get_jaad_meta(clip["name"])

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        frame_idx += 1
        vis = frame.copy()

        # TLD-READY
        tl_state = TrafficLightClassifier.detect_from_frame(frame)
        if tl_state != "UNKNOWN":
            tl_states.append(tl_state)

        # Draw TL detections — RED/GREEN boxes ONLY on traffic lights, not people
        tl_results = tl_model.predict(frame, verbose=False, conf=0.10, imgsz=640)
        if tl_results and tl_results[0].boxes is not None:
            for box in tl_results[0].boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                state = _map_class(cls_id)
                if state in ("OFF", "UNKNOWN"):
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                bw = x2 - x1
                bh = y2 - y1
                # Skip if box looks like a person (tall and narrow) — likely misclassification
                aspect = bh / bw if bw > 0 else 0
                if aspect > 3.0 and bh > 80:
                    continue  # skip — probably a person, not a TL
                color = C_TL_RED if state == "RED" else C_TL_GREEN if state == "GREEN" else C_TL_YELLOW
                cv2.rectangle(vis, (int(x1),int(y1)), (int(x2),int(y2)), color, 2)
                cv2.putText(vis, f"TL:{state} {conf:.2f}", (int(x1),int(y1)-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        # YOLO tracking
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
                    cx = ((x1+x2)/2) / W
                    person_tracks[tid].append(cx)
                    on_road = 0.15 < cx < 0.85
                    box_color = C_CYAN if on_road else C_GRAY
                    cv2.rectangle(vis, (x1,y1), (x2,y2), box_color, 2)
                    label = f"PED ID:{tid}"
                    cv2.putText(vis, label, (x1, y1-8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2)
                elif yolo_id == 2:  # car
                    cv2.rectangle(vis, (x1,y1), (x2,y2), (180,180,180), 1)
                    cv2.putText(vis, "CAR", (x1, y1-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180,180,180), 1)

        # --- v3 Classification ---
        has_crossing = False
        for tid, xs in person_tracks.items():
            if len(xs) < 3:
                continue
            x_range = max(xs) - min(xs)
            road = sum(1 for x in xs if 0.15 < x < 0.85)
            if x_range > 0.10 and road >= 2:
                has_crossing = True
                break

        tl_counts = Counter(tl_states)
        has_red = tl_counts.get("RED", 0) > 0
        has_green = tl_counts.get("GREEN", 0) > 0
        red_count = tl_counts.get("RED", 0)
        green_count = tl_counts.get("GREEN", 0)
        colored = red_count + green_count
        red_ratio = red_count / colored if colored > 0 else 0

        if not has_crossing:
            label, reason, color = "NOT CROSSING", "no crossing detected", C_UNKNOWN
        elif has_red and red_ratio > 0.15:
            label, reason, color = "JAYWALKING", f"RED + crossing (ratio={red_ratio:.0%})", C_JAYWALK
        elif has_green and has_red and green_count >= 10:
            label, reason, color = "COMPLIANT", f"GREEN light ({green_count}G/{red_count}R)", C_COMPLIANT
        elif has_green and not has_red:
            label, reason, color = "JAYWALKING", f"phantom GREEN ({green_count}G, 0R)", C_JAYWALK
        else:
            label, reason, color = "JAYWALKING", f"no signal ({red_count}R/{green_count}G)", C_JAYWALK

        # === HUD ===
        draw_banner(vis, f"  {label}  ", color, 40, 1.4, 4)
        cv2.putText(vis, reason, (15, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.65, C_WHITE, 2)

        # GT
        gt_text = f"GT: {clip['gt'].upper()}"
        gt_color = C_COMPLIANT if clip["gt"] == "compliant" else C_JAYWALK
        cv2.putText(vis, gt_text, (10, H-55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, gt_color, 2)
        sig_text = f"Signalized: {'YES' if meta.get('is_signalized') else 'NO'}"
        cv2.putText(vis, sig_text, (10, H-30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_WHITE, 1)
        cv2.putText(vis, f"Frame {frame_idx}", (10, H-8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_GRAY, 1)

        # Logic panel
        px = W - 340
        cv2.putText(vis, f"Crossing: {'YES' if has_crossing else 'no'}", (px, H-100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, C_CYAN if has_crossing else C_UNKNOWN, 2)
        cv2.putText(vis, f"Red:   {red_count}  ({red_ratio:.0%})", (px, H-78),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, C_RED if has_red else C_UNKNOWN, 2)
        cv2.putText(vis, f"Green: {green_count}", (px, H-56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, C_GREEN if has_green else C_UNKNOWN, 2)
        cv2.putText(vis, f"Phantom: {'YES (green-only)' if (has_green and not has_red) else 'no'}", (px, H-34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_YELLOW if (has_green and not has_red) else C_GRAY, 1)

        # Color legend (top right)
        lx = W - 200
        ly = 25
        cv2.putText(vis, "LEGEND:", (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_WHITE, 1)
        ly += 18
        cv2.rectangle(vis, (lx, ly-10), (lx+12, ly), C_CYAN, -1)
        cv2.putText(vis, "= Pedestrian", (lx+16, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_CYAN, 1)
        ly += 16
        cv2.rectangle(vis, (lx, ly-10), (lx+12, ly), C_RED, -1)
        cv2.putText(vis, "= TL RED", (lx+16, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_RED, 1)
        ly += 16
        cv2.rectangle(vis, (lx, ly-10), (lx+12, ly), C_GREEN, -1)
        cv2.putText(vis, "= TL GREEN", (lx+16, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_GREEN, 1)
        ly += 16
        cv2.rectangle(vis, (lx, ly-10), (lx+12, ly), C_GRAY, -1)
        cv2.putText(vis, "= CAR / off-road", (lx+16, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_GRAY, 1)

        writer.write(vis)
        if frame_idx % 60 == 0:
            print(f"{frame_idx}...", end=" ", flush=True)

    cap.release()
    writer.release()

    jaywalk_f = sum(1 for s in ["JAYWALKING"] if True)  # placeholder
    tl_counts_final = Counter(tl_states)
    print(f"done ({frame_idx}f)")
    return {"name": clip["name"], "gt": clip["gt"], "path": str(out_path)}


def main():
    clips = get_clips()
    print(f"Annotating {len(clips)} clips (v3 logic)...\n")
    t0 = time.time()
    for clip in clips:
        run_clip(clip)
    print(f"\nDone in {time.time()-t0:.0f}s")
    print(f"Videos: {OUTPUT_DIR}/")
    print("Replace old saves with: mpv <path>")


if __name__ == "__main__":
    main()
