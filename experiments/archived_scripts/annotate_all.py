#!/usr/bin/env python3
"""
Annotate all 13 reviewed clips with detailed per-frame logic.
Clear visual indicators so user can verify jaywalking vs not.
"""
import sys, cv2, torch, csv, time
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from ultralytics import YOLO

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from utils.crossing.pose import PoseEstimator
from utils.crossing.zebra import ZebraDetector
from utils.crossing.traffic_light import TrafficLightClassifier, _get_model, _map_class

REVIEW_CSV = ROOT / "evaluation" / "review_results.csv"
YOLO_MODEL = str(ROOT / "yolo11x.pt")
CONFIDENCE = 0.5
OUTPUT_DIR = ROOT / "evaluation" / "annotated_all"
OUTPUT_DIR.mkdir(exist_ok=True)

# Colors
C_RED = (0, 0, 255)
C_GREEN = (0, 255, 0)
C_YELLOW = (0, 200, 255)
C_WHITE = (255, 255, 255)
C_BLACK = (0, 0, 0)
C_CYAN = (255, 255, 0)
C_ZEBRA = (255, 200, 0)
C_JAYWALK = (0, 0, 255)     # red banner = jaywalking
C_COMPLIANT = (0, 180, 0)   # green banner = compliant
C_UNKNOWN = (128, 128, 128)

TL_COLORS = {"RED": C_RED, "GREEN": C_GREEN, "YELLOW": C_YELLOW, "UNKNOWN": C_UNKNOWN, "OFF": (80, 80, 80)}


def get_clips():
    clips = []
    with open(REVIEW_CSV) as f:
        for row in csv.DictReader(f):
            if row["reviewer_verdict"] not in ("correct", "wrong", "ambiguous"):
                continue
            for clip_dir in [ROOT/"evaluation/jaad_positive", ROOT/"evaluation/jaad_negative"]:
                path = clip_dir / row["clip_name"]
                if path.exists():
                    clips.append({"path": str(path), "name": row["clip_name"],
                                  "gt": row["auto_label"], "verdict": row["reviewer_verdict"]})
                    break
    return clips


def draw_banner(frame, text, color, y_pos, font_scale=1.2, thickness=3):
    """Draw a colored banner with text."""
    h, w = frame.shape[:2]
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    x = 10
    cv2.rectangle(frame, (x-5, y_pos-th-10), (x+tw+10, y_pos+10), color, -1)
    cv2.putText(frame, text, (x, y_pos), cv2.FONT_HERSHEY_SIMPLEX, font_scale, C_WHITE, thickness)
    return frame


def draw_zebra_overlay(frame, polygon):
    if polygon is None or len(polygon) == 0:
        return frame
    overlay = frame.copy()
    pts = np.array(polygon, dtype=np.int32)
    cv2.fillPoly(overlay, [pts], C_ZEBRA)
    return cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)


def classify_frame(has_crossing, has_red, has_green, has_zebra):
    """Return (label, reason, color) for current frame state."""
    if not has_crossing:
        return "NOT CROSSING", "person not on road", C_UNKNOWN
    if has_red:
        return "JAYWALKING", "RED light + crossing", C_JAYWALK
    if has_green:
        return "COMPLIANT", "GREEN light = legal", C_COMPLIANT
    if not has_zebra:
        return "JAYWALKING", "NO CROSSWALK + crossing", C_JAYWALK
    return "COMPLIANT", "on zebra, no red", C_COMPLIANT


def run_clip(clip):
    print(f"  {clip['name']}...", end=" ", flush=True)
    out_path = OUTPUT_DIR / f"{clip['name']}_annotated.mp4"

    yolo = YOLO(YOLO_MODEL).to(0 if torch.cuda.is_available() else "cpu")
    tl_model = _get_model()
    ZebraDetector.reset_cache()
    TrafficLightClassifier.reset()

    cap = cv2.VideoCapture(clip["path"])
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (W, H))

    frame_idx = 0
    person_tracks = defaultdict(list)
    tl_states_all = []
    zebra_frame_count = 0
    verdicts = []  # per-frame verdict

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        frame_idx += 1
        vis = frame.copy()

        # --- Zebra ---
        zebra_poly = ZebraDetector.get_zebra_polygon(frame)
        zebra_present = zebra_poly is not None and len(zebra_poly) > 0
        if zebra_present:
            zebra_frame_count += 1
            vis = draw_zebra_overlay(vis, zebra_poly)

        # --- TLD-READY ---
        tl_state = TrafficLightClassifier.detect_from_frame(frame)
        tl_states_all.append(tl_state)

        # Draw TL detections on frame
        tl_results = tl_model.predict(frame, verbose=False, conf=0.15, imgsz=640)
        if tl_results and tl_results[0].boxes is not None:
            for box in tl_results[0].boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                state = _map_class(cls_id)
                if state in ("OFF", "UNKNOWN"):
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                color = TL_COLORS.get(state, C_UNKNOWN)
                cv2.rectangle(vis, (int(x1),int(y1)), (int(x2),int(y2)), color, 2)
                cv2.putText(vis, f"{state} {conf:.2f}", (int(x1),int(y1)-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # --- YOLO tracking ---
        results = yolo.track(frame, tracker="bytetrack.yaml", persist=True,
                             conf=CONFIDENCE, verbose=False,
                             device=0 if torch.cuda.is_available() else "cpu")

        road_persons = 0
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
                    color = C_CYAN if on_road else (150, 150, 150)
                    cv2.rectangle(vis, (x1,y1), (x2,y2), color, 2)
                    cv2.putText(vis, f"ID:{tid}", (x1, y1-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    if on_road:
                        road_persons += 1
                elif yolo_id == 2:
                    cv2.rectangle(vis, (x1,y1), (x2,y2), (100,100,100), 1)

        # --- Per-frame classification ---
        has_crossing = False
        for tid, xs in person_tracks.items():
            if len(xs) < 3:
                continue
            x_range = max(xs) - min(xs)
            road = sum(1 for x in xs if 0.15 < x < 0.85)
            if x_range > 0.12 and road >= 3:
                has_crossing = True
                break

        tl_counts = Counter(tl_states_all)
        has_red = tl_counts.get("RED", 0) > 0
        has_green = tl_counts.get("GREEN", 0) > 0
        has_zebra = zebra_frame_count > frame_idx * 0.3

        label, reason, color = classify_frame(has_crossing, has_red, has_green, has_zebra)
        verdicts.append(label)

        # === DRAW HUD ===

        # TOP BANNER: verdict
        draw_banner(vis, f"  {label}  ", color, 40, 1.4, 4)

        # Reason below banner
        cv2.putText(vis, reason, (15, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, C_WHITE, 2)

        # Left panel: GT + frame
        cv2.putText(vis, f"GT: {clip['gt'].upper()}", (10, H-60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, C_WHITE, 2)
        cv2.putText(vis, f"Frame {frame_idx}", (10, H-30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, C_WHITE, 2)

        # Right panel: logic state
        px = W - 350
        cv2.putText(vis, f"Crossing: {'YES' if has_crossing else 'no'}", (px, H-120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, C_CYAN if has_crossing else C_UNKNOWN, 2)
        cv2.putText(vis, f"Red TL:   {'YES' if has_red else 'no'}", (px, H-95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, C_RED if has_red else C_UNKNOWN, 2)
        cv2.putText(vis, f"Green TL: {'YES' if has_green else 'no'}", (px, H-70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, C_GREEN if has_green else C_UNKNOWN, 2)
        cv2.putText(vis, f"Zebra:    {'YES' if has_zebra else 'no'}", (px, H-45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, C_ZEBRA if has_zebra else C_UNKNOWN, 2)

        # TL state counter top-right
        if tl_counts:
            state_str = " ".join(f"{s}:{c}" for s, c in tl_counts.most_common() if s != "UNKNOWN")
            if state_str:
                cv2.putText(vis, f"TL: {state_str}", (W-400, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, C_WHITE, 2)

        writer.write(vis)
        if frame_idx % 50 == 0:
            print(f"{frame_idx}...", end=" ", flush=True)

    cap.release()
    writer.release()
    ZebraDetector.reset_cache()

    # Final verdict
    jaywalk_frames = verdicts.count("JAYWALKING")
    compliant_frames = verdicts.count("COMPLIANT")
    not_crossing_frames = verdicts.count("NOT CROSSING")
    final = "JAYWALKING" if jaywalk_frames > compliant_frames else "COMPLIANT"

    print(f"done ({frame_idx}f)")
    return {
        "name": clip["name"],
        "gt": clip["gt"],
        "verdict": clip["verdict"],
        "final_pred": final.lower(),
        "jaywalk_frames": jaywalk_frames,
        "compliant_frames": compliant_frames,
        "not_crossing_frames": not_crossing_frames,
        "total_frames": frame_idx,
        "match": final.lower() == clip["gt"],
        "path": str(out_path),
    }


def main():
    clips = get_clips()
    print(f"Annotating {len(clips)} clips with detailed per-frame logic...\n")

    results = []
    t0 = time.time()
    for clip in clips:
        r = run_clip(clip)
        results.append(r)

    elapsed = time.time() - t0

    # Summary table
    print(f"\n{'='*100}")
    print(f"{'Clip':<22} {'GT':<12} {'Predicted':<12} {'Match':<6} {'J frames':<10} {'C frames':<10} {'NC frames':<10} {'Total':<7}")
    print(f"{'-'*100}")
    for r in results:
        m = "✓" if r["match"] else "✗"
        print(f"{r['name']:<22} {r['gt']:<12} {r['final_pred']:<12} {m:<6} {r['jaywalk_frames']:<10} {r['compliant_frames']:<10} {r['not_crossing_frames']:<10} {r['total_frames']:<7}")

    matches = sum(1 for r in results if r["match"])
    total = len(results)
    print(f"{'-'*100}")
    print(f"Accuracy: {matches}/{total} ({matches/total*100:.0f}%)")
    print(f"Time: {elapsed:.0f}s")

    # Save
    out_csv = ROOT / "evaluation" / "annotated_all" / "results.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name","gt","verdict","final_pred","match",
                                                "jaywalk_frames","compliant_frames","not_crossing_frames","total_frames","path"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults: {out_csv}")
    print(f"\nVideos saved to: {OUTPUT_DIR}/")
    print("Open each with: mpv <path>")
    print("\nREVIEW EACH VIDEO — then tell me:")
    print("  'video_XXXX is jaywalking' or 'video_XXXX is compliant'")
    print("  for clips where the algorithm got it wrong.")


if __name__ == "__main__":
    main()
