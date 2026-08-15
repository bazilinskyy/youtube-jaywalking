#!/usr/bin/env python3
"""
Improved pipeline with multiple fixes:
1. Zebra is primary signal (more reliable than TL)
2. TL only matters when no zebra detected
3. Higher TL confidence threshold (reduce false positives)
4. Better crossing detection (wider road threshold)
5. Use JAAD metadata for evaluation comparison
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
EVENTS_CSV = ROOT / "evaluation" / "jaad_events_summary.csv"
YOLO_MODEL = str(ROOT / "yolo11x.pt")
CONFIDENCE = 0.5

# TLD-READY confidence threshold — raise to reduce false positives
TL_CONF_THRESHOLD = 0.25


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


def get_jaad_meta(clip_name):
    """Get JAAD metadata for a clip."""
    with open(EVENTS_CSV) as f:
        for row in csv.DictReader(f):
            if row["clip_name"] == clip_name:
                return {
                    "is_signalized": row["is_signalized"] == "True",
                    "has_zebra_gt": row["has_zebra"] == "True",
                    "has_red_gt": row["has_red_light"] == "True",
                    "has_green_gt": row["has_green_light"] == "True",
                    "violation_type": row["violation_type"],
                    "is_designated": row["is_designated"] == "True",
                    "road_type": row["road_type"],
                    "num_lanes": int(row["num_lanes"]),
                }
    return {}


def run_clip(clip):
    """Run pipeline with improved classification."""
    yolo = YOLO(YOLO_MODEL).to(0 if torch.cuda.is_available() else "cpu")
    tl_model = _get_model()
    ZebraDetector.reset_cache()
    TrafficLightClassifier.reset()

    cap = cv2.VideoCapture(clip["path"])
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_idx = 0
    person_tracks = defaultdict(list)
    tl_states = []
    zebra_frames = 0
    tl_detections_raw = []  # (frame, state, conf, x1,y1,x2,y2)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        frame_idx += 1

        # Zebra detection
        zebra_poly = ZebraDetector.get_zebra_polygon(frame)
        if zebra_poly is not None and len(zebra_poly) > 0:
            zebra_frames += 1

        # TLD-READY full frame — store raw detections with confidence
        tl_results = tl_model.predict(frame, verbose=False, conf=0.10, imgsz=640)
        if tl_results and tl_results[0].boxes is not None:
            for box in tl_results[0].boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                state = _map_class(cls_id)
                if state in ("OFF", "UNKNOWN"):
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                tl_detections_raw.append((frame_idx, state, conf, x1, y1, x2, y2))

                # Only use high-confidence detections for classification
                if conf >= TL_CONF_THRESHOLD:
                    tl_states.append(state)

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
                if int(c) == 0:  # pedestrian
                    cx = ((box[0]+box[2])/2) / W
                    person_tracks[tid].append(cx)

    cap.release()
    ZebraDetector.reset_cache()

    # --- Classification (IMPROVED LOGIC) ---

    # 1. Detect crossing
    has_crossing = False
    crossing_tracks = []
    for tid, xs in person_tracks.items():
        if len(xs) < 3:
            continue
        x_range = max(xs) - min(xs)
        road = sum(1 for x in xs if 0.15 < x < 0.85)
        if x_range > 0.10 and road >= 2:  # More lenient threshold
            has_crossing = True
            crossing_tracks.append(tid)

    # 2. Zebra (primary signal — more reliable)
    total_frames = max(frame_idx, 1)
    has_zebra = zebra_frames > total_frames * 0.25  # Lower threshold

    # 3. Traffic light (secondary — only if confident AND signalized context)
    tl_counts = Counter(tl_states)
    has_red = tl_counts.get("RED", 0) > 0
    has_green = tl_counts.get("GREEN", 0) > 0

    # Count high-conf red/green separately
    high_conf_red = sum(1 for s in tl_states if s == "RED")
    high_conf_green = sum(1 for s in tl_states if s == "GREEN")

    # 4. Classify
    if not has_crossing:
        pred = "compliant"
        reason = "no crossing detected"
        confidence = "high"
    elif has_zebra:
        # Zebra present — crossing on zebra is legal
        # Check TL: if RED is minority (<30% of colored detections), it's noise
        colored = high_conf_red + high_conf_green
        red_ratio = high_conf_red / colored if colored > 0 else 0

        if has_red and red_ratio > 0.3 and high_conf_red >= 5:
            # RED is significant (>30% of detections, >=5 frames)
            pred = "jaywalking"
            reason = f"SIGNAL_VIOLATION: red_ratio={red_ratio:.0%} ({high_conf_red}/{colored})"
            confidence = "medium"
        else:
            pred = "compliant"
            reason = f"on zebra ({zebra_frames}/{total_frames}f)"
            confidence = "high"
    else:
        # No zebra — check TL
        colored = high_conf_red + high_conf_green
        red_ratio = high_conf_red / colored if colored > 0 else 0

        if has_red and red_ratio > 0.2 and high_conf_red >= 3:
            pred = "jaywalking"
            reason = f"SIGNAL_VIOLATION: red_ratio={red_ratio:.0%} ({high_conf_red}/{colored}), no zebra"
            confidence = "medium"
        elif has_green and high_conf_green >= 5:
            pred = "compliant"
            reason = f"green({high_conf_green}) = legal crossing"
            confidence = "medium"
        else:
            pred = "jaywalking"
            reason = f"NO_CROSSWALK: no zebra, no clear signal"
            confidence = "high"

    return {
        "pred": pred,
        "reason": reason,
        "confidence": confidence,
        "frames": frame_idx,
        "persons": len(person_tracks),
        "crossing_persons": len(crossing_tracks),
        "tl_red_highconf": high_conf_red,
        "tl_green_highconf": high_conf_green,
        "tl_red_total": tl_counts.get("RED", 0),
        "tl_green_total": tl_counts.get("GREEN", 0),
        "tl_unknown": tl_counts.get("UNKNOWN", 0),
        "zebra_pct": round(zebra_frames / total_frames * 100),
    }


def main():
    clips = get_clips()
    print(f"Running improved pipeline on {len(clips)} clips...")
    print(f"TL confidence threshold: {TL_CONF_THRESHOLD}\n")

    results = []
    t0 = time.time()
    for i, clip in enumerate(clips):
        meta = get_jaad_meta(clip["name"])
        print(f"[{i+1}/{len(clips)}] {clip['name']} (signalized={meta.get('is_signalized','?')}, zebra_gt={meta.get('has_zebra_gt','?')})...", end=" ", flush=True)
        r = run_clip(clip)
        gt = clip["gt"]
        match = r["pred"] == gt
        r["name"] = clip["name"]
        r["gt"] = gt
        r["verdict"] = clip["verdict"]
        r["match"] = match
        r["meta"] = meta
        results.append(r)
        symbol = "✓" if match else "✗"
        print(f"GT={gt:<12} pred={r['pred']:<12} {symbol}  ({r['reason']})")

    elapsed = time.time() - t0

    # Summary
    print(f"\n{'='*110}")
    print(f"{'Clip':<22} {'GT':<12} {'Pred':<12} {'Match':<6} {'Reason':<35} {'Red':<5} {'Green':<5} {'Zebra%':<7} {'Sig':<5}")
    print(f"{'-'*110}")
    for r in results:
        m = "✓" if r["match"] else "✗"
        sig = "Y" if r["meta"].get("is_signalized") else "N"
        print(f"{r['name']:<22} {r['gt']:<12} {r['pred']:<12} {m:<6} {r['reason']:<35} {r['tl_red_highconf']:<5} {r['tl_green_highconf']:<5} {r['zebra_pct']}%{'':<4} {sig}")

    matches = sum(1 for r in results if r["match"])
    total = len(results)
    print(f"{'-'*110}")
    print(f"Accuracy: {matches}/{total} ({matches/total*100:.0f}%)")
    print(f"Time: {elapsed:.0f}s ({elapsed/total:.1f}s per clip)")

    # Confusion
    tp = sum(1 for r in results if r["gt"] == "jaywalking" and r["pred"] == "jaywalking")
    tn = sum(1 for r in results if r["gt"] == "compliant" and r["pred"] == "compliant")
    fp = sum(1 for r in results if r["gt"] == "compliant" and r["pred"] == "jaywalking")
    fn = sum(1 for r in results if r["gt"] == "jaywalking" and r["pred"] == "compliant")
    print(f"\nConfusion: TP={tp} TN={tn} FP={fp} FN={fn}")
    if tp+fp > 0:
        print(f"Precision: {tp/(tp+fp)*100:.0f}%")
    if tp+fn > 0:
        print(f"Recall:    {tp/(tp+fn)*100:.0f}%")
    if tp+fp > 0 and tp+fn > 0:
        p, r_ = tp/(tp+fp), tp/(tp+fn)
        print(f"F1:        {2*p*r_/(p+r_)*100:.0f}%")

    # Analyze failures
    print(f"\n--- FAILURE ANALYSIS ---")
    for r in results:
        if not r["match"]:
            meta = r["meta"]
            print(f"  {r['name']}: GT={r['gt']}, pred={r['pred']}")
            print(f"    signalized={meta.get('is_signalized')}, zebra_gt={meta.get('has_zebra_gt')}, red_gt={meta.get('has_red_gt')}, green_gt={meta.get('has_green_gt')}")
            print(f"    algo: zebra={r['zebra_pct']}%, red_det={r['tl_red_highconf']}, green_det={r['tl_green_highconf']}")
            print(f"    reason: {r['reason']}")

    print(f"{'='*110}")

    # Save
    out_csv = ROOT / "evaluation" / "pipeline_results_v2.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name","gt","pred","match","verdict","reason","confidence",
                                                "frames","persons","crossing_persons",
                                                "tl_red_highconf","tl_green_highconf",
                                                "tl_red_total","tl_green_total","tl_unknown",
                                                "zebra_pct"])
        writer.writeheader()
        for r in results:
            row = {k: r.get(k, "") for k in writer.fieldnames}
            writer.writerow(row)
    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()
