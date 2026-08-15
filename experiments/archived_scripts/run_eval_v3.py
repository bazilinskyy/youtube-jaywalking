#!/usr/bin/env python3
"""
Pipeline v3: Drop zebra detection (unreliable).
Use only: crossing detection + TLD-READY traffic lights.
"""
import sys, cv2, torch, csv, time
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from ultralytics import YOLO

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from utils.crossing.pose import PoseEstimator
from utils.crossing.traffic_light import TrafficLightClassifier, _get_model, _map_class

REVIEW_CSV = ROOT / "evaluation" / "review_results.csv"
EVENTS_CSV = ROOT / "evaluation" / "jaad_events_summary.csv"
YOLO_MODEL = str(ROOT / "yolo11x.pt")
CONFIDENCE = 0.5
TL_CONF = 0.25


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
                    "has_red_gt": row["has_red_light"] == "True",
                    "has_green_gt": row["has_green_light"] == "True",
                    "violation_type": row["violation_type"],
                    "is_designated": row["is_designated"] == "True",
                }
    return {}


def run_clip(clip):
    yolo = YOLO(YOLO_MODEL).to(0 if torch.cuda.is_available() else "cpu")
    tl_model = _get_model()
    TrafficLightClassifier.reset()

    cap = cv2.VideoCapture(clip["path"])
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_idx = 0
    person_tracks = defaultdict(list)
    tl_states = []

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        frame_idx += 1

        # TLD-READY
        tl_state = TrafficLightClassifier.detect_from_frame(frame)
        if tl_state != "UNKNOWN":
            tl_states.append(tl_state)

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
                if int(c) == 0:
                    cx = ((box[0]+box[2])/2) / W
                    person_tracks[tid].append(cx)

    cap.release()

    # Crossing detection
    has_crossing = False
    for tid, xs in person_tracks.items():
        if len(xs) < 3:
            continue
        x_range = max(xs) - min(xs)
        road = sum(1 for x in xs if 0.15 < x < 0.85)
        if x_range > 0.10 and road >= 2:
            has_crossing = True
            break

    # TL state
    tl_counts = Counter(tl_states)
    has_red = tl_counts.get("RED", 0) > 0
    has_green = tl_counts.get("GREEN", 0) > 0
    red_count = tl_counts.get("RED", 0)
    green_count = tl_counts.get("GREEN", 0)
    colored = red_count + green_count
    red_ratio = red_count / colored if colored > 0 else 0

    # Classification — NO ZEBRA, just crossing + TL
    if not has_crossing:
        pred = "compliant"
        reason = "no crossing"
    elif has_red and red_ratio > 0.15:
        # RED detected with meaningful ratio → SIGNAL_VIOLATION
        pred = "jaywalking"
        reason = f"SIGNAL_VIOLATION: red_ratio={red_ratio:.0%} ({red_count}/{colored})"
    elif has_green and has_red and green_count >= 10:
        # Both RED and GREEN detected → TL is real, GREEN means legal
        pred = "compliant"
        reason = f"GREEN light ({green_count}G/{red_count}R)"
    elif has_green and not has_red:
        # GREEN only, NO red at all → phantom detection, unreliable
        # Treat as "no signal" → classify as jaywalking
        pred = "jaywalking"
        reason = f"NO_SIGNAL: green-only ({green_count}G, 0R = phantom)"
    else:
        # No clear TL signal → jaywalking
        pred = "jaywalking"
        reason = f"NO_SIGNAL: no clear TL ({red_count}R/{green_count}G)"

    return {
        "pred": pred, "reason": reason,
        "frames": frame_idx, "persons": len(person_tracks),
        "red": red_count, "green": green_count, "red_ratio": round(red_ratio, 2),
    }


def main():
    clips = get_clips()
    print(f"Pipeline v3: crossing + TL only (no zebra)\n")

    results = []
    t0 = time.time()
    for i, clip in enumerate(clips):
        meta = get_jaad_meta(clip["name"])
        r = run_clip(clip)
        match = r["pred"] == clip["gt"]
        r.update({"name": clip["name"], "gt": clip["gt"], "verdict": clip["verdict"],
                   "match": match, "meta": meta})
        results.append(r)
        s = "✓" if match else "✗"
        print(f"[{i+1}/{len(clips)}] {clip['name']}: GT={clip['gt']:<12} pred={r['pred']:<12} {s}  {r['reason']}")

    elapsed = time.time() - t0
    print(f"\n{'='*100}")
    print(f"{'Clip':<22} {'GT':<12} {'Pred':<12} {'Match':<6} {'Reason':<40} {'R':<5} {'G':<5} {'R%':<6}")
    print(f"{'-'*100}")
    for r in results:
        m = "✓" if r["match"] else "✗"
        print(f"{r['name']:<22} {r['gt']:<12} {r['pred']:<12} {m:<6} {r['reason']:<40} {r['red']:<5} {r['green']:<5} {r['red_ratio']:<6}")

    matches = sum(1 for r in results if r["match"])
    total = len(results)
    print(f"{'-'*100}")
    print(f"Accuracy: {matches}/{total} ({matches/total*100:.0f}%)  Time: {elapsed:.0f}s")

    tp = sum(1 for r in results if r["gt"] == "jaywalking" and r["pred"] == "jaywalking")
    tn = sum(1 for r in results if r["gt"] == "compliant" and r["pred"] == "compliant")
    fp = sum(1 for r in results if r["gt"] == "compliant" and r["pred"] == "jaywalking")
    fn = sum(1 for r in results if r["gt"] == "jaywalking" and r["pred"] == "compliant")
    print(f"TP={tp} TN={tn} FP={fp} FN={fn}")
    if tp+fp > 0: print(f"Precision: {tp/(tp+fp)*100:.0f}%")
    if tp+fn > 0: print(f"Recall:    {tp/(tp+fn)*100:.0f}%")
    if tp+fp > 0 and tp+fn > 0:
        p, r_ = tp/(tp+fp), tp/(tp+fn)
        print(f"F1:        {2*p*r_/(p+r_)*100:.0f}%")

    print(f"\n--- FAILURES ---")
    for r in results:
        if not r["match"]:
            m = r["meta"]
            print(f"  {r['name']}: GT={r['gt']}, pred={r['pred']}, sig={m.get('is_signalized')}, zebra_gt={m.get('has_zebra_gt')}, red_gt={m.get('has_red_gt')}")
    print(f"{'='*100}")

    out = ROOT / "evaluation" / "pipeline_results_v3.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["name","gt","pred","match","reason","frames","persons","red","green","red_ratio"])
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k, "") for k in w.fieldnames})
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
