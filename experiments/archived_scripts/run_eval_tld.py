#!/usr/bin/env python3
"""
Run TLD-READY integrated pipeline on all 13 reviewed clips.
Output: GT vs prediction comparison table.
"""
import sys, cv2, torch, csv, time
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


def get_clips():
    """Load reviewed clips with GT labels."""
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


def get_gt_details(clip_name):
    """Get JAAD ground truth details for a clip."""
    with open(EVENTS_CSV) as f:
        for row in csv.DictReader(f):
            if row["clip_name"] == clip_name:
                return row
    return {}


def run_clip(clip):
    """Run pipeline on a single clip. Returns prediction + stats."""
    yolo = YOLO(YOLO_MODEL).to(0 if torch.cuda.is_available() else "cpu")
    tl_model = _get_model()
    ZebraDetector.reset_cache()
    TrafficLightClassifier.reset()

    cap = cv2.VideoCapture(clip["path"])
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_idx = 0
    person_tracks = defaultdict(list)
    tl_states = []
    zebra_frames = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        frame_idx += 1

        # Skip every other frame for speed (still enough data)
        if frame_idx % 2 == 0 and frame_idx > 10:
            # Still need to count frame for zebra
            zebra_poly = ZebraDetector.get_zebra_polygon(frame)
            if zebra_poly is not None and len(zebra_poly) > 0:
                zebra_frames += 1
            continue

        # Zebra
        zebra_poly = ZebraDetector.get_zebra_polygon(frame)
        if zebra_poly is not None and len(zebra_poly) > 0:
            zebra_frames += 1

        # TLD-READY on full frame
        tl_state = TrafficLightClassifier.detect_from_frame(frame)
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
            h, w = frame.shape[:2]

            for box, c, tid in zip(boxes, classes, tids):
                if int(c) == 0:
                    cx = ((box[0]+box[2])/2) / w
                    person_tracks[tid].append(cx)

    cap.release()
    ZebraDetector.reset_cache()

    # --- Classification ---
    has_crossing = False
    for tid, xs in person_tracks.items():
        if len(xs) < 3:
            continue
        x_range = max(xs) - min(xs)
        road = sum(1 for x in xs if 0.15 < x < 0.85)
        if x_range > 0.12 and road >= 3:
            has_crossing = True
            break

    tl_counts = Counter(tl_states)
    has_red = tl_counts.get("RED", 0) > 0
    has_green = tl_counts.get("GREEN", 0) > 0
    total_frames = max(frame_idx, 1)
    has_zebra = zebra_frames > total_frames * 0.3

    if not has_crossing:
        pred = "compliant"
        reason = "no crossing"
    elif has_red:
        pred = "jaywalking"
        reason = "SIGNAL_VIOLATION"
    elif has_green:
        pred = "compliant"
        reason = "green light"
    elif not has_zebra:
        pred = "jaywalking"
        reason = "NO_CROSSWALK"
    else:
        pred = "compliant"
        reason = "on zebra, no red"

    return {
        "pred": pred,
        "reason": reason,
        "frames": frame_idx,
        "persons": len(person_tracks),
        "tl_red": tl_counts.get("RED", 0),
        "tl_green": tl_counts.get("GREEN", 0),
        "tl_unknown": tl_counts.get("UNKNOWN", 0),
        "zebra_pct": round(zebra_frames / total_frames * 100),
    }


def main():
    clips = get_clips()
    print(f"Running pipeline on {len(clips)} clips...\n")

    results = []
    t0 = time.time()
    for i, clip in enumerate(clips):
        print(f"[{i+1}/{len(clips)}] {clip['name']}...", end=" ", flush=True)
        r = run_clip(clip)
        gt = clip["gt"]
        match = r["pred"] == gt
        r["name"] = clip["name"]
        r["gt"] = gt
        r["verdict"] = clip["verdict"]
        r["match"] = match
        results.append(r)
        symbol = "✓" if match else "✗"
        print(f"GT={gt:<12} pred={r['pred']:<12} {symbol}  ({r['reason']})")

    elapsed = time.time() - t0

    # --- Summary table ---
    print(f"\n{'='*90}")
    print(f"{'Clip':<25} {'GT':<12} {'Predicted':<12} {'Match':<6} {'Reason':<25} {'TL_R':<5} {'TL_G':<5} {'Zebra%':<7}")
    print(f"{'-'*90}")
    for r in results:
        m = "✓" if r["match"] else "✗"
        print(f"{r['name']:<25} {r['gt']:<12} {r['pred']:<12} {m:<6} {r['reason']:<25} {r['tl_red']:<5} {r['tl_green']:<5} {r['zebra_pct']}%")

    matches = sum(1 for r in results if r["match"])
    total = len(results)
    print(f"{'-'*90}")
    print(f"Accuracy: {matches}/{total} ({matches/total*100:.0f}%)")
    print(f"Time: {elapsed:.0f}s ({elapsed/total:.1f}s per clip)")

    # Breakdown by GT
    for gt_label in ["jaywalking", "compliant"]:
        subset = [r for r in results if r["gt"] == gt_label]
        correct = sum(1 for r in subset if r["match"])
        print(f"  {gt_label}: {correct}/{len(subset)} correct")

    # Confusion matrix
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
    print(f"{'='*90}")

    # Save CSV
    out_csv = ROOT / "evaluation" / "pipeline_results_tld_ready.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name","gt","pred","match","reason","verdict",
                                                "frames","persons","tl_red","tl_green","tl_unknown","zebra_pct"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()
