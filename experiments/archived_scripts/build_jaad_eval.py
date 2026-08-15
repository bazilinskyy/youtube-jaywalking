#!/usr/bin/env python3
"""
Parse JAAD annotations and build evaluation set.

JAAD provides:
- Main XML: bounding boxes + behavioral tags (cross=crossing/not-crossing)
- Traffic XML: ped_crossing, traffic_light, stop_sign per frame
- Attributes XML: crossing=1/0/-1, designated, signalized, intersection

Maps to your pipeline:
- crossing=1 + signalized + traffic_light=red → SIGNAL_VIOLATION
- crossing=1 + no ped_crossing → NO_CROSSWALK  
- crossing=1 + has ped_crossing + green → compliant crossing (negative)
- crossing=0 / -1 → did not cross (not relevant for evaluation)

Usage:
    python build_jaad_eval.py
    python build_jaad_eval.py --max-clips 50
"""
import os
import sys
import csv
import shutil
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).parent
JAAD_DIR = ROOT / "data" / "JAAD_clips" / "JAAD_clips"
JAAD_ANN = ROOT / "data" / "JAAD_annotations" / "annotations"
JAAD_TRAFFIC = ROOT / "data" / "JAAD_annotations" / "annotations_traffic"
JAAD_ATTRS = ROOT / "data" / "JAAD_annotations" / "annotations_attributes"
EVAL_DIR = ROOT / "evaluation"
SUMMARY_CSV = EVAL_DIR / "jaad_events_summary.csv"


def parse_traffic(xml_path):
    """Parse traffic XML → per-frame dict of scene attributes."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    frames = {}
    road_type = None

    road_el = root.find("road_type")
    if road_el is not None:
        road_type = road_el.text

    for frame_el in root.findall("frame"):
        fid = int(frame_el.get("id"))
        frames[fid] = {
            "ped_crossing": frame_el.get("ped_crossing") == "1",
            "ped_sign": frame_el.get("ped_sign") == "1",
            "stop_sign": frame_el.get("stop_sign") == "1",
            "traffic_light": frame_el.get("traffic_light", "n/a"),
        }
    return frames, road_type


def parse_attributes(xml_path):
    """Parse attributes XML → per-pedestrian attributes."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    peds = {}
    for ped_el in root.findall("pedestrian"):
        pid = ped_el.get("id")
        peds[pid] = {
            "crossing": int(ped_el.get("crossing", "-1")),
            "designated": ped_el.get("designated", "ND"),
            "signalized": ped_el.get("signalized", "n/a"),
            "intersection": ped_el.get("intersection", "no"),
            "num_lanes": int(ped_el.get("num_lanes", "2")),
            "gender": ped_el.get("gender", "n/a"),
            "age": ped_el.get("age", "n/a"),
            "group_size": int(ped_el.get("group_size", "1")),
            "motion_direction": ped_el.get("motion_direction", "n/a"),
            "traffic_direction": ped_el.get("traffic_direction", "n/a"),
        }
    return peds


def parse_main_annotation(xml_path):
    """Parse main XML → list of pedestrian behavioral records."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    pedestrians = []

    for track in root.findall(".//track"):
        label = track.get("label", "")
        if label != "pedestrian":
            continue

        boxes = track.findall("box")
        if not boxes:
            continue

        # Get behavioral attributes from first keyframe
        behavior = {}
        for box in boxes:
            for attr in box.findall("attribute"):
                name = attr.get("name")
                behavior[name] = attr.text
            if "cross" in behavior:
                break

        if "cross" not in behavior:
            continue

        # Get frame range
        frames = [int(b.get("frame")) for b in boxes]

        pedestrians.append({
            "id": behavior.get("id", ""),
            "cross": behavior.get("cross", ""),
            "look": behavior.get("look", ""),
            "action": behavior.get("action", ""),
            "occlusion": behavior.get("occlusion", ""),
            "start_frame": min(frames),
            "end_frame": max(frames),
            "num_frames": len(frames),
        })

    return pedestrians


def classify_event(ped, traffic_frames, road_type, attrs):
    """
    Classify a crossing event.
    
    Returns: (label, reason, details_dict)
    label: 'jaywalking', 'compliant', or 'not_crossing'
    """
    cross_status = ped["cross"]
    attr = attrs.get(ped["id"], {})

    # Not crossing — skip
    if cross_status == "not-crossing":
        return "not_crossing", "pedestrian did not cross", {}

    # crossing=1 means the pedestrian crossed
    # Now determine if it was jaywalking or compliant

    # Get traffic context during the crossing
    crossing_frames = list(range(ped["start_frame"], ped["end_frame"] + 1))
    relevant_traffic = {f: traffic_frames.get(f, {}) for f in crossing_frames if f in traffic_frames}

    has_zebra = any(t.get("ped_crossing", False) for t in relevant_traffic.values())
    traffic_lights = set(t.get("traffic_light", "n/a") for t in relevant_traffic.values())
    has_red = "red" in traffic_lights
    has_green = "green" in traffic_lights
    has_stop_sign = any(t.get("stop_sign", False) for t in relevant_traffic.values())
    is_signalized = attr.get("signalized", "n/a") not in ("n/a", "NS", "")
    is_designated = attr.get("designated", "ND") == "D"

    details = {
        "has_zebra": has_zebra,
        "has_red_light": has_red,
        "has_green_light": has_green,
        "has_stop_sign": has_stop_sign,
        "is_signalized": is_signalized,
        "is_designated": is_designated,
        "road_type": road_type or "unknown",
        "num_lanes": attr.get("num_lanes", 2),
        "intersection": attr.get("intersection", "no"),
        "traffic_lights": list(traffic_lights),
    }

    # Jaywalking conditions:
    # 1. Signal violation: crossing at red light
    if has_red:
        return "jaywalking", "SIGNAL_VIOLATION", details

    # 2. No crosswalk: crossing where there's no zebra/sign/stop
    if not has_zebra and not has_stop_sign and not is_signalized:
        return "jaywalking", "NO_CROSSWALK", details

    # 3. Non-designated crossing at signalized location
    if is_signalized and not is_designated:
        return "jaywalking", "UNSIGNALIZED_CROSSING", details

    # Compliant: crossing at zebra/green light/designated spot
    return "compliant", "compliant_crossing", details


def main():
    parser = argparse.ArgumentParser(description="Build JAAD evaluation set")
    parser.add_argument("--max-clips", type=int, default=None,
                        help="Max clips per category")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    pos_dir = EVAL_DIR / "jaad_positive"
    neg_dir = EVAL_DIR / "jaad_negative"

    if not args.dry_run:
        pos_dir.mkdir(exist_ok=True)
        neg_dir.mkdir(exist_ok=True)

    all_events = []
    video_ids = sorted([f.stem.replace("video_", "")
                        for f in JAAD_ANN.glob("video_*.xml")])

    print(f"Parsing {len(video_ids)} JAAD videos...")

    for vid in video_ids:
        vid_name = f"video_{vid}"
        main_xml = JAAD_ANN / f"{vid_name}.xml"
        traffic_xml = JAAD_TRAFFIC / f"{vid_name}_traffic.xml"
        attrs_xml = JAAD_ATTRS / f"{vid_name}_attributes.xml"
        clip_path = JAAD_DIR / f"{vid_name}.mp4"

        if not main_xml.exists():
            continue

        # Parse all three annotation files
        pedestrians = parse_main_annotation(str(main_xml))
        traffic_frames = {}
        road_type = None
        if traffic_xml.exists():
            traffic_frames, road_type = parse_traffic(str(traffic_xml))
        ped_attrs = {}
        if attrs_xml.exists():
            ped_attrs = parse_attributes(str(attrs_xml))

        for ped in pedestrians:
            label, reason, details = classify_event(ped, traffic_frames, road_type, ped_attrs)

            if label == "not_crossing":
                continue

            event = {
                "video_id": vid_name,
                "clip_name": f"{vid_name}.mp4",
                "ped_id": ped["id"],
                "eval_label": label,
                "violation_type": reason if label == "jaywalking" else "",
                "cross_action": ped["cross"],
                "look": ped["look"],
                "action": ped["action"],
                "start_frame": ped["start_frame"],
                "end_frame": ped["end_frame"],
                "num_frames": ped["num_frames"],
                "has_zebra": details.get("has_zebra", False),
                "has_red_light": details.get("has_red_light", False),
                "has_green_light": details.get("has_green_light", False),
                "is_signalized": details.get("is_signalized", False),
                "is_designated": details.get("is_designated", False),
                "road_type": details.get("road_type", ""),
                "num_lanes": details.get("num_lanes", 2),
                "intersection": details.get("intersection", ""),
                "clip_exists": clip_path.exists(),
            }
            all_events.append(event)

    # Separate positive/negative
    positive = [e for e in all_events if e["eval_label"] == "jaywalking"]
    negative = [e for e in all_events if e["eval_label"] == "compliant"]

    print(f"\nFound {len(all_events)} crossing events:")
    print(f"  Jaywalking (positive): {len(positive)}")
    print(f"  Compliant (negative):  {len(negative)}")

    # Breakdown by violation type
    if positive:
        print("\n  Jaywalking breakdown:")
        by_type = defaultdict(int)
        for e in positive:
            by_type[e["violation_type"]] += 1
        for vtype, count in sorted(by_type.items()):
            print(f"    {vtype}: {count}")

    # Save summary
    if all_events and not args.dry_run:
        cols = list(all_events[0].keys())
        with open(SUMMARY_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=cols)
            writer.writeheader()
            writer.writerows(all_events)
        print(f"\nSaved summary to {SUMMARY_CSV}")

    # Copy clips
    if args.dry_run:
        print("\n[DRY RUN] Would copy clips to:")
        print(f"  {pos_dir} ({len(positive)} clips)")
        print(f"  {neg_dir} ({len(negative)} clips)")
        return

    # Deduplicate: one clip may have multiple pedestrians
    # Copy each clip once, tag with all pedestrian labels
    pos_clips = set()
    neg_clips = set()

    for e in positive:
        if e["clip_exists"] and e["clip_name"] not in pos_clips:
            src = JAAD_DIR / e["clip_name"]
            dst = pos_dir / e["clip_name"]
            if not dst.exists():
                shutil.copy2(str(src), str(dst))
            pos_clips.add(e["clip_name"])

    for e in negative:
        if e["clip_exists"] and e["clip_name"] not in neg_clips:
            src = JAAD_DIR / e["clip_name"]
            dst = neg_dir / e["clip_name"]
            if not dst.exists():
                shutil.copy2(str(src), str(dst))
            neg_clips.add(e["clip_name"])

    print(f"\nCopied clips:")
    print(f"  {pos_dir}: {len(pos_clips)} unique clips")
    print(f"  {neg_dir}: {len(neg_clips)} unique clips")

    # Also create a per-clip summary for the review tool
    clip_summary = {}
    for e in all_events:
        cn = e["clip_name"]
        if cn not in clip_summary:
            clip_summary[cn] = {
                "clip_name": cn,
                "eval_label": e["eval_label"],
                "violation_type": e["violation_type"],
                "video_id": e["video_id"],
                "ped_id": e["ped_id"],
                "has_zebra": e["has_zebra"],
                "has_red_light": e["has_red_light"],
                "is_signalized": e["is_signalized"],
                "is_designated": e["is_designated"],
                "road_type": e["road_type"],
            }
        else:
            # Multiple pedestrians — keep jaywalking label if any
            if e["eval_label"] == "jaywalking":
                clip_summary[cn]["eval_label"] = "jaywalking"
                clip_summary[cn]["violation_type"] = e["violation_type"]

    print(f"\nTotal unique clips for evaluation: {len(clip_summary)}")


if __name__ == "__main__":
    main()
