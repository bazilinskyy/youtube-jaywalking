#!/usr/bin/env python3
"""
Interactive CLI video reviewer for dataset ground truth labeling.
Usage:
    python scripts/review_clips.py
    python scripts/review_clips.py --label unlabeled
"""
import argparse
import csv
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.data_loader import load_ground_truth_records


def play_clip(video_path: str):
    """Attempts to open video in the default media player."""
    try:
        if sys.platform.startswith("darwin"):
            subprocess.Popen(["open", video_path])
        elif sys.platform.startswith("linux"):
            subprocess.Popen(["xdg-open", video_path])
        elif sys.platform.startswith("win"):
            subprocess.Popen(["start", video_path], shell=True)
    except Exception as e:
        print(f"Could not open media player: {e}")


def main():
    parser = argparse.ArgumentParser(description="Interactive video review tool")
    parser.add_argument("--gt", default="data/ground_truth.csv", help="Path to ground truth CSV")
    parser.add_argument("--label", choices=["all", "unlabeled", "ambiguous", "jaywalking", "compliant"], default="all")
    args = parser.parse_args()

    gt_path = Path(args.gt)
    if not gt_path.exists():
        print(f"Error: {gt_path} does not exist.")
        sys.exit(1)

    records = load_ground_truth_records(gt_path, only_evaluable=False)
    if args.label != "all":
        records = [r for r in records if r["ground_truth"] == args.label]

    print(f"\nLoaded {len(records)} records for review.")
    print("Commands: [j] Jaywalking, [c] Compliant, [a] Ambiguous, [s] Skip, [p] Play again, [q] Quit\n")

    updates = {}
    for idx, r in enumerate(records, 1):
        v_path = r["video_path"]
        print(f"[{idx}/{len(records)}] {r['clip_name']} | Current Label: {r['ground_truth']} | Notes: {r['notes']}")
        play_clip(v_path)

        while True:
            choice = input("Decision [j/c/a/s/p/q]: ").strip().lower()
            if choice == "j":
                updates[r["clip_name"]] = ("jaywalking", "True", "Yes")
                break
            elif choice == "c":
                updates[r["clip_name"]] = ("compliant", "True", "No")
                break
            elif choice == "a":
                updates[r["clip_name"]] = ("ambiguous", "False", "Cannot be concluded")
                break
            elif choice == "s":
                break
            elif choice == "p":
                play_clip(v_path)
            elif choice == "q":
                break
            else:
                print("Invalid key.")

        if choice == "q":
            break

    if updates:
        print(f"\nSaving {len(updates)} label updates to {gt_path}...")
        all_rows = []
        with open(gt_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            for row in reader:
                cn = row["clip_name"]
                if cn in updates:
                    gt, is_eval, raw = updates[cn]
                    row["ground_truth"] = gt
                    row["is_evaluated"] = is_eval
                    row["raw_label"] = raw
                all_rows.append(row)

        with open(gt_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        print("Updated successfully.")


if __name__ == "__main__":
    main()
