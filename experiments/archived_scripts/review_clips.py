#!/usr/bin/env python3
"""
Interactive clip reviewer for evaluation clips.

Opens each clip in the default video player, shows metadata,
and lets you mark it as correct/incorrect/ambiguous via keyboard input.

Usage:
    python review_clips.py                    # review all
    python review_clips.py --label jaywalking  # review only positive clips
    python review_clips.py --label compliant   # review only negative clips
    python review_clips.py --resume            # skip already-reviewed clips

Controls:
    c = correct label
    w = wrong label (misclassified)
    a = ambiguous / hard to tell
    s = skip (come back later)
    q = quit and save progress
    n = next clip (same as skip)
    o = replay the clip
"""
import os
import sys
import csv
import subprocess
import argparse
import platform
import signal
import time
from pathlib import Path

ROOT = Path(__file__).parent
EVAL_DIR = ROOT / "evaluation"
CLIPS_POS = EVAL_DIR / "positive"
CLIPS_NEG = EVAL_DIR / "negative"
SUMMARY_CSV = EVAL_DIR / "events_summary.csv"
REVIEW_CSV = EVAL_DIR / "review_results.csv"

REVIEW_COLUMNS = [
    "clip_name", "auto_label", "reviewer_verdict", "reviewer_notes",
    "source", "video_id", "person_uid", "violation_type", "x_range",
    "has_zebra", "has_red_light", "road_frames"
]


def get_video_paths(label_filter=None, source="all"):
    """Collect all clip paths with metadata from events_summary.csv and JAAD."""
    clips = []

    # --- YouTube-derived clips ---
    if source in ("all", "youtube"):
        meta = {}
        if SUMMARY_CSV.exists():
            with open(SUMMARY_CSV) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    meta[row["clip_name"]] = row

        for clip_dir, auto_label in [(CLIPS_POS, "jaywalking"), (CLIPS_NEG, "compliant")]:
            if not clip_dir.exists():
                continue
            if label_filter and auto_label != label_filter:
                continue
            for clip_path in sorted(clip_dir.glob("*.mp4")):
                name = clip_path.name
                info = meta.get(name, {})
                clips.append({
                    "path": clip_path,
                    "name": name,
                    "auto_label": auto_label,
                    "source": "youtube",
                    "meta": info,
                })

    # --- JAAD clips ---
    if source in ("all", "jaad"):
        jaad_summary = EVAL_DIR / "jaad_events_summary.csv"
        jaad_meta = {}
        if jaad_summary.exists():
            with open(jaad_summary) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    cn = row["clip_name"]
                    # Keep jaywalking label if any pedestrian in clip is jaywalking
                    if cn not in jaad_meta or row["eval_label"] == "jaywalking":
                        jaad_meta[cn] = row

        jaad_pos = EVAL_DIR / "jaad_positive"
        jaad_neg = EVAL_DIR / "jaad_negative"

        for clip_dir, auto_label in [(jaad_pos, "jaywalking"), (jaad_neg, "compliant")]:
            if not clip_dir.exists():
                continue
            if label_filter and auto_label != label_filter:
                continue
            for clip_path in sorted(clip_dir.glob("*.mp4")):
                name = clip_path.name
                info = jaad_meta.get(name, {})
                clips.append({
                    "path": clip_path,
                    "name": name,
                    "auto_label": auto_label,
                    "source": "jaad",
                    "meta": info,
                })

    return clips


def load_reviewed():
    """Load already-reviewed clip names."""
    reviewed = {}
    if REVIEW_CSV.exists():
        with open(REVIEW_CSV) as f:
            reader = csv.DictReader(f)
            for row in reader:
                reviewed[row["clip_name"]] = row["reviewer_verdict"]
    return reviewed


def save_review(review_data, mode="a"):
    """Append a single review row to the CSV."""
    file_exists = REVIEW_CSV.exists() and mode == "a"
    with open(REVIEW_CSV, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REVIEW_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(review_data)


def open_video(path):
    """Open video with mpv (non-blocking, auto-closes on end)."""
    try:
        proc = subprocess.Popen(
            ["mpv", "--no-terminal", "--keep-open=no", "--no-osd-bar",
             str(path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return proc
    except FileNotFoundError:
        pass
    # Fallback
    try:
        proc = subprocess.Popen(
            ["xdg-open", str(path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return proc
    except FileNotFoundError:
        print("  No player found. Install mpv: sudo apt install mpv")
        return None


def close_video(proc):
    """Kill the video player process."""
    if proc is None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=2)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def print_clip_info(clip, idx, total):
    """Print clip metadata in a readable format."""
    m = clip["meta"]
    print(f"\n{'='*70}")
    print(f"  Clip {idx+1}/{total}")
    print(f"{'='*70}")
    print(f"  File:         {clip['name']}")
    print(f"  Source:       {clip.get('source', '?').upper()}")
    print(f"  Auto label:   {clip['auto_label'].upper()}")
    print(f"  Video:        {m.get('video_id', '?')}")
    if clip.get("source") == "jaad":
        print(f"  Ped ID:       {m.get('ped_id', '?')}")
        print(f"  Action:       {m.get('action', '?')}")
        print(f"  Look:         {m.get('look', '?')}")
        print(f"  Designated:   {m.get('is_designated', '?')}")
        print(f"  Signalized:   {m.get('is_signalized', '?')}")
        print(f"  Intersection: {m.get('intersection', '?')}")
        print(f"  Road type:    {m.get('road_type', '?')}")
        print(f"  Lanes:        {m.get('num_lanes', '?')}")
    else:
        print(f"  Person ID:    {m.get('person_uid', '?')}")
        print(f"  Frames:       {m.get('start_frame', '?')} → {m.get('end_frame', '?')}")
        print(f"  Time:         {m.get('start_time', '?')}s → {m.get('end_time', '?')}s")
        print(f"  X-range:      {m.get('x_range', '?')}")
        print(f"  Road frames:  {m.get('road_frames', '?')}/{m.get('total_frames', '?')}")
    print(f"  Zebra:        {m.get('has_zebra', '?')}")
    print(f"  Red light:    {m.get('has_red_light', '?')}")
    print(f"  Violation:    {m.get('violation_type', 'none')}")
    print(f"{'='*70}")


def get_input():
    """Get a single keypress from the user."""
    try:
        # Python 3
        import termios
        import tty
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        return ch.lower()
    except ImportError:
        # Windows fallback
        import msvcrt
        return msvcrt.getch().decode().lower()


def main():
    parser = argparse.ArgumentParser(description="Review evaluation clips")
    parser.add_argument("--label", choices=["jaywalking", "compliant"],
                        help="Only review one label type")
    parser.add_argument("--source", choices=["all", "youtube", "jaad"],
                        default="all", help="Which clips to review")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-reviewed clips")
    parser.add_argument("--shuffle", action="store_true",
                        help="Randomize review order")
    args = parser.parse_args()

    clips = get_video_paths(args.label, source=args.source)
    if not clips:
        print("No clips found.")
        sys.exit(1)

    reviewed = load_reviewed() if args.resume else {}
    if args.shuffle:
        import random
        random.shuffle(clips)

    total = len(clips)
    skipped = 0
    reviewed_count = len(reviewed)

    print(f"\nFound {total} clips to review.")
    print(f"Already reviewed: {reviewed_count}")
    print(f"\nControls:")
    print(f"  c = correct label     w = wrong label")
    print(f"  a = ambiguous         s/n = skip")
    print(f"  o = replay clip       q = quit & save")
    print()

    for idx, clip in enumerate(clips):
        if args.resume and clip["name"] in reviewed:
            skipped += 1
            continue

        print_clip_info(clip, idx - skipped, total - skipped)

        # Open the video
        proc = open_video(clip["path"])
        if proc is None:
            continue

        # Wait for user decision
        print("  [c]orrect  [w]rong  [a]mbiguous  [s]kip  [o]pen again  [q]uit")
        print("  > ", end="", flush=True)

        while True:
            key = get_input()
            print(key)

            if key in ("c", "w", "a", "s", "n", "q", "o"):
                break
            print("  Invalid key. Try again: ", end="", flush=True)

        if key == "q":
            close_video(proc)
            print("\nSaving progress and quitting...")
            break
        elif key == "o":
            # Close current, replay
            close_video(proc)
            proc = open_video(clip["path"])
            print("  Replay opened. Press any key to close & continue...")
            get_input()
            close_video(proc)
            # Re-ask
            print("  > ", end="", flush=True)
            while True:
                key = get_input()
                print(key)
                if key in ("c", "w", "a", "s", "n", "q"):
                    break
                print("  Invalid key. Try again: ", end="", flush=True)
            if key == "q":
                break
        else:
            # Close player after decision
            close_video(proc)

        verdict_map = {"c": "correct", "w": "wrong", "a": "ambiguous", "s": "skip", "n": "skip"}
        verdict = verdict_map.get(key, "skip")

        if verdict != "skip":
            m = clip["meta"]
            review_row = {
                "clip_name": clip["name"],
                "auto_label": clip["auto_label"],
                "reviewer_verdict": verdict,
                "reviewer_notes": "",
                "source": clip.get("source", ""),
                "video_id": m.get("video_id", ""),
                "person_uid": m.get("person_uid", m.get("ped_id", "")),
                "violation_type": m.get("violation_type", ""),
                "x_range": m.get("x_range", ""),
                "has_zebra": m.get("has_zebra", ""),
                "has_red_light": m.get("has_red_light", ""),
                "road_frames": m.get("road_frames", ""),
            }
            save_review(review_row)
            reviewed_count += 1
            print(f"  Saved: {verdict} ({reviewed_count} reviewed)")

    # Summary
    print(f"\n{'='*70}")
    print(f"  REVIEW SUMMARY")
    print(f"{'='*70}")
    print(f"  Total clips:    {total}")
    print(f"  Reviewed:       {reviewed_count}")

    if REVIEW_CSV.exists():
        with open(REVIEW_CSV) as f:
            reader = csv.DictReader(f)
            results = list(reader)
            correct = sum(1 for r in results if r["reviewer_verdict"] == "correct")
            wrong = sum(1 for r in results if r["reviewer_verdict"] == "wrong")
            ambiguous = sum(1 for r in results if r["reviewer_verdict"] == "ambiguous")
            print(f"  Correct:        {correct}")
            print(f"  Wrong:          {wrong}")
            print(f"  Ambiguous:      {ambiguous}")
            if correct + wrong > 0:
                acc = correct / (correct + wrong) * 100
                print(f"  Auto-label accuracy: {acc:.1f}% (excluding ambiguous)")

    print(f"\n  Results saved to: {REVIEW_CSV}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
