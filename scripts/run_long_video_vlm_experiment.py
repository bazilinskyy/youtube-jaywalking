#!/usr/bin/env python3
"""
Long Video Multi-Event VLM Baseline Experiment with Temporal Event Merging

Reuses YOLO11x + ByteTrack tracking to extract pedestrian candidate tracks,
applies a generic temporal IoU event-merging mechanism to coalesce overlapping
tracks into distinct physical crossing events, and runs the VLM baseline
(qwen2.5vl:7b via FullVideoVLMDetector) once per merged event.

Usage:
    python scripts/run_long_video_vlm_experiment.py --video data/raw_clips/video_0006.mp4
"""

from pathlib import Path
import sys
import time

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

import argparse  # noqa: E402
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from ultralytics import YOLO  # noqa: E402
from src.vlm.client import encode_frame_to_base64  # noqa: E402
from src.vlm.alpamayo_detector import FullVideoVLMDetector  # noqa: E402


def extract_candidate_events(video_path: str, min_duration: float = 1.0, min_displacement: float = 0.10):
    """Tracks pedestrians using YOLO11x + ByteTrack and returns candidate event intervals."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO("models/yolo11x.pt").to(device)

    tracks = {}
    track_frames = {}
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        results = model.track(
            frame,
            tracker="bytetrack.yaml",
            persist=True,
            classes=[0],
            conf=0.3,
            verbose=False,
            device=device,
        )

        if results and len(results) > 0 and results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            tids = results[0].boxes.id.int().cpu().tolist()
            for box, tid in zip(boxes, tids):
                cx = ((box[0] + box[2]) / 2.0) / max(w, 1)
                if tid not in tracks:
                    tracks[tid] = []
                    track_frames[tid] = []
                tracks[tid].append(cx)
                track_frames[tid].append(frame_idx)

    cap.release()

    candidates = []
    cand_idx = 1
    for tid in sorted(tracks.keys()):
        xs = tracks[tid]
        f_list = track_frames[tid]
        disp = abs(xs[-1] - xs[0])
        s_f = f_list[0]
        e_f = f_list[-1]
        s_ts = round(s_f / fps, 2)
        e_ts = round(e_f / fps, 2)
        dur = round(e_ts - s_ts, 2)

        if dur >= min_duration and disp >= min_displacement:
            candidates.append({
                "candidate_id": f"cand_{cand_idx:03d}",
                "track_id": tid,
                "start_frame": s_f,
                "end_frame": e_f,
                "start_timestamp": s_ts,
                "end_timestamp": e_ts,
                "duration": dur,
                "displacement": round(disp, 3),
            })
            cand_idx += 1

    return total_frames, fps, round(total_frames / fps, 2), candidates


def merge_overlapping_events(candidates, fps: float, tiou_thresh: float = 0.35, overlap_ratio_thresh: float = 0.50):
    """Merges overlapping candidate pedestrian tracks representing the same physical temporal crossing activity."""
    if not candidates:
        return []

    sorted_cand = sorted(candidates, key=lambda x: (x["start_frame"], x["end_frame"]))
    merged = []

    for cand in sorted_cand:
        if not merged:
            merged.append({
                "event_id": "event_001",
                "track_ids": [cand["track_id"]],
                "start_frame": cand["start_frame"],
                "end_frame": cand["end_frame"],
            })
        else:
            last = merged[-1]
            s1, e1 = last["start_frame"], last["end_frame"]
            s2, e2 = cand["start_frame"], cand["end_frame"]

            inter = max(0, min(e1, e2) - max(s1, s2))
            union = max(e1, e2) - min(s1, s2)
            tiou = inter / union if union > 0 else 0.0
            min_len = min(e1 - s1, e2 - s2)
            overlap_ratio = inter / min_len if min_len > 0 else 0.0

            if tiou >= tiou_thresh or overlap_ratio >= overlap_ratio_thresh:
                last["start_frame"] = min(s1, s2)
                last["end_frame"] = max(e1, e2)
                if cand["track_id"] not in last["track_ids"]:
                    last["track_ids"].append(cand["track_id"])
            else:
                idx = len(merged) + 1
                merged.append({
                    "event_id": f"event_{idx:03d}",
                    "track_ids": [cand["track_id"]],
                    "start_frame": cand["start_frame"],
                    "end_frame": cand["end_frame"],
                })

    for m in merged:
        s_f = m["start_frame"]
        e_f = m["end_frame"]
        s_ts = round(s_f / fps, 2)
        e_ts = round(e_f / fps, 2)
        m["start_timestamp"] = s_ts
        m["end_timestamp"] = e_ts
        m["duration"] = round(e_ts - s_ts, 2)

    return merged


def run_experiment(video_path: str):
    print("=" * 75)
    print("LONG-VIDEO MULTI-EVENT EXPERIMENT WITH TEMPORAL EVENT MERGING")
    print(f"Target Video: {video_path}")
    print("=" * 75)

    t_start_all = time.time()

    # 1. Detection & Extraction
    total_frames, fps, video_duration, candidates = extract_candidate_events(video_path)

    print("\n[1. ORIGINAL CANDIDATE TRACK EXTRACTION (YOLO11x + ByteTrack)]")
    print(f"  Video Duration: {video_duration}s ({total_frames} total frames at {fps:.2f} FPS)")
    print(f"  Original Candidate Tracks Extracted: {len(candidates)}")
    for c in candidates:
        print(
            f"   - {c['candidate_id']}: Track {c['track_id']} | "
            f"Frames {c['start_frame']}–{c['end_frame']} ({c['start_timestamp']}s–{c['end_timestamp']}s, "
            f"dur {c['duration']}s, disp {c['displacement']})"
        )

    # 2. Merging Mechanism
    merged_events = merge_overlapping_events(candidates, fps=fps)

    print("\n[2. TEMPORAL EVENT MERGING & GROUPING]")
    print(f"  Event Count Before Merging: {len(candidates)}")
    print(f"  Event Count After Merging:  {len(merged_events)}")
    for m in merged_events:
        print(
            f"   - {m['event_id']}: Tracks {m['track_ids']} | "
            f"Frames {m['start_frame']}–{m['end_frame']} ({m['start_timestamp']}s–{m['end_timestamp']}s, "
            f"duration {m['duration']}s)"
        )

    if not merged_events:
        print("  No crossing events remaining after merging.")
        return

    # 3. VLM Baseline Inference Per Merged Event
    print(f"\n[3. EXECUTING VLM BASELINE ({len(merged_events)} RUNS FOR MERGED EVENTS)]")
    detector = FullVideoVLMDetector()
    event_results = []
    cap = cv2.VideoCapture(video_path)

    t_vlm_start = time.time()

    for m in merged_events:
        s_f = m["start_frame"]
        e_f = m["end_frame"]
        sample_indices = np.linspace(s_f, e_f, num=5, dtype=int).tolist()

        frames = []
        for idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx - 1)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)

        t0 = time.time()
        b64_list = [encode_frame_to_base64(f, quality=85) for f in frames]
        raw_response = detector.client.generate_chat(prompt=detector.coc_prompt, base64_images=b64_list)
        parsed = detector.parse_coc_response(raw_response)
        elapsed = round(time.time() - t0, 3)

        verdict = parsed["prediction"]
        coc_text = parsed["chain_of_causation"]

        event_results.append({
            "event_id": m["event_id"],
            "track_ids": m["track_ids"],
            "start_frame": s_f,
            "end_frame": e_f,
            "start_timestamp": m["start_timestamp"],
            "end_timestamp": m["end_timestamp"],
            "duration": m["duration"],
            "verdict": verdict,
            "reasoning": coc_text,
            "inference_time": elapsed,
        })

        print(
            f"\n  --- {m['event_id']} (Tracks {m['track_ids']}, Frames {s_f}–{e_f}, "
            f"{m['start_timestamp']}s–{m['end_timestamp']}s) ---"
        )
        print(f"  VLM Verdict: {verdict}")
        print(f"  Inference Latency: {elapsed}s")
        print(f"  VLM Chain-of-Causation Reasoning:\n{coc_text}")

    cap.release()
    total_vlm_latency = round(time.time() - t_vlm_start, 2)
    total_execution_time = round(time.time() - t_start_all, 2)

    # 4. Aggregation
    has_jaywalking = any(r["verdict"].upper() == "JAYWALKING" for r in event_results)
    final_video_verdict = "JAYWALKING" if has_jaywalking else "COMPLIANT"

    print("\n" + "=" * 75)
    print("EXPERIMENT 14 SUMMARY REPORT: VIDEO_0006.MP4")
    print("=" * 75)
    print(f"Input Video File:          {video_path}")
    print(f"Video Duration:            {video_duration}s ({total_frames} total frames at {fps:.2f} FPS)")
    print(f"Candidate Tracks Before:   {len(candidates)}")
    print(f"Merged Crossing Events:    {len(merged_events)}")
    for r in event_results:
        print(
            f"  * {r['event_id']}: Frames {r['start_frame']}–{r['end_frame']} "
            f"({r['start_timestamp']}s–{r['end_timestamp']}s) -> {r['verdict']} ({r['inference_time']}s)"
        )
    print(f"Final Aggregated Verdict:  {final_video_verdict}")
    print(f"Total VLM Inference Time:  {total_vlm_latency}s")
    print(f"Total Script Time:         {total_execution_time}s")
    print("=" * 75)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Long Video VLM Baseline Experiment with Event Merging")
    parser.add_argument("--video", type=str, default="data/raw_clips/video_0006.mp4", help="Input video file path")
    args = parser.parse_args()

    run_experiment(args.video)
