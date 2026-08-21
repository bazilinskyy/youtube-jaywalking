#!/usr/bin/env python3
"""
CLI tool to generate an annotated video with visual bounding boxes and VLM decision overlay.
Usage:
    python scripts/annotate_video.py --video data/raw_clips/video_0014.mp4 --output outputs/annotated_videos/video_0014_annotated.mp4
"""
import argparse
import sys
from pathlib import Path
import cv2
import numpy as np
import torch
from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.config import get_cv_config
from src.vlm.alpamayo_detector import AlpamayoFullVideoDetector


def parse_coc_summary(coc_text: str) -> dict:
    """Parses Chain-of-Causation text into clean summary fields for HUD rendering."""
    summary = {
        "trajectory": "Extracted from visual sequence",
        "infrastructure": "Observed in scene",
        "vehicle_response": "Observed in scene",
        "final_classification": "UNKNOWN",
    }
    for line in coc_text.split("\n"):
        line_s = line.strip()
        if "1. **Pedestrian Trajectory" in line_s or "Trajectory & Location" in line_s:
            summary["trajectory"] = line_s.split(":", 1)[-1].strip() if ":" in line_s else line_s
        elif "2. **Infrastructure" in line_s or "Infrastructure & Right-of-Way" in line_s:
            summary["infrastructure"] = line_s.split(":", 1)[-1].strip() if ":" in line_s else line_s
        elif "3. **Vehicle Kinematic" in line_s or "Vehicle Response" in line_s:
            summary["vehicle_response"] = line_s.split(":", 1)[-1].strip() if ":" in line_s else line_s
        elif "5. **Final Classification" in line_s or "Final Classification" in line_s:
            summary["final_classification"] = line_s.split(":", 1)[-1].strip() if ":" in line_s else line_s
    return summary


def main():
    parser = argparse.ArgumentParser(description="Render Alpamayo visual reasoning annotations on video")
    parser.add_argument("--video", type=str, required=True, help="Input video path")
    parser.add_argument("--output", type=str, default=None, help="Output annotated video path")
    args = parser.parse_args()

    input_path = Path(args.video)
    if not input_path.exists():
        print(f"Error: Video file not found: {input_path}")
        sys.exit(1)

    out_path = Path(args.output) if args.output else ROOT_DIR / "outputs" / "annotated_videos" / f"{input_path.stem}_annotated.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Running Alpamayo visual reasoning on {input_path.name}...")
    alpamayo = AlpamayoFullVideoDetector()
    res = alpamayo.predict(input_path)
    pred_label = res["prediction"].upper()
    coc_text = res.get("chain_of_causation", "")
    coc_summary = parse_coc_summary(coc_text)

    print(f"VLM Decision: {pred_label}")
    print(f"Rendering annotated video to: {out_path}")

    cv_cfg = get_cv_config()
    yolo_model = YOLO(cv_cfg.get("yolo_model")).to(0 if torch.cuda.is_available() else "cpu")

    cap = cv2.VideoCapture(str(input_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    banner_color = (0, 0, 220) if pred_label == "JAYWALKING" else (0, 180, 0)

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # Track pedestrians and vehicles for visual context
        results = yolo_model.track(frame, tracker="bytetrack.yaml", persist=True, conf=0.4, verbose=False)
        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()
            tids = results[0].boxes.id
            track_ids = tids.int().cpu().tolist() if tids is not None else [-1] * len(classes)

            for box, c, tid in zip(boxes, classes, track_ids):
                cid = int(c)
                x1, y1, x2, y2 = [int(v) for v in box]
                if cid == 0:  # Pedestrian
                    color = (255, 255, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"Ped {tid}" if tid != -1 else "Ped", (x1, max(y1 - 5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                elif cid in (2, 3, 5, 7):  # Vehicles
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (120, 120, 120), 1)

        # Draw Top VLM Decision Banner
        text = f"ALPAMAYO PREDICTION: {pred_label}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.rectangle(frame, (10, 10), (10 + tw + 20, 10 + th + 20), banner_color, -1)
        cv2.putText(frame, text, (20, 10 + th + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # Draw Bottom Alpamayo Reasoning Panel (HUD)
        bot_y = h - 140
        cv2.rectangle(frame, (10, bot_y), (w - 10, h - 10), (20, 20, 20), -1)
        cv2.rectangle(frame, (10, bot_y), (w - 10, h - 10), banner_color, 2)

        cv2.putText(frame, f"1. Trajectory: {coc_summary['trajectory'][:85]}", (20, bot_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
        cv2.putText(frame, f"2. Infrastructure: {coc_summary['infrastructure'][:85]}", (20, bot_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
        cv2.putText(frame, f"3. Vehicle Response: {coc_summary['vehicle_response'][:85]}", (20, bot_y + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
        cv2.putText(frame, f"4. Final Verdict: {pred_label} | Frame {frame_idx}/{total_frames}", (20, bot_y + 105), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        out_writer.write(frame)

    cap.release()
    out_writer.release()
    print(f"Successfully saved: {out_path}")


if __name__ == "__main__":
    main()
