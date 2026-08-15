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
from src.vlm.detector import VLMJaywalkingDetector


def main():
    parser = argparse.ArgumentParser(description="Render visual annotations on video")
    parser.add_argument("--video", type=str, required=True, help="Input video path")
    parser.add_argument("--output", type=str, default=None, help="Output annotated video path")
    args = parser.parse_args()

    input_path = Path(args.video)
    if not input_path.exists():
        print(f"Error: Video file not found: {input_path}")
        sys.exit(1)

    out_path = Path(args.output) if args.output else ROOT_DIR / "outputs" / "annotated_videos" / f"{input_path.stem}_annotated.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Running VLM inference on {input_path.name}...")
    vlm = VLMJaywalkingDetector()
    vlm_res = vlm.predict(input_path)
    pred_label = vlm_res["prediction"].upper()
    confidence = vlm_res["confidence"].upper()

    print(f"VLM Decision: {pred_label} ({confidence})")
    print(f"Rendering annotated video to: {out_path}")

    cv_cfg = get_cv_config()
    yolo_model = YOLO(cv_cfg.get("yolo_model")).to(0 if torch.cuda.is_available() else "cpu")

    cap = cv2.VideoCapture(str(input_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    banner_color = (0, 0, 220) if pred_label == "JAYWALKING" else (0, 180, 0)

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # Track pedestrians and cars
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
                    cv2.putText(frame, f"Ped {tid}", (x1, max(y1 - 5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                elif cid in (2, 3, 5, 7):  # Vehicles
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (120, 120, 120), 1)

        # Draw decision banner
        text = f"VLM PREDICTION: {pred_label} [{confidence}]"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.rectangle(frame, (10, 10), (10 + tw + 20, 10 + th + 20), banner_color, -1)
        cv2.putText(frame, text, (20, 10 + th + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        out_writer.write(frame)

    cap.release()
    out_writer.release()
    print(f"Successfully saved: {out_path}")


if __name__ == "__main__":
    main()
