#!/usr/bin/env python3
"""Debug: check what YOLO detects in test clips."""
import sys, cv2, torch
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from utils.crossing.traffic_light import _get_model

CLIPS = [
    ROOT/"evaluation/jaad_positive/video_0003.mp4",
    ROOT/"evaluation/jaad_positive/video_0035.mp4",
    ROOT/"evaluation/jaad_negative/video_0014.mp4",
]

yolo = YOLO(str(ROOT/"yolo11x.pt"))
tl_model = _get_model()

for clip_path in CLIPS:
    print(f"\n--- {clip_path.name} ---")
    cap = cv2.VideoCapture(str(clip_path))
    frame_idx = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        frame_idx += 1

        # Main YOLO: check for class 9 (traffic light)
        results = yolo(frame, conf=0.3, verbose=False, device=0 if torch.cuda.is_available() else "cpu")
        if results and results[0].boxes is not None:
            classes = results[0].boxes.cls.cpu().numpy()
            if 9 in classes:
                print(f"  Frame {frame_idx}: YOLO sees traffic light (class 9)")
                # Get the TL crop and run TLD-READY
                boxes = results[0].boxes.xyxy.cpu().numpy()
                for box, c in zip(boxes, classes):
                    if int(c) == 9:
                        x1,y1,x2,y2 = box
                        tl_crop = frame[max(0,int(y1)):min(frame.shape[0],int(y2)),
                                        max(0,int(x1)):min(frame.shape[1],int(x2))]
                        if tl_crop.size > 0:
                            tl_results = tl_model.predict(tl_crop, verbose=False, conf=0.1, imgsz=64)
                            if tl_results and len(tl_results) > 0 and tl_results[0].boxes is not None and len(tl_results[0].boxes) > 0:
                                for tb in tl_results[0].boxes:
                                    cls_id = int(tb.cls[0])
                                    conf = float(tb.conf[0])
                                    print(f"    TLD-READY: class={cls_id} conf={conf:.2f} name={tl_model.names[cls_id]}")
                            else:
                                print(f"    TLD-READY: no detection in TL crop")
                        break
        if frame_idx >= 60:
            break
    cap.release()
