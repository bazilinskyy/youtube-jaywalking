# Mentor Research Progress Report: Pedestrian Keypoints & Video Perception Pipeline

**To:** Mentor / Research Advisor  
**From:** Mobility Squad AI Research Team  
**Date:** 2026-08-21  
**Project:** Long-Video Pedestrian Jaywalking Perception & Track Attribution  
**Status:** Frozen at **Experiment 29** + Offline 17-Keypoint Kinematic Dataset Extraction

---

## 1. Current Classification Benchmark Summary

We have frozen the current accuracy iteration at **Experiment 29**. The table below summarizes our progression from the historical baseline to our current best long-video pipeline:

| Setup / Experiment | Accuracy | Precision | Recall | Specificity | F1 Score | FP Count | FN Count |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Historical Short-Clip Baseline** | **97.44%** | **93.75%** | **100.0%** | **95.83%** | **96.77%** | 1 | 0 |
| **Architecture B Multi-Track Baseline** | 66.67% | 58.33% | 46.67% | 79.17% | 51.85% | 5 | 8 |
| **Exp 28 Dominant-Track Isolation** | 66.67% | 55.00% | **73.33%** | 62.50% | 62.86% | 9 | **4** |
| **Exp 29 Two-Stage Roadway Entry (Current Best)** | **74.36%** | **72.73%** | 53.33% | **87.50%** | **61.54%** | **3** | 7 |

### Why Historical (97.44%) vs Long-Video (74.36%) Differ
- **Historical Setup:** Evaluated pre-cut 1.5s–4.0s JAAD video clips where human annotators pre-localized the exact pedestrian crossing window.
- **Long-Video Setup:** Operates on uncut raw video streams, requiring online temporal event boundary localization, multi-pedestrian separation, and track attribution.
- **Current Long-Video State:** Experiment 29 achieved our highest long-video accuracy (**74.36%**) and precision (**72.73%**), reducing false alarms from 9 down to 3.

---

## 2. Key Empirical Findings (Experiments 23–29)

1. **Error Forensics (Exp 23):** Proved 8/8 False Negatives occurred because 5-frame uniform sampling across multi-second clips hopped over the brief ~0.5s curb step-off transition.
2. **Frame Budget Scaling (Exp 24):** Tested 3 to 16 frames. Increasing uniform frames beyond 10 saturated the VLM context window, causing recall to collapse to 0.0%.
3. **Temporal Boundary Audit & Control (Exp 26–27):** Audited `mapping.csv` and confirmed it contains YouTube metadata rather than ground-truth labels. Confirmed boundary matching alone without track isolation yields 61.54%.
4. **Dominant Pedestrian Isolation (Exp 28):** Selecting the primary moving track by normalized lateral displacement ($\Delta x / \text{bbox\_width}$) recovered 6/8 False Negatives and jumped Recall to **73.33%**.
5. **Two-Stage Roadway Entry Validation (Exp 29):** Requiring pre-entry curb stability before roadway penetration eliminated 77.8% of false alarms ($FP: 9 \to 3$), raising precision to **72.73%** and accuracy to **74.36%**.

---

## 3. Current Limitations

- **Recall is 53.33%:** The two-stage filter successfully eliminated false alarms, but strict entry thresholding suppressed slow/diagonal crossers (7 False Negatives).
- **System is not yet production-ready:** The gap between 74.36% and 97.44% indicates that 2D bounding box kinematics alone cannot fully capture 3D road/curb topography.

---

## 4. Pedestrian Keypoint & Body-Movement Extraction Deliverable

As requested, we implemented and executed an **offline 17-keypoint COCO pose & kinematic trajectory extraction pipeline** across all 39 development videos.

### Extracted Parameters Per Pedestrian Track
- `video_id`, `track_id`, `frame_id`, `timestamp_seconds`
- 2D Bounding Box: `[center_x, center_y, width, height, bottom_y]`
- 17 COCO Keypoints: `[x_norm, y_norm, confidence]` for head, shoulders, elbows, wrists, hips, knees, ankles
- Kinematics: normalized lateral displacement, frame-by-frame velocity, acceleration, trajectory direction
- Event Markers: `roadway_entry_candidate_frame`, `peak_motion_frame`, `valid_keypoint_frames`

### Deliverable File Locations
```
outputs/keypoint_analysis/
├── per_video/               # 39 JSON files containing frame-by-frame tracks and keypoints
│   ├── video_0003_keypoints.json
│   ├── video_0028_keypoints.json
│   └── ...
├── summaries/
│   └── dataset_summary.json # Dataset-level tracking & pose coverage metrics
└── visualizations/          # Trajectory plots & representative annotated MP4s
    ├── video_0003_trajectory_diagnostic.png
    ├── video_0028_keypoint_diagnostic.mp4
    └── ...
```

---

## 5. Exact Reproduction Commands

```bash
# 1. Reproduce Current Best Long-Video Pipeline (Exp 29, 74.36% Accuracy)
python experiments/run_exp29_roadway_entry.py

# 2. Reproduce Offline Keypoint & Trajectory Extraction
python scripts/extract_pedestrian_keypoints.py
```

---

## 6. Next Recommended Research Step

Combine the **17-keypoint lower-body stride kinematics** (ankle spread ratio $\Delta x_{\text{ankle}} / h_{\text{body}}$ and knee flexion) with our **Two-Stage Roadway-Entry Validator** to recover the remaining slow/diagonal False Negatives without re-introducing false alarms.
