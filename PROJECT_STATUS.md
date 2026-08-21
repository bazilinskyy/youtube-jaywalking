# Project Status Report: Crowd Jaywalking Video Perception Pipeline

**Date:** 2026-08-21  
**Repository State:** Frozen at **Experiment 29**  
**Pipeline Framework:** YOLO11x + ByteTrack → Roadway-Entry Kinematic Validation → Qwen2.5-VL-7B (VLM Baseline)

---

## 1. Executive Summary & Core Results

| Setup / Experiment | Accuracy | Precision | Recall | Specificity | F1 Score | Notes |
|---|:---:|:---:|:---:|:---:|:---:|---|
| **Historical Short-Clip Baseline** | **97.44%** (38/39) | 93.75% | **100.00%** | 95.83% | 96.77% | Pre-cut short clips (1.5s - 4.0s), human-localized crossing intervals |
| **Architecture B Multi-Track Baseline** | 66.67% (26/39) | 58.33% | 46.67% | 79.17% | 51.85% | Online multi-track merged temporal envelope + 5 uniform frames |
| **Exp 28 Dominant-Track Isolation** | 66.67% (26/39) | 55.00% | **73.33%** | 62.50% | 62.86% | Isolates track with max lateral displacement; recovers 6/8 FNs |
| **Exp 29 Two-Stage Roadway Entry (Current Best)** | **74.36%** (29/39) | **72.73%** | 53.33% | **87.50%** | 61.54% | Filters curb-dwellers (FP: 9 → 3); highest long-video accuracy to date |

> **Understanding the 97.44% vs 74.36% Gap:**  
> The historical 97.44% baseline operated on **pre-cut short clips** where human annotators tightly bounded the pedestrian crossing interaction. In the long-video pipeline, the system must perform online temporal event localization and track attribution from raw uncut footage. The current best long-video accuracy is **74.36% (29/39 clips)**.

---

## 2. Experimental Progression (Experiments 23–29)

1. **Experiment 23 (Error Forensics):** Analyzed the 11 misclassified clips. Proved 8/8 False Negatives were caused by uniform sampling jumping over the 0.5s curb-stepping transition moment.
2. **Experiment 24 (Frame Budget Study):** Tested budgets N in {3, 5, 8, 10, 12, 16}. Proved increasing uniform frames beyond 10 triggers VLM context saturation (Recall collapsed to 0.0% at N=16).
3. **Experiment 25 (Controlled Frame Selection):** Proved sampling at kinematic velocity peaks (Δx / width) outperforms uniform sampling, raising F1 to 57.14%.
4. **Experiment 26 (Historical Boundary Audit):** Audited mapping.csv and confirmed it contains YouTube metadata without crossing boundaries. Confirmed the 97.44% baseline evaluated human-curated JAAD clips.
5. **Experiment 27 (Boundary-Controlled Reproduction):** Proved feeding exact JAAD clip boundaries into single-call VLM yields 61.54% with uniform sampling due to background dwell false alarms.
6. **Experiment 28 (Single-Pedestrian Track Isolation):** Isolated the dominant pedestrian track using normalized lateral displacement, boosting Recall from 46.67% to **73.33%** and F1 to **62.86%**.
7. **Experiment 29 (Roadway-Entry Validation):** Implemented two-stage geometric validation (curb stability → roadway penetration), crushing False Positives from 9 to 3 and achieving **74.36% Accuracy** and **72.73% Precision**.

---

## 3. Current System Architecture (Experiment 29 Endpoint)

```
Uncut Video Stream
       │
       ▼
1. YOLO11x + ByteTrack (Pedestrian Tracking)
       │
       ▼
2. Kinematic Motion Profiler (Lateral Displacement D = |Δx| / bbox_width)
       │
       ▼
3. Dominant Track Isolation (Selects Primary Crossing Candidate)
       │
       ▼
4. Two-Stage Roadway-Entry Validation (Pre-entry stability -> Roadway penetration)
       │
       ▼ (5 Key-State Frames: Pre-entry, Entry, Peak Velocity, Post-entry, Context)
5. VLM Baseline: Qwen2.5-VL-7B (5-Step Chain-of-Causation Reasoning)
       │
       ▼
   Output: { "verdict": "JAYWALKING" / "COMPLIANT", "responsible_track_id": ID }
```

---

## 4. Current Limitations & Remaining Bottlenecks

1. **Recall is 53.33% (7 False Negatives):** The two-stage filter successfully eliminated false alarms, but aggressive entry thresholding suppressed true crossings where pedestrians enter slowly or diagonally (video_0030, video_0110, video_0122, video_0139).
2. **Camera Ego-Motion Confounding:** On vehicle turning clips (video_0160), camera rotation induces apparent lateral displacement on sidewalk pedestrians.
3. **Spatial Ambiguity in Pedestrian Plazas:** Clips with shared commercial driveways (video_0003) remain visually ambiguous without 3D curb height cues.

---

## 5. Reproduction Instructions

To reproduce the validated Experiment 29 pipeline and generate evaluation metrics:

```bash
# 1. Activate Python Environment
source myenv/bin/activate

# 2. Run Experiment 29 Benchmark
python experiments/run_exp29_roadway_entry.py

# 3. View Artifacts & Machine-Readable Results
cat outputs/exp29_roadway_entry/exp29_results.json
cat outputs/exp29_roadway_entry/false_positive_forensics.json
```
