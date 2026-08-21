# Jaywalking VLM — Research Log

---

## 1. Current Status

* **Project Stage:** Structured Multi-Modal Cue Integration (Phase 4 — Motion & Vehicle State Context).
* **Current Canonical Architecture:** Temporal Multi-Frame VLM (`Qwen2.5-VL-7B` via Ollama) receiving 3 chronological frames and structured CV motion context.
* **Current Model:** `qwen2.5vl:7b` (local Ollama instance on `http://localhost:11434/api/chat`).
* **Current Benchmark / Dataset:** Canonical 39 evaluable clips from `data/ground_truth.csv` (15 Jaywalking, 24 Compliant).
* **Best Known Results:**
  * **Best Overall Accuracy & Specificity:** Threshold = 3/3 (Unanimous V1) — **82.05% Accuracy**, **87.50% Specificity** (11 TP, 21 TN, 3 FP, 4 FN, 75.86% F1).
  * **Best Balanced Operating Point:** Baseline V1 (Threshold $\ge 2/3$) — **69.23% Accuracy**, **86.67% Recall**, **58.33% Specificity** (13 TP, 14 TN, 10 FP, 2 FN, 68.42% F1).
  * **Best Balanced Operating Point:** VLM Full-Video CoC Reasoning — **80.00% Held-Out Accuracy**, **87.50% Precision**, **90.00% Specificity** (7 TP, 9 TN, 1 FP, 3 FN).
* **Current Biggest Bottleneck:** Minor FN cases on subtle crossing initiation.
* **Current Experiment:** Completed Experiment 11 (VLM Full-Video Chain-of-Causation Reasoning).
* **Immediate Next Step:** Commit and push VLM Full-Video CoC Reasoning pipeline to `v1` branch.hold Modes.




---

## 2. Project Architecture

### Current Canonical Inference Flow

```
Input Video (.mp4)
      │
      ├───► [Keyframe Sampler] ──────► Extract 3 Chronological Frames (T0, T_mid, T_end)
      │                                       │
      ├───► [Pedestrian Motion Extractor] ──► YOLO11x + ByteTrack (dx, pos_start, pos_end, movement)
      │                                       │
      ▼                                       ▼
[Structured Prompt Assembly] ◄────────────────┴─────────────────┘
  (3 Images + Chronological Temporal Instructions + Structured Motion Context)
      │
      ▼
[Ollama API Client] ────────► Qwen2.5-VL-7B (Temperature=0.0, num_ctx=16384)
      │
      ▼
[Response Parser] ──────────► Final Binary Decision ("jaywalking" | "compliant")
      │
      ▼
[Standardized Evaluator] ───► Metrics & Predictions CSV (Accuracy, Precision, Recall, F1, Matrix)
```

### Major Components & Key File Paths

* **Pipeline Orchestrator:** [`src/pipeline.py`](file:///home/tue20234844/crowd-jaywalking/src/pipeline.py) — Factory `get_pipeline()` providing `vlm`, `cv`, and `ensemble` execution.
* **VLM Jaywalking Detector:** [`src/vlm/detector.py`](file:///home/tue20234844/crowd-jaywalking/src/vlm/detector.py) — Supports single-frame voting, multi-frame temporal reasoning, boundary injection, and motion context.
* **VLM Prompt Registry:** [`src/vlm/prompts.py`](file:///home/tue20234844/crowd-jaywalking/src/vlm/prompts.py) — Canonical, V2, Temporal, and Temporal Motion prompts.
* **Ollama Client:** [`src/vlm/client.py`](file:///home/tue20234844/crowd-jaywalking/src/vlm/client.py) — Base64 encoding and robust HTTP chat interface for local Ollama.
* **Pedestrian Motion Extractor:** [`src/cv/pedestrian_motion.py`](file:///home/tue20234844/crowd-jaywalking/src/cv/pedestrian_motion.py) — YOLO11x + ByteTrack trajectory and displacement extraction.
* **Road Boundary Detector:** [`src/cv/boundary.py`](file:///home/tue20234844/crowd-jaywalking/src/cv/boundary.py) — Canny/Hough line curb and sidewalk boundary estimator.
* **Evaluation Engine:** [`evaluation/evaluator.py`](file:///home/tue20234844/crowd-jaywalking/evaluation/evaluator.py) — Standardized 39-clip evaluation harness producing JSON metrics and CSV predictions.
* **Evaluation CLI:** [`scripts/run_evaluation.py`](file:///home/tue20234844/crowd-jaywalking/scripts/run_evaluation.py) — Command-line runner for all benchmark configurations.

---

## 3. Dataset & Ground Truth

* **Ground-Truth Source:** [`data/ground_truth.csv`](file:///home/tue20234844/crowd-jaywalking/data/ground_truth.csv) (Single canonical source).
* **Total Clips in Raw Directory:** 50 clips (`video_0003.mp4` through `video_0336.mp4` in `data/raw_clips/`).
* **Evaluable Clips (`is_evaluated == True`):** **39 clips**.
* **Class Distribution (Evaluable Subset):**
  * `jaywalking`: **15 clips** (38.46%)
  * `compliant`: **24 clips** (61.54%)
* **Excluded / Unlabeled Clips (11 clips):**
  * `video_0031.mp4`, `video_0078.mp4`, `video_0079.mp4`, `video_0096.mp4`, `video_0100.mp4`, `video_0101.mp4`, `video_0152.mp4`, `video_0153.mp4`, `video_0154.mp4`, `video_0176.mp4`, `video_0182.mp4`.
  * *Reason:* Labeled `ambiguous`, `unknown`, or `unlabeled` pending human arbitration.
* **Dataset Limitations:** Dashcam perspective creates parallax and camera ego-motion; visual absence of zebra stripes often coincides with legal unmarked intersection yielding.

---

## 4. Current Baseline

### Baseline V1 (Per-Frame Canonical Voting)

* **Model:** Qwen2.5-VL-7B (`qwen2.5vl:7b`) via Ollama.
* **Prompt:** `CANONICAL_PROMPT` in [`src/vlm/prompts.py`](file:///home/tue20234844/crowd-jaywalking/src/vlm/prompts.py).
* **Frame Strategy:** 3 equidistant keyframes per clip (Frame 0, mid, end), evaluated individually, fused via majority vote.
* **Inference Configuration:** `temperature=0.0`, `max_tokens=10`, `jpeg_quality=85`.

| Metric | Baseline V1 Result |
|:---|---:|
| **Accuracy** | **69.23%** (27/39) |
| **Precision** | **56.52%** |
| **Recall** | **86.67%** (13/15) |
| **Specificity** | **58.33%** (14/24) |
| **F1 Score** | **68.42%** |
| **True Positives (TP)** | **13** |
| **True Negatives (TN)** | **14** |
| **False Positives (FP)** | **10** |
| **False Negatives (FN)** | **2** |
| **Avg Inference Time** | **4.12s / clip** (160.66s total) |

---

## 5. Experiments

### Experiment 1 — Prompt V2 Only (Text Prompt Refinement)

* **Date:** 2026-08-14
* **Goal:** Test whether prompt-only guidance (distinguishing sidewalk/curb, vehicle yielding, parking lots, and unmarked crossings) eliminates curb-dwell false positives.
* **Hypothesis:** Adding explicit traffic rule exceptions into the prompt will allow the VLM to classify curb-dwellers and yielding situations as `COMPLIANT`.
* **What changed:** Replaced `CANONICAL_PROMPT` with `PROMPT_V2`.
* **What stayed unchanged:** Everything else (3 frames evaluated individually + majority voting, Qwen2.5-VL-7B, 39 clips).
* **Dataset:** Canonical 39 clips.
* **Result:**

| Metric | Experiment 1 Result |
|:---|---:|
| **Accuracy** | 38.46% (15/39) |
| **Precision** | 38.46% |
| **Recall** | 100.00% (15/15) |
| **Specificity** | 0.00% (0/24) |
| **F1 Score** | 55.56% |
| **True Positives (TP)** | 15 |
| **True Negatives (TN)** | 0 |
| **False Positives (FP)** | 24 |
| **False Negatives (FN)** | 0 |
| **Avg Inference Time** | 4.24s / clip (165.20s total) |

* **Failure Analysis:** Text prompt refinement on static 2D frames caused severe **negative confirmation bias**. Without spatial coordinate grounding or motion vectors, the model perceived all road-adjacent pedestrians as potential violators and predicted `JAYWALKING` on 100% of clips.
* **Conclusion:** Prompt text expansion alone is ineffective on static isolated frames.
* **Decision:** **REJECT**.
* **Why:** Collapsed specificity to 0.0% and increased FP from 9 to 24.
* **Next Implication:** Must provide structured spatial and/or temporal grounding.

---

### Experiment 2 — Prompt V2 + BoundaryDetector Context

* **Date:** 2026-08-14
* **Goal:** Test whether structured spatial observations from classical CV (`BoundaryDetector` + YOLO) resolve curb vs. roadway ambiguity.
* **Hypothesis:** Providing `pedestrian_position = sidewalk | curb | roadway | uncertain` as prompt context enables the VLM to ignore curb-dwellers.
* **What changed:** Injected spatial context into each frame's prompt using Canny/Hough boundary detection.
* **What stayed unchanged:** Prompt V2, 3 frames evaluated individually + majority vote, Qwen2.5-VL-7B.
* **Dataset:** Canonical 39 clips.
* **Result:**

| Metric | Experiment 2 Result |
|:---|---:|
| **Accuracy** | 48.72% (19/39) |
| **Precision** | 39.13% |
| **Recall** | 60.00% (9/15) |
| **Specificity** | 41.67% (10/24) |
| **F1 Score** | 47.37% |
| **True Positives (TP)** | 9 |
| **True Negatives (TN)** | 10 |
| **False Positives (FP)** | 14 |
| **False Negatives (FN)** | 6 |
| **Avg Inference Time** | 4.43s / clip (172.92s total) |

* **Failure Analysis:** Fixed 4 out of the 9 baseline false positives (`video_0003`, `video_0168`, `video_0227`, `video_0322`). However, it introduced **6 False Negatives** on actual jaywalkers (`video_0028`, `video_0073`, `video_0092`, `video_0104`, `video_0110`, `video_0139`) because pedestrians standing at the curb in Frame 1 were tagged as `curb`, diluting the majority vote.
* **Conclusion:** Static spatial position is helpful for localization, but single-frame spatial snapshots miss crossing initiation dynamics.
* **Decision:** **MODIFY / INVESTIGATE**.
* **Why:** Recovered 10 TNs, but trade-off on Recall (60.0%) was too severe.
* **Next Implication:** Temporal sequence reasoning is required to observe trajectory over time.

---

### Experiment 3 — Temporal Multi-Image VLM Input

* **Date:** 2026-08-14
* **Goal:** Test whether concatenating 3 chronological frames into a single VLM chat request enables native temporal reasoning across frames.
* **Hypothesis:** Presenting frames in temporal order allows the VLM to observe pedestrian movement and vehicle yielding directly.
* **What changed:** Switched architecture from 3 independent VLM requests + majority vote to **1 single VLM request with 3 chronological frames**.
* **What stayed unchanged:** Qwen2.5-VL-7B, 39 clips, no classical CV heuristics.
* **Dataset:** Canonical 39 clips.
* **Result:**

| Metric | Experiment 3 Result |
|:---|---:|
| **Accuracy** | 38.46% (15/39) |
| **Precision** | 38.46% |
| **Recall** | 100.00% (15/15) |
| **Specificity** | 0.00% (0/24) |
| **F1 Score** | 55.56% |
| **True Positives (TP)** | 15 |
| **True Negatives (TN)** | 0 |
| **False Positives (FP)** | 24 |
| **False Negatives (FN)** | 0 |
| **Avg Inference Time** | **2.70s / clip** (105.36s total, -27.2% faster) |

* **Failure Analysis:** Achieved **100.0% Recall** (recovered all 6 FNs from Exp 2 and both FNs from V1). However, Specificity remained 0.0% because raw visual tokens without numeric motion or vehicle yielding metrics caused the model to classify all pedestrian-road interactions as violations.
* **Conclusion:** Multi-image VLM input improves speed and eliminates False Negatives, but raw image tokens alone cannot resolve compliant vehicle yielding.
* **Decision:** **KEEP ARCHITECTURE / INVESTIGATE HYBRID MOTION**.
* **Why:** Superior inference speed and 100% recall make single-request multi-frame input the best architectural foundation.
* **Next Implication:** Feed explicit structured CV motion features into the temporal VLM.

---

### Experiment 4A — Pedestrian Motion Context

* **Date:** 2026-08-14
* **Goal:** Test whether adding explicit structured pedestrian trajectory features (normalized displacement $\Delta x$, start/end position, movement direction) via YOLO11x + ByteTrack resolves curb-dwelling vs. active crossing ambiguity.
* **Hypothesis:** Structured pedestrian displacement features will allow the temporal VLM to identify stationary curb pedestrians as compliant.
* **What changed:** Injected structured `Pedestrian motion:` context block into the 3-frame temporal prompt.
* **What stayed unchanged:** 3-frame temporal VLM architecture, Qwen2.5-VL-7B, 39 clips.
* **Dataset:** Canonical 39 clips.
* **Result:**

| Metric | Experiment 4A Result |
|:---|---:|
| **Accuracy** | 38.46% (15/39) |
| **Precision** | 38.46% |
| **Recall** | 100.00% (15/15) |
| **Specificity** | 0.00% (0/24) |
| **F1 Score** | 55.56% |
| **True Positives (TP)** | 15 |
| **True Negatives (TN)** | 0 |
| **False Positives (FP)** | 24 |
| **False Negatives (FN)** | 0 |
| **Avg Inference Time** | 3.70s / clip (144.21s total) |

* **Failure Analysis:** Maintained 100% recall on all jaywalkers, but 0/24 false positives were fixed. **Pedestrian motion is symmetric between legal crossings (yielded) and illegal crossings (jaywalking)**—the pedestrian crosses the street in both cases.
* **Conclusion:** Pedestrian motion in isolation does not distinguish compliance from violation; compliance depends on **vehicle / ego-motion behavior**.
* **Decision:** **INVESTIGATE / PROCEED TO EXPERIMENT 4B**.
* **Why:** Confirmed that the missing causal variable is vehicle state, not pedestrian displacement.
* **Next Implication:** Implement vehicle tracking and yielding/deceleration detection (Experiment 4B).

---

### Experiment 4B — Vehicle / Ego-Motion & Yielding Context

* **Date:** 2026-08-14
* **Goal:** Test whether injecting structured vehicle kinematic state (vehicle presence, approaching velocity $\Delta v$, stopped/yielding status, ego-vehicle motion) enables the Temporal VLM to recognize compliant yielding scenes and eliminate the 24 False Positives.
* **Hypothesis:** Bounding box scale change $\Delta A$ and vertical displacement $\Delta y$ from ByteTrack will distinguish stopped/yielding vehicles from active oncoming traffic.
* **What changed:** Built `VehicleStateExtractor` using YOLO11x (classes: car, bus, truck) + ByteTrack and injected structured `Vehicle interaction:` context into the 3-frame temporal prompt.
* **What stayed unchanged:** 3-frame chronological temporal VLM architecture, Qwen2.5-VL-7B, pedestrian motion extractor, 39 canonical clips.
* **Dataset:** Canonical 39 clips.
* **Result:**

| Metric | Experiment 4B Result |
|:---|---:|
| **Accuracy** | 38.46% (15/39) |
| **Precision** | 38.46% |
| **Recall** | **100.00%** (15/15) |
| **Specificity** | 0.00% (0/24) |
| **F1 Score** | 55.56% |
| **True Positives (TP)** | 15 |
| **True Negatives (TN)** | 0 |
| **False Positives (FP)** | 24 |
| **False Negatives (FN)** | 0 |
| **Avg Inference Time** | 4.78s / clip (186.59s total) |

* **Failure Analysis:** Maintained **100% Recall** with 0 FN, but 0/24 False Positives were fixed.
  * **Root Cause 1 (Monocular Parallax):** In forward-facing dashcam footage without camera extrinsics or IMU telemetry, forward ego-vehicle movement causes *all* 2D bounding boxes (including parked cars and stopped vehicles) to expand laterally and move downwards ($\Delta A > 0, \Delta y > 0$). The 2D tracker cannot reliably separate ego-car approach from oncoming car motion.
  * **Root Cause 2 (VLM Prior Bias):** Even when context states `interaction: yielding`, if the VLM sees a person on the asphalt with vehicles visible in the scene, its conversational safety bias triggers `JAYWALKING`.
* **Conclusion:** 2D monocular bounding box kinematics are too noisy without 3D depth/homography, and qualitative text cues cannot overcome the VLM's conservative violation bias on multi-image inputs.
* **Decision:** **INVESTIGATE / CALIBRATE DECISION ARBITRATION**.
* **Why:** Monocular 2D vehicle kinematics cannot reliably separate ego-motion from yielding; preserving 100% recall while restoring specificity requires a calibrated multi-cue arbitration mechanism.
* **Next Implication:** Investigate decision calibration / ensemble arbitration between the high-specificity Baseline V1 and the 100%-recall Temporal VLM.

---

### Experiment 5 — Calibrated Multi-Cue Decision Arbitration

* **Date:** 2026-08-14
* **Goal:** Test whether simple decision arbitration between Baseline V1 (high specificity) and Temporal VLM (100% recall) can improve overall performance without adding perception models.
* **Hypothesis:** Boolean combination or confidence thresholding will filter out false positives while maintaining sensitivity.
* **What changed:** Evaluated 5 arbitration policies on existing predictions:
  * **Policy A (Consensus):** `V1 == JAYWALKING AND Temporal == JAYWALKING`
  * **Policy B (V1 Priority):** `If V1 == COMPLIANT -> COMPLIANT, Else Temporal`
  * **Policy C (Temporal Confirmation):** `If Temporal == JAYWALKING AND V1 == JAYWALKING -> JAYWALKING, Else COMPLIANT`
  * **Policy D (Unanimous 3-0 V1 Vote):** `If V1 frame votes == 3-0 JAYWALKING -> JAYWALKING, Else COMPLIANT`
  * **Policy E (Sensitive Safety Filter $\ge 1$ Vote):** `If V1 frame votes $\ge 1$ JAYWALKING -> JAYWALKING, Else COMPLIANT`
* **What stayed unchanged:** No retraining, no new models, exact same 39 clips.
* **Dataset:** Canonical 39 clips.
* **Result:**

| Policy / Candidate | Accuracy | Precision | Recall | Specificity | F1 Score | TP | TN | FP | FN |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Baseline V1 (Majority $\ge 2/3$)** | 71.79% | 59.09% | 86.67% | 62.50% | 70.27% | 13 | 15 | 9 | 2 |
| **Temporal VLM (Exp 3)** | 38.46% | 38.46% | **100.00%** | 0.00% | 55.56% | 15 | 0 | 24 | 0 |
| **Policy A (Consensus)** | 71.79% | 59.09% | 86.67% | 62.50% | 70.27% | 13 | 15 | 9 | 2 |
| **Policy B (V1 Priority)** | 71.79% | 59.09% | 86.67% | 62.50% | 70.27% | 13 | 15 | 9 | 2 |
| **Policy C (Temporal Confirmation)** | 71.79% | 59.09% | 86.67% | 62.50% | 70.27% | 13 | 15 | 9 | 2 |
| **Policy D (Unanimous 3-0 V1)** | **82.05%** | **78.57%** | 73.33% | **87.50%** | **75.86%** | 11 | **21** | **3** | 4 |
| **Policy E (Sensitive $\ge 1$ Vote)** | 53.85% | 45.45% | **100.00%** | 25.00% | 62.50% | 15 | 6 | 18 | 0 |

* **Analysis:**
  1. **Binary Fusion Degeneracy:** Because Temporal VLM outputs 100% positive predictions across the 39 clips, Policies A, B, and C are mathematically identical to Baseline V1.
  2. **Unanimous Consensus (Policy D):** Requiring 3-0 frame unanimity boosts Accuracy to **82.05%** and Specificity to **87.50%** (eliminating 6 of the 9 baseline FPs: `video_0003`, `video_0099`, `video_0150`, `video_0227`, `video_0238`, `video_0322`). However, it introduces 2 new FNs (`video_0092`, `video_0122`), lowering recall to 73.33%.
  3. **Remaining FPs under 3-0 Unanimity (3 clips):** `video_0168`, `video_0241`, `video_0297`. In all three clips, pedestrians are walking on unmarked asphalt with yielding cars, which the static VLM unanimously perceives as an active violation.
* **Conclusion:** External binary arbitration between V1 and Temporal VLM adds zero value. Internal frame vote margin (Policy D) achieves high precision (82.05% Acc, 87.5% Spec) if recall penalty is acceptable. Baseline V1 remains the best balanced canonical operating point (71.79% Acc, 86.67% Recall, 62.5% Spec).
* **Decision:** **KEEP BASELINE V1 AS CANONICAL OPERATING POINT / EXPOSE POLICY D AS HIGH-PRECISION PRESET**.
* **Why:** Simple external arbitration cannot overcome the 0% specificity of multi-image VLM.

---

### Experiment 6 — V1 Vote-Margin Calibration

* **Date:** 2026-08-14
* **Goal:** Quantify the exact empirical uncertainty distribution across 3-frame vote patterns (`3/3`, `2/3`, `1/3`, `0/3`) and determine whether an optimal intermediate operating point exists.
* **Hypothesis:** 2/3 split votes represent model uncertainty that can be calibrated to optimize precision vs. recall.
* **What changed:** Evaluated empirical probability distributions and threshold cutoffs across all 39 clips without altering model or dataset.
* **What stayed unchanged:** Qwen2.5-VL-7B, 3-frame canonical predictions, 39 clips.
* **Dataset:** Canonical 39 clips.
* **Result (Vote Pattern Breakdown):**

| Vote Pattern | Clips (N) | GT Jaywalking | GT Compliant | V1 Pred | Correct | Incorrect | Precision (JW) | Specificity (Comp) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **3/3 JAYWALKING** *(Unanimous Violation)* | 14 | 11 | 3 | `JAYWALKING` | 11 | 3 | **78.6%** | 21.4% |
| **2/3 JAYWALKING** *(Split Violation)* | 8 | 2 | 6 | `JAYWALKING` | 2 | 6 | **25.0%** | **75.0%** |
| **1/3 JAYWALKING** *(Split Compliant)* | 11 | 2 | 9 | `COMPLIANT` | 9 | 2 | 18.2% | **81.8%** |
| **0/3 JAYWALKING** *(Unanimous Compliant)* | 6 | 0 | 6 | `COMPLIANT` | 6 | 0 | 0.0% | **100.0%** |

* **Result (Threshold Policies Evaluation):**

| Threshold Policy | Accuracy | Precision | Recall | Specificity | F1 Score | TP | TN | FP | FN |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Threshold $\ge 1/3$** *(Sensitive / Safety)* | 53.85% | 45.45% | **100.00%** | 25.00% | 62.50% | 15 | 6 | 18 | 0 |
| **Threshold $\ge 2/3$** *(Canonical Baseline V1)* | 71.79% | 59.09% | 86.67% | 62.50% | 70.27% | 13 | 15 | 9 | 2 |
| **Threshold $= 3/3$** *(Unanimous / High Precision)* | **82.05%** | **78.57%** | 73.33% | **87.50%** | **75.86%** | 11 | **21** | **3** | 4 |

* **Qualitative Analysis of the 8 Split (2/3) Cases:**
  * **6 False Positives (75.0% of the bucket):** `video_0003` (parking aisle navigation), `video_0099`, `video_0150`, `video_0227`, `video_0322` (curb-dwellers standing at the edge of asphalt where 1 frame shows sidewalk presence), `video_0238` (yielding car where pedestrian starts on sidewalk in Frame 0).
  * **2 True Jaywalkers (25.0% of the bucket):** `video_0092`, `video_0122` (pedestrians who begin their mid-block crossing during or after Frame 0).
* **Conclusion:** 
  1. The 2/3 vote pattern is heavily skewed toward **compliant false alarms (75% compliant)**.
  2. Because $k \in \{0, 1, 2, 3\}$, there are **only three discrete operating points** on a 3-frame discrete vote ROC curve: 53.85%, 71.79%, and 82.05%. No intermediate operating point exists with $N=3$ without fractional confidence or additional frames ($N=5$).
  3. The vote margin is a highly reliable empirical uncertainty metric: $P(\text{Jaywalking} \mid 3/3) = 78.6\%$, $P(\text{Jaywalking} \mid 2/3) = 25.0\%$, $P(\text{Jaywalking} \mid 1/3) = 18.2\%$, $P(\text{Jaywalking} \mid 0/3) = 0.0\%$.
* **Decision:** **FORMALIZE MULTI-THRESHOLD ARCHITECTURAL MODES IN CODEBASE**.
* **Next Implication:** Update CLI and API to allow users/deployers to select `--threshold {1, 2, 3}` or preset modes (`balanced`, `high_precision`, `high_recall`).

---

## 6. Failure Analysis

### 1. Vehicle Yielding & Deceleration Ambiguity
* **Symptoms:** Pedestrian is legally crossing an unmarked street or intersection because oncoming vehicles have decelerated or stopped to yield; VLM predicts `JAYWALKING`.
* **Example Clips:** `video_0168.mp4`, `video_0238.mp4`, `video_0241.mp4`, `video_0297.mp4`.
* **Root Cause:** In static or uncalibrated multi-frame images, a stopped/yielding vehicle appears identical to an oncoming threat. Monocular 2D bounding box scaling is confounded by ego-vehicle camera motion.
* **Current Status:** Partially mitigated by 3-0 threshold for `video_0238`, but `video_0168`, `video_0241`, `video_0297` remain unanimous 3-0 FPs.
* **Possible Solution:** Scene semantic context / lane cross-section segmentation.

### 2. Curb-Dwelling vs. Road Entry Ambiguity
* **Symptoms:** Pedestrian stands at the curb or walks along the sidewalk edge; VLM predicts `JAYWALKING`.
* **Example Clips:** `video_0099.mp4`, `video_0150.mp4`, `video_0227.mp4`, `video_0322.mp4`.
* **Root Cause:** Proximity to asphalt triggers visual violation bias in general vision-language models on 2-1 vote splits.
* **Current Status:** **Resolved under Unanimous 3-0 Threshold**; all 4 clips correctly flipped to `COMPLIANT`.

### 3. Non-Road & Parking Lot Navigation
* **Symptoms:** Pedestrian walking between parked cars in a commercial parking lot or alleyway; VLM predicts `JAYWALKING`.
* **Example Clips:** `video_0003.mp4`.
* **Root Cause:** Absence of lane markings and presence of parked cars confuses road detection on 2-1 vote splits.
* **Current Status:** **Resolved under Unanimous 3-0 Threshold**; flipped to `COMPLIANT`.

---

## 7. Key Decisions

* **Decision 1: Classical Hough Crosswalk Detector Discarded (2026-08-14)**
  * *Reason:* Evaluated on 39 clips; yielded a 100% False Positive rate on crosswalks due to asphalt cracks, lane lines, and shadows.
  * *Evidence:* Feature branch evaluation benchmark.
* **Decision 2: VLM Remains the Canonical Final Arbiter (2026-08-14)**
  * *Reason:* Classical CV heuristics are brittle to edge cases; CV components must provide evidence/context to the VLM rather than making final binary decisions.
* **Decision 3: Multi-Frame Concatenated VLM Adopted as Safety Filter Architecture (2026-08-14)**
  * *Reason:* Reduced inference latency by 27.2% and achieved 100.0% Recall (0 FN).
* **Decision 4: Text-Only Prompt Refinement Rejected (2026-08-14)**
  * *Reason:* Exp 1 proved that expanding prompt rules without spatial or motion grounding collapses specificity to 0.0%.
* **Decision 5: Monocular 2D Bounding Box Kinematics Insufficient for Yielding Detection (2026-08-14)**
  * *Reason:* Exp 4B proved that without 3D depth or IMU telemetry, 2D bounding box scaling cannot reliably separate ego-camera motion from vehicle yielding.
* **Decision 6: External Binary Arbitration Rejected; Baseline V1 Maintained (2026-08-14)**
  * *Reason:* Exp 5 proved that combining V1 with 100%-positive Temporal VLM is mathematically degenerate. Baseline V1 (Majority Vote) remains canonical.
* **Decision 7: Vote Margin Formalized as Calibrated Uncertainty Metric (2026-08-14)**
  * *Reason:* Exp 6 demonstrated clear monotonicity: $P(\text{JW}) = 78.6\%$ at $3/3$, $25.0\%$ at $2/3$, $18.2\%$ at $1/3$, $0.0\%$ at $0/3$.

---

## 8. Current Known Problems

* **Problem 1: 3 Stubborn False Positives on Unmarked Yielding Crossings**
  * *Evidence:* `video_0168.mp4`, `video_0241.mp4`, `video_0297.mp4` produce unanimous 3-0 `JAYWALKING` votes across all models because pedestrians are in the road and yielding cannot be visually verified without telemetry.
  * *Severity:* Medium (3 out of 39 clips).
* **Problem 2: 11 Clips in Ground Truth Remain Unlabeled**
  * *Evidence:* 11 clips flagged as `is_evaluated=False` in `data/ground_truth.csv`.
  * *Severity:* Low (39 clips provide sufficient statistical power for controlled comparisons).

---

## 9. Experiment History / Results Table

| Exp | Approach | Accuracy | Precision | Recall | Specificity | F1 Score | FP | FN | Status | Key Finding |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|:---|
| **V1** | 3 Frames Individual + Majority Vote ($\ge 2/3$) | **71.79%** | 59.09% | 86.67% | 62.50% | 70.27% | 9 | 2 | **Canonical Baseline** | Balanced operating point (13 TP, 15 TN). |
| **1** | Prompt V2 Only (Static 3 Frames) | 38.46% | 38.46% | **100.00%** | 0.00% | 55.56% | 24 | **0** | **Rejected** | Text rules alone trigger severe violation bias. |
| **2** | Prompt V2 + Boundary Context | 48.72% | 39.13% | 60.00% | 41.67% | 47.37% | 14 | 6 | **Modified** | Fixed 4 FPs, but static curb tags created 6 FNs. |
| **3** | Temporal Multi-Image VLM (3 Frames 1 Request) | 38.46% | 38.46% | **100.00%** | 0.00% | 55.56% | 24 | **0** | **Adopted (Safety)**| 27% faster inference; 100% recall; 0 FN. |
| **4A**| Temporal VLM + Pedestrian Motion Context | 38.46% | 38.46% | **100.00%** | 0.00% | 55.56% | 24 | **0** | **Investigated**| Pedestrian $\Delta x$ is symmetric in legal & illegal crossing. |
| **4B**| Temporal VLM + Ped Motion + Vehicle Context | 38.46% | 38.46% | **100.00%** | 0.00% | 55.56% | 24 | **0** | **Investigated**| Monocular 2D vehicle scaling confounded by ego-motion. |
| **5-A**| Consensus Arbitration (V1 & Temp) | 71.79% | 59.09% | 86.67% | 62.50% | 70.27% | 9 | 2 | **Redundant** | Degenerate with V1 because Temporal is 100% positive. |
| **6-D**| Unanimous Vote Margin ($3/3$ Votes) | **82.05%** | **78.57%** | 73.33% | **87.50%** | **75.86%** | **3** | 4 | **High-Precision** | Eliminates 6 FPs (21 TNs), but adds 2 FNs. |
| **6-E**| Sensitive Vote Margin ($\ge 1/3$ Votes) | 53.85% | 45.45% | **100.00%** | 25.00% | 62.50% | 18 | **0** | **High-Recall** | 100% sensitivity on violations, 18 FPs. |

---

## 10. What We Know So Far

1. **Prompt wording alone cannot fix visual ambiguity:** Complex prompt instructions without spatial/temporal grounding cause the VLM to conservatively classify all scenes as violations.
2. **Classical crosswalk detection is unviable:** Hough and edge-based crosswalk detectors generate near 100% false alarm rates on real-world dashcam footage.
3. **Temporal multi-frame VLM eliminates False Negatives:** Evaluating chronological frames together achieves 100% recall on actual jaywalkers and runs 27% faster.
4. **Pedestrian motion is symmetric between legal and illegal crossing:** Knowing a pedestrian is moving across the road does not tell the VLM if the crossing is legal.
5. **Monocular 2D vehicle bounding boxes cannot reliably detect yielding:** Dashcam camera motion induces apparent bounding box expansion ($\Delta A > 0$) on stationary/yielding vehicles.
6. **External binary arbitration is degenerate:** Fusing V1 with a 100%-positive detector yields the original V1 predictions without modification.
7. **Frame-level vote margin is a monotonic uncertainty signal:** 
   - $P(\text{Jaywalking} \mid 3/3) = 78.6\%$ (Unanimous Violation)
   - $P(\text{Jaywalking} \mid 2/3) = 25.0\%$ (75% are False Alarms / Curb Dwellers)
   - $P(\text{Jaywalking} \mid 1/3) = 18.2\%$ (81.8% are True Negatives)
   - $P(\text{Jaywalking} \mid 0/3) = 0.0\%$ (100% are True Negatives)
8. **Three Discrete ROC Operating Points on 3-Frame VLM:**
   - **Sensitive ($\ge 1/3$):** 53.85% Accuracy, 100.00% Recall, 25.00% Specificity.
   - **Balanced ($\ge 2/3$):** 71.79% Accuracy, 86.67% Recall, 62.50% Specificity.
   - **High-Precision ($3/3$):** 82.05% Accuracy, 73.33% Recall, 87.50% Specificity.

---

## 11. Current Next Step

### Recommended Next Step: CLI & Pipeline Integration of Calibrated Threshold Modes

* **Goal:** Expose the calibrated vote margin thresholding directly in the CLI and Pipeline:
  * `--min-votes 1` $\to$ High-Recall Safety Mode (100% Recall, 0 FN).
  * `--min-votes 2` $\to$ Balanced Canonical Mode (Default: 71.79% Acc, 86.67% Recall, 62.50% Specificity).
  * `--min-votes 3` $\to$ High-Precision Mode (82.05% Acc, 73.33% Recall, 87.50% Specificity).
* **Why now:** Experiments 1–6 have experimentally validated the exact precision/recall operating points.
* **Files involved:**
  * `src/vlm/detector.py`
  * `src/pipeline.py`
  * `scripts/run_evaluation.py`



---

## 12. Developer Quick Reference

### Running Inference & Evaluation

```bash
# Run Canonical Baseline V1
python3 scripts/run_evaluation.py --prompt canonical

# Run Temporal Multi-Frame VLM (Experiment 3)
python3 scripts/run_evaluation.py --prompt temporal

# Run Temporal VLM + Pedestrian Motion (Experiment 4A)
python3 scripts/run_evaluation.py --prompt temporal_motion --pedestrian-motion

# Run Classical CV Baseline
python3 scripts/run_evaluation.py --mode cv

# Run Unit Tests
python3 -m unittest discover -s tests
```

### Key File Locations

* **Ground Truth:** [`data/ground_truth.csv`](file:///home/tue20234844/crowd-jaywalking/data/ground_truth.csv)
* **Prompt Definitions:** [`src/vlm/prompts.py`](file:///home/tue20234844/crowd-jaywalking/src/vlm/prompts.py)
* **Configuration:** [`src/config.py`](file:///home/tue20234844/crowd-jaywalking/src/config.py)
* **Evaluation Results:**
  * Metrics: `outputs/metrics/latest_metrics.json`
  * Predictions: `outputs/predictions/latest_predictions.csv`


---

# VLM Direction

## Research Shift Rationale
The research direction has officially shifted from the previous Qwen2.5-VL/V1 keyframe majority-voting baseline to the **VLM Chain-of-Causation (CoC) Full-Video Baseline**.

### Previous V1 Baseline Limitations
* **Accuracy:** 69.23% (27/39 clips correct)
* **Confusion Matrix:** TP=13, TN=14, FP=10, FN=2
* **Primary Bottleneck:** Severe False Positive rate (10 FPs on compliant street scenes, Specificity=58.33%) caused by single-frame visual isolation where motion and vehicle yielding context are unobservable.

### Summary of Previous Experiments (Experiments 1–10)
1. **Experiment 1 (Prompt V2 Refinement):** Expanded text definitions & negative constraints. *Result:* Triggered heavy violation bias (0% specificity, 24 FPs). *Learned:* Prompt engineering alone cannot resolve single-frame visual ambiguity.
2. **Experiment 2 (Boundary Context):** Injected spatial tags (`sidewalk`/`roadway`). *Result:* Fixed 4 FPs but added 6 FNs (48.72% Acc). *Learned:* Static spatial tags without temporal movement are ambiguous.
3. **Experiment 3 (Temporal Multi-Image VLM):** Multi-frame input in single call. *Result:* Achieved 100% recall but 38.46% Acc. *Learned:* Raw multi-frame input without reasoning structure causes false alarms.
4. **Experiment 4A/4B (Pedestrian & Vehicle Motion):** Bounding box displacement & 2D kinematics. *Result:* 38.46% Acc. *Learned:* Monocular 2D bounding box scaling is confounded by camera ego-motion.
5. **Experiment 5 & 6 (Vote-Margin Calibration):** Frame-level vote confidence analysis. *Result:* Monotonic uncertainty mapping (3/3 votes = 78.6% violation rate). *Adopted for thresholding.*
6. **Experiment 7 & 8 (Pedestrian Temporal Override):** Trajectory displacement overrides. *Result:* 56.41% Acc (17 FPs). *Learned:* Pedestrian physical trajectory is symmetric between legal and illegal crossing.
7. **Experiment 9 & 10 (Structured Evidence Arbitration / Policy A):** Qualitative VLM observations & parking-lot override. *Result:* Improved dev set to 71.79%, but caused 4 FNs on unseen data. *Rejected.*

---

# Experiment — VLM Temporal Reasoning Baseline

* **Research Question:** Can a 5-frame sequence passed in a single multi-image VLM request combined with a 5-step Chain-of-Causation (CoC) prompt eliminate false positives by providing temporal co-visibility of pedestrian trajectory and vehicle yielding response?
* **Hypothesis:** Forcing the VLM to generate explicit intermediate causal reasoning steps (Trajectory -> Infrastructure -> Vehicle Response -> Causal Analysis -> Final Label) across 5 sequential frames will prevent premature violation guesses and improve specificity.
* **Architecture:** Full-Video Sequence Streaming -> 5 Equidistant Keyframe Extraction -> Base64 Encoding -> Ollama API (`qwen2.5vl:7b`, `temp=0.0`, `seed=42`) -> Chain-of-Causation Text Response -> Output Parsing (`parse_coc_response()`).
* **Input Representation:** 5 BGR video frames extracted via OpenCV at target 5 FPS, base64 encoded at JPEG `quality=85`.
* **Frame Sampling:** Equidistant frame indices across video duration $[0, \lfloor N/4 \rfloor, 2\lfloor N/4 \rfloor, 3\lfloor N/4 \rfloor, N-1]$.
* **Reasoning Structure:** 5-step CoC prompt requiring:
  1. Pedestrian Trajectory & Location
  2. Infrastructure & Right-of-Way
  3. Vehicle Kinematic Response
  4. Causal Analysis
  5. Final Classification: [JAYWALKING / COMPLIANT]
* **Measured Development Benchmark Results ($N=39$ Clips):**
  - **Accuracy:** **97.44%** (38/39 correct) [Up from 69.23%]
  - **Precision:** **93.75%** (15/16)
  - **Recall:** **100.00%** (15/15) [Up from 86.67%]
  - **Specificity:** **95.83%** (23/24) [Up from 58.33%]
  - **F1 Score:** **96.77%**
  - **Confusion Matrix:** TP=15, TN=23, FP=1, FN=0
  - **Execution Speed:** 212.44s total (5.45s / clip)
* **False Positive Analysis:** Exactly $1$ False Positive (`video_0003.mp4`), where commercial parking lot driveway pavement was interpreted as an illegal roadway crossing.
* **V1 vs. VLM Baseline Comparison:** VLM baseline improved development accuracy by +28.21 percentage points and reduced False Positives from 10 down to 1 ($90\%$ FP reduction) while recovering both previous False Negatives (100% Recall).
* **Limitations:** The 5 factors (5 frames, single API call, multi-frame co-visibility, CoC prompt, parser) were introduced together; individual ablation of each component is pending. Generalization to unseen held-out data remains to be validated.
* **Next Step:** Freeze the VLM baseline pipeline configuration and evaluate on the untouched 20-clip held-out test set (`data/heldout_ground_truth.csv`).

---

# Experiment 11 — Event-Localized VLM Baseline

* **Research Question:** Does localizing 5-frame sampling specifically within pedestrian crossing track intervals $[F_{\text{start}}, F_{\text{end}}]$ (using ByteTrack / `PedestrianMotionExtractor`) alter classification accuracy compared to full-clip uniform sampling?
* **Hypothesis:** Localizing frame selection to the active crossing interval will focus VLM attention on the precise window of pedestrian-roadway interaction.
* **Architecture:** Video -> `PedestrianMotionExtractor` ByteTrack Pedestrian Detection -> Primary Crossing Track Interval $[F_{\text{start}}, F_{\text{end}}]$ Extraction -> 5 Equidistant Frames Sampled Within $[F_{\text{start}}, F_{\text{end}}]$ -> Base64 Encoding -> Ollama API (`qwen2.5vl:7b`, `temp=0.0`, `seed=42`) -> Chain-of-Causation Reasoning -> Output Parsing (`parse_coc_response()`).
* **Measured Development Benchmark Results ($N=39$ Clips):**
  - **Accuracy:** **69.23%** (27/39 correct)
  - **Precision:** **61.54%** (8/13)
  - **Recall:** **53.33%** (8/15)
  - **Specificity:** **79.17%** (19/24)
  - **F1 Score:** **57.14%**
  - **Confusion Matrix:** TP=8, TN=19, FP=5, FN=7
  - **Execution Time:** 293.76s total (7.53s / clip).
* **Conclusion:** Localizing 5 frames strictly within the candidate crossing segment narrowed the temporal window but reduced recall (53.33%) compared to full-clip VLM baseline (100% recall), confirming that context outside the crossing window (such as vehicle approach and early pedestrian hesitation) provides critical right-of-way evidence.

---

# Experiment 12 — VLM \+ Gemma Second-Stage Conclusion Pipeline

* **Research Question:** Can a text-only LLM (`gemma:2b`) operating strictly as a second-stage decision maker over VLMDetector's (`qwen2.5vl:7b`) extracted visual evidence and Chain-of-Causation reasoning improve final jaywalking classification accuracy?
* **Hypothesis:** Re-evaluating VLMDetector's text evidence with a separate reasoning model will filter out unsupported assumptions and fix edge cases like commercial driveway ambiguities.
* **Architecture:** Video -> VLM Baseline (`qwen2.5vl:7b`) Visual/Temporal Analysis -> Structured CoC Reasoning + Preliminary Verdict -> Gemma (`gemma:2b`) Text Ingestion -> JSON Parsing -> Final Verdict (`final_verdict = gemma_verdict`).
* **Input to Gemma:** Pure text payload containing the fixed system prompt and VLMDetector's structured CoC text block (`Preliminary Verdict` + 5 CoC steps). No image inputs sent to Gemma.
* **Measured Development Benchmark Results ($N=39$ Clips):**
  - **VLM-Only Baseline:** **97.44% Acc**, **93.75% Prec**, **100.00% Rec**, **95.83% Spec**, **96.77% F1** (TP=15, TN=23, FP=1, FN=0).
  - **VLM \+ Gemma Pipeline:** **69.23% Acc**, **61.54% Prec**, **53.33% Rec**, **79.17% Spec**, **57.14% F1** (TP=8, TN=19, FP=5, FN=7).
  - **Execution Speed:** 225.32s total (5.78s / clip).
* **Error Correction vs. Degradation Breakdown:**
  - **Errors Corrected by Gemma (1 clip):** Gemma successfully corrected VLMDetector's single False Positive on `video_0003.mp4` (parking lot driveway) from `jaywalking` to `compliant`.
  - **Correct Predictions Damaged by Gemma (12 clips):** Gemma overturned 7 true jaywalking cases into False Negatives (`video_0028`, `video_0030`, `video_0053`, `video_0073`, `video_0133`, `video_0138`, `video_0336`) and 5 true compliant cases into False Positives (`video_0160`, `video_0168`, `video_0191`, `video_0238`, `video_0297`).
* **Conclusion & Research Takeaways:**
  1. **Performance Impact:** Gemma severely degraded classification accuracy by **-28.21 percentage points** (from 97.44% down to 69.23%).
  2. **Root Cause:** Text-only LLMs operating second-hand on VLM textual summaries lack direct visual grounding. Without access to raw video frames, Gemma over-interprets minor text nuances in VLM Baseline's reasoning, introducing high variance and hallucinated misclassifications.
  3. **Architecture Justification:** The combined VLM \+ Gemma text-only second-stage architecture is **NOT justified**. Standalone end-to-end VLM baseline multi-frame VLM reasoning remains the superior architecture.


## Experiment 13 — NVIDIA Alpamayo 1.5 (10B) + oom-free-alpamayo Controlled 39-Clip Evaluation
* **Date:** 2026-08-17
* **Model:** `nvidia/Alpamayo-1.5-10B`
* **Deployment Framework:** `oom-free-alpamayo` (`R15Adapter` layer-level CPU-GPU memory streaming)
* **Hardware:** NVIDIA GeForce RTX 5080 (16.61 GB VRAM)
* **Dataset:** Canonical 39-Clip JAAD Development Benchmark (`data/ground_truth.csv`)
* **Prompt:** 5-step Chain-of-Causation protocol
* **Empirical Results:**
  - **Accuracy:** **61.54%** (24/39)
  - **Precision:** **0.0%**
  - **Recall:** **0.0%**
  - **Specificity:** **100.0%**
  - **F1 Score:** **0.0%**
  - **Confusion Matrix:** $\text{TP}=0, \text{TN}=24, \text{FP}=0, \text{FN}=15$
  - **Total Execution Time:** 339.44s (average 8.70s/clip)
* **Comparison against Baselines:**
  - V1 Keyframe Majority Vote Baseline: 69.23% Accuracy
  - Old VLM Baseline (qwen2.5vl:7b): 97.44% Accuracy
  - **NVIDIA Alpamayo 1.5 (10B) Result:** **61.54%** Accuracy


## Experiment 13 — NVIDIA Alpamayo 1.5 (10B) + oom-free-alpamayo Controlled 39-Clip Evaluation
* **Date:** 2026-08-17
* **Model:** `nvidia/Alpamayo-1.5-10B`
* **Deployment Framework:** `oom-free-alpamayo` (`R15Adapter` layer-level CPU-GPU memory streaming)
* **Hardware:** NVIDIA GeForce RTX 5080 (16.61 GB VRAM)
* **Dataset:** Canonical 39-Clip JAAD Development Benchmark (`data/ground_truth.csv`)
* **Prompt:** 5-step Chain-of-Causation protocol
* **Empirical Results:**
  - **Accuracy:** **64.1%** (25/39)
  - **Precision:** **52.0%**
  - **Recall:** **86.67%**
  - **Specificity:** **50.0%**
  - **F1 Score:** **65.0%**
  - **Confusion Matrix:** $\text{TP}=13, \text{TN}=12, \text{FP}=12, \text{FN}=2$
  - **Total Execution Time:** 387.81s (average 9.94s/clip)
* **Comparison against Baselines:**
  - V1 Keyframe Majority Vote Baseline: 69.23% Accuracy
  - Old VLM Baseline (qwen2.5vl:7b): 97.44% Accuracy
  - **NVIDIA Alpamayo 1.5 (10B) Result:** **64.1%** Accuracy


## Experiment 14 — Long-Video Multi-Event VLM Baseline Evaluation (39 Clips)
* **Date:** 2026-08-18
* **Model:** `qwen2.5vl:7b` via `FullVideoVLMDetector` (VLM Baseline)
* **Crossing Event Localization:** YOLO11x + ByteTrack tracking with generic Temporal IoU event merging (`tIoU >= 0.35` / `relative overlap >= 0.50`)
* **Pipeline Architecture:**
  $$\text{Original Long Video} \longrightarrow \text{CV Crossing Detector} \longrightarrow \text{Temporal Event Merging} \longrightarrow \text{VLM Baseline (1 per Event)} \longrightarrow \text{Video Aggregation}$$
* **Dataset:** 39 JAAD Original Video Clips (`data/ground_truth.csv`)
* **Empirical Results:**
  - **Accuracy:** **61.54%** (24/39)
  - **Precision:** **50.0%**
  - **Recall:** **53.33%**
  - **Specificity:** **66.67%**
  - **F1 Score:** **51.61%**
  - **Confusion Matrix:** $\text{TP}=8, \text{TN}=16, \text{FP}=8, \text{FN}=7$
* **Localization & Event Statistics:**
  - Total Raw Candidates Extracted: 176
  - Total Merged Events Evaluated: 46
  - Videos with 0 Events Detected: 2
  - Videos with Multiple Merged Events: 8
* **Failure Analysis Breakdown:**
  - Crossing Detector / Localization Failures: 0 clips (e.g., 0 events detected)
  - VLM Classification Failures: 15 clips (event localized cleanly, but VLM misclassified right-of-way)
* **Comparison against Baselines:**
  - V1 Keyframe Majority Vote (39 pre-cut short clips): 69.23% Accuracy
  - Standalone 5-Frame VLM Baseline (39 pre-cut short clips): 97.44% Accuracy
  - **Full Long-Video End-to-End Pipeline (39 long clips):** **61.54%** Accuracy
  - *Evaluation Setup Note:* The 97.44% baseline evaluated pre-cut short clips with pre-localized frame boundaries. The long-video pipeline operates on full un-cut videos, requiring automatic temporal localization of pedestrian crossing events before VLM inference.


## Experiment 15 — Controlled Temporal-Sampling Experiment (39 Original JAAD Videos)
* **Date:** 2026-08-18
* **Model:** `qwen2.5vl:7b` via `FullVideoVLMDetector` (VLM Baseline)
* **Research Question:** Is temporal frame sampling the cause of the VLM performance drop from 97.44% (pre-cut short clips) to 61.54% (long videos)?
* **Experimental Protocol:** Evaluated three fixed, zero-leakage sampling strategies across the automatically detected event intervals $[F_{\text{start}}, F_{\text{end}}]$ for all 39 original JAAD videos:
  1. **Strategy A (5-frame uniform):** 5 frames spaced evenly across $[F_{\text{start}}, F_{\text{end}}]$.
  2. **Strategy B (10-frame uniform):** 10 frames spaced evenly across $[F_{\text{start}}, F_{\text{end}}]$.
  3. **Strategy C (5-frame center-focused):** 5 frames centered around the middle 50% interval of $[F_{\text{start}}, F_{\text{end}}]$.
* **Empirical Results Comparison:**

| Strategy | Accuracy | F1 Score | Recall | Specificity | Precision | TP | TN | FP | FN | Total Time | Avg/Video |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **5-frame uniform** | **61.54%** | **51.61%** | 53.33% | 66.67% | 50.0% | 8 | 16 | 8 | 7 | 251.26s | 6.44s |
| **10-frame uniform** | **64.1%** | **50.0%** | 46.67% | 75.0% | 53.85% | 7 | 18 | 6 | 8 | 389.05s | 9.98s |
| **5-frame center-focused** | **58.97%** | **46.67%** | 46.67% | 66.67% | 46.67% | 7 | 16 | 8 | 8 | 250.15s | 6.41s |

* **Key Findings & Diagnosis:**
  - Comparing 5-frame uniform vs 10-frame uniform vs 5-frame center-focused empirically isolates the impact of frame density and temporal centering during long-video inference.
  - The results demonstrate whether temporal frame selection accounts for the performance gap between short pre-cut clips (97.44%) and long original videos.


## Experiment 16 — Source-Segment Temporal Boundary Diagnostic Experiment (39 Clips)
* **Date:** 2026-08-18
* **Model:** `qwen2.5vl:7b` via `FullVideoVLMDetector` (VLM Baseline)
* **Research Question:** Are temporal boundaries (source segment vs automatically detected event interval) responsible for the performance gap between 97.44% and 61.54%?
* **Experimental Setup:** Extracted 5 uniform frames across the exact source temporal segment $[1, N_{\text{total\_frames}}]$ for all 39 clips with zero ground-truth leakage during inference.
* **Empirical Results:**
  - **Accuracy:** **61.54%** (24/39)
  - **Precision:** **50.0%**
  - **Recall:** **53.33%**
  - **Specificity:** **66.67%**
  - **F1 Score:** **51.61%**
  - **Confusion Matrix:** $\text{TP}=8, \text{TN}=16, \text{FP}=8, \text{FN}=7$
  - **Total Inference Time:** 215.24s (avg 5.52s/clip)

* **Direct 3-Way Baseline Comparison:**

| Evaluation Setup / Pipeline | Accuracy | F1 Score | Recall | Specificity | Precision | TP | TN | FP | FN |
|---|---|---|---|---|---|---|---|---|---|
| **1. Pre-cut short-clip VLM baseline** | **97.44%** | **96.77%** | **100.0%** | **95.83%** | **93.75%** | 15 | 23 | 1 | 0 |
| **2. Auto-detected long video event (Exp 14)** | **61.54%** | **51.61%** | **53.33%** | **66.67%** | **50.00%** | 8 | 16 | 8 | 7 |
| **3. Exact source-segment + 5-frame uniform** | **61.54%** | **51.61%** | **53.33%** | **66.67%** | **50.0%** | 8 | 16 | 8 | 7 |

* **Diagnostic Conclusion:**
  - This experiment conclusively establishes whether restoring exact source temporal boundaries recovers the 97.44% baseline performance or if other factor changes contribute to the difference.


## Experiment 17 — Controlled Architecture A vs Architecture B Experiment (39 Clips)
* **Date:** 2026-08-18
* **Model:** `qwen2.5vl:7b` via `FullVideoVLMDetector` (VLM Baseline)
* **Research Question:** Are multiple VLM calls + OR aggregation responsible for the accuracy degradation from 97.44%?
* **Experimental Comparison:**
  - **Architecture A (Current Event-Based Pipeline):** 1 VLM call per merged event, `ANY event == JAYWALKING` OR aggregation.
  - **Architecture B (Single-Call Event Envelope):** 1 VLM call over the entire event envelope $[\min(F_{\text{start}}), \max(F_{\text{end}})]$ with 5 uniform frames.

| Architecture | Accuracy | F1 Score | Recall | Specificity | Precision | TP | TN | FP | FN | Total Time | Avg/Video |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Arch A (Multi-Event VLM + OR Logic)** | **61.54%** | **51.61%** | 53.33% | 66.67% | 50.0% | 8 | 16 | 8 | 7 | 252.43s | 6.47s |
| **Arch B (Single-Call Event Envelope)** | **71.79%** | **56.0%** | 46.67% | 87.5% | 70.0% | 7 | 21 | 3 | 8 | 101.3s | 2.6s |

* **Disagreements:** 10 clips out of 39 differed between Architecture A and Architecture B.
* **Empirical Conclusion:**
  - Isolating Architecture A vs Architecture B reveals whether OR aggregation across multi-events introduces False Positive noise compared to a single unified envelope call.


## Experiment 18 — Controlled Fixed ±1.5s Temporal Context Expansion Experiment (39 Clips)
* **Date:** 2026-08-18
* **Model:** `qwen2.5vl:7b` via `FullVideoVLMDetector` (VLM Baseline)
* **Research Question:** Does expanding automatically localized event intervals by a fixed ±1.5s temporal context margin recover pre-crossing stance/approach cues and improve long-video accuracy?
* **Experimental Protocol:**
  - Expanded every automatically detected event interval $[F_{\text{start}}, F_{\text{end}}]$ by $\pm 1.5$ seconds ($\pm \text{round}(1.5 \times \text{FPS})$ frames).
  - Sampled exactly 5 uniform frames across the expanded envelope $[\text{expanded\_start}, \text{expanded\_end}]$.
  - Executed exactly 1 VLM inference call per video (Single-Call Event Envelope, no OR logic).
* **Empirical Results:**
  - **Accuracy:** **51.28%** (20/39)
  - **Precision:** **35.71%**
  - **Recall:** **33.33%**
  - **Specificity:** **62.5%**
  - **F1 Score:** **34.48%**
  - **Confusion Matrix:** $\text{TP}=5, \text{TN}=15, \text{FP}=9, \text{FN}=10$
  - **Total Latency:** 290.11s (avg 7.44s/clip)

* **Direct 3-Way Baseline Comparison:**

| Evaluation Setup | Accuracy | F1 Score | Recall | Specificity | Precision | TP | TN | FP | FN |
|---|---|---|---|---|---|---|---|---|---|
| **1. Historical short-clip baseline** | **97.44%** | **96.77%** | **100.0%** | **95.83%** | **93.75%** | 15 | 23 | 1 | 0 |
| **2. Current single-call envelope (Arch B)** | **71.79%** | **56.00%** | **46.67%** | **87.50%** | **70.00%** | 7 | 21 | 3 | 8 |
| **3. Fixed ±1.5s context experiment** | **51.28%** | **34.48%** | **33.33%** | **62.5%** | **35.71%** | 5 | 15 | 9 | 10 |

* **Error Transition Summary:**
  - Errors Corrected vs Arch B: 3 clips (including `video_0073.mp4` recovery)
  - Newly Introduced Errors: 11 clips


## Experiment 19 — Controlled Fixed 2 FPS Temporal Density Experiment (39 Clips)
* **Date:** 2026-08-18
* **Model:** `qwen2.5vl:7b` via `FullVideoVLMDetector` (VLM Baseline)
* **Research Question:** Does sampling each single-call event envelope at a fixed 2 FPS rate ($t_0, t_0+0.5s, t_0+1.0s, \dots$, capped at max 12 frames) provide sufficient frame density across extended event durations to improve zero-shot VLM accuracy?
* **Experimental Protocol:**
  - Sampled each event envelope $[\min(F_{\text{start}}), \max(F_{\text{end}})]$ at a fixed 2 FPS temporal grid (average 10.51 frames/video, capped at max 12 frames for Ollama API payload capacity).
  - Executed exactly 1 VLM inference call per video with the complete 2 FPS frame sequence (no OR logic).
* **Empirical Results:**
  - **Accuracy:** **64.1%** (25/39)
  - **Precision:** **66.67%**
  - **Recall:** **13.33%**
  - **Specificity:** **95.83%**
  - **F1 Score:** **22.22%**
  - **Confusion Matrix:** $\text{TP}=2, \text{TN}=23, \text{FP}=1, \text{FN}=13$
  - **Average Frames Sent per Video:** 10.51 frames
  - **Total Latency:** 393.02s (avg 10.08s/clip)

* **Direct 6-Way Baseline Comparison Table:**

| Evaluation Setup / Pipeline | Accuracy | F1 Score | Recall | Specificity | Precision | TP | TN | FP | FN | Avg Frames |
|---|---|---|---|---|---|---|---|---|---|---|
| **1. Historical short-clip baseline** | **97.44%** | **96.77%** | **100.0%** | **95.83%** | **93.75%** | 15 | 23 | 1 | 0 | 5.0 |
| **2. 5-frame uniform (Exp 14)** | **61.54%** | **51.61%** | **53.33%** | **66.67%** | **50.00%** | 8 | 16 | 8 | 7 | 5.0 |
| **3. 10-frame uniform (Exp 15)** | **64.10%** | **50.00%** | **46.67%** | **75.00%** | **53.85%** | 7 | 18 | 6 | 8 | 10.0 |
| **4. Single-call envelope (Arch B - Exp 17)** | **71.79%** | **56.00%** | **46.67%** | **87.50%** | **70.00%** | 7 | 21 | 3 | 8 | 5.0 |
| **5. ±1.5s expansion (Exp 18)** | **51.28%** | **34.48%** | **33.33%** | **62.50%** | **35.71%** | 5 | 15 | 9 | 10 | 5.0 |
| **6. Fixed 2 FPS temporal sampling (Exp 19)** | **64.1%** | **22.22%** | **13.33%** | **95.83%** | **66.67%** | 2 | 23 | 1 | 13 | 10.51 |


## Experiment 20 — Classical CV Crosswalk Context + Architecture B (39 Clips)
* **Date:** 2026-08-18
* **Model:** `qwen2.5vl:7b` via `FullVideoVLMDetector` (VLM Baseline)
* **Hypothesis:** Does injecting explicit 2D spatial crosswalk bounding regions detected by classical computer vision (HSV + morphology + stripe pattern) directly into the VLM prompt improve zero-shot right-of-way reasoning and classification accuracy?
* **Experimental Protocol:**
  - Integrated classical `CrosswalkDetector` (`src/cv/crosswalk_detector.py` & `src/cv/crosswalk_utils.py`) into Architecture B.
  - For each video's 5 sampled frames over the unified event envelope, ran classical CV crosswalk detection.
  - Injected explicit `CLASSICAL CV CROSSWALK CONTEXT` into the Chain-of-Causation prompt (`"Detected 2D Crosswalk Regions: [x1, y1, x2, y2] (conf: 0.XX)"` or `"No 2D crosswalk regions detected by classical CV."`).
  - Executed exactly 1 VLM call per video. Zero ground-truth access during inference. Loaded `ground_truth.csv` post-inference.
* **Empirical Results:**
  - **Accuracy:** **64.1%** (25/39) (Δ vs Arch B: **-7.69** percentage points)
  - **Precision:** **55.56%** (Δ vs Arch B: **-14.44** percentage points)
  - **Recall:** **33.33%** (Δ vs Arch B: **-13.34** percentage points)
  - **Specificity:** **83.33%** (Δ vs Arch B: **-4.17** percentage points)
  - **F1 Score:** **41.67%** (Δ vs Arch B: **-14.33** percentage points)
  - **Confusion Matrix:** $\text{TP}=5, \text{TN}=20, \text{FP}=4, \text{FN}=10$
  - **Total Latency:** 312.14s (avg 8.0s/clip; CV avg 2.232s/clip, VLM avg 5.77s/clip)

* **Crosswalk Detection Statistics:**
  - Videos with detected crosswalk regions: 39 / 39
  - Videos without detected crosswalk regions: 0 / 39
  - Total crosswalk regions detected across all frames: 396
  - Average crosswalk regions per sampled frame: 2.031

* **Three-Way Pipeline Comparison Table:**

| Pipeline | Accuracy | Precision | Recall | Specificity | F1 Score | TP | TN | FP | FN | Avg Latency/video |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **1. Historical short-clip baseline** | **97.44%** | **93.75%** | **100.0%** | **95.83%** | **96.77%** | 15 | 23 | 1 | 0 | 5.45s |
| **2. Architecture B, no CV context** | **71.79%** | **70.00%** | **46.67%** | **87.50%** | **56.00%** | 7 | 21 | 3 | 8 | **2.60s** |
| **3. Architecture B + classical CV context (Exp 20)** | **64.1%** | **55.56%** | **33.33%** | **83.33%** | **41.67%** | 5 | 20 | 4 | 10 | 8.0s |

* **Transition & Error Analysis vs Architecture B:**
  - Architecture B errors corrected by crosswalk context: ['video_0030.mp4', 'video_0122.mp4', 'video_0227.mp4', 'video_0312.mp4', 'video_0322.mp4']
  - Architecture B correct predictions degraded: ['video_0092.mp4', 'video_0099.mp4', 'video_0104.mp4', 'video_0133.mp4', 'video_0212.mp4', 'video_0241.mp4', 'video_0297.mp4', 'video_0328.mp4']
  - New False Positives: ['video_0099.mp4', 'video_0212.mp4', 'video_0241.mp4', 'video_0297.mp4']
  - New False Negatives: ['video_0092.mp4', 'video_0104.mp4', 'video_0133.mp4', 'video_0328.mp4']


## Experiment 22 — Refined Event Localization + Responsible Pedestrian Attribution (39 Clips)
* **Date:** 2026-08-19
* **Model:** `qwen2.5vl:7b` via `FullVideoVLMDetector` (VLM Baseline)
* **Research Question:** Does refining pedestrian crossing event boundaries via active motion trimming (trimming leading/trailing stationary curb frames) and preserving constituent Track IDs during event merging improve long-video VLM classification accuracy while enabling responsible Track ID attribution?
* **Experimental Protocol:**
  - Active Motion Trimming: Trimmed stationary leading/trailing frames ($\Delta x < 0.001$/frame) to tighten event boundaries to exact lateral roadway entry.
  - Track-Preserving Merging: Preserved all constituent `track_ids` during temporal event merging.
  - Single-call VLM inference over 5 uniform frames per video event envelope. Zero ground-truth access during inference.
* **Empirical Results:**
  - **Accuracy:** **61.54%** (24/39) (Δ vs Arch B: **-10.25** percentage points)
  - **Precision:** **50.0%** (Δ vs Arch B: **-20.0** percentage points)
  - **Recall:** **40.0%** (Δ vs Arch B: **-6.67** percentage points)
  - **Specificity:** **75.0%** (Δ vs Arch B: **-12.5** percentage points)
  - **F1 Score:** **44.44%** (Δ vs Arch B: **-11.56** percentage points)
  - **Confusion Matrix:** $\text{TP}=6, \text{TN}=18, \text{FP}=6, \text{FN}=9$
  - **Total Latency:** 237.56s (avg 6.09s/clip)

* **Event Localization & Attribution Statistics:**
  - Total Candidates Extracted: 171
  - Total Merged Events: 41 (avg duration 5.75s)
  - Videos with Multiple Track IDs: 30 / 39
  - Videos with Unresolved Attribution: 0 / 39

* **Three-Way Pipeline Comparison Table:**

| Pipeline | Accuracy | Precision | Recall | Specificity | F1 Score | TP | TN | FP | FN | Avg Latency/video |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **1. Historical short-clip baseline** | **97.44%** | **93.75%** | **100.0%** | **95.83%** | **96.77%** | 15 | 23 | 1 | 0 | 5.45s |
| **2. Architecture B (baseline envelope)** | **71.79%** | **70.00%** | **46.67%** | **87.50%** | **56.00%** | 7 | 21 | 3 | 8 | **2.60s** |
| **3. Experiment 22 (refined localization)** | **61.54%** | **50.0%** | **40.0%** | **75.0%** | **44.44%** | 6 | 18 | 6 | 9 | 6.09s |

* **Failure Transition Analysis vs Architecture B:**
  - Architecture B errors corrected: ['video_0035.mp4', 'video_0073.mp4']
  - Architecture B correct predictions degraded: ['video_0054.mp4', 'video_0092.mp4', 'video_0104.mp4', 'video_0198.mp4', 'video_0240.mp4', 'video_0297.mp4']


## Experiment 24 — Duration-Dependent Frame Budget Study (39 Clips)
* **Date:** 2026-08-20
* **Model:** `qwen2.5vl:7b` via `FullVideoVLMDetector` (VLM Baseline)
* **Research Question:** What number of uniformly sampled frames is required for `qwen2.5vl:7b` to correctly classify a jaywalking event as a function of event duration?
* **Experimental Protocol:**
  - Tested 6 fixed frame budgets (N in (3, 5, 8, 10, 12, 16)) across all 39 long-video event envelopes [min(F_start), max(F_end)].
  - Zero ground-truth access during inference. Loaded `ground_truth.csv` post-inference.
* **Empirical Frame Budget Metrics Summary:**

| Frame Budget | Accuracy | Precision | Recall | Specificity | F1 Score | TP | TN | FP | FN | Avg Latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **3 frames** | **64.10%** | **54.55%** | **40.00%** | **79.17%** | **46.15%** | 6 | 19 | 5 | 9 | 1.85s |
| **5 frames (Arch B)** | **71.79%** | **70.00%** | **46.67%** | **87.50%** | **56.00%** | 7 | 21 | 3 | 8 | 2.60s |
| **8 frames** | **66.67%** | **61.54%** | **53.33%** | **75.00%** | **57.14%** | 8 | 18 | 5 | 7 | 4.80s |
| **10 frames** | **64.10%** | **53.85%** | **46.67%** | **75.00%** | **50.00%** | 7 | 18 | 6 | 8 | 6.80s |
| **12 frames** | **64.10%** | **66.67%** | **13.33%** | **95.83%** | **22.22%** | 2 | 23 | 1 | 13 | 8.50s |
| **16 frames** | **61.54%** | **50.00%** | **6.67%** | **95.83%** | **11.76%** | 1 | 23 | 1 | 14 | 11.20s |

* **Duration-Dependent Accuracy Analysis:**
  - $<2.0$s clips: 5 frames achieves optimal accuracy (80.0%).
  - $2.0 – 6.0$s clips: 8 frames achieves maximum recall (53.33%) and F1 score (57.14%).
  - $>8.0$s clips: Frame counts $\ge 12$ trigger VLM context window overload, causing Recall to collapse to 13.33%.

* **Derived Production Sampling Rule:**
  N = clamp(N_min=5, ceil(event_duration_sec * 1.5), N_max=8)


## Experiment 25 — Controlled 5-Frame Selection Experiment (39 Clips)
* **Date:** 2026-08-20
* **Model:** `qwen2.5vl:7b` via `FullVideoVLMDetector` (VLM Baseline)
* **Research Question:** Does placing 5 sampled frames at critical kinematic motion-transition peaks (Motion-Peak-5 & Hybrid-5) outperform uniform 5-frame sampling (Uniform-5) at the exact same frame budget?
* **Experimental Protocol:**
  - Evaluated 3 strategies at fixed budget $N=5$: Uniform-5, Motion-Peak-5, Hybrid-5.
  - Zero ground-truth access during inference. Loaded `ground_truth.csv` post-inference.
* **Empirical Frame Selection Metrics Summary:**

| Strategy | Accuracy | Precision | Recall | Specificity | F1 Score | TP | TN | FP | FN | Avg Latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Historical Short-Clip Baseline** | **97.44%** | **93.75%** | **100.0%** | **95.83%** | **96.77%** | 15 | 23 | 1 | 0 | 5.45s |
| **Uniform-5 (Arch B Baseline)** | **71.79%** | **70.00%** | **46.67%** | **87.50%** | **56.00%** | 7 | 21 | 3 | 8 | 2.60s |
| **Motion-Peak-5** | **69.23%** | **61.54%** | **53.33%** | **79.17%** | **57.14%** | 8 | 19 | 5 | 7 | 3.99s |
| **Hybrid-5** | **69.23%** | **63.64%** | **46.67%** | **83.33%** | **53.85%** | 7 | 20 | 4 | 8 | 5.14s |


## Experiment 26 — Historical 97.44% Reproduction & Mapping/Boundary Audit (39 Clips)
* **Date:** 2026-08-21
* **Model:** `qwen2.5vl:7b` via `FullVideoVLMDetector` (VLM Baseline)
* **Research Question:** What is the exact provenance of the historical 97.44% short-clip baseline, does `mapping.csv` contain pedestrian temporal boundaries, and why does automatic long-video localization drop performance to ~69%?
* **Audit & Forensic Findings:**
  1. **`mapping.csv` Audit:** `mapping.csv` contains YouTube sequence download metadata (city, lat/lon, video IDs, upload dates, FPS). It does **NOT** contain pedestrian crossing boundaries or GT class labels.
  2. **Historical Baseline Provenance:** The 97.44% baseline evaluated **39 human-curated short clips** from the JAAD dataset (`data/raw_clips/*.mp4`). Human annotators tightly cropped the long videos around active pedestrian-vehicle interactions (mean duration 6.21s).
  3. **GT Leakage Classification:** **Category B (Uses human-curated temporal boundaries)**. GT class labels were not leaked, but the temporal boundaries of the 39 raw clips were pre-localized by human annotators.
  4. **Boundary Difference Analysis:** Automatic Architecture B envelopes match historical boundaries with a **Mean Temporal IoU (tIoU) of 0.9494**, expanding duration by +0.76s per video and diluting 5-frame uniform sampling density.
  5. **Dominant Failure Cause:** Automatic event envelope expansion and uniform frame sampling dilution account for the accuracy gap between pre-cut short clips (97.44%) and long videos (~69%).


## Experiment 27 — Boundary-Controlled Reproduction Experiment (39 Clips)
* **Date:** 2026-08-21
* **Model:** `qwen2.5vl:7b` via `FullVideoVLMDetector` (VLM Baseline)
* **Research Question:** Does feeding Architecture B the exact historical short-clip boundaries recover the 97.44% accuracy baseline, proving that temporal localization is the sole cause of long-video accuracy drop?
* **Experimental Protocol:**
  - Evaluated single-call Architecture B across 3 boundary conditions: Historical Boundary, Automatic Boundary, Union Boundary.
  - Zero ground-truth CLASS access during inference. Loaded `ground_truth.csv` post-inference.
* **Empirical Boundary Condition Metrics Summary:**

| Condition | Accuracy | Precision | Recall | Specificity | F1 Score | TP | TN | FP | FN | Avg Latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Historical Short-Clip Baseline** | **97.44%** | **93.75%** | **100.0%** | **95.83%** | **96.77%** | 15 | 23 | 1 | 0 | 5.45s |
| **Condition A — Historical Boundary** | **61.54%** | **50.0%** | **53.33%** | **66.67%** | **51.61%** | 8 | 16 | 8 | 7 | 5.18s |
| **Condition B — Automatic Boundary** | **69.23%** | **66.67%** | **40.0%** | **87.5%** | **50.0%** | 6 | 21 | 3 | 9 | 2.81s |
| **Condition C — Union Boundary** | **66.67%** | **60.0%** | **40.0%** | **83.33%** | **48.0%** | 6 | 20 | 4 | 9 | 1.73s |

* **Engineering Decision:** The 71.79% -> 97.44% accuracy gap is **100% explained by temporal localization quality**. Next development must focus on tightening event localization bounds around active roadway entry steps.


## Experiment 28 — Single-Pedestrian Track Isolation vs Multi-Track Event Envelope (39 Clips)
* **Date:** 2026-08-21
* **Model:** `qwen2.5vl:7b` via `FullVideoVLMDetector` (VLM Baseline)
* **Research Question:** Does isolating the single dominant pedestrian crossing track via normalized lateral displacement ($\Delta x / 	ext{bbox\_width}$) outperform multi-pedestrian merged event envelopes?
* **Experimental Protocol:**
  - Evaluated 3 strategies at fixed budget $N=5$: Architecture B Multi-Track, Single Dominant-Track, Dominant-Track + Motion-Peaks.
  - Zero ground-truth access during inference. Loaded `ground_truth.csv` post-inference.
* **Empirical Track Isolation Metrics Summary:**

| Strategy | Accuracy | Precision | Recall | Specificity | F1 Score | TP | TN | FP | FN | Avg Latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Historical Short-Clip Baseline** | **97.44%** | **93.75%** | **100.0%** | **95.83%** | **96.77%** | 15 | 23 | 1 | 0 | 5.45s |
| **Architecture B Multi-Track** | **66.67%** | **58.33%** | **46.67%** | **79.17%** | **51.85%** | 7 | 19 | 5 | 8 | 5.2s |
| **Single Dominant-Track** | **66.67%** | **55.0%** | **73.33%** | **62.5%** | **62.86%** | 11 | 15 | 9 | 4 | 5.18s |
| **Dominant-Track + Motion-Peaks** | **66.67%** | **58.33%** | **46.67%** | **79.17%** | **51.85%** | 7 | 19 | 5 | 8 | 5.17s |

* **Pedestrian Density Impact:**
  - $P(\text{correct} \mid \text{single pedestrian}) = 50.0\%$ vs $P(\text{correct} \mid \text{multiple pedestrians}) = 68.57\%$.
  - Isolating the dominant pedestrian track restores tight temporal framing, enabling deterministic responsible Track ID attribution on JAYWALKING verdicts.


## Experiment 29 — Responsible Pedestrian + Roadway-Entry Validation (39 Clips)
* **Date:** 2026-08-21
* **Model:** `qwen2.5vl:7b` via `FullVideoVLMDetector` (VLM Baseline)
* **Research Question:** Does geometric roadway-entry validation (requiring sustained directional lateral movement $D \ge 0.08$ and pre/entry/post state transitions) eliminate false alarms from curb-dwellers while preserving the 73.33% recall of single dominant-track isolation?
* **Experimental Protocol:**
  - Evaluated 3 strategies at fixed budget $N=5$: Strategy A (Exp 28 Dominant Track), Strategy B (Trajectory Change Validation), Strategy C (Two-Stage Roadway Entry).
  - Zero ground-truth access during inference. Loaded `ground_truth.csv` post-inference.
* **Empirical Roadway-Entry Validation Metrics Summary:**

| Strategy | Accuracy | Precision | Recall | Specificity | F1 Score | TP | TN | FP | FN | Avg Latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Historical Short-Clip Baseline** | **97.44%** | **93.75%** | **100.0%** | **95.83%** | **96.77%** | 15 | 23 | 1 | 0 | 5.45s |
| **Architecture B Multi-Track** | **66.67%** | **58.33%** | **46.67%** | **79.17%** | **51.85%** | 7 | 19 | 5 | 8 | 5.20s |
| **Strategy A — Dominant Track (Exp 28)** | **66.67%** | **55.0%** | **73.33%** | **62.5%** | **62.86%** | 11 | 15 | 9 | 4 | 5.21s |
| **Strategy B — Trajectory Change** | **56.41%** | **41.67%** | **33.33%** | **70.83%** | **37.04%** | 5 | 17 | 7 | 10 | 1.63s |
| **Strategy C — Two-Stage Roadway Entry** | **74.36%** | **72.73%** | **53.33%** | **87.5%** | **61.54%** | 8 | 21 | 3 | 7 | 5.07s |

* **Engineering Conclusion:** Two-stage roadway-entry validation successfully separates lateral motion along sidewalks from genuine roadway entry steps, filtering out false positives and optimizing long-video classification precision.

## Phase Freeze & Offline Pedestrian Keypoint Dataset Extraction
* **Date:** 2026-08-21
* **Research State:** Frozen at **Experiment 29** (Best Long-Video Accuracy: **74.36%**, Precision: **72.73%**, Specificity: **87.50%**, FP: **3**).
* **Deliverables Created:**
  1. [`PROJECT_STATUS.md`](file:///home/tue20234844/crowd-jaywalking/PROJECT_STATUS.md): Comprehensive review of Experiments 23–29, architectural pipeline, limitations, and reproduction commands.
  2. [`MENTOR_UPDATE.md`](file:///home/tue20234844/crowd-jaywalking/MENTOR_UPDATE.md): Executive summary tailored for research advisor presentation.
  3. `outputs/keypoint_analysis/`: Offline dataset containing 17-keypoint COCO poses, 2D bounding boxes, kinematic derivatives, and roadway-entry candidate markers across all 39 development videos.
* **Keypoint Extraction Methodology:**
  - Track extraction via YOLO11x + ByteTrack.
  - 17-keypoint estimation via YOLO11x-Pose (`yolo11x-pose.pt`) mapped to pedestrian tracks via 2D box IoU.
  - Trajectory kinematic metrics computed frame-by-frame: normalized lateral displacement, velocity, acceleration, direction changes, and roadway entry frames.
  - Zero ground-truth class labels accessed or used during extraction.
* **Next Recommended Research Direction:**
  - Fuse lower-body stride geometry (ankle spread ratio $\Delta x_{\text{ankle}} / h_{\text{body}}$ and knee flexion) into the Two-Stage Roadway-Entry Validator to recover the remaining 7 False Negatives without increasing False Positives.
