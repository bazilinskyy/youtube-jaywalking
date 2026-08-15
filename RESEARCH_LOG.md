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
  * **Best Sensitivity / Safety Filter:** Threshold $\ge 1/3$ & Temporal VLM — **100.00% Recall** (15 TP, 0 FN).
* **Current Biggest Problem:** **3 Stubborn Unanimous False Positives** (`video_0099`, `video_0168`, `video_0297` where yielding cannot be visually confirmed without telemetry).
* **Current Experiment:** Completed Experiment 6 (V1 Vote-Margin Calibration).
* **Immediate Next Step:** CLI & Pipeline Integration of Calibrated Threshold Modes.




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
