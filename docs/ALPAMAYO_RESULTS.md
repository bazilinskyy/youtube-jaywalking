# VLM Baseline (CoC) Results

---

## 1. Objective

This experiment evaluated the **VLM Chain-of-Causation (CoC) Full-Video Baseline** for pedestrian street-crossing compliance and jaywalking violation detection. The objective was to test whether replacing isolated single-frame majority voting with a 5-frame temporal sequence input and a 5-step Chain-of-Causation reasoning prompt eliminates false positives and improves overall benchmark accuracy.

---

## 2. Dataset

* **Source:** Derived from the open-source JAAD (Joint Attention in Autonomous Driving) benchmark.
* **Benchmark Split:** Canonical 39-clip Development Benchmark (`data/ground_truth.csv`).
* **Class Distribution:** 15 Jaywalking clips, 24 Compliant clips.
* **Ground-Truth Provenance:** Derived from frame-level JAAD XML metadata in `experiments/previous_results/jaad_events_summary.csv`.
* **Benchmark Status:** **DEVELOPMENT BENCHMARK**. This dataset was used during algorithm development and is NOT the held-out test set.

---

## 3. Pipeline Architecture

```
Video Clip (.mp4)
  │
  ▼
5 Temporal Frames (Equidistant Extraction)
  │
  ▼
VLM Temporal Reasoning Baseline (Single Multi-Image Request)
  │
  ▼
Chain-of-Causation Reasoning (5-Step CoC Prompt)
  │
  ▼
Final Classification Parsing (parse_coc_response())
  │
  ▼
JAYWALKING / COMPLIANT
```

---

## 4. Temporal Sampling

* **Number of Frames:** 5 frames per video clip.
* **Selection Method:** Equidistant frame sampling across the video duration using OpenCV `cv2.VideoCapture`.
* **Sampling Formula:** `step = max(1, int(fps / target_fps))`, selecting 5 equidistant indices `indices[::step_idx][:5]`.
* **Traced Example Frame Indices:**
  - `video_0003.mp4` (175 frames, 30 FPS, 5.83s duration): Selected frames `[0, 35, 70, 105, 140]`.
  - `video_0028.mp4` (240 frames, 30 FPS, 8.00s duration): Selected frames `[0, 48, 96, 144, 192]`.
  - `video_0014.mp4` (257 frames, 30 FPS, 8.57s duration): Selected frames `[0, 50, 100, 150, 200]`.
* **Timestamps:** Equidistant timestamps spanning T0 (start of clip) to Tend (end of clip).

---

## 5. Reasoning Process

The current implementation utilizes a 5-step **Chain-of-Causation (CoC)** prompt protocol enforced via `FullVideoVLMDetector.coc_prompt`:

1. **Pedestrian Trajectory & Location:** [sidewalk / curb / roadway]
2. **Infrastructure & Right-of-Way:** [marked crosswalk / signal / none]
3. **Vehicle Kinematic Response:** [yielding / decelerating / accelerating / none]
4. **Causal Analysis:** Explain why the crossing is legal compliance or an illegal violation.
5. **Final Classification:** Output EXACTLY either JAYWALKING or COMPLIANT.

*Note on Implementation:* In the current codebase, Chain-of-Causation is an explicit **prompt protocol** enforced on the underlying vision-language backbone (`qwen2.5vl:7b` via Ollama). It structures the model's generation sequence to prevent premature binary label guessing.

---

## 6. Benchmark Results

Comparison of Canonical Baseline V1 vs. VLM CoC Baseline on the 39-clip development benchmark:

| Metric | V1 | VLM Baseline |
|---|---:|---:|
| Accuracy | 69.23% | 97.44% |
| Precision | 56.52% | 93.75% |
| Recall | 86.67% | 100.00% |
| Specificity | 58.33% | 95.83% |
| F1 | 68.42% | 96.77% |
| TP | 13 | 15 |
| TN | 14 | 23 |
| FP | 10 | 1 |
| FN | 2 | 0 |

VLM baseline improved development accuracy by **28.21 percentage points** (from 69.23% to 97.44%) and reduced false positives from **10 down to 1** while recovering both previous false negatives (achieving **100% Recall**). Total benchmark execution time was 212.44 seconds (average 5.45 seconds/clip).

---

## 7. False Positive Analysis (`video_0003.mp4`)

Only a single clip failed on the 39-clip development benchmark:

* **Clip ID:** `video_0003.mp4` (Ground Truth: `compliant`).
* **Visual Scenario:** Pedestrian walking near a commercial store entrance / parking lot driveway adjacent to a parked vehicle.
* **Model CoC Reasoning:**
  - *Pedestrian Trajectory:* Pedestrian walking on roadway crossing in front of a parked car.
  - *Infrastructure:* No marked crosswalk or traffic signal detected.
  - *Vehicle Response:* Vehicle appears slow or stopped.
  - *Causal Analysis:* Model concluded that crossing on pavement without a marked crosswalk constitutes jaywalking.
* **Diagnosis:** The model misinterpreted a commercial parking lot shared space as an active public roadway without a crosswalk. This remains an unresolved semantic/contextual ambiguity between parking lot driveways and public streets.

---

## 8. Why VLM Baseline Appears Better Than V1

The observed performance jump from 69.23% to 97.44% stems from several combined architectural differences:

1. **Independent single-frame classification vs. Temporal multi-frame reasoning:** V1 classified frames in total isolation; VLM Baseline receives the sequence together.
2. **3 frames vs. 5 frames:** 5 frames provide finer temporal granularity of pedestrian movement.
3. **Independent votes vs. Joint temporal reasoning:** V1 majority-voted separate binary guesses; VLM Baseline performs joint reasoning across all frames simultaneously.
4. **Limited frame-level context vs. Temporal co-visibility:** Passing 5 frames in 1 multi-image prompt gives the model co-visibility of pedestrian trajectory and vehicle deceleration over time.
5. **Majority voting vs. Model-generated final reasoning:** Chain-of-Causation forcing step-by-step analysis prevents hasty violation calls.

> **SCIENTIFIC NOTE:** These factors were changed together, so the current experiment does not isolate the contribution of each factor.

---

## 9. What We Have NOT Proven

* We have **NOT** proven that CoC prompting alone caused the improvement (it was introduced alongside 5-frame joint input).
* We have **NOT** proven that VLM baseline generalizes to unseen held-out data.
* We have **NOT** proven that Gemma adds value (Gemma is not active in the current 97.44% pipeline).
* We have **NOT** proven that we have solved all right-of-way ambiguity (as evidenced by `video_0003.mp4`).
* We have **NOT** established performance on longer video sequences or multi-pedestrian dense crowds.

---

## 10. Current Bottleneck

The current bottleneck is **Generalization and robustness validation**.

While the development-set perception and reasoning performance improved substantially (97.44%), the 39 clips remain the development benchmark. The critical open question is whether this high performance holds on unseen video sequences.

---

## 11. Next Experiment

The exact next experiment is:

```
FREEZE VLM BASELINE CONFIGURATION
        │
        ▼
EVALUATE ON UNTOUCHED HELD-OUT TEST SET (20 CLIPS)
        │
        ▼
COMPARE AGAINST FROZEN V1
        │
        ▼
ANALYZE FAILURE CASES
```

*Do not run this evaluation until explicitly instructed.*

---

## 12. Future VLM \+ Gemma Direction

A future experiment will investigate integrating **Google Gemma / PaliGemma 2** for enhanced fine-grained vision-language reasoning and VLA trajectory planning. This will be treated as a **NEW, distinct experiment** and will not be mixed into the current VLM baseline result.
