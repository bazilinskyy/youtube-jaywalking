# Current Research State

---

## Current Primary Pipeline
**VLM Chain-of-Causation (CoC) Full-Video Baseline** (`FullVideoVLMDetector` in [`src/vlm/alpamayo_detector.py`](src/vlm/alpamayo_detector.py)).
* **Configuration:** 5 equidistant frames per clip, single multi-image API call (`qwen2.5vl:7b`, `temp=0.0`, `seed=42`), 5-step Chain-of-Causation reasoning prompt protocol.

---

## Current Best Development Result
* **Benchmark:** Canonical 39-clip JAAD Development Benchmark (`data/ground_truth.csv`).
* **Metrics:** **97.44% Accuracy** (38/39 correct), **93.75% Precision**, **100.00% Recall**, **95.83% Specificity**, **96.77% F1**.
* **Confusion Matrix:** $\text{TP}=15, \text{TN}=23, \text{FP}=1, \text{FN}=0$.
* **Execution Time:** 212.44 seconds total (average 5.45 seconds/clip).

---

## Historical Baseline
* **Pipeline:** Baseline V1 (3 equidistant frames, independent classification, majority vote $\ge 2/3$).
* **Metrics:** 69.23% Accuracy, 56.52% Precision, 86.67% Recall, 58.33% Specificity, 68.42% F1 ($\\text{TP}=13, \\text{TN}=14, \\text{FP}=10, \\text{FN}=2$).

---

## Dataset
* **Source:** JAAD-derived dashcam video sequences.
* **Development Benchmark:** 39 clips ($15$ Jaywalking, $24$ Compliant) in `data/ground_truth.csv`.

---

## Known Failure
* **`video_0003.mp4` (GT: `compliant`):** Commercial store entrance / parking lot driveway misclassified as an active roadway crossing without a crosswalk (`jaywalking`).

---

## What Has Been Tried
* Single-frame prompt expansion (Prompt V2)
* Static boundary context tags
* Raw temporal multi-image prompts without CoC
* 2D bounding box pedestrian & vehicle kinematics
* Pedestrian trajectory motion overrides
* Qualitative structured VLM evidence arbitration rules
* Parking lot domain heuristics (Policy A)

---

## What Has Been Rejected
* **Prompt V2:** 38.46% Acc (0% Specificity, 24 FPs)
* **Raw Temporal VLM:** 38.46% Acc (false alarm bias)
* **2D Vehicle Motion:** 38.46% Acc (confounded by camera ego-motion)
* **Pedestrian Trajectory Overrides:** 56.41% Acc (trajectory is symmetric between legal and illegal crossing)
* **Parking Lot Override (Policy A):** Over-corrected true parking-lot jaywalking on unseen clips.

---

## What Must NOT Be Done
1. **Do NOT tune prompts or models on the held-out set.**
2. **Do NOT modify the frozen VLM baseline configuration before held-out evaluation.**
3. **Do NOT claim generalization to unseen data based solely on the 39-clip development benchmark.**
4. **Do NOT mix Gemma results or claims into the current VLM baseline result.**
5. **Do NOT modify ground-truth labels during experiments.**

---

## Immediate Next Step
**FREEZE CURRENT VLM BASELINE PIPELINE $\rightarrow$ EVALUATE ON UNTOUCHED HELD-OUT TEST SET.**

---
