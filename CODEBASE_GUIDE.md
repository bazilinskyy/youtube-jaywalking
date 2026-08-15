# CODEBASE GUIDE — VLM Jaywalking Detection

---

## 1. Current Status

* **Current Canonical Pipeline:** Baseline V1 (3 Equidistant Keyframes + Independent Qwen2.5-VL-7B Inference + Majority Vote $\ge 2/3$).
* **Current Model:** `qwen2.5vl:7b` running locally via Ollama (`temperature: 0.0`, `seed: 42`).
* **Current Datasets:**
  - **Development Benchmark:** 39 evaluable JAAD video clips ([`data/ground_truth.csv`](file:///home/tue20234844/crowd-jaywalking/data/ground_truth.csv)).
  - **Held-Out Test Set:** 20 unseen JAAD video clips ([`data/heldout_ground_truth.csv`](file:///home/tue20234844/crowd-jaywalking/data/heldout_ground_truth.csv)).
* **Current Benchmark Status:**
  - **Development (39 clips):** **69.23% Accuracy**, 56.52% Precision, 86.67% Recall, 58.33% Specificity, 68.42% F1 (TP=13, TN=14, FP=10, FN=2).
  - **Held-Out Test (20 clips):** **60.00% Accuracy**, 56.25% Precision, 90.00% Recall, 30.00% Specificity, 69.23% F1 (TP=9, TN=3, FP=7, FN=1).
* **Current Bottleneck:** Poor Specificity on compliant street crossings ($30.00\%$ on held-out test data), driven by **7 False Positives** where VLM predicts Jaywalking on legal zebra/signalized crossings.
* **Current Next Step:** Perform a deep visual error analysis on the 7 held-out false positive clips (`video_0023`, `video_0025`, `video_0041`, `video_0050`, `video_0076`, `video_0095`, `video_0100`) to identify why compliant crossings are misclassified.

---

## 2. Canonical Architecture

```
                                  [Video Clip (.mp4)]
                                           │
                                           ▼
                                 [sample_keyframes()]
                    Extracts 3 Equidistant Keyframes (T0, T_mid, T_end)
                                           │
                                           ▼
                               [encode_frame_to_base64()]
                     Converts BGR NumPy Arrays to JPEG Base64
                                           │
                                           ▼
                                 [Ollama API Client]
                     http://localhost:11434/api/chat (Qwen2.5-VL-7B)
                   Temperature = 0.0, Seed = 42, Max Tokens = 10
                                           │
                        ┌──────────────────┼──────────────────┐
                        ▼                  ▼                  ▼
                    [Frame 0]          [Frame 1]          [Frame 2]
                    Prediction         Prediction         Prediction
                        │                  │                  │
                        └──────────────────┼──────────────────┘
                                           ▼
                                 [parse_response()]
                   Extracts "JAYWALKING" or "COMPLIANT" from Raw Text
                                           │
                                           ▼
                                  [Majority Vote]
                     Threshold: >= 2/3 Jaywalking Votes -> JAYWALKING
                     Otherwise                              -> COMPLIANT
                                           │
                                           ▼
                                [Evaluator / CLI Output]
                 Outputs Prediction CSV, Metrics JSON, & Summary Report
```

---

## 3. Exact Code Flow

```
scripts/run_evaluation.py
    │
    ▼
src/pipeline.py  (get_pipeline())
    │
    ▼
src/vlm/detector.py  (VLMJaywalkingDetector)
    ├── sample_keyframes()  -> [Frame 0, Frame 1, Frame 2]
    ├── classify_frame()    -> Frame-level Base64 request
    ├── parse_response()    -> "jaywalking" / "compliant"
    └── predict()           -> Majority Vote (>= 2/3)
    │
    ▼
src/vlm/client.py  (OllamaClient)
    │
    ▼
Local Ollama Server  (qwen2.5vl:7b via HTTP POST)
    │
    ▼
evaluation/evaluator.py  (Evaluator)
    │
    ▼
evaluation/metrics.py    (compute_metrics())
```

---

## 4. Important Files

| File | What it does | Important functions | Called By | Calls |
|---|---|---|---|---|
| [`src/vlm/detector.py`](file:///home/tue20234844/crowd-jaywalking/src/vlm/detector.py) | Core VLM multi-frame detector | `VLMJaywalkingDetector`, `predict()`, `sample_keyframes()`, `parse_response()` | [`src/pipeline.py`](file:///home/tue20234844/crowd-jaywalking/src/pipeline.py), [`scripts/run_evaluation.py`](file:///home/tue20234844/crowd-jaywalking/scripts/run_evaluation.py) | [`src/vlm/client.py`](file:///home/tue20234844/crowd-jaywalking/src/vlm/client.py), [`src/vlm/prompts.py`](file:///home/tue20234844/crowd-jaywalking/src/vlm/prompts.py) |
| [`src/vlm/client.py`](file:///home/tue20234844/crowd-jaywalking/src/vlm/client.py) | Ollama HTTP API client | `OllamaClient`, `generate_chat()`, `encode_frame_to_base64()` | [`src/vlm/detector.py`](file:///home/tue20234844/crowd-jaywalking/src/vlm/detector.py) | `requests`, `cv2`, `base64` |
| [`src/vlm/prompts.py`](file:///home/tue20234844/crowd-jaywalking/src/vlm/prompts.py) | Prompt registry | `CANONICAL_PROMPT`, `get_prompt()` | [`src/vlm/detector.py`](file:///home/tue20234844/crowd-jaywalking/src/vlm/detector.py) | None |
| [`src/pipeline.py`](file:///home/tue20234844/crowd-jaywalking/src/pipeline.py) | Pipeline factory | `get_pipeline()` | [`scripts/run_inference.py`](file:///home/tue20234844/crowd-jaywalking/scripts/run_inference.py), [`scripts/run_evaluation.py`](file:///home/tue20234844/crowd-jaywalking/scripts/run_evaluation.py) | [`src/vlm/detector.py`](file:///home/tue20234844/crowd-jaywalking/src/vlm/detector.py) |
| [`src/config.py`](file:///home/tue20234844/crowd-jaywalking/src/config.py) | Config & path loader | `get_vlm_config()`, `get_paths()` | [`src/vlm/detector.py`](file:///home/tue20234844/crowd-jaywalking/src/vlm/detector.py), [`evaluation/evaluator.py`](file:///home/tue20234844/crowd-jaywalking/evaluation/evaluator.py) | `configs/config.json` |
| [`src/data_loader.py`](file:///home/tue20234844/crowd-jaywalking/src/data_loader.py) | Ground-truth loader | `load_ground_truth_records()` | [`evaluation/evaluator.py`](file:///home/tue20234844/crowd-jaywalking/evaluation/evaluator.py), [`tests/test_pipeline.py`](file:///home/tue20234844/crowd-jaywalking/tests/test_pipeline.py) | `pandas`, [`src/config.py`](file:///home/tue20234844/crowd-jaywalking/src/config.py) |
| [`evaluation/evaluator.py`](file:///home/tue20234844/crowd-jaywalking/evaluation/evaluator.py) | Benchmark execution harness | `Evaluator`, `run_evaluation()` | [`scripts/run_evaluation.py`](file:///home/tue20234844/crowd-jaywalking/scripts/run_evaluation.py) | [`src/data_loader.py`](file:///home/tue20234844/crowd-jaywalking/src/data_loader.py), [`evaluation/metrics.py`](file:///home/tue20234844/crowd-jaywalking/evaluation/metrics.py) |
| [`evaluation/metrics.py`](file:///home/tue20234844/crowd-jaywalking/evaluation/metrics.py) | Metrics calculation | `compute_metrics()` | [`evaluation/evaluator.py`](file:///home/tue20234844/crowd-jaywalking/evaluation/evaluator.py) | None |
| [`scripts/run_inference.py`](file:///home/tue20234844/crowd-jaywalking/scripts/run_inference.py) | Single/Batch inference CLI | `main()` | CLI User | [`src/pipeline.py`](file:///home/tue20234844/crowd-jaywalking/src/pipeline.py) |
| [`scripts/run_evaluation.py`](file:///home/tue20234844/crowd-jaywalking/scripts/run_evaluation.py) | Evaluation CLI | `main()` | CLI User | [`evaluation/evaluator.py`](file:///home/tue20234844/crowd-jaywalking/evaluation/evaluator.py) |

---

## 5. Configuration

* **Model:** `qwen2.5vl:7b` (Configured in `configs/config.json` -> `vlm.model`)
* **Temperature:** `0.0` (Configured in `configs/config.json` -> `vlm.temperature`)
* **Seed:** `42` (Configured in `configs/config.json` -> `vlm.seed`)
* **Keyframes:** `3` (Configured in `configs/config.json` -> `vlm.num_frames`)
* **Vote Threshold:** `2` ($\ge 2/3$ Jaywalking votes) (Configured in `src/pipeline.py` -> `min_votes_for_jaywalking`)
* **Ollama Endpoint:** `http://localhost:11434/api/chat` (Configured in `configs/config.json` -> `vlm.ollama_url`)
* **Ground-Truth Path:** `data/ground_truth.csv` (Default in `evaluation/evaluator.py`, overridable via `--gt`)

---

## 6. Dataset & Ground Truth

### Development Benchmark
* **Dataset File:** [`data/ground_truth.csv`](file:///home/tue20234844/crowd-jaywalking/data/ground_truth.csv)
* **Total Clips:** 39 evaluable JAAD video clips ($15$ Jaywalking, $24$ Compliant).

### Held-Out Test Set
* **Dataset File:** [`data/heldout_ground_truth.csv`](file:///home/tue20234844/crowd-jaywalking/data/heldout_ground_truth.csv)
* **Total Clips:** 20 unseen JAAD video clips ($10$ Jaywalking, $10$ Compliant).
* **SHA256 Hash:** `1e3679866a597d3a7a22388de92ed6e1cefbaf815ed78573270c8188ed58d4e0`

### Ground-Truth Provenance
JAAD raw dashcam videos -> Official JAAD XML metadata -> Event summary [`experiments/previous_results/jaad_events_summary.csv`](file:///home/tue20234844/crowd-jaywalking/experiments/previous_results/jaad_events_summary.csv) -> Canonical CSV ground-truth files.

> **CRITICAL RULE:** Never modify ground-truth labels during algorithm experiments.

---

## 7. Canonical Baseline

### Development Benchmark (39 Clips)
* **Accuracy:** **69.23%** (27/39)
* **Precision:** 56.52%
* **Recall:** 86.67%
* **Specificity:** 58.33%
* **F1 Score:** 68.42%
* **Confusion Matrix:** $	ext{TP}=13, 	ext{TN}=14, 	ext{FP}=10, 	ext{FN}=2$

### Held-Out Test Set (20 Clips)
* **Accuracy:** **60.00%** (12/20)
* **Precision:** 56.25%
* **Recall:** **90.00%**
* **Specificity:** 30.00%
* **F1 Score:** 69.23%
* **Confusion Matrix:** $	ext{TP}=9, 	ext{TN}=3, 	ext{FP}=7, 	ext{FN}=1$

This is the **CURRENT reproducible baseline**.

---

## 8. Current Bottleneck

Primary problem: **FALSE POSITIVES on compliant scenes**.

The model has high recall ($90.00\%$) but poor specificity ($30.00\%$).

### Held-Out False Positive Clips ($N=7$):
* `video_0023.mp4` (GT: compliant — marked zebra crossing)
* `video_0025.mp4` (GT: compliant — marked zebra crossing)
* `video_0041.mp4` (GT: compliant — marked zebra crossing)
* `video_0050.mp4` (GT: compliant — parking lot aisle)
* `video_0076.mp4` (GT: compliant — marked zebra crossing)
* `video_0095.mp4` (GT: compliant — marked zebra crossing)
* `video_0100.mp4` (GT: compliant — marked zebra crossing)

*Do not assume the cause.* These 7 clips must be visually analyzed frame-by-frame before designing the next experiment.

---

## 9. Experiments Already Tried

| Experiment | Purpose | Result | Decision |
|---|---|---|:---:|
| **Baseline V1** | 3-frame independent classification + majority vote | **69.23% Dev / 60.00% Held-Out** | **KEEP (Canonical Baseline)** |
| **Prompt V2** | Detailed prompt with definitions & negative constraints | 38.46% Dev Acc (0% Specificity, 24 FPs) | **REJECT** |
| **Temporal VLM** | Single-request 3-image prompt | 38.46% Dev Acc | **REJECT** |
| **Boundary Context** | Static spatial tags (`roadway`/`sidewalk`) in prompt | 48.72% Dev Acc (4 FPs fixed, 6 FNs added) | **REJECT** |
| **Pedestrian Motion** | Bounding box displacement & velocity vectors in prompt | 38.46% Dev Acc | **REJECT** |
| **Vehicle Motion** | 2D bounding box scaling & vehicle kinematics | 38.46% Dev Acc (Confounded by camera motion) | **REJECT** |
| **Motion Override** | Rule-based pedestrian trajectory override | 56.41% Dev Acc (17 FPs) | **REJECT** |
| **Structured Arbitration** | 7-field structured VLM scene observation rules | Collapsed Recall to 0% due to zero-shot text bias | **REJECT** |
| **Parking-Lot Correction (Policy A)** | `parking_lot` context override to `COMPLIANT` | 71.79% Dev Acc, but reduced Held-Out to 80% with 4 FNs | **REJECT** |

For detailed logs and frame-by-frame analysis:
→ [`RESEARCH_LOG.md`](file:///home/tue20234844/crowd-jaywalking/RESEARCH_LOG.md)

---

## 10. What We Learned

1. More prompt rules do not solve visual ambiguity.
2. Raw multi-frame VLM prompts trigger strong violation bias.
3. Pedestrian trajectory displacement alone cannot distinguish legal crossing from jaywalking.
4. Monocular 2D vehicle kinematics are fundamentally unreliable due to camera ego-motion.
5. Broad heuristic overrides cause severe collateral errors.
6. Simple V1 remains the single canonical baseline.
7. The core research problem is improving specificity on compliant street scenes without destroying recall.

---

## 11. Known Dead / Archived Code

All archived code and legacy modules are moved to `experiments/`:
* `experiments/archived_scripts/`: Historical single-run test scripts (`review_clips.py`, `build_jaad_eval.py`, `test_tld_ready.py`, `run_eval_v3.py`, comparison scripts).
* `experiments/legacy/`: Legacy codebase modules (`common.py`, `analysis.py`, `sample_daylight.py`, `process_batch.py`, `utils/`).

**DO NOT** use archived modules for active development or inference.

---

## 12. Reproducibility Reference

* **Model Name:** `qwen2.5vl:7b` via Ollama
* **Ollama Options:** `temperature: 0.0`, `seed: 42`, `max_tokens: 10`
* **Keyframe Strategy:** 3 equidistant frames $[0, \lfloor N/2 \rfloor, N-1]$
* **Vote Threshold:** $\ge 2/3$ Jaywalking votes
* **Dev Dataset:** `data/ground_truth.csv` (39 clips)
* **Held-Out Dataset:** `data/heldout_ground_truth.csv` (20 clips, SHA256 `1e3679866a597d3a7a22388de92ed6e1cefbaf815ed78573270c8188ed58d4e0`)

---

## 13. Commands

### Single Video Inference
```bash
python3 scripts/run_inference.py --video data/raw_clips/video_0014.mp4
```

### Canonical Development Evaluation (39 Clips)
```bash
python3 scripts/run_evaluation.py --mode balanced
```

### Held-Out Test Evaluation (20 Clips)
```bash
python3 scripts/run_evaluation.py --mode balanced --gt data/heldout_ground_truth.csv
```

### Run Unit Tests
```bash
python3 -m unittest discover -s tests
```

---

## 14. Research Continuation Protocol

Before starting any new experiment:

1. Read `CODEBASE_GUIDE.md`.
2. Read the latest section of `RESEARCH_LOG.md`.
3. Freeze the canonical V1 baseline.
4. Define **ONE hypothesis**.
5. Change **ONE major variable**.
6. Run development benchmark (`data/ground_truth.csv`).
7. Analyze changed/error clips.
8. **Do NOT modify held-out data.**
9. Evaluate held-out set **only** after the experiment logic is completely frozen.
10. Record in `RESEARCH_LOG.md`:
    - Hypothesis
    - Implementation
    - Exact config
    - Dev & Held-Out metrics
    - Changed clips table
    - Failure analysis
    - Conclusion & next decision.

---

## 15. Current Next Step

**DO NOT start another architecture experiment yet.**

First analyze the 7 held-out false positives (`video_0023`, `video_0025`, `video_0041`, `video_0050`, `video_0076`, `video_0095`, `video_0100`) and determine whether the failure is:
* Spatial ambiguity (crosswalk markings worn out or partially out of frame)
* Temporal ambiguity (pedestrian standing at curb vs. stepping onto roadway)
* Missing legal/contextual information (traffic light status or vehicle yielding)
* Dataset/label ambiguity
* VLM reasoning failure (Qwen2.5-VL hallucinating lack of crosswalk)

Then choose the next experiment based on that empirical evidence.

---

## 16. Mentor Discussion

Questions for thesis advisor / mentor review:

1. Is our JAAD -> binary label definition (`jaywalking` vs `compliant`) appropriate for ego-vehicle decision making?
2. Is $90\%$ Recall / $30\%$ Specificity an acceptable operating point for safety-critical pedestrian interaction, or must specificity be prioritized?
3. Do we need more compliant zebra-crossing examples in the development set?
4. Should the primary objective of the next phase be Specificity improvement?
5. Is RGB-only visual information sufficient for determining right-of-way without vehicle telemetry?
6. Should we investigate fine-tuning or a different VLM architecture (e.g. LLaVA-NeXT, InternVL2) rather than rule-based heuristics?
