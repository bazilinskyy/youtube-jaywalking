# CODEBASE GUIDE — VLM Jaywalking Detection

---

## 1. Current Status

* **Current Primary Baseline:** VLM Full-Video Chain-of-Causation (CoC) Reasoning Baseline (`FullVideoVLMDetector`).
* **Current Model:** `qwen2.5vl:7b` running locally via Ollama (`temperature: 0.0`, `seed: 42`).
* **Historical Baseline:** V1 Keyframe Majority-Voting Baseline (3 equidistant frames, min_votes=2).
* **Current Dataset:** Canonical 39-clip JAAD Development Benchmark (`data/ground_truth.csv`, 15 jaywalking, 24 compliant).
* **Current Measured Result (VLM Baseline (CoC)):**
  - **Accuracy:** **97.44%** (38/39 correct)
  - **Precision:** **93.75%** (15/16)
  - **Recall:** **100.00%** (15/15)
  - **Specificity:** **95.83%** (23/24)
  - **F1 Score:** **96.77%**
  - **Confusion Matrix:** TP=15, TN=23, FP=1, FN=0
  - **Execution Time:** 212.44s total (5.45s / clip)
* **Current Bottleneck:** Generalization & robustness validation on unseen video sequences.
* **Current Next Step:** Freeze current VLM baseline configuration and evaluate on untouched held-out test set.

---

## 2. Active & Historical File Reference Table

| File | Role | Important Functions | Status |
|---|---|---|:---:|
| [`src/vlm/alpamayo_detector.py`](file:///home/tue20234844/crowd-jaywalking/src/vlm/alpamayo_detector.py) | **VLM Baseline CoC Detector** | `FullVideoVLMDetector`, `extract_full_video_frames()`, `parse_coc_response()`, `predict()` | **ACTIVE (PRIMARY)** |
| [`src/vlm/client.py`](file:///home/tue20234844/crowd-jaywalking/src/vlm/client.py) | Ollama HTTP API Client | `OllamaClient`, `generate_chat()`, `encode_frame_to_base64()` | **ACTIVE** |
| [`src/vlm/detector.py`](file:///home/tue20234844/crowd-jaywalking/src/vlm/detector.py) | V1 Keyframe Voter | `VLMJaywalkingDetector`, `sample_keyframes()`, `parse_response()`, `predict()` | **HISTORICAL BASELINE** |
| [`src/vlm/prompts.py`](file:///home/tue20234844/crowd-jaywalking/src/vlm/prompts.py) | Prompt Registry | `CANONICAL_PROMPT`, `get_prompt()` | **HISTORICAL BASELINE** |
| [`src/pipeline.py`](file:///home/tue20234844/crowd-jaywalking/src/pipeline.py) | Pipeline Factory | `get_pipeline()` | **ACTIVE** |
| [`src/config.py`](file:///home/tue20234844/crowd-jaywalking/src/config.py) | Config & Path Loader | `get_vlm_config()`, `get_paths()` | **ACTIVE** |
| [`src/data_loader.py`](file:///home/tue20234844/crowd-jaywalking/src/data_loader.py) | Ground Truth Loader | `load_ground_truth_records()` | **ACTIVE** |
| [`evaluation/evaluator.py`](file:///home/tue20234844/crowd-jaywalking/evaluation/evaluator.py) | Benchmark Evaluator | `Evaluator`, `run_evaluation()` | **ACTIVE** |
| [`evaluation/metrics.py`](file:///home/tue20234844/crowd-jaywalking/evaluation/metrics.py) | Metrics Calculator | `compute_metrics()` | **ACTIVE** |
| [`scripts/run_inference.py`](file:///home/tue20234844/crowd-jaywalking/scripts/run_inference.py) | Inference CLI | `main()` | **ACTIVE** |
| [`scripts/run_evaluation.py`](file:///home/tue20234844/crowd-jaywalking/scripts/run_evaluation.py) | Evaluation CLI | `main()` | **ACTIVE** |
| [`tests/test_pipeline.py`](file:///home/tue20234844/crowd-jaywalking/tests/test_pipeline.py) | Test Suite | `TestJaywalkingPipeline` | **ACTIVE** |

---

## 3. Canonical Architecture & Execution Path

```
scripts/run_evaluation.py --mode alpamayo --gt data/ground_truth.csv
    │
    ▼
src/pipeline.py (get_pipeline(mode="alpamayo"))
    │
    ▼
src/vlm/alpamayo_detector.py (FullVideoVLMDetector)
    ├── extract_full_video_frames(max_frames=5) -> [Frame_0 .. Frame_4]
    ├── src/vlm/client.py: encode_frame_to_base64() -> 5 Base64 Strings
    ├── src/vlm/client.py: OllamaClient.generate_chat(prompt=coc_prompt, base64_images=b64_list)
    │     └──> Ollama HTTP API (http://localhost:11434/api/chat, qwen2.5vl:7b)
    ├── parse_coc_response(raw_text) -> Extract classification from Step 5
    └── predict() -> Return prediction, CoC text, timing
    │
    ▼
evaluation/evaluator.py (Evaluator) -> compute_metrics() -> 97.44% Acc Summary
```

---

## 4. Model & Sampling Configuration

* **Model Name:** `qwen2.5vl:7b` (via local Ollama daemon).
* **Generation Parameters:** `temperature=0.0`, `seed=42`, `max_tokens=300`.
* **Frame Sampling:** 5 equidistant frames per clip sampled by `extract_full_video_frames()`.
* **Prompt Location:** `FullVideoVLMDetector.coc_prompt` in `src/vlm/alpamayo_detector.py`.
* **Output Parsing:** `FullVideoVLMDetector.parse_coc_response()` in `src/vlm/alpamayo_detector.py`.

---

## 5. Execution Commands

### 1. Run 39-Clip Primary VLM Baseline Evaluation (97.44% Accuracy)
```bash
python3 scripts/run_evaluation.py --mode alpamayo --gt data/ground_truth.csv
```

### 2. Single Video VLM Baseline Inference
```bash
python3 scripts/run_inference.py --video data/raw_clips/video_0014.mp4 --mode alpamayo
```

### 3. Historical Baseline V1 Evaluation
```bash
python3 scripts/run_evaluation.py --mode balanced --gt data/ground_truth.csv
```

### 4. Run Unit Tests
```bash
python3 -m unittest discover -s tests
```
