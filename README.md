# Jaywalking VLM — Pedestrian Compliance & Violation Detection

Vision-Language Model (VLM) framework for detecting pedestrian street-crossing compliance and jaywalking violations from egocentric dashcam video.

Powered by **Qwen2.5-VL-7B** running locally via Ollama with deterministic multi-frame keyframe voting.

---

## Quick Start

### 1. Requirements & Installation
* Python 3.10+
* Local Ollama instance with `qwen2.5vl:7b` pulled (`ollama pull qwen2.5vl:7b`)

```bash
python3 -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt
```

### 2. Single Video Inference
```bash
python3 scripts/run_inference.py --video data/raw_clips/video_0014.mp4
```

### 3. Batch Directory Inference
```bash
python3 scripts/run_inference.py --dir data/raw_clips/ --limit 5
```

### 4. Benchmark Evaluation
* **Canonical 39-Clip Development Evaluation:**
  ```bash
  python3 scripts/run_evaluation.py --mode balanced
  ```
* **Held-Out 20-Clip Test Evaluation:**
  ```bash
  python3 scripts/run_evaluation.py --mode balanced --gt data/heldout_ground_truth.csv
  ```

### 5. Run Unit Tests
```bash
python3 -m unittest discover -s tests
```

---

## Canonical Pipeline Overview

* **Input:** Dashcam `.mp4` video clip.
* **Keyframes:** 3 equidistant frames ($T_0, T_{\text{mid}}, T_{\text{end}}$).
* **Classifier:** `qwen2.5vl:7b` (`temperature=0.0`, `seed=42`).
* **Decision Rule:** Independent 3-frame classification + Majority Vote ($\ge 2/3$ Jaywalking votes $	o$ `jaywalking`, else `compliant`).

---

## Performance Summary

| Benchmark | Dataset | Clips | Accuracy | Precision | Recall | Specificity | F1 Score |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Canonical Baseline V1** | Development (`data/ground_truth.csv`) | 39 | **69.23%** | 56.52% | 86.67% | 58.33% | 68.42% |
| **Held-Out Test Set** | Unseen (`data/heldout_ground_truth.csv`) | 20 | **60.00%** | 56.25% | **90.00%** | 30.00% | 69.23% |

---

## Key Documentation

* 📖 **[CODEBASE_GUIDE.md](CODEBASE_GUIDE.md):** Complete architecture reference, file table, data flow, configuration guide, and troubleshooting.
* 🧪 **[RESEARCH_LOG.md](RESEARCH_LOG.md):** Chronological log of all 10 experiments, hypothesis testing, failure analysis, and decision trade-offs.
