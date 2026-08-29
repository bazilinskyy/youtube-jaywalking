# Crowd-Jaywalking: Multimodal Context-Aware Jaywalking Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![CI](https://github.com/crowd-dataset/crowd-jaywalking/actions/workflows/linter.yml/badge.svg)](https://github.com/crowd-dataset/crowd-jaywalking/actions/workflows/linter.yml)

An end-to-end multimodal perception and reasoning framework for detecting pedestrian jaywalking in real-world urban driving video sequences.

---

## 1. Overview & Problem Definition

Jaywalking detection requires understanding whether a pedestrian crossing a roadway is doing so lawfully (at a designated marked crosswalk, signalized intersection, or yielding junction) or unlawfully (unregulated mid-block crossing, crossing against a red signal).

### Core Challenges Addressed:
- **Severe Occlusions & Scale Variance:** Pedestrians stepping out from behind parked vehicles or crossing at long distances ($>40\text{ m}$).
- **Degraded Visibility:** Nighttime illumination, snow-covered asphalt, and headlight glare.
- **Crosswalk Striping Ambiguity:** Faded or snow-covered zebra markings where legal crossing infrastructure is partially obscured.
- **Shared Urban Spaces:** Curbless cobblestone streets, private parking lots, and garage ramps where drivable road segmentation erroneously spans building-to-building.
- **Camera Ego-Motion:** Distinguishing camera vehicle motion from true pedestrian transverse translation.

---

## 2. Final System Architecture (Exp 57 / Exp 58)

The **Refined Context Synergy Architecture** combines kinematic tracking, road semantic segmentation, multi-frame vision-language consensus, and wide-scene context verification:

```text
Monocular Video (.mp4)
         │
         ├──► Pedestrian Pose Tracking (YOLO26x-Pose + BoT-SORT)
         ├──► Multi-Temporal Road Surface Segmentation (SegFormer-B0 Cityscapes)
         └──► Temporal Frame Sampler (3 Keyframes: 0%, 50%, 100%)
                     │
                     ▼
         Vision-Language Classification (Qwen2.5-VL-7B)
                     │
                     ▼
         Unanimous Consensus & Tracker-Independent Persistence
                     │
                     ▼
         Wide Context Verification Routers
           ├── Crosswalk & Zebra Markings Verifier
           ├── Public Roadway Structure Verifier
           └── Intersection Junction Legal Crossing Verifier
                     │
                     ▼
         Production Decision Engine
                     │
                     ▼
            JAYWALKING / COMPLIANT
```

---

## 3. Benchmark Protocol & Evaluation Results

All experiments followed a strict separation protocol:
- **Development Set (69 videos, SHA-locked):** Used exclusively for design and optimization.
- **Locked Test Set (30 unseen videos, SHA-locked):** Evaluated strictly once after complete freeze.

### Official Benchmark Comparison

| Benchmark Partition | Videos | Accuracy | Precision | Recall | Specificity | F1 Score | TP | TN | FP | FN | Role |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **Locked Test Set (Exp 58)** | **30** | **83.33%** | **75.00%** | **81.82%** | **84.21%** | **78.26%** | **9** | **16** | **3** | **2** | **Unseen Evaluation** |
| **Development Set (Exp 57)** | 69 | **92.75%** | 83.33% | 100.0% | 88.64% | 90.91% | 25 | 39 | 5 | 0 | Optimization Set |
| **Canonical JAAD (Exp 52)** | 39 | **89.74%** | 100.0% | 73.33% | 100.0% | 84.61% | 11 | 24 | 0 | 4 | Initial Suite |

> **Scientific Generalization Statement:**  
> *"The final frozen architecture achieved 83.33% accuracy on a completely unseen locked test set. Performance decreased relative to the development set, demonstrating a measurable generalization gap while maintaining balanced recall and specificity."*

---

## 4. Installation & Environment Setup (uv)

This project uses [`uv`](https://github.com/astral-sh/uv) for fast, deterministic Python environment management.

### 1. Install uv
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Clone and Setup Environment
```bash
git clone https://github.com/crowd-dataset/crowd-jaywalking.git
cd crowd-jaywalking
uv sync
```

### 3. Start Local VLM Daemon
```bash
ollama run qwen2.5vl:7b
```

---

## 5. Configuration & Secret Management

The project uses a clean separation of configuration parameters and secrets:
- Non-sensitive parameters reside in `default.config` (or user override `config`).
- Secret tokens (API keys) reside in `default.secret` (template) or `secret` (ignored by Git).
- Load configuration programmatically via `common.get_configs()` and `common.get_secrets()`.

```python
import common
from custom_logger import CustomLogger

logger = CustomLogger(__name__)
vlm_cfg = common.get_configs("vlm")
logger.info("Loaded VLM Model: {}", vlm_cfg.get("model"))
```

---

## 6. Video Dataset Setup

Due to size constraints, raw video binaries are excluded from Git version control. Place external video files in the designated directories:
- **Canonical 39 Videos:** Place in `videos/` (e.g. `videos/video_0001.mp4`)
- **JAAD 100 Dataset:** Place in `jaad_pedestrian_100/videos/`

Dataset manifests mapping clips to ground truth labels are preserved in:
- `datasets/manifests/development_manifest.csv`
- `datasets/manifests/locked_test_manifest.csv`
- `experiments/legacy/mapping.csv`

---

## 7. Running Inference & Evaluations

### Run Single Video CLI Inference:
```bash
uv run python scripts/run_inference.py --video path/to/video.mp4
```

### Run Benchmark Evaluations:
```bash
# Evaluate on Canonical 39-video benchmark
uv run python scripts/run_canonical_evaluation.py

# Evaluate on Locked 30-video test benchmark
uv run python scripts/run_locked_evaluation.py
```

---

## 8. Code Quality & Linting

All active production code enforces strict PEP 8 compliance (119-character limit) and YAML validation:

```bash
# Run Flake8 linter on maintained codebase
uv run flake8 --config=.github/linters/.flake8 .

# Run Yamllint on configuration files
uv run yamllint -c .github/linters/.yamllint .github/ configs/

# Compile Python bytecodes
uv run python -m compileall src scripts tests evaluation common.py custom_logger.py logmod.py
```

---

## 9. Repository Structure

```text
├── common.py                # Core configuration & secret access utilities
├── custom_logger.py         # Standardized logger with brace-formatting
├── logmod.py                # Global logging configuration & handler setup
├── default.config           # Baseline project configuration template
├── default.secret           # Safe template for API credentials
├── pyproject.toml           # Project dependencies & packaging specification
├── uv.lock                  # Deterministic dependency lockfile
├── src/
│   ├── pipeline/            # End-to-end Jaywalking pipeline & decision logic
│   ├── perception/          # YOLO26x-Pose, SegFormer-B0, Qwen2.5-VL interfaces
│   └── utils/               # Metric computation, video extraction, Plotly plotting
├── scripts/                 # CLI inference & benchmark runner scripts
├── datasets/manifests/      # SHA-256 verified benchmark split manifests
├── results/                 # Complete benchmark summaries & per-video records
├── docs/                    # Architecture documentation & full project reports
└── experiments/             # Historical experimental archives & logs
```

---

## 10. License & Citation

This project is licensed under the MIT License.
