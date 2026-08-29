# Crowd-Jaywalking: Multimodal Context-Aware Jaywalking Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

An end-to-end multimodal perception and reasoning framework for detecting pedestrian jaywalking in real-world urban driving video sequences.

---

## 1. Problem Definition

Jaywalking detection requires understanding whether a pedestrian crossing a roadway is doing so lawfully (at a designated marked crosswalk, signalized intersection, or yielding junction) or unlawfully (unregulated mid-block crossing, crossing against a red signal).

### Why It Is Difficult:
- **Severe Occlusions & Small Scale:** Pedestrians stepping out from behind parked vehicles or crossing at long distances ($>40\text{ m}$).
- **Degraded Visibility:** Nighttime low illumination, snow-covered asphalt, and headlight glare.
- **Crosswalk Striping Ambiguity:** Faded or snow-covered zebra markings where legal crossing infrastructure is partially obscured.
- **Shared Urban Spaces:** Curbless cobblestone streets, private parking lots, and garage ramps where drivable road segmentation erroneously spans building-to-building.
- **Camera Ego-Motion:** Distinguishing camera vehicle motion from true pedestrian transverse translation.

---

## 2. Final System Architecture

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

## 3. Research Evolution Milestones

| Milestone | Architecture / Innovation | Key Finding & Contribution |
|---|---|---|
| **Exp 42** | Segmentation + VLM Baseline | Established base pipeline combining SegFormer road segmentation with VLM zero-shot voting. |
| **Exp 50** | Specialist Routing | Introduced failure-aware semantic routing to resolve off-road false alarms. |
| **Exp 52** | Diagonal Trajectory Recovery | Integrated BoT-SORT trajectory tracking to recover diagonal crossers ($89.74\%$ on canonical suite). |
| **Exp 53** | Multi-Temporal Road Verification | Sampled road surface overlap across multiple temporal phases ($[25\%, 50\%, 75\%]$), recovering false dropouts ($81.16\%$). |
| **Exp 55** | Context-Aware Verification | Introduced wide-scene Crosswalk and Shared-Street context verifiers to eliminate False Positives ($85.51\%$). |
| **Exp 56** | Tracker-Independent Persistence | Persisted unanimous 3/3 VLM votes across public roadways to recover tracker dropout FNs ($89.86\%$). |
| **Exp 57** | Refined Context Synergy Architecture | Refined residential through-street connectivity and intersection junction verification, achieving **92.75% Accuracy and 100.0% Recall** on the development set. |
| **Exp 58** | Final Locked Unseen Evaluation | Single zero-leakage evaluation on 30 sequestered test videos: **83.33% Accuracy**, $81.82\%$ Recall, $84.21\%$ Specificity. |

---

## 4. Benchmark Protocol & Stratification

The benchmark consists of 99 labeled videos from the JAAD Pedestrian Dataset, partitioned using a stratified split (fixed random seed `42`):
- **Development Set (69 videos, ~70%):** Used exclusively for iterative hypothesis testing and ablation studies.
- **Locked Test Set (30 videos, ~30%):** Kept 100% sequestered and uninspected throughout development. Evaluated strictly once as Experiment 58.

---

## 5. Final Benchmark Results

| Benchmark Partition | Videos | Accuracy | Precision | Recall | Specificity | F1 Score | TP | TN | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Canonical JAAD (Exp 52)** | 39 | **89.74%** | 100.0% | 73.33% | 100.0% | 84.61% | 11 | 24 | 0 | 4 |
| **Development Set (Exp 57)** | 69 | **92.75%** | 83.33% | 100.0% | 88.64% | 90.91% | 25 | 39 | 5 | 0 |
| **Locked Test Set (Exp 58)** | 30 | **83.33%** | 75.00% | 81.82% | 84.21% | 78.26% | 9 | 16 | 3 | 2 |

> **Scientific Evaluation Statement:**  
> The final frozen architecture achieved 83.33% accuracy on a completely unseen locked test set. Performance decreased by 9.42 percentage points relative to the development set, indicating a measurable generalization gap. However, the model maintained balanced recall and specificity on unseen videos.

---

## 6. Installation & Quick Start

### Installation
```bash
# 1. Clone repository
git clone https://github.com/your-username/crowd-jaywalking.git
cd crowd-jaywalking

# 2. Create and activate virtual environment
python3 -m venv myenv
source myenv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Ensure local Ollama VLM daemon is running
ollama run qwen2.5vl:7b
```

### Run Single-Video Inference
```bash
python3 scripts/run_inference.py --video videos/video_0001.mp4
```

### Run Benchmark Evaluations
```bash
# Evaluate Canonical 39-Video Benchmark
python3 scripts/run_canonical_evaluation.py

# Evaluate Final Locked 30-Video Unseen Test Benchmark
python3 scripts/run_locked_evaluation.py
```

---

## 7. Repository Structure

```text
crowd-jaywalking/
├── src/
│   ├── pipeline/
│   │   ├── jaywalking_pipeline.py    # Unified end-to-end detection pipeline
│   │   ├── frame_sampler.py          # Multi-temporal keyframe extractor
│   │   ├── decision_engine.py        # Production decision rules
│   │   └── context_router.py         # Wide-scene context verifiers
│   ├── perception/
│   │   ├── vlm_classifier.py         # Qwen2.5-VL-7B client interface
│   │   ├── pedestrian_tracking.py    # YOLO26x-Pose + BoT-SORT tracker
│   │   └── road_segmentation.py      # SegFormer-B0 Cityscapes segmenter
│   └── utils/
│       ├── metrics.py                # Classification metrics calculator
│       └── video_utils.py            # Video decoding and encoding helpers
├── scripts/
│   ├── run_inference.py              # CLI for single-video inference
│   ├── run_canonical_evaluation.py   # Benchmark runner for 39 canonical videos
│   └── run_locked_evaluation.py      # Benchmark runner for 30 locked test videos
├── datasets/
│   └── manifests/
│       ├── development_manifest.csv  # 69-video dev set manifest (SHA-256 locked)
│       └── locked_test_manifest.csv  # 30-video locked test manifest (SHA-256 locked)
├── results/
│   ├── canonical/                    # Canonical benchmark artifacts
│   ├── development/                  # Development benchmark artifacts
│   └── locked_test/                  # Final locked test artifacts
├── docs/
│   ├── ARCHITECTURE.md               # Detailed system design
│   ├── BENCHMARK_PROTOCOL.md         # Scientific splitting protocol
│   └── PROJECT_REPORT.md             # Complete research report
├── README.md
├── requirements.txt
└── RESEARCH_LOG.md
```

---

## 8. License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
