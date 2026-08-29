# Comprehensive Research Project Report

---

## 1. Problem Definition & Challenges

Detecting jaywalking in real-world urban driving environments is a challenging task in intelligent transportation and autonomous driving perception. Key challenges include:
- **Severe Occlusions & Small Scale:** Pedestrians entering roadways from behind parked delivery vans, trees, or distant sidewalks ($>40\text{ m}$).
- **Degraded Visual Conditions:** Nighttime low illumination, snow-covered pavement, and headlight glare.
- **Crosswalk Striping Ambiguity:** Faded paint markings, snow concealment, and unmarked suburban intersections where pedestrians cross legally.
- **Shared Urban Topologies:** Narrow curbless cobblestone alleys, parking garage ramps, and gas station aprons where drivable road segmentation erroneously spans building-to-building.

---

## 2. Research Evolution & Key Milestones

| Milestone | Architecture / Innovation | Key Finding & Contribution |
|---|---|---|
| **Exp 42** | Single-Frame SegFormer + VLM Baseline | Established multimodal baseline combining Cityscapes road segmentation with VLM zero-shot voting. |
| **Exp 50** | Specialist Routing ($p_{\text{sem}}$) | Introduced failure-aware routing to resolve off-road false alarms. |
| **Exp 52** | Diagonal Trajectory & Kinematic Tracking | Integrated BoT-SORT trajectory tracking to recover diagonal crossers ($89.74\%$ on canonical suite). |
| **Exp 53** | Multi-Temporal Road Surface Integration | Replaced single-frame static masks with multi-temporal temporal sampling ($[25\%, 50\%, 75\%]$), recovering false off-road dropouts ($81.16\%$ on 69 dev videos). |
| **Exp 55** | Dual Context Verification | Introduced wide-scene Crosswalk and Shared-Street context verifiers to eliminate False Positives ($85.51\%$). |
| **Exp 56** | Tracker-Independent Persistence | Persisted unanimous 3/3 VLM votes across public roadways to recover tracker dropout FNs ($89.86\%$). |
| **Exp 57** | Refined Context Synergy Architecture | Refined residential through-street connectivity and intersection junction verification, achieving **92.75% Accuracy and 100.0% Recall** on the development set. |
| **Exp 58** | Final Locked Unseen Evaluation | Single zero-leakage evaluation on 30 sequestered test videos: **83.33% Accuracy**, $81.82\%$ Recall, $84.21\%$ Specificity. |

---

## 3. Final Benchmark Results

| Benchmark Partition | Video Count | Accuracy | Precision | Recall | Specificity | F1 Score | TP | TN | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Canonical JAAD (Exp 52)** | 39 | **89.74%** | 100.0% | 73.33% | 100.0% | 84.61% | 11 | 24 | 0 | 4 |
| **Development Set (Exp 57)** | 69 | **92.75%** | 83.33% | 100.0% | 88.64% | 90.91% | 25 | 39 | 5 | 0 |
| **Locked Test Set (Exp 58)** | 30 | **83.33%** | 75.00% | 81.82% | 84.21% | 78.26% | 9 | 16 | 3 | 2 |

> **Scientific Evaluation Statement:**
> The final frozen architecture achieved 83.33% accuracy on a completely unseen locked test set. Performance decreased by 9.42 percentage points relative to the development set, indicating a measurable generalization gap. However, the model maintained balanced recall and specificity on unseen videos.

---

## 4. Limitations & Future Work

1. **Long-Distance / Small-Scale Targets:** Pedestrians crossing at extreme distances ($>45\text{ m}$) under low contrast remain susceptible to split frame voting.
2. **Dense Occlusion Emergence:** Pedestrians stepping out suddenly between large parked vehicles require multi-stage temporal reasoning before entering the road.
3. **End-to-End Multimodal Representation Learning:** Future iterations can explore joint fine-tuning of vision-language tokens directly on temporal video bounding cylinders.
