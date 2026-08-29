# Scientific Benchmark & Generalization Protocol

---

## 1. Dataset Overview and Stratification

The benchmark consists of 99 labeled video sequences from the JAAD Pedestrian Dataset:
- **Total Valid Labeled Videos:** 99
- **Total Jaywalking Videos:** 36 (36.36%)
- **Total Compliant Videos:** 63 (63.64%)

To guarantee unbiased out-of-distribution generalization, a **stratified split** was created using fixed random seed `42`:

| Partition | Filename | Video Count | Jaywalking | Compliant | Purpose |
|---|---|---:|---:|---:|---|
| **Development Set (~70%)** | [`datasets/manifests/development_manifest.csv`](datasets/manifests/development_manifest.csv) | **69** | 25 (36.23%) | 44 (63.77%) | Architecture tuning & ablation studies |
| **Locked Test Set (~30%)** | [`datasets/manifests/locked_test_manifest.csv`](datasets/manifests/locked_test_manifest.csv) | **30** | 11 (36.67%) | 19 (63.33%) | Single unbiased final evaluation |

### Manifest Checksums (SHA-256)
- **Development Manifest:** `fd3cd23f81fe6ca0a72295ab974ea95ddb5bfbb029e8e369ba7547b2ba553723`
- **Locked Test Manifest:** `0ba8541a9ba09dfaa03fa130064be2bc5d7024a6b7f4dc9bbb8e38ee4ae07269`
- **Partition Overlap:** Exactly 0 overlapping videos.

---

## 2. Strict Research Governance Rules

1. **Test Set Sequestration:** The locked test set was never opened, loaded, inspected, or evaluated during model development (Experiments 53–57).
2. **Generic Mechanisms Only:** Prohibited any rules, branching, or thresholds tuned to individual video filenames or IDs.
3. **Reproducibility Locking:** Every ablation was required to evaluate all 69 development videos with full transition matrices (measuring both recoveries and regressions).
4. **Single Final Benchmark:** The locked test set was evaluated strictly once on the frozen Exp57 architecture (as Experiment 58) with zero post-hoc parameter adjustments.

---

## 3. Evaluation Metrics

All experiments report:
- **Accuracy:** $(TP + TN) / N$
- **Precision:** $TP / (TP + FP)$
- **Recall (Sensitivity):** $TP / (TP + FN)$
- **Specificity:** $TN / (TN + FP)$
- **F1 Score:** $2 \cdot (\text{Precision} \cdot \text{Recall}) / (\text{Precision} + \text{Recall})$
- **Confusion Matrix:** $TP, TN, FP, FN$ counts.
