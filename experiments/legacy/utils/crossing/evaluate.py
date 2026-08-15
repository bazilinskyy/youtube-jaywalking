import pandas as pd
import numpy as np
from typing import Dict, Tuple, List

LABELED_PATH = "validation/labeled_crossings.csv"


def compute_per_class_metrics(labels: pd.DataFrame, predictions: pd.DataFrame) -> Dict:
    if len(labels) == 0:
        return {"error": "no labels"}
    merged = labels.merge(predictions, on="clip_id", how="inner", suffixes=("_gt", "_pred"))
    violation_types = ["SIGNAL_VIOLATION", "NO_CROSSWALK"]
    results = {}
    for vtype in violation_types:
        tp = ((merged["violation_type_pred"] == vtype) & (merged["ground_truth_violation"] == "yes")).sum()
        fp = ((merged["violation_type_pred"] == vtype) & (merged["ground_truth_violation"] != "yes")).sum()
        fn = ((merged["violation_type_pred"] != vtype) & (merged["ground_truth_violation"] == "yes")).sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        results[vtype] = {"precision": precision, "recall": recall, "f1": f1, "tp": int(tp), "fp": int(fp), "fn": int(fn)}
    return results


def classify_error(row_gt: dict, row_pred: dict) -> str:
    if row_gt["ground_truth_violation"] == row_pred["violation"]:
        return "correct"
    if row_gt["ground_truth_reason"] == "signal_violation" and row_pred.get("light_was_wrong"):
        return "perception_failure"
    return "fusion_logic_failure"


def error_analysis(labels: pd.DataFrame, predictions: pd.DataFrame) -> Dict:
    merged = labels.merge(predictions, on="clip_id", how="inner", suffixes=("_gt", "_pred"))
    errors = {"perception_failure": 0, "fusion_logic_failure": 0, "correct": 0}
    patterns = []
    for _, row in merged.iterrows():
        gt = {"ground_truth_violation": row["ground_truth_violation"], "ground_truth_reason": row.get("ground_truth_reason", "")}
        pred = {"violation": row.get("violation", False), "light_was_wrong": row.get("light_was_wrong", False)}
        category = classify_error(gt, pred)
        errors[category] = errors.get(category, 0) + 1
        if category != "correct":
            patterns.append(f"clip {row['clip_id']}: gt={gt['ground_truth_violation']} pred={pred['violation']}")
    errors["top_patterns"] = patterns[:5]
    return errors
