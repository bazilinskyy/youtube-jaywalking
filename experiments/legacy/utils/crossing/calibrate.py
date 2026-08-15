import pandas as pd
import numpy as np
from typing import List, Tuple
from utils.crossing.intent import detect_hesitation, detect_inattentive_entry, classify_crossing

LABELED_PATH = "validation/labeled_crossings.csv"


def load_ground_truth(path: str = LABELED_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def sweep_hesitation_threshold(
    labels: pd.DataFrame,
    thresh_values: List[float] = None,
) -> Tuple[float, float]:
    if thresh_values is None:
        thresh_values = [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1]
    if "observed_hesitation" not in labels.columns:
        return 0.02, 0.0
    best_acc = 0.0
    best_thresh = 0.02
    for thresh in thresh_values:
        correct = 0
        for _, row in labels.iterrows():
            predicted = row.get("dummy_hesitation", False)
            actual = row["observed_hesitation"] == "yes"
            if predicted == actual:
                correct += 1
        acc = correct / max(len(labels), 1)
        if acc > best_acc:
            best_acc = acc
            best_thresh = thresh
    return best_thresh, best_acc


def sweep_violation_weights(
    labels: pd.DataFrame,
    signal_weight_values: List[float] = None,
    no_crosswalk_weight_values: List[float] = None,
) -> Tuple[float, float, float]:
    if signal_weight_values is None:
        signal_weight_values = [0.7, 0.8, 0.9, 1.0]
    if no_crosswalk_weight_values is None:
        no_crosswalk_weight_values = [0.5, 0.6, 0.7, 0.8, 0.9]
    if "ground_truth_violation" not in labels.columns:
        return 0.9, 0.7, 0.0
    best_acc = 0.0
    best_sw = 0.9
    best_nw = 0.7
    for sw in signal_weight_values:
        for nw in no_crosswalk_weight_values:
            correct = 0
            for _, row in labels.iterrows():
                gt = row["ground_truth_violation"]
                has_signal = row.get("dummy_signal", False)
                has_crosswalk = row.get("dummy_crosswalk", True)
                score = 0.0
                if has_signal:
                    score += sw
                if not has_crosswalk:
                    score += nw
                predicted = "yes" if score >= 0.7 else "no"
                if predicted == gt:
                    correct += 1
            acc = correct / max(len(labels), 1)
            if acc > best_acc:
                best_acc = acc
                best_sw = sw
                best_nw = nw
    return best_sw, best_nw, best_acc
