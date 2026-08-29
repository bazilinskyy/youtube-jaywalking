"""
Metrics and evaluation helper utilities.
"""

from typing import Dict, List


def calculate_classification_metrics(
    y_true: List[str],
    y_pred: List[str],
    pos_label: str = "JAYWALKING",
    neg_label: str = "COMPLIANT",
) -> Dict[str, float]:
    """
    Computes standard binary classification metrics:
      - Accuracy
      - Precision
      - Recall
      - Specificity
      - F1 Score
      - TP, TN, FP, FN counts
    """
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == pos_label and yp == pos_label)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == neg_label and yp == neg_label)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == neg_label and yp == pos_label)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == pos_label and yp == neg_label)

    total = len(y_true)
    acc = (tp + tn) / max(1, total) * 100.0
    prec = tp / max(1, tp + fp) * 100.0 if (tp + fp) > 0 else 0.0
    rec = tp / max(1, tp + fn) * 100.0 if (tp + fn) > 0 else 0.0
    spec = tn / max(1, tn + fp) * 100.0 if (tn + fp) > 0 else 0.0
    f1 = 2 * prec * rec / max(1e-6, prec + rec) if (prec + rec) > 0 else 0.0

    return {
        "accuracy": round(acc, 2),
        "precision": round(prec, 2),
        "recall": round(rec, 2),
        "specificity": round(spec, 2),
        "f1_score": round(f1, 2),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "total": total,
    }
