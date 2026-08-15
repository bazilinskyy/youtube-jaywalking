from typing import Any, Dict, List


def compute_metrics(
    eval_records: List[Dict[str, Any]], pred_key: str = "prediction", gt_key: str = "ground_truth"
) -> Dict[str, Any]:
    """
    Computes comprehensive binary classification metrics.
    Positive class: 'jaywalking'
    Negative class: 'compliant'
    """
    tp = sum(1 for r in eval_records if r[gt_key] == "jaywalking" and r[pred_key] == "jaywalking")
    tn = sum(1 for r in eval_records if r[gt_key] == "compliant" and r[pred_key] == "compliant")
    fp = sum(1 for r in eval_records if r[gt_key] == "compliant" and r[pred_key] == "jaywalking")
    fn = sum(1 for r in eval_records if r[gt_key] == "jaywalking" and r[pred_key] == "compliant")
    unknown = sum(1 for r in eval_records if r[pred_key] not in ("jaywalking", "compliant"))

    valid_total = tp + tn + fp + fn
    correct = tp + tn

    accuracy = (correct / valid_total * 100.0) if valid_total > 0 else 0.0
    precision = (tp / (tp + fp) * 100.0) if (tp + fp) > 0 else 0.0
    recall = (tp / (tp + fn) * 100.0) if (tp + fn) > 0 else 0.0
    specificity = (tn / (tn + fp) * 100.0) if (tn + fp) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    fpr = (fp / (fp + tn) * 100.0) if (fp + tn) > 0 else 0.0
    fnr = (fn / (fn + tp) * 100.0) if (fn + tp) > 0 else 0.0

    return {
        "accuracy": round(accuracy, 2),
        "precision": round(precision, 2),
        "recall": round(recall, 2),
        "specificity": round(specificity, 2),
        "f1_score": round(f1, 2),
        "false_positive_rate": round(fpr, 2),
        "false_negative_rate": round(fnr, 2),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "unknown_predictions": unknown,
        "correct": correct,
        "total_evaluated": valid_total,
        "confusion_matrix": {
            "predicted_jaywalking": {"actual_jaywalking": tp, "actual_compliant": fp},
            "predicted_compliant": {"actual_jaywalking": fn, "actual_compliant": tn},
        },
    }
