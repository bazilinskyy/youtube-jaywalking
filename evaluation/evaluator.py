import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from evaluation.metrics import compute_metrics
from src.config import get_paths
from src.data_loader import load_ground_truth_records


class Evaluator:
    """Evaluates a jaywalking detection pipeline against canonical ground truth."""

    def __init__(
        self,
        pipeline: Any,
        ground_truth_path: Optional[Union[str, Path]] = None,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        self.pipeline = pipeline
        paths = get_paths()
        self.gt_path = Path(ground_truth_path) if ground_truth_path else paths["ground_truth"]
        self.output_dir = Path(output_dir) if output_dir else paths["output_dir"]
        self.pred_dir = self.output_dir / "predictions"
        self.metrics_dir = self.output_dir / "metrics"
        self.pred_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

    def run_evaluation(
        self, limit: Optional[int] = None, only_evaluable: bool = True
    ) -> Dict[str, Any]:
        """Runs batch evaluation on ground truth dataset."""
        records = load_ground_truth_records(self.gt_path, only_evaluable=only_evaluable)
        if limit:
            records = records[:limit]

        print("\n==================================================")
        print(f"Starting Evaluation on {len(records)} clips")
        print(f"Ground Truth Source: {self.gt_path}")
        print(f"Pipeline: {self.pipeline.__class__.__name__}")
        print("==================================================\n")

        results: List[Dict[str, Any]] = []
        t0 = time.time()

        for idx, rec in enumerate(records, start=1):
            clip_name = rec["clip_name"]
            gt = rec["ground_truth"]
            video_path = rec["video_path"]

            print(f"[{idx}/{len(records)}] Evaluating {clip_name} (GT={gt})...", end=" ", flush=True)

            if not Path(video_path).exists():
                print("SKIPPED (Video file not found)")
                continue

            try:
                pred_out = self.pipeline.predict(video_path)
                pred = pred_out.get("prediction", "unknown").lower()
                conf = pred_out.get("confidence", "unknown")
                reason = pred_out.get("reason", "")
                elapsed = pred_out.get("elapsed_seconds", 0.0)
            except Exception as e:
                pred = "error"
                conf = "none"
                reason = f"Exception during prediction: {e}"
                elapsed = 0.0

            # Match status
            is_match = (pred == gt) if gt in ("jaywalking", "compliant") else None
            if is_match:
                failure_type = "CORRECT"
            elif gt == "compliant" and pred == "jaywalking":
                failure_type = "FP"
            elif gt == "jaywalking" and pred == "compliant":
                failure_type = "FN"
            else:
                failure_type = "OTHER"

            status_icon = "✓" if is_match else ("✗" if is_match is False else "-")
            print(f"Pred={pred:<11} [{status_icon}] ({elapsed:.1f}s)")

            result_entry = {
                "clip_id": rec["clip_id"],
                "clip_name": clip_name,
                "ground_truth": gt,
                "prediction": pred,
                "is_correct": is_match,
                "failure_type": failure_type,
                "confidence": conf,
                "reason": reason,
                "elapsed_seconds": elapsed,
                "video_path": video_path,
                "notes": rec["notes"],
            }
            results.append(result_entry)

        total_elapsed = time.time() - t0
        evaluable_subset = [r for r in results if r["ground_truth"] in ("jaywalking", "compliant")]
        metrics = compute_metrics(evaluable_subset)
        metrics["total_wall_time_seconds"] = round(total_elapsed, 2)
        metrics["avg_time_per_clip_seconds"] = round(total_elapsed / max(len(results), 1), 2)

        # Print report
        self._print_report(results, metrics)

        # Save files
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        pred_file = self.pred_dir / f"predictions_{timestamp}.csv"
        metrics_file = self.metrics_dir / f"metrics_{timestamp}.json"

        self._save_results(results, metrics, pred_file, metrics_file)

        return {
            "results": results,
            "metrics": metrics,
            "predictions_file": str(pred_file),
            "metrics_file": str(metrics_file),
        }

    def _print_report(self, results: List[Dict[str, Any]], metrics: Dict[str, Any]) -> None:
        print(f"\n{'='*75}")
        print(f"{'EVALUATION RESULTS SUMMARY':^75}")
        print(f"{'='*75}")
        print(f"{'Clip Name':<18} | {'GT':<11} | {'Pred':<11} | {'Status':<7} | {'Time':<6} | {'Reason'}")
        print(f"{'-'*75}")
        for r in results:
            status = "PASS" if r["is_correct"] else ("FAIL" if r["is_correct"] is False else "N/A")
            print(
                f"{r['clip_name']:<18} | {r['ground_truth']:<11} | {r['prediction']:<11} | "
                f"{status:<7} | {r['elapsed_seconds']:>4.1f}s | {r['reason'][:30]}"
            )

        print(f"{'='*75}")
        print(f"Accuracy:        {metrics['accuracy']}% ({metrics['correct']}/{metrics['total_evaluated']})")
        print(f"Precision:       {metrics['precision']}%")
        print(f"Recall:          {metrics['recall']}%")
        print(f"Specificity:     {metrics['specificity']}%")
        print(f"F1 Score:        {metrics['f1_score']}%")
        print(f"False Pos Rate:  {metrics['false_positive_rate']}% (FP={metrics['fp']})")
        print(f"False Neg Rate:  {metrics['false_negative_rate']}% (FN={metrics['fn']})")
        print(f"Confusion Matrix: TP={metrics['tp']}, TN={metrics['tn']}, FP={metrics['fp']}, FN={metrics['fn']}")
        print(f"Total Time:      {metrics['total_wall_time_seconds']}s ({metrics['avg_time_per_clip_seconds']}s/clip)")
        print(f"{'='*75}\n")

    def _save_results(
        self,
        results: List[Dict[str, Any]],
        metrics: Dict[str, Any],
        pred_file: Path,
        metrics_file: Path,
    ) -> None:
        if results:
            keys = list(results[0].keys())
            with open(pred_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(results)

        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        # Also update latest files for convenience
        latest_pred = self.pred_dir / "latest_predictions.csv"
        latest_metrics = self.metrics_dir / "latest_metrics.json"
        if results:
            with open(latest_pred, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(results)
        with open(latest_metrics, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
