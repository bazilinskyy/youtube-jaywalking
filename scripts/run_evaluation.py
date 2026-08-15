#!/usr/bin/env python3
"""
CLI entry point to evaluate the jaywalking detection system against canonical ground truth.
Usage:
    python scripts/run_evaluation.py
    python scripts/run_evaluation.py --mode cv
    python scripts/run_evaluation.py --mode ensemble
    python scripts/run_evaluation.py --limit 10
"""
import argparse
import sys
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from evaluation.evaluator import Evaluator
from src.pipeline import get_pipeline


def main():
    parser = argparse.ArgumentParser(description="Evaluate Jaywalking Detection Pipeline against Ground Truth")
    parser.add_argument(
        "--mode",
        choices=["vlm", "balanced", "high_precision", "safety", "high_recall", "cv", "ensemble"],
        default="balanced",
        help="Pipeline mode ('balanced', 'high_precision', 'safety', 'vlm', 'cv', 'ensemble')",
    )
    parser.add_argument("--min-votes", type=int, choices=[1, 2, 3], default=None, help="Explicit vote threshold: min frames required for jaywalking (1=safety, 2=balanced, 3=high_precision)")
    parser.add_argument("--prompt", type=str, default="canonical", help="VLM prompt preset ('canonical', 'v2', 'temporal', 'temporal_motion', 'v4b', 'right_of_way')")
    parser.add_argument("--boundary-context", action="store_true", help="Inject Road Boundary and Pedestrian spatial context into VLM prompt")
    parser.add_argument("--pedestrian-motion", action="store_true", help="Inject structured Pedestrian Motion features (tracking, displacement, direction) into VLM prompt")
    parser.add_argument("--vehicle-context", action="store_true", help="Inject structured Vehicle Interaction & Ego-Motion context into VLM prompt")
    parser.add_argument("--gt", type=str, default=None, help="Path to ground truth CSV (default: data/ground_truth.csv)")
    parser.add_argument("--limit", type=int, default=None, help="Limit evaluation to first N clips")
    parser.add_argument("--all", action="store_true", help="Include unreviewed / unlabeled clips")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save evaluation results (default: outputs/)")
    args = parser.parse_args()

    pipeline = get_pipeline(
        args.mode,
        prompt_name=args.prompt,
        use_boundary_context=args.boundary_context,
        use_pedestrian_motion=args.pedestrian_motion,
        use_vehicle_context=args.vehicle_context,
        min_votes=args.min_votes,
    )



    evaluator = Evaluator(
        pipeline=pipeline,
        ground_truth_path=args.gt,
        output_dir=args.output_dir,
    )

    evaluator.run_evaluation(
        limit=args.limit,
        only_evaluable=not args.all,
    )


if __name__ == "__main__":
    main()
