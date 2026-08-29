import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.config import get_paths


def load_ground_truth_records(
    csv_path: Optional[Union[str, Path]] = None,
    only_evaluable: bool = True,
) -> List[Dict[str, Any]]:
    """
    Loads canonical ground truth records and maps them to local video paths.

    Args:
        csv_path: Path to ground truth CSV. Defaults to path from config (data/ground_truth.csv).
        only_evaluable: If True, filters only records marked with is_evaluated=True.

    Returns:
        List of dicts representing each dataset record.
    """
    paths = get_paths()
    path = Path(csv_path) if csv_path else paths["ground_truth"]
    videos_dir = paths["videos_dir"]

    if not path.exists():
        raise FileNotFoundError(f"Ground truth file not found at: {path}")

    records: List[Dict[str, Any]] = []
    with open(path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            is_eval = row.get("is_evaluated", "False").strip().lower() == "true"
            if only_evaluable and not is_eval:
                continue

            clip_name = row.get("clip_name", "").strip()
            video_path = Path(row.get("video_path", ""))

            # Resolve video path if not existing as recorded
            if not video_path.exists():
                candidate = paths["root"] / video_path
                if candidate.exists():
                    video_path = candidate
                else:
                    candidate_dir = videos_dir / clip_name
                    if candidate_dir.exists():
                        video_path = candidate_dir

            records.append({
                "clip_id": row.get("clip_id", "").strip(),
                "clip_name": clip_name,
                "ground_truth": row.get("ground_truth", "unlabeled").strip().lower(),
                "raw_label": row.get("raw_label", "").strip(),
                "is_evaluated": is_eval,
                "notes": row.get("notes", "").strip(),
                "video_path": str(video_path),
                "video_exists": Path(video_path).exists(),
            })

    return records
