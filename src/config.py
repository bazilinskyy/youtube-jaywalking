import json
from pathlib import Path
from typing import Any, Dict

ROOT_DIR = Path(__file__).resolve().parents[1]
CONFIG_FILE = ROOT_DIR / "configs" / "config.json"


def load_config() -> Dict[str, Any]:
    """Loads JSON configuration from configs/config.json with path resolution."""
    if not CONFIG_FILE.exists():
        # Fallback to legacy config.json in root if needed
        legacy_path = ROOT_DIR / "config.json"
        if legacy_path.exists():
            with open(legacy_path) as f:
                return json.load(f)
        return {}

    with open(CONFIG_FILE) as f:
        config = json.load(f)
    return config


CONFIG = load_config()


def get_vlm_config() -> Dict[str, Any]:
    return CONFIG.get("vlm", {
        "provider": "ollama",
        "model": "qwen2.5vl:7b",
        "ollama_url": "http://localhost:11434/api/chat",
        "temperature": 0.0,
        "max_tokens": 10,
        "num_frames": 3,
        "jpeg_quality": 85,
        "timeout_seconds": 60,
    })


def get_cv_config() -> Dict[str, Any]:
    return CONFIG.get("cv", {
        "yolo_model": str(ROOT_DIR / "models" / "yolo11x.pt"),
        "pose_model": str(ROOT_DIR / "models" / "yolo11x-pose.pt"),
        "seg_model": str(ROOT_DIR / "models" / "yolo11x-seg.pt"),
        "confidence_threshold": 0.5,
    })


def get_paths() -> Dict[str, Path]:
    paths = CONFIG.get("paths", {})
    return {
        "root": ROOT_DIR,
        "ground_truth": ROOT_DIR / paths.get("ground_truth", "data/ground_truth.csv"),
        "videos_dir": ROOT_DIR / paths.get("videos_dir", "data/raw_clips"),
        "output_dir": ROOT_DIR / paths.get("output_dir", "outputs"),
        "models_dir": ROOT_DIR / "models",
    }
