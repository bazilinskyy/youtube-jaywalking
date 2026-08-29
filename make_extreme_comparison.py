#!/usr/bin/env python3
"""
Create a side by side video of two extreme pedestrian behaviour examples:

1. Jaywalking pedestrian with the highest early event body movement.
2. Non jaywalking pedestrian with the lowest early event body movement.

The selection is based on the first EARLY_EVENT_PERCENT of each labelled
jaywalking event using the responsible pedestrian identified in
`data/results_summary.csv`.

Required inputs:
    data/body_movement_event_frame_level.csv
    data/results_summary.csv
    data/per_video/<video_id>_keypoints.json
    source video files referenced by the keypoint JSON or available in the repo

Outputs:
    data/extreme_behaviour_examples.csv
    figures/body_movement/extreme_jay_vs_nonjay_side_by_side.mp4

The output video contains a joint title card and then shows the selected
Jaywalking and Non jaywalking examples side by side, with metadata and event
progress overlaid on each half.
"""

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


# -------------------------------------------------------------------------
# PATHS
# -------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
KEYPOINT_DIR = DATA_DIR / "per_video"
FIGURES_DIR = PROJECT_ROOT / "figures" / "body_movement"

FRAME_LEVEL_PATH = DATA_DIR / "body_movement_event_frame_level.csv"
RESULTS_SUMMARY_PATH = DATA_DIR / "results_summary.csv"

SELECTION_OUTPUT = DATA_DIR / "extreme_behaviour_examples.csv"
VIDEO_OUTPUT = FIGURES_DIR / "extreme_jay_vs_nonjay_side_by_side.mp4"


# -------------------------------------------------------------------------
# SETTINGS
# -------------------------------------------------------------------------

# "Starting" behaviour is defined as the first 20% of the labelled event.
EARLY_EVENT_PERCENT = 20.0

# Use the smoothed pose measure when available.
PREFERRED_MOTION_COLUMN = "pose_motion_smoothed"
FALLBACK_MOTION_COLUMN = "pose_motion_speed"

# Small amount of visual context around the event window.
PADDING_SECONDS = 0.50

# Standard output settings for the comparison clip.
OUTPUT_WIDTH = 1280
OUTPUT_HEIGHT = 720
OUTPUT_FPS = 30.0
TITLE_CARD_SECONDS = 1.75

# Video codec used by OpenCV.
VIDEO_CODEC = "mp4v"

# Side by side layout.
PANE_GAP = 8
PANE_WIDTH = (OUTPUT_WIDTH - PANE_GAP) // 2
PANE_HEIGHT = OUTPUT_HEIGHT


# -------------------------------------------------------------------------
# LABEL HELPERS
# -------------------------------------------------------------------------


def normalise_video_id(video_name):
    """Convert video_0003.mp4 to video_0003."""

    return Path(str(video_name)).stem


def ground_truth_to_behaviour(value):
    """Map project labels into Jaywalking / Non jaywalking."""

    label = str(value).strip().upper()

    if label == "COMPLIANT":
        return "Non jaywalking"

    if label in {
        "JAYWALKING",
        "JAYWALK",
        "NON_COMPLIANT",
        "NON-COMPLIANT",
        "NONCOMPLIANT",
    }:
        return "Jaywalking"

    return str(value).strip()


# -------------------------------------------------------------------------
# EXTREME CASE SELECTION
# -------------------------------------------------------------------------


def load_analysis_inputs():
    """Load frame level body movement and event labels."""

    if not FRAME_LEVEL_PATH.exists():
        raise FileNotFoundError(
            f"Frame level analysis not found: {FRAME_LEVEL_PATH}\n"
            "Run analysis.py first."
        )

    if not RESULTS_SUMMARY_PATH.exists():
        raise FileNotFoundError(
            f"Results summary not found: {RESULTS_SUMMARY_PATH}"
        )

    frame_data = pd.read_csv(FRAME_LEVEL_PATH)
    labels = pd.read_csv(RESULTS_SUMMARY_PATH)

    required_frame_columns = {
        "video_id",
        "track_id",
        "frame_id",
        "event_progress_percent",
    }

    missing_frame_columns = required_frame_columns.difference(
        frame_data.columns
    )

    if missing_frame_columns:
        raise ValueError(
            "Frame level movement file is missing required columns: "
            f"{sorted(missing_frame_columns)}"
        )

    required_label_columns = {
        "video",
        "ground_truth",
        "responsible_track_id",
        "event_start",
        "event_end",
    }

    missing_label_columns = required_label_columns.difference(
        labels.columns
    )

    if missing_label_columns:
        raise ValueError(
            "results_summary.csv is missing required columns: "
            f"{sorted(missing_label_columns)}"
        )

    labels = labels.copy()
    labels["video_id"] = labels["video"].map(normalise_video_id)
    labels["track_id"] = pd.to_numeric(
        labels["responsible_track_id"],
        errors="coerce",
    )
    labels["event_start"] = pd.to_numeric(
        labels["event_start"],
        errors="coerce",
    )
    labels["event_end"] = pd.to_numeric(
        labels["event_end"],
        errors="coerce",
    )
    labels["behaviour"] = labels["ground_truth"].map(
        ground_truth_to_behaviour
    )

    labels = labels.dropna(
        subset=["track_id", "event_start", "event_end"]
    ).copy()
    labels["track_id"] = labels["track_id"].astype(int)

    return frame_data, labels


def choose_motion_column(frame_data):
    """Choose the best available frame level pose movement measure."""

    if PREFERRED_MOTION_COLUMN in frame_data.columns:
        return PREFERRED_MOTION_COLUMN

    if FALLBACK_MOTION_COLUMN in frame_data.columns:
        return FALLBACK_MOTION_COLUMN

    raise ValueError(
        "Could not find either pose_motion_smoothed or pose_motion_speed "
        "in the frame level analysis output."
    )


def calculate_early_movement(frame_data, labels):
    """
    Calculate one early movement score per responsible pedestrian.

    Early movement is the mean pose movement during the first
    EARLY_EVENT_PERCENT percent of the labelled event.
    """

    motion_column = choose_motion_column(frame_data)

    early_frames = frame_data[
        frame_data["event_progress_percent"].between(
            0.0,
            EARLY_EVENT_PERCENT,
            inclusive="both",
        )
    ].copy()

    early_frames[motion_column] = pd.to_numeric(
        early_frames[motion_column],
        errors="coerce",
    )

    early_frames = early_frames.dropna(subset=[motion_column])

    early_summary = (
        early_frames
        .groupby(["video_id", "track_id"], as_index=False)
        .agg(
            early_pose_motion=(motion_column, "mean"),
            early_pose_motion_median=(motion_column, "median"),
            early_motion_frames=(motion_column, "count"),
        )
    )

    labelled = labels.merge(
        early_summary,
        on=["video_id", "track_id"],
        how="inner",
        validate="one_to_one",
    )

    labelled["early_event_percent"] = EARLY_EVENT_PERCENT
    labelled["motion_measure"] = motion_column

    return labelled


def select_extreme_examples(labelled):
    """
    Select the most visually contrasting pair.

    Jaywalking:
        highest early pose movement.

    Non jaywalking:
        lowest early pose movement.
    """

    jaywalking = labelled[
        labelled["behaviour"] == "Jaywalking"
    ].dropna(subset=["early_pose_motion"])

    non_jaywalking = labelled[
        labelled["behaviour"] == "Non jaywalking"
    ].dropna(subset=["early_pose_motion"])

    if jaywalking.empty:
        raise ValueError(
            "No Jaywalking candidate with valid early pose movement was found."
        )

    if non_jaywalking.empty:
        raise ValueError(
            "No Non jaywalking candidate with valid early pose movement was found."
        )

    jay_example = jaywalking.loc[
        jaywalking["early_pose_motion"].idxmax()
    ].copy()

    non_jay_example = non_jaywalking.loc[
        non_jaywalking["early_pose_motion"].idxmin()
    ].copy()

    selected = pd.DataFrame(
        [jay_example, non_jay_example]
    ).reset_index(drop=True)

    selected["selection_rule"] = [
        "Highest early pose movement among Jaywalking candidates",
        "Lowest early pose movement among Non jaywalking candidates",
    ]

    return selected


# -------------------------------------------------------------------------
# SOURCE VIDEO LOOKUP
# -------------------------------------------------------------------------


def source_video_from_keypoint_json(video_id):
    """Try to recover the original video path from the keypoint JSON."""

    json_path = KEYPOINT_DIR / f"{video_id}_keypoints.json"

    if not json_path.exists():
        return None

    try:
        with open(json_path, "r", encoding="utf-8") as file:
            payload = json.load(file)
    except (OSError, json.JSONDecodeError):
        return None

    video_path = payload.get("video_path")

    if not video_path:
        return None

    candidate = Path(str(video_path)).expanduser()

    if candidate.is_absolute() and candidate.exists():
        return candidate

    relative_candidate = PROJECT_ROOT / candidate

    if relative_candidate.exists():
        return relative_candidate

    return None


def find_source_video(video_name, video_id):
    """Locate the original source MP4 for a selected event."""

    json_video_path = source_video_from_keypoint_json(video_id)

    if json_video_path is not None:
        return json_video_path

    common_candidates = [
        # External JAAD clips drive
        Path("/Volumes/Alam/JAAD_clips") / video_name,

        # Local project locations
        PROJECT_ROOT / video_name,
        PROJECT_ROOT / "videos" / video_name,
        DATA_DIR / video_name,
        DATA_DIR / "videos" / video_name,
        PROJECT_ROOT / "data" / "raw" / video_name,
    ]

    for candidate in common_candidates:
        if candidate.exists():
            return candidate

    search_roots = [
        PROJECT_ROOT,
        Path("/Volumes/Alam/JAAD_clips"),
    ]

    matches = []

    for root in search_roots:
        if root.exists():
            matches.extend(
                [
                    path
                    for path in root.rglob(video_name)
                    if path.is_file()
                ]
            )

    if matches:
        return matches[0]

    raise FileNotFoundError(
        f"Could not locate source video {video_name}. "
        "The script checked the keypoint JSON video_path, the configured "
        "JAAD clips folder, and common repository video folders."
    )


# -------------------------------------------------------------------------
# VIDEO RENDERING
# -------------------------------------------------------------------------


def fit_frame_to_canvas(frame, target_width, target_height):
    """Resize and letterbox a frame into the requested canvas."""

    source_height, source_width = frame.shape[:2]

    scale = min(
        target_width / source_width,
        target_height / source_height,
    )

    resized_width = max(1, int(round(source_width * scale)))
    resized_height = max(1, int(round(source_height * scale)))

    resized = cv2.resize(
        frame,
        (resized_width, resized_height),
        interpolation=cv2.INTER_AREA,
    )

    canvas = np.zeros(
        (target_height, target_width, 3),
        dtype=np.uint8,
    )

    x_offset = (target_width - resized_width) // 2
    y_offset = (target_height - resized_height) // 2

    canvas[
        y_offset:y_offset + resized_height,
        x_offset:x_offset + resized_width,
    ] = resized

    return canvas


def draw_text_with_background(
    frame,
    text,
    origin,
    font_scale=0.58,
    thickness=2,
):
    """Draw readable text with a dark background rectangle."""

    x, y = origin
    font = cv2.FONT_HERSHEY_SIMPLEX

    text_size, baseline = cv2.getTextSize(
        text,
        font,
        font_scale,
        thickness,
    )

    width, height = text_size

    cv2.rectangle(
        frame,
        (x - 8, y - height - 8),
        (x + width + 8, y + baseline + 8),
        (0, 0, 0),
        -1,
    )

    cv2.putText(
        frame,
        text,
        (x, y),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def annotate_pane(frame, example, progress_percent, side_label):
    """Overlay behaviour metadata and early movement score on one pane."""

    behaviour = str(example["behaviour"])
    video_id = str(example["video_id"])
    track_id = int(example["track_id"])
    score = float(example["early_pose_motion"])

    lines = [
        f"{side_label}: {behaviour}",
        f"{video_id} | Track {track_id}",
        f"Early pose motion: {score:.2f}",
        f"Event progress: {progress_percent:.0f}%",
    ]

    y = 34

    for line in lines:
        draw_text_with_background(
            frame,
            line,
            (18, y),
            font_scale=0.56,
            thickness=2,
        )
        y += 32

    bar_x = 18
    bar_y = PANE_HEIGHT - 34
    bar_width = PANE_WIDTH - 36
    bar_height = 12

    cv2.rectangle(
        frame,
        (bar_x, bar_y),
        (bar_x + bar_width, bar_y + bar_height),
        (90, 90, 90),
        2,
    )

    fill_width = int(
        np.clip(progress_percent / 100.0, 0.0, 1.0) * bar_width
    )

    cv2.rectangle(
        frame,
        (bar_x, bar_y),
        (bar_x + fill_width, bar_y + bar_height),
        (255, 255, 255),
        -1,
    )

    return frame


def create_side_by_side_title_card(left_example, right_example):
    """Create a single title card describing both selected examples."""

    frame = np.zeros(
        (OUTPUT_HEIGHT, OUTPUT_WIDTH, 3),
        dtype=np.uint8,
    )

    title = "EXTREME EARLY BODY MOVEMENT COMPARISON"
    subtitle = f"First {EARLY_EVENT_PERCENT:.0f}% of the labelled event"

    font = cv2.FONT_HERSHEY_SIMPLEX

    for index, line in enumerate([title, subtitle]):
        font_scale = 1.0 if index == 0 else 0.70
        thickness = 2
        text_size, _ = cv2.getTextSize(
            line,
            font,
            font_scale,
            thickness,
        )
        text_width = text_size[0]
        x = max(20, (OUTPUT_WIDTH - text_width) // 2)
        y = 90 + index * 45

        cv2.putText(
            frame,
            line,
            (x, y),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    left_x0 = 40
    left_x1 = OUTPUT_WIDTH // 2 - 20
    right_x0 = OUTPUT_WIDTH // 2 + 20
    right_x1 = OUTPUT_WIDTH - 40

    cv2.rectangle(frame, (left_x0, 180), (left_x1, 500), (255, 255, 255), 2)
    cv2.rectangle(frame, (right_x0, 180), (right_x1, 500), (255, 255, 255), 2)

    def draw_block(example, x0, header):
        lines = [
            header,
            str(example["behaviour"]).upper(),
            str(example["video_id"]),
            f"Track {int(example['track_id'])}",
            f"Mean early pose motion: {float(example['early_pose_motion']):.2f}",
        ]
        start_y = 235
        for idx, line in enumerate(lines):
            font_scale = 0.85 if idx == 0 else 0.72
            thickness = 2
            cv2.putText(
                frame,
                line,
                (x0 + 20, start_y + idx * 45),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )

    draw_block(left_example, left_x0, "LEFT PANEL")
    draw_block(right_example, right_x0, "RIGHT PANEL")

    footer = "Jaywalking and Non jaywalking clips are shown simultaneously."
    text_size, _ = cv2.getTextSize(footer, font, 0.65, 2)
    cv2.putText(
        frame,
        footer,
        ((OUTPUT_WIDTH - text_size[0]) // 2, 610),
        font,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return frame


def write_title_card(writer, left_example, right_example):
    """Write the side by side title card for TITLE_CARD_SECONDS."""

    frame = create_side_by_side_title_card(left_example, right_example)
    frame_count = int(round(TITLE_CARD_SECONDS * OUTPUT_FPS))

    for _ in range(frame_count):
        writer.write(frame)


def build_event_clip_frames(example, video_path, side_label):
    """
    Render one example into a list of annotated half-width frames.

    This preserves the real duration of the event clip while converting it to
    OUTPUT_FPS.
    """

    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise OSError(f"Could not open source video: {video_path}")

    source_fps = float(capture.get(cv2.CAP_PROP_FPS))

    if source_fps <= 0:
        source_fps = OUTPUT_FPS

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    event_start = int(round(float(example["event_start"])))
    event_end = int(round(float(example["event_end"])))

    padding_frames = int(round(PADDING_SECONDS * source_fps))

    clip_start = max(0, event_start - padding_frames)
    clip_end = min(total_frames - 1, event_end + padding_frames)

    if clip_end < clip_start:
        capture.release()
        raise ValueError(
            f"Invalid event frame interval for {example['video_id']}: "
            f"{clip_start} ... {clip_end}"
        )

    capture.set(cv2.CAP_PROP_POS_FRAMES, clip_start)

    source_index = clip_start
    output_frames_written = 0
    rendered_frames = []

    while source_index <= clip_end:
        success, frame = capture.read()

        if not success:
            break

        elapsed_source_frames = source_index - clip_start + 1
        target_output_count = int(
            round(
                elapsed_source_frames
                * OUTPUT_FPS
                / source_fps
            )
        )

        event_denominator = max(1, event_end - event_start)
        progress_percent = (
            (source_index - event_start)
            / event_denominator
            * 100.0
        )

        output_frame = fit_frame_to_canvas(
            frame,
            PANE_WIDTH,
            PANE_HEIGHT,
        )
        output_frame = annotate_pane(
            output_frame,
            example,
            progress_percent,
            side_label,
        )

        while output_frames_written < target_output_count:
            rendered_frames.append(output_frame.copy())
            output_frames_written += 1

        source_index += 1

    capture.release()

    if not rendered_frames:
        raise ValueError(
            f"No output frames were rendered for {example['video_id']}"
        )

    return rendered_frames


def combine_panes(left_frame, right_frame):
    """Combine two pane frames into one side by side output frame."""

    output = np.zeros(
        (OUTPUT_HEIGHT, OUTPUT_WIDTH, 3),
        dtype=np.uint8,
    )

    output[:, :PANE_WIDTH] = left_frame
    output[:, PANE_WIDTH:PANE_WIDTH + PANE_GAP] = 0
    output[:, PANE_WIDTH + PANE_GAP:] = right_frame

    return output


def create_comparison_video(selected):
    """Create the final side by side comparison MP4."""

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    ordered = selected.copy()
    ordered["behaviour_order"] = ordered["behaviour"].map(
        {
            "Jaywalking": 0,
            "Non jaywalking": 1,
        }
    )
    ordered = ordered.sort_values("behaviour_order").reset_index(drop=True)

    left_example = ordered.iloc[0]
    right_example = ordered.iloc[1]

    left_video_path = find_source_video(
        str(left_example["video"]),
        str(left_example["video_id"]),
    )
    right_video_path = find_source_video(
        str(right_example["video"]),
        str(right_example["video_id"]),
    )

    print(
        f"Rendering left panel ({left_example['behaviour']}): "
        f"{left_video_path} | early pose motion="
        f"{float(left_example['early_pose_motion']):.3f}"
    )
    print(
        f"Rendering right panel ({right_example['behaviour']}): "
        f"{right_video_path} | early pose motion="
        f"{float(right_example['early_pose_motion']):.3f}"
    )

    left_frames = build_event_clip_frames(
        left_example,
        left_video_path,
        side_label="LEFT",
    )
    right_frames = build_event_clip_frames(
        right_example,
        right_video_path,
        side_label="RIGHT",
    )

    fourcc = cv2.VideoWriter_fourcc(*VIDEO_CODEC)
    writer = cv2.VideoWriter(
        str(VIDEO_OUTPUT),
        fourcc,
        OUTPUT_FPS,
        (OUTPUT_WIDTH, OUTPUT_HEIGHT),
    )

    if not writer.isOpened():
        raise OSError(
            f"Could not create output video: {VIDEO_OUTPUT}"
        )

    try:
        write_title_card(writer, left_example, right_example)

        total_frames = max(len(left_frames), len(right_frames))
        left_last = left_frames[-1]
        right_last = right_frames[-1]

        for frame_index in range(total_frames):
            left_frame = (
                left_frames[frame_index]
                if frame_index < len(left_frames)
                else left_last
            )
            right_frame = (
                right_frames[frame_index]
                if frame_index < len(right_frames)
                else right_last
            )

            combined = combine_panes(left_frame, right_frame)
            writer.write(combined)
    finally:
        writer.release()


# -------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------


def main():
    print("=" * 80)
    print("EXTREME JAYWALKING VS NON JAYWALKING VIDEO COMPARISON")
    print(
        f"Early movement definition: first {EARLY_EVENT_PERCENT:.0f}% "
        "of the labelled event"
    )
    print("Output mode: side by side")
    print("=" * 80)

    frame_data, labels = load_analysis_inputs()

    labelled = calculate_early_movement(
        frame_data,
        labels,
    )

    selected = select_extreme_examples(
        labelled,
    )

    output_columns = [
        "behaviour",
        "video",
        "video_id",
        "ground_truth",
        "track_id",
        "event_start",
        "event_end",
        "early_event_percent",
        "early_pose_motion",
        "early_pose_motion_median",
        "early_motion_frames",
        "motion_measure",
        "selection_rule",
    ]

    selected[output_columns].to_csv(
        SELECTION_OUTPUT,
        index=False,
    )

    print()
    print("Selected examples:")

    for _, example in selected.iterrows():
        print(
            f"  {example['behaviour']}: {example['video']} | "
            f"track={int(example['track_id'])} | "
            f"early pose motion={float(example['early_pose_motion']):.3f}"
        )

    create_comparison_video(
        selected,
    )

    print()
    print("=" * 80)
    print("COMPLETE")
    print("=" * 80)
    print(f"Selection details: {SELECTION_OUTPUT}")
    print(f"Comparison video: {VIDEO_OUTPUT}")


if __name__ == "__main__":
    main()
