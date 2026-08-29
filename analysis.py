#!/usr/bin/env python3
"""
Analyse body movement from pedestrian COCO keypoint tracks.

The script reads JSON files produced by:
    extract_pedestrian_keypoints.py

It separates:

    1. Global pedestrian translation
       Movement of the whole person through the image.

    2. Articulated body movement
       Changes in arms, legs, torso, and overall body pose after removing
       global body translation and body size.

Outputs:

    data/body_movement_frame_level.csv
    data/body_movement_track_level.csv

    figures/body_movement/
        mean_pose_motion_histogram.html
        movement_frame_ratio_histogram.html
        pose_motion_vs_translation_scatter.html
        video_level_mean_pose_motion_bar.html
        representative_track_pose_motion.html

The main metric is:

    pose_motion_speed

which is approximately the amount of body pose change per second,
expressed relative to the pedestrian's torso length.
"""

import json
import math
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd

import common


# -------------------------------------------------------------------------
# PATHS
# -------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
INPUT_DIR = DATA_DIR / "per_video"
FIGURES_DIR = PROJECT_ROOT / "figures" / "body_movement"

FRAME_OUTPUT = DATA_DIR / "body_movement_frame_level.csv"
TRACK_OUTPUT = DATA_DIR / "body_movement_track_level.csv"


# -------------------------------------------------------------------------
# ANALYSIS SETTINGS
# -------------------------------------------------------------------------

# Temporary video resolution assumption.
# All source videos are treated as 720p landscape frames.
ANALYSIS_VERSION = "jaywalking-comparison-720p-v3"
VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720

# Ignore unreliable YOLO pose keypoints below this confidence.
KEYPOINT_CONFIDENCE_THRESHOLD = 0.35

# Minimum number of corresponding valid body points required between
# two frames before calculating body movement.
MIN_VALID_KEYPOINTS = 4

# Do not compare observations separated by very large temporal gaps.
MAX_FRAME_GAP_SECONDS = 0.50

# Median smoothing window for frame level motion.
SMOOTHING_WINDOW = 5

# Initial heuristic only.
#
# This should eventually be calibrated using visual inspection of your
# jaywalking videos.
#
# Units are approximately torso lengths per second.
BODY_MOTION_THRESHOLD = 0.30

# Number of bars in the video-level figure.
MAX_VIDEOS_IN_BAR_FIGURE = 20

# Static image export settings for Plotly PNG output.
PNG_EXPORT_SCALE = 2


# -------------------------------------------------------------------------
# COCO KEYPOINT INDICES
# -------------------------------------------------------------------------

NOSE = 0

LEFT_EYE = 1
RIGHT_EYE = 2

LEFT_EAR = 3
RIGHT_EAR = 4

LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6

LEFT_ELBOW = 7
RIGHT_ELBOW = 8

LEFT_WRIST = 9
RIGHT_WRIST = 10

LEFT_HIP = 11
RIGHT_HIP = 12

LEFT_KNEE = 13
RIGHT_KNEE = 14

LEFT_ANKLE = 15
RIGHT_ANKLE = 16


UPPER_BODY = [
    LEFT_SHOULDER,
    RIGHT_SHOULDER,
    LEFT_ELBOW,
    RIGHT_ELBOW,
    LEFT_WRIST,
    RIGHT_WRIST,
]

LOWER_BODY = [
    LEFT_HIP,
    RIGHT_HIP,
    LEFT_KNEE,
    RIGHT_KNEE,
    LEFT_ANKLE,
    RIGHT_ANKLE,
]

ARMS = [
    LEFT_ELBOW,
    RIGHT_ELBOW,
    LEFT_WRIST,
    RIGHT_WRIST,
]

LEGS = [
    LEFT_KNEE,
    RIGHT_KNEE,
    LEFT_ANKLE,
    RIGHT_ANKLE,
]


# -------------------------------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------------------------------


def midpoint(point_a, point_b):
    """
    Return midpoint between two 2D points.
    """

    return (point_a + point_b) / 2.0



def euclidean_distance(point_a, point_b):
    """
    Euclidean distance between two 2D points.
    """

    return float(np.linalg.norm(point_a - point_b))



def keypoints_to_pixels(
    keypoints,
    image_width,
    image_height,
    confidence_threshold,
):
    """
    Convert stored normalised keypoints back into pixel coordinates.

    Input format for every keypoint:

        [normalised_x, normalised_y, confidence]

    Returns:

        coordinates: shape (17, 2)
        valid:       shape (17,)
    """

    coordinates = np.full((17, 2), np.nan, dtype=float)
    valid = np.zeros(17, dtype=bool)

    if keypoints is None:
        return coordinates, valid

    for index, keypoint in enumerate(keypoints):

        if index >= 17:
            break

        x_normalised = float(keypoint[0])
        y_normalised = float(keypoint[1])
        confidence = float(keypoint[2])

        if confidence < confidence_threshold:
            continue

        x_pixel = x_normalised * image_width
        y_pixel = y_normalised * image_height

        coordinates[index] = [x_pixel, y_pixel]
        valid[index] = True

    return coordinates, valid



def get_body_centre(coordinates, valid):
    """
    Find a stable body centre.

    Preference:

        1. midpoint of left and right hips
        2. midpoint of left and right shoulders
        3. median of all visible keypoints
    """

    if valid[LEFT_HIP] and valid[RIGHT_HIP]:

        return midpoint(
            coordinates[LEFT_HIP],
            coordinates[RIGHT_HIP],
        )

    if valid[LEFT_SHOULDER] and valid[RIGHT_SHOULDER]:

        return midpoint(
            coordinates[LEFT_SHOULDER],
            coordinates[RIGHT_SHOULDER],
        )

    visible_points = coordinates[valid]

    if len(visible_points) == 0:
        return None

    return np.median(visible_points, axis=0)



def get_body_scale(
    coordinates,
    valid,
    bbox,
    image_height,
):
    """
    Estimate body scale.

    The preferred measure is torso length:

        shoulder midpoint -> hip midpoint

    If shoulders or hips are unavailable, bounding box height is used.
    """

    shoulders_available = (
        valid[LEFT_SHOULDER]
        and valid[RIGHT_SHOULDER]
    )

    hips_available = (
        valid[LEFT_HIP]
        and valid[RIGHT_HIP]
    )

    if shoulders_available and hips_available:

        shoulder_midpoint = midpoint(
            coordinates[LEFT_SHOULDER],
            coordinates[RIGHT_SHOULDER],
        )

        hip_midpoint = midpoint(
            coordinates[LEFT_HIP],
            coordinates[RIGHT_HIP],
        )

        torso_length = euclidean_distance(
            shoulder_midpoint,
            hip_midpoint,
        )

        if torso_length > 5:
            return torso_length

    bbox_height_pixels = float(bbox["height"]) * image_height

    if bbox_height_pixels > 5:
        return bbox_height_pixels

    return None



def normalise_pose(
    coordinates,
    valid,
    bbox,
    image_height,
):
    """
    Remove body translation and body scale.

    After this transformation:

        pelvis approximately becomes (0, 0)

    and distances are measured relative to torso size.

    Therefore a person moving through the image does not automatically
    produce a high articulated body motion score.
    """

    body_centre = get_body_centre(
        coordinates,
        valid,
    )

    if body_centre is None:
        return None

    body_scale = get_body_scale(
        coordinates,
        valid,
        bbox,
        image_height,
    )

    if body_scale is None:
        return None

    normalised = np.full_like(
        coordinates,
        np.nan,
        dtype=float,
    )

    normalised[valid] = (
        coordinates[valid] - body_centre
    ) / body_scale

    return normalised



def subset_motion(
    pose_previous,
    pose_current,
    valid_previous,
    valid_current,
    indices,
    delta_time,
):
    """
    Calculate median movement for a selected set of body keypoints.

    Examples:

        upper body
        lower body
        arms
        legs
    """

    distances = []

    for index in indices:

        if not (
            valid_previous[index]
            and valid_current[index]
        ):
            continue

        if (
            np.any(np.isnan(pose_previous[index]))
            or np.any(np.isnan(pose_current[index]))
        ):
            continue

        distance = euclidean_distance(
            pose_previous[index],
            pose_current[index],
        )

        distances.append(distance)

    if len(distances) == 0:
        return np.nan

    return float(np.median(distances) / delta_time)



def overall_pose_motion(
    pose_previous,
    pose_current,
    valid_previous,
    valid_current,
    delta_time,
):
    """
    Calculate overall articulated body movement.

    The median is preferred over the mean because an individual noisy
    keypoint can otherwise create a very large false motion estimate.
    """

    common_valid = (
        valid_previous
        & valid_current
    )

    valid_indices = np.where(common_valid)[0]

    distances = []

    for index in valid_indices:

        previous = pose_previous[index]
        current = pose_current[index]

        if (
            np.any(np.isnan(previous))
            or np.any(np.isnan(current))
        ):
            continue

        distances.append(
            euclidean_distance(previous, current)
        )

    if len(distances) < MIN_VALID_KEYPOINTS:
        return np.nan, len(distances)

    motion = float(
        np.median(distances) / delta_time
    )

    return motion, len(distances)



def calculate_torso_lean(
    coordinates,
    valid,
):
    """
    Calculate torso lean in degrees.

    Zero means approximately vertical.

    Positive and negative values represent opposite lateral directions.
    """

    shoulders_available = (
        valid[LEFT_SHOULDER]
        and valid[RIGHT_SHOULDER]
    )

    hips_available = (
        valid[LEFT_HIP]
        and valid[RIGHT_HIP]
    )

    if not (
        shoulders_available
        and hips_available
    ):
        return np.nan

    shoulder_midpoint = midpoint(
        coordinates[LEFT_SHOULDER],
        coordinates[RIGHT_SHOULDER],
    )

    hip_midpoint = midpoint(
        coordinates[LEFT_HIP],
        coordinates[RIGHT_HIP],
    )

    dx = (
        shoulder_midpoint[0]
        - hip_midpoint[0]
    )

    dy = (
        hip_midpoint[1]
        - shoulder_midpoint[1]
    )

    angle = math.degrees(
        math.atan2(dx, dy)
    )

    return float(angle)



def calculate_global_translation(
    previous_frame,
    current_frame,
    image_width,
    image_height,
    delta_time,
):
    """
    Calculate movement of the whole pedestrian through the image.

    Bounding box centre displacement is divided by bounding box height.

    This metric is deliberately separate from articulated pose motion.
    """

    previous_bbox = previous_frame["bbox"]
    current_bbox = current_frame["bbox"]

    previous_centre = np.array(
        [
            previous_bbox["center_x"] * image_width,
            previous_bbox["center_y"] * image_height,
        ],
        dtype=float,
    )

    current_centre = np.array(
        [
            current_bbox["center_x"] * image_width,
            current_bbox["center_y"] * image_height,
        ],
        dtype=float,
    )

    displacement = euclidean_distance(
        previous_centre,
        current_centre,
    )

    previous_height = (
        previous_bbox["height"]
        * image_height
    )

    current_height = (
        current_bbox["height"]
        * image_height
    )

    scale = np.mean(
        [
            previous_height,
            current_height,
        ]
    )

    if scale <= 5:
        return np.nan

    return float(
        displacement
        / scale
        / delta_time
    )


# -------------------------------------------------------------------------
# TRACK ANALYSIS
# -------------------------------------------------------------------------


def analyse_track(
    video_id,
    track,
    fps,
    image_width,
    image_height,
):
    """
    Analyse one ByteTrack pedestrian trajectory.
    """

    frames = track["frames"]

    results = []

    previous_data = None

    for frame in frames:

        frame_id = int(frame["frame_id"])

        timestamp = float(
            frame["timestamp_seconds"]
        )

        keypoints = frame.get("keypoints")

        coordinates, valid = keypoints_to_pixels(
            keypoints,
            image_width,
            image_height,
            KEYPOINT_CONFIDENCE_THRESHOLD,
        )

        pose = normalise_pose(
            coordinates,
            valid,
            frame["bbox"],
            image_height,
        )

        torso_lean = calculate_torso_lean(
            coordinates,
            valid,
        )

        row = {
            "video_id": video_id,
            "track_id": track["track_id"],
            "frame_id": frame_id,
            "timestamp_seconds": timestamp,
            "valid_keypoints": int(valid.sum()),
            "pose_motion_speed": np.nan,
            "upper_body_motion": np.nan,
            "lower_body_motion": np.nan,
            "arm_motion": np.nan,
            "leg_motion": np.nan,
            "global_translation_speed": np.nan,
            "torso_lean_degrees": torso_lean,
            "torso_lean_change": np.nan,
            "comparable_keypoints": 0,
        }

        current_data = {
            "frame": frame,
            "coordinates": coordinates,
            "valid": valid,
            "pose": pose,
            "torso_lean": torso_lean,
        }

        if (
            previous_data is not None
            and pose is not None
            and previous_data["pose"] is not None
        ):

            previous_frame = previous_data["frame"]

            frame_difference = (
                frame_id
                - int(previous_frame["frame_id"])
            )

            delta_time = (
                frame_difference / fps
            )

            if (
                delta_time > 0
                and delta_time <= MAX_FRAME_GAP_SECONDS
            ):

                pose_motion, comparable = overall_pose_motion(
                    previous_data["pose"],
                    pose,
                    previous_data["valid"],
                    valid,
                    delta_time,
                )

                row["pose_motion_speed"] = pose_motion
                row["comparable_keypoints"] = comparable

                row["upper_body_motion"] = subset_motion(
                    previous_data["pose"],
                    pose,
                    previous_data["valid"],
                    valid,
                    UPPER_BODY,
                    delta_time,
                )

                row["lower_body_motion"] = subset_motion(
                    previous_data["pose"],
                    pose,
                    previous_data["valid"],
                    valid,
                    LOWER_BODY,
                    delta_time,
                )

                row["arm_motion"] = subset_motion(
                    previous_data["pose"],
                    pose,
                    previous_data["valid"],
                    valid,
                    ARMS,
                    delta_time,
                )

                row["leg_motion"] = subset_motion(
                    previous_data["pose"],
                    pose,
                    previous_data["valid"],
                    valid,
                    LEGS,
                    delta_time,
                )

                row["global_translation_speed"] = (
                    calculate_global_translation(
                        previous_frame,
                        frame,
                        image_width,
                        image_height,
                        delta_time,
                    )
                )

                previous_lean = previous_data[
                    "torso_lean"
                ]

                if (
                    not np.isnan(previous_lean)
                    and not np.isnan(torso_lean)
                ):
                    row["torso_lean_change"] = (
                        abs(
                            torso_lean
                            - previous_lean
                        )
                        / delta_time
                    )

        results.append(row)

        previous_data = current_data

    return results


# -------------------------------------------------------------------------
# TRACK LEVEL SUMMARY
# -------------------------------------------------------------------------


def summarise_tracks(frame_dataframe):
    """
    Convert frame level movement measurements into one summary per person.
    """

    summaries = []

    grouped = frame_dataframe.groupby(
        [
            "video_id",
            "track_id",
        ]
    )

    for (
        video_id,
        track_id,
    ), group in grouped:

        group = group.sort_values(
            "frame_id"
        ).copy()

        group["pose_motion_smoothed"] = (
            group["pose_motion_speed"]
            .rolling(
                window=SMOOTHING_WINDOW,
                center=True,
                min_periods=1,
            )
            .median()
        )

        valid_motion = (
            group["pose_motion_smoothed"]
            .dropna()
        )

        if len(valid_motion) == 0:
            continue

        movement_frames = (
            valid_motion
            >= BODY_MOTION_THRESHOLD
        )

        movement_ratio = float(
            movement_frames.mean()
        )

        mean_motion = float(
            valid_motion.mean()
        )

        median_motion = float(
            valid_motion.median()
        )

        p95_motion = float(
            valid_motion.quantile(0.95)
        )

        max_motion = float(
            valid_motion.max()
        )

        upper_motion = (
            group["upper_body_motion"]
            .dropna()
        )

        lower_motion = (
            group["lower_body_motion"]
            .dropna()
        )

        arm_motion = (
            group["arm_motion"]
            .dropna()
        )

        leg_motion = (
            group["leg_motion"]
            .dropna()
        )

        translation = (
            group["global_translation_speed"]
            .dropna()
        )

        lean_change = (
            group["torso_lean_change"]
            .dropna()
        )

        body_movement_detected = (
            movement_ratio >= 0.20
        )

        summary = {
            "video_id": video_id,
            "track_id": track_id,
            "analysed_frames": len(group),
            "mean_pose_motion": mean_motion,
            "median_pose_motion": median_motion,
            "p95_pose_motion": p95_motion,
            "max_pose_motion": max_motion,
            "mean_upper_body_motion": (
                float(upper_motion.mean())
                if len(upper_motion)
                else np.nan
            ),
            "mean_lower_body_motion": (
                float(lower_motion.mean())
                if len(lower_motion)
                else np.nan
            ),
            "mean_arm_motion": (
                float(arm_motion.mean())
                if len(arm_motion)
                else np.nan
            ),
            "mean_leg_motion": (
                float(leg_motion.mean())
                if len(leg_motion)
                else np.nan
            ),
            "mean_global_translation": (
                float(translation.mean())
                if len(translation)
                else np.nan
            ),
            "mean_torso_lean_change": (
                float(lean_change.mean())
                if len(lean_change)
                else np.nan
            ),
            "movement_frame_ratio": movement_ratio,
            "body_movement_detected": body_movement_detected,
        }

        summaries.append(summary)

        frame_dataframe.loc[
            group.index,
            "pose_motion_smoothed",
        ] = group[
            "pose_motion_smoothed"
        ]

    return pd.DataFrame(summaries)


# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# LABELLED JAYWALKING COMPARISON
# -------------------------------------------------------------------------

RESULTS_SUMMARY_PATH = DATA_DIR / "results_summary.csv"

ALL_FRAME_OUTPUT = DATA_DIR / "body_movement_all_tracks_frame_level.csv"
ALL_TRACK_OUTPUT = DATA_DIR / "body_movement_all_tracks_track_level.csv"

FRAME_OUTPUT = DATA_DIR / "body_movement_event_frame_level.csv"
TRACK_OUTPUT = DATA_DIR / "body_movement_event_track_level.csv"
GROUP_OUTPUT = DATA_DIR / "body_movement_group_summary.csv"


def normalise_video_id(video_name):
    """
    Convert names such as video_0003.mp4 to video_0003 so they match the
    keypoint JSON filenames.
    """

    return Path(str(video_name)).stem


def ground_truth_to_behaviour(value):
    """
    Convert the project ground-truth label into a reader-facing behaviour
    group used in the figures.

    Unknown labels are preserved rather than silently forced into a group.
    """

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


def load_event_labels():
    """
    Load ground truth and responsible pedestrian IDs from results_summary.csv.
    """

    if not RESULTS_SUMMARY_PATH.exists():
        raise FileNotFoundError(
            f"Results summary not found: {RESULTS_SUMMARY_PATH}"
        )

    labels = pd.read_csv(RESULTS_SUMMARY_PATH)

    required_columns = {
        "video",
        "ground_truth",
        "responsible_track_id",
        "event_start",
        "event_end",
    }

    missing_columns = required_columns.difference(labels.columns)

    if missing_columns:
        raise ValueError(
            "results_summary.csv is missing required columns: "
            f"{sorted(missing_columns)}"
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

    if "entry_frame" in labels.columns:
        labels["entry_frame"] = pd.to_numeric(
            labels["entry_frame"],
            errors="coerce",
        )

    if "peak_motion_frame" in labels.columns:
        labels["peak_motion_frame"] = pd.to_numeric(
            labels["peak_motion_frame"],
            errors="coerce",
        )

    labels["behaviour"] = labels["ground_truth"].map(
        ground_truth_to_behaviour
    )

    labels = labels.dropna(subset=["track_id"])
    labels["track_id"] = labels["track_id"].astype(int)

    keep_columns = [
        "video_id",
        "video",
        "ground_truth",
        "behaviour",
        "track_id",
        "event_start",
        "event_end",
    ]

    for optional_column in [
        "entry_frame",
        "peak_motion_frame",
        "prediction",
        "correct",
    ]:
        if optional_column in labels.columns:
            keep_columns.append(optional_column)

    return labels[keep_columns].drop_duplicates(
        subset=["video_id", "track_id"],
        keep="first",
    )


def select_responsible_event_frames(all_frame_dataframe, labels):
    """
    Keep only the responsible pedestrian from each labelled video and restrict
    the analysis to that video's event_start ... event_end interval.
    """

    frame_data = all_frame_dataframe.copy()
    frame_data["track_id"] = pd.to_numeric(
        frame_data["track_id"],
        errors="coerce",
    )

    merged = frame_data.merge(
        labels,
        on=["video_id", "track_id"],
        how="inner",
        validate="many_to_one",
    )

    if merged.empty:
        return merged

    start_available = merged["event_start"].notna()
    end_available = merged["event_end"].notna()

    keep = pd.Series(True, index=merged.index)
    keep &= (
        ~start_available
        | (merged["frame_id"] >= merged["event_start"])
    )
    keep &= (
        ~end_available
        | (merged["frame_id"] <= merged["event_end"])
    )

    merged = merged.loc[keep].copy()

    event_duration_frames = (
        merged["event_end"] - merged["event_start"]
    )

    valid_duration = event_duration_frames > 0

    merged["event_progress_percent"] = np.nan
    merged.loc[valid_duration, "event_progress_percent"] = (
        (
            merged.loc[valid_duration, "frame_id"]
            - merged.loc[valid_duration, "event_start"]
        )
        / event_duration_frames.loc[valid_duration]
        * 100.0
    )

    return merged


def attach_labels_to_track_summary(track_dataframe, labels):
    """
    Attach ground truth and event metadata to one-row-per-track summaries.
    """

    metadata_columns = [
        column
        for column in labels.columns
        if column not in {"track_id"}
    ]

    return track_dataframe.merge(
        labels[["track_id", *metadata_columns]],
        on=["video_id", "track_id"],
        how="left",
        validate="one_to_one",
    )


def save_plotly_figure(figure, html_path):
    """
    Save a Plotly figure as both interactive HTML and static PNG.

    PNG export uses Plotly's static image engine. If the local environment
    does not have the required dependency available, HTML export still
    succeeds and a warning is printed.
    """

    figure.update_layout(
        template="plotly_white",
        font=dict(size=14),
        legend_title_text="Behaviour",
    )

    html_path = Path(html_path)
    png_path = html_path.with_suffix(".png")

    figure.write_html(
        html_path,
        include_plotlyjs="cdn",
        full_html=True,
    )

    output_paths = [html_path]

    try:
        figure.write_image(
            png_path,
            format="png",
            scale=PNG_EXPORT_SCALE,
        )
        output_paths.append(png_path)
    except Exception as error:
        print(
            f"   [WARNING] Could not save PNG for {html_path.name}: {error}"
        )
        print(
            "   [WARNING] Install/enable kaleido for Plotly static image export."
        )

    return output_paths


def plot_overall_pose_motion(track_dataframe):
    data = track_dataframe.dropna(
        subset=["behaviour", "mean_pose_motion"]
    )

    if data.empty:
        return None

    figure = px.box(
        data,
        x="behaviour",
        y="mean_pose_motion",
        color="behaviour",
        points="all",
        hover_data=[
            "video_id",
            "track_id",
            "ground_truth",
            "analysed_frames",
        ],
        title="Overall body pose motion: Jaywalking vs non jaywalking",
        labels={
            "behaviour": "Behaviour",
            "mean_pose_motion": "Mean pose motion",
        },
    )

    return save_plotly_figure(
        figure,
        FIGURES_DIR / "overall_pose_motion_by_behaviour.html",
    )


def plot_body_region_motion(track_dataframe):
    metric_map = {
        "mean_upper_body_motion": "Upper body",
        "mean_lower_body_motion": "Lower body",
        "mean_arm_motion": "Arms",
        "mean_leg_motion": "Legs",
    }

    available_metrics = [
        metric
        for metric in metric_map
        if metric in track_dataframe.columns
    ]

    if not available_metrics:
        return None

    long_data = track_dataframe.melt(
        id_vars=[
            "video_id",
            "track_id",
            "ground_truth",
            "behaviour",
        ],
        value_vars=available_metrics,
        var_name="metric",
        value_name="motion",
    )

    long_data["body_region"] = long_data["metric"].map(metric_map)
    long_data = long_data.dropna(subset=["behaviour", "motion"])

    if long_data.empty:
        return None

    figure = px.box(
        long_data,
        x="body_region",
        y="motion",
        color="behaviour",
        points="all",
        hover_data=["video_id", "track_id", "ground_truth"],
        title="Body region movement by jaywalking behaviour",
        labels={
            "body_region": "Body region",
            "motion": "Mean normalised motion",
            "behaviour": "Behaviour",
        },
    )

    figure.update_layout(boxmode="group")

    return save_plotly_figure(
        figure,
        FIGURES_DIR / "body_region_motion_by_behaviour.html",
    )


def plot_pose_motion_vs_translation(track_dataframe):
    data = track_dataframe.dropna(
        subset=[
            "behaviour",
            "mean_pose_motion",
            "mean_global_translation",
        ]
    )

    if data.empty:
        return None

    figure = px.scatter(
        data,
        x="mean_global_translation",
        y="mean_pose_motion",
        color="behaviour",
        hover_data=[
            "video_id",
            "track_id",
            "ground_truth",
            "mean_arm_motion",
            "mean_leg_motion",
        ],
        title="Body pose motion versus pedestrian translation",
        labels={
            "mean_global_translation": "Mean global translation",
            "mean_pose_motion": "Mean pose motion",
            "behaviour": "Behaviour",
        },
    )

    figure.update_traces(marker=dict(size=10, opacity=0.8))

    return save_plotly_figure(
        figure,
        FIGURES_DIR / "pose_motion_vs_translation_by_behaviour.html",
    )


def plot_torso_change(track_dataframe):
    data = track_dataframe.dropna(
        subset=["behaviour", "mean_torso_lean_change"]
    )

    if data.empty:
        return None

    figure = px.box(
        data,
        x="behaviour",
        y="mean_torso_lean_change",
        color="behaviour",
        points="all",
        hover_data=["video_id", "track_id", "ground_truth"],
        title="Torso orientation change: Jaywalking vs non jaywalking",
        labels={
            "behaviour": "Behaviour",
            "mean_torso_lean_change": "Mean torso lean change (degrees/s)",
        },
    )

    return save_plotly_figure(
        figure,
        FIGURES_DIR / "torso_change_by_behaviour.html",
    )


def plot_event_timecourse(frame_dataframe):
    required = {
        "video_id",
        "track_id",
        "behaviour",
        "event_progress_percent",
        "pose_motion_smoothed",
    }

    if not required.issubset(frame_dataframe.columns):
        return None

    data = frame_dataframe.dropna(
        subset=[
            "behaviour",
            "event_progress_percent",
            "pose_motion_smoothed",
        ]
    ).copy()

    if data.empty:
        return None

    data = data[
        data["event_progress_percent"].between(0, 100)
    ].copy()

    data["event_bin"] = (
        np.floor(data["event_progress_percent"] / 10.0) * 10.0
    ).clip(upper=90.0)

    track_bin = (
        data
        .groupby(
            ["video_id", "track_id", "behaviour", "event_bin"],
            as_index=False,
        )
        .agg(pose_motion=("pose_motion_smoothed", "mean"))
    )

    group_bin = (
        track_bin
        .groupby(["behaviour", "event_bin"], as_index=False)
        .agg(
            mean_pose_motion=("pose_motion", "mean"),
            std_pose_motion=("pose_motion", "std"),
            n_tracks=("track_id", "count"),
        )
    )

    group_bin["sem"] = (
        group_bin["std_pose_motion"]
        / np.sqrt(group_bin["n_tracks"].clip(lower=1))
    )
    group_bin["event_midpoint_percent"] = group_bin["event_bin"] + 5.0

    figure = px.line(
        group_bin,
        x="event_midpoint_percent",
        y="mean_pose_motion",
        color="behaviour",
        markers=True,
        error_y="sem",
        hover_data=["n_tracks"],
        title="Body pose motion across the labelled event",
        labels={
            "event_midpoint_percent": "Event progress (%)",
            "mean_pose_motion": "Mean smoothed pose motion",
            "behaviour": "Behaviour",
        },
    )

    figure.update_xaxes(range=[0, 100])

    return save_plotly_figure(
        figure,
        FIGURES_DIR / "event_timecourse_pose_motion_by_behaviour.html",
    )


def create_group_summary(track_dataframe):
    metrics = [
        "mean_pose_motion",
        "median_pose_motion",
        "mean_upper_body_motion",
        "mean_lower_body_motion",
        "mean_arm_motion",
        "mean_leg_motion",
        "mean_global_translation",
        "mean_torso_lean_change",
    ]

    available_metrics = [
        metric
        for metric in metrics
        if metric in track_dataframe.columns
    ]

    rows = []

    for behaviour, group in track_dataframe.groupby("behaviour"):
        for metric in available_metrics:
            values = group[metric].dropna()

            if values.empty:
                continue

            rows.append(
                {
                    "behaviour": behaviour,
                    "metric": metric,
                    "n": len(values),
                    "mean": float(values.mean()),
                    "median": float(values.median()),
                    "std": float(values.std()) if len(values) > 1 else np.nan,
                    "q25": float(values.quantile(0.25)),
                    "q75": float(values.quantile(0.75)),
                }
            )

    return pd.DataFrame(rows)


def generate_behaviour_figures(frame_dataframe, track_dataframe):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    generated = []

    for figure_function in [
        plot_overall_pose_motion,
        plot_body_region_motion,
        plot_pose_motion_vs_translation,
        plot_torso_change,
    ]:
        paths = figure_function(track_dataframe)
        if paths is not None:
            generated.extend(paths)

    paths = plot_event_timecourse(frame_dataframe)
    if paths is not None:
        generated.extend(paths)

    return generated


# -------------------------------------------------------------------------
# VIDEO FILE ANALYSIS
# -------------------------------------------------------------------------


def analyse_video_file(json_path):
    """
    Analyse all pedestrian tracks from one keypoint JSON file.

    A valid keypoint file must contain fps and tracks. Video resolution is
    temporarily assumed to be 1280 x 720.
    """

    with open(
        json_path,
        "r",
        encoding="utf-8",
    ) as file:
        video_data = json.load(file)

    required_fields = {
        "fps",
        "tracks",
    }

    missing_fields = required_fields.difference(video_data)

    if missing_fields:
        print(
            f"   [SKIPPED] {json_path.name}: missing required fields "
            f"{sorted(missing_fields)}"
        )
        return []

    video_id = json_path.stem.replace(
        "_keypoints",
        "",
    )

    fps = float(video_data["fps"])
    image_width = VIDEO_WIDTH
    image_height = VIDEO_HEIGHT

    all_rows = []

    for track in video_data.get("tracks", []):
        track_rows = analyse_track(
            video_id=video_id,
            track=track,
            fps=fps,
            image_width=image_width,
            image_height=image_height,
        )
        all_rows.extend(track_rows)

    return all_rows


# -------------------------------------------------------------------------
# MAIN PIPELINE
# -------------------------------------------------------------------------


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_DIR.exists():
        print(f"Keypoint directory not found: {INPUT_DIR}")
        return

    json_files = sorted(INPUT_DIR.glob("*.json"))

    if not json_files:
        print(f"No keypoint JSON files found in {INPUT_DIR}")
        return

    try:
        labels = load_event_labels()
    except (OSError, ValueError) as error:
        print(f"Unable to load results_summary.csv: {error}")
        return

    print("=" * 80)
    print("JAYWALKING VS NON JAYWALKING BODY MOVEMENT ANALYSIS")
    print(f"Analysis version: {ANALYSIS_VERSION}")
    print(f"Assumed video resolution: {VIDEO_WIDTH} x {VIDEO_HEIGHT} (720p)")
    print(f"Labelled events in results_summary.csv: {len(labels)}")
    print("=" * 80)

    all_frame_rows = []

    for index, json_path in enumerate(json_files, start=1):
        print(
            f"[{index}/{len(json_files)}] "
            f"Analysing {json_path.name}"
        )

        try:
            rows = analyse_video_file(json_path)
        except (
            OSError,
            ValueError,
            TypeError,
            KeyError,
            json.JSONDecodeError,
        ) as error:
            print(f"   [SKIPPED] {json_path.name}: {error}")
            continue

        all_frame_rows.extend(rows)

    if not all_frame_rows:
        print("No valid pedestrian keypoint tracks were found.")
        return

    all_frame_dataframe = pd.DataFrame(all_frame_rows)
    all_track_dataframe = summarise_tracks(all_frame_dataframe)

    all_frame_dataframe.to_csv(ALL_FRAME_OUTPUT, index=False)
    all_track_dataframe.to_csv(ALL_TRACK_OUTPUT, index=False)

    event_frame_dataframe = select_responsible_event_frames(
        all_frame_dataframe,
        labels,
    )

    if event_frame_dataframe.empty:
        print(
            "No responsible tracks from results_summary.csv matched the "
            "keypoint tracks. Check video IDs and responsible_track_id values."
        )
        return

    event_track_dataframe = summarise_tracks(event_frame_dataframe)
    event_track_dataframe = attach_labels_to_track_summary(
        event_track_dataframe,
        labels,
    )

    event_frame_dataframe.to_csv(FRAME_OUTPUT, index=False)
    event_track_dataframe.to_csv(TRACK_OUTPUT, index=False)

    group_summary = create_group_summary(event_track_dataframe)
    group_summary.to_csv(GROUP_OUTPUT, index=False)

    generated_figures = generate_behaviour_figures(
        frame_dataframe=event_frame_dataframe,
        track_dataframe=event_track_dataframe,
    )

    matched_events = event_track_dataframe[
        ["video_id", "track_id"]
    ].drop_duplicates()

    print()
    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"All detected pedestrian tracks: {len(all_track_dataframe)}")
    print(f"Labelled responsible tracks analysed: {len(matched_events)}")

    behaviour_counts = (
        event_track_dataframe["behaviour"]
        .value_counts(dropna=False)
    )

    for behaviour, count in behaviour_counts.items():
        print(f"{behaviour}: {count}")

    print()
    print(f"Event frame output: {FRAME_OUTPUT}")
    print(f"Event track output: {TRACK_OUTPUT}")
    print(f"Group summary: {GROUP_OUTPUT}")
    print(f"Figures directory: {FIGURES_DIR}")

    for figure_path in generated_figures:
        print(f"Generated figure: {figure_path}")


if __name__ == "__main__":
    main()
