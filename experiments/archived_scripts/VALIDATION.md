# Validation Report — Intelligent Jaywalking Detection

## Sample

| Metric | Value |
|--------|-------|
| Total clips | TBD |
| Total annotated crossing events | TBD |
| Annotator confidence distribution | TBD |

## Per-Class Precision / Recall / F1

| Violation Type | Precision | Recall | F1 | TP | FP | FN |
|----------------|-----------|--------|----|----|----|----|
| SIGNAL_VIOLATION | TBD | TBD | TBD | TBD | TBD | TBD |
| NO_CROSSWALK | TBD | TBD | TBD | TBD | TBD | TBD |

*Pooled metrics are deliberately omitted — pooling hides which signal is driving errors.*

## Top-3 Failure Patterns

1. TBD
2. TBD
3. TBD

## Weakest Module

TBD — based on per-error classification into perception vs. fusion-logic failures.

## Calibrated Thresholds

| Parameter | Value |
|-----------|-------|
| Hesitation low_motion_thresh | 0.02 |
| SIGNAL_VIOLATION weight | 0.9 |
| NO_CROSSWALK weight | 0.7 |
| Violation confidence threshold | 0.7 |

## Known Limitations

- Traffic light module uses hand-weighted 1x1 conv + temporal smoothing (5-frame average, 0.8 commit threshold)
- Zebra crossing uses YOLO segmentation for dynamic ROI but Hough-line heuristic for crosswalk detection
- No dedicated crosswalk segmentation model was trained (deferred to Day 7+)
- Pose-based hesitation and inattentive-entry signals are experimental
