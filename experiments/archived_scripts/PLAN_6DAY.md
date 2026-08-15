# 6-Day Execution Plan — Intelligent Jaywalking Detection

## Reality check before you start

The original ambition (segmentation + pose + rule reasoning + multi-country
validation + benchmark comparison) is a 12-day scope. Compressed into 6 days,
something has to give. This plan keeps the parts that produce a genuinely
better, defensible system and cuts the parts that are nice-to-have:

**Kept:** pose-based intent signals, a real temporal state machine, ground-truth
validation, one perception-module upgrade (whichever tests worse), final demo.
**Cut/deferred:** training a full custom zebra-segmentation model from scratch,
full JAAD/PIE benchmark leaderboard comparison, night-vs-day traffic-light
calibration, cross-dataset generalization study. These get a "Day 7+" placeholder
at the bottom instead of being squeezed in badly.

**Parallelization is the key to fitting this in 6 days:** ground-truth annotation
(a human task) and code changes (an engineering task) run in parallel, not
sequentially. Day 1 kicks off annotation immediately so it's not blocking Day 3.

---

## Day 1 — Fetch videos, kick off annotation in parallel, start wiring

### Morning: fetch videos (do this first, it unblocks everything else)

1. **Own dataset videos** (for diversity/scale): pull 10–15 video IDs from your
   existing `mapping.csv`, prioritizing `time_of_day = 0` (daytime, cleaner
   visibility) across different countries. Use `yt-dlp` to download.
2. **JAAD clips** (for crossing density — every clip is pre-trimmed to a crossing
   moment): clone `github.com/ykotseruba/JAAD` (branch `JAAD_2.0`), download
   15–20 clips (not all 346 — enough for a real validation set without eating the
   whole day). Sequential filenames `video_0001.mp4` ... `video_0346.mp4`.
3. Put everything in `data/raw_clips/`, one file each, keep names traceable
   (`mapping_<video_id>.mp4`, `jaad_<clip_num>.mp4`).

**Target: 25–30 total clips landed locally by midday.** This matches the
validation-set size from the earlier 3-day plan — don't over-collect, more
clips just means more manual annotation time later that you don't have.

### Midday: hand off for annotation (runs in background for the rest of the week)

Give yourself (or whoever's annotating) this exact CSV template to fill in per
clip, one row per pedestrian-crossing event (a clip may have 0, 1, or several):

```
clip_id,video_id_or_jaad_num,ped_track_hint,ground_truth_violation,ground_truth_reason,observed_hesitation,observed_facing,signalized,has_zebra,annotator_confidence,notes
```

- `ground_truth_violation`: yes / no / ambiguous
- `ground_truth_reason`: signal_violation / no_crosswalk / both / none
- `observed_hesitation`: yes/no — did the pedestrian visibly pause mid-crossing
- `observed_facing`: looked_at_traffic / did_not_look / unclear
- `signalized`: yes/no — was there a traffic light at all
- `has_zebra`: yes/no — was there a marked crossing at all
- `annotator_confidence`: high/medium/low

Save as `validation/labeled_crossings.csv`. **This file does not need to be
finished today** — annotation continues through Day 2–3 while code work proceeds
in parallel. Just get it started now, since it's the slowest part of the week.

### Afternoon: start the pose-intent module (engineering, doesn't need labels yet)

1. Create `utils/crossing/intent.py`.
2. Implement `detect_hesitation(stride_series, road_mask, fps, low_motion_thresh)`
   — rolling window (~0.5s) where `stride_ratio` stays below a threshold while
   on-road. Leave the threshold as a named constant, not final yet.
3. Implement `detect_inattentive_entry(facing_at_entry)` — flags
   `SIDE_VIEW`/`BACK_VIEW` at road-entry frame; `UNKNOWN` is a non-signal, never
   a positive flag.
4. Reuse `Detection.build_states` from `detection.py` to find the road-entry
   frame — do not re-derive this boundary logic.

**End of Day 1 checkpoint:** clips downloaded, annotation started (even if not
finished), `intent.py` skeleton exists and runs against dummy data without
crashing.

---

## Day 2 — Traffic light + zebra: apply published methods, not custom training

Given the time budget, full model retraining is out of scope. Apply the
cheapest version of what published work recommends instead of hand-tuned
heuristics:

### Morning: traffic light

1. Replace `TrafficLightCNN`'s fixed color-projection matrix with a pretrained
   YOLO-based detector-classifier if one is readily available (check Ultralytics
   hub / a pretrained traffic-light-state model) rather than training your own —
   training from scratch in a morning is not realistic.
2. If no suitable pretrained model is found quickly (timebox this to 1.5 hrs,
   don't rabbit-hole), fall back to calibrating the existing hand-weighted conv
   against whatever annotation labels exist so far (`signalized` + observed
   state), rather than the current magic constants.
3. Add temporal smoothing regardless of which path you take: average
   classification probability over 5 consecutive frames, only commit to a state
   change above a 0.8 threshold (this is a ~10-line change, cheap insurance
   against single-frame flicker, borrowed directly from published work on this
   exact problem).

### Afternoon: zebra crossing

1. `zebra.py` already loads `yolo11x-seg.pt` but never calls `.predict()` on it
   for crosswalks — it does classical Hough-line detection in a fixed ROI
   instead. Actually invoke the segmentation model on the road region and use
   its output instead of/alongside the Hough-line result.
2. If the seg model's classes don't include "crosswalk" specifically (likely,
   since it's a general COCO/general-purpose model), use its "road" segmentation
   to constrain a dynamic ROI (replacing the hardcoded `0.55h–0.95h` band), then
   run the existing Hough-line logic inside that better-constrained region rather
   than a fixed pixel band. This is a realistic middle ground given no time to
   train a dedicated crosswalk-segmentation model this week.
3. Do NOT attempt to fine-tune a segmentation model on custom crosswalk masks
   this week — that's the Day 7+ deferred item. Ship the dynamic-ROI improvement
   instead; it is achievable today and still a real upgrade over the fixed band.

**End of Day 2 checkpoint:** both perception modules changed and re-run on a
handful of clips to visually sanity-check (not full evaluation yet — that's
Day 5, once labels exist).

---

## Day 3 — Build the temporal state machine and fusion logic

### Morning: state machine

Extend `Detection.build_states` (already computes LEFT/ROAD/RIGHT) into six
named states without rewriting the underlying boundary math:

| State | Definition |
|---|---|
| `CURB_LEFT` | left of `min_x` (existing state 0) |
| `APPROACHING` | buffer zone before `min_x`, moving toward road |
| `COMMITTED` | first frame inside `[min_x, max_x]` |
| `ON_ROAD` | remainder of existing state 1 |
| `EXITING` | buffer zone near `max_x` (or `min_x` if reversing) |
| `CURB_RIGHT` | right of `max_x` (existing state 2) |

### Afternoon: `classify_crossing()` — the single fusion function

```python
def classify_crossing(track_df, light_states, zebra_result, fps,
                       hesitated: bool, inattentive_entry: bool) -> dict:
    # join track states with light_states and zebra_result on frame-count
    # find committed_frame = first frame in COMMITTED state
    # only frames >= committed_frame count toward violations
    #   (a red light 10s before the pedestrian stepped off the curb
    #   is not this pedestrian's violation)
    # violations: SIGNAL_VIOLATION if red observed post-commit,
    #             NO_CROSSWALK if never on a zebra polygon post-commit
    # risk_factors (NOT violations): HESITATION, INATTENTIVE_ENTRY
    # violation_confidence: weighted score, not a hard boolean
    #   (start weights: SIGNAL_VIOLATION=0.9, NO_CROSSWALK=0.7,
    #   scaled by underlying classifier confidence where available)
    ...
```

Wire this into `analysis.py`'s `crossing_event_wt_traffic_equipment`, replacing
the old two-`if` classifier.

**End of Day 3 checkpoint:** `classify_crossing()` runs end-to-end on the Day 1
clips and produces structured, confidence-scored output — even if thresholds
are still provisional, since final calibration depends on annotation progress.

---

## Day 4 — Calibrate against whatever ground truth exists, close the annotation loop

1. Check annotation progress. If `validation/labeled_crossings.csv` isn't fully
   done, prioritize finishing it today — everything from here depends on it.
2. Sweep `low_motion_thresh` (hesitation) and the inattentive-entry logic
   against `observed_hesitation`/`observed_facing` columns; pick the
   best-separating values. This replaces Day 1's placeholder constants with
   evidence-based ones.
3. Sweep the violation-confidence weights (`SIGNAL_VIOLATION`/`NO_CROSSWALK`)
   against `ground_truth_violation` similarly.
4. Re-run `classify_crossing()` on all annotated clips with the new calibrated
   values.

**End of Day 4 checkpoint:** every threshold in the system has been checked
against at least one real label, not left as a guess.

---

## Day 5 — Evaluation and error analysis

1. Compute precision/recall/F1 **per violation type**
   (`SIGNAL_VIOLATION`, `NO_CROSSWALK`) against `labeled_crossings.csv` — not one
   pooled number, since pooling hides which signal is driving errors.
2. For every false positive/negative, classify the error as perception failure
   (wrong light/zebra/pose reading) vs. fusion-logic failure (right signals,
   wrong weight) — this tells you whether remaining time on Day 6 should go to
   code or documentation.
3. If time allows (timebox to 2 hrs, don't let this eat the day): run the same
   clips through JAAD's own annotation format as a sanity cross-check, since
   JAAD clips already ship with official ground truth you can compare your own
   labels against — skip full PedestrianActionBenchmark integration, that's a
   Day 7+ item.
4. Write `VALIDATION.md`: sample size, per-class precision/recall, top 3 failure
   patterns, weakest module.

**End of Day 5 checkpoint:** real, documented numbers exist. This is the
artifact that separates "looks right on a few clips" from an actual result.

---

## Day 6 — Demo, polish, documentation

1. Run `tracking_mode` (or the updated pipeline) on the 2–3 best-looking clips
   from Day 1 (highest crossing-event density, cleanest visuals) to produce the
   final demo video(s) — reuse the annotated-overlay function, don't build a new
   visualization from scratch this late.
2. Trim to the most convincing 15–20 seconds per clip using the existing
   `trim_video` helper.
3. Update `README.md` with: what's a learned/pretrained model vs. a heuristic
   (be explicit — the traffic light and zebra modules are still partially
   heuristic even after this week's upgrades), the `VALIDATION.md` numbers, and
   a short architecture summary.
4. Package: final demo video(s), `VALIDATION.md`, updated `README.md`,
   `labeled_crossings.csv` — these four artifacts are what you present.

**End of Day 6 checkpoint:** a working, calibrated, evaluated system with an
honest validation report and a demo video, not just a demo video alone.

---

## Deferred to "Day 7+" (do not attempt this week)

- Training a dedicated crosswalk-segmentation model on custom-labeled masks.
- Full PedestrianActionBenchmark leaderboard comparison against published
  baselines.
- Night-vs-day-specific traffic-light recalibration.
- Cross-dataset generalization testing (train on JAAD/PIE, test on own data).
- LSTM-based temporal traffic-light state modeling (the 5-frame averaging trick
  from Day 2 is the correct-for-this-week substitute).

## What to feed opencode each day

Point it at this file and tell it which day's section to execute, e.g.:

```
Read PLAN_6DAY.md, section "Day 3 -- Build the temporal state machine and
fusion logic". Implement exactly what's described, reusing
Detection.build_states from detection.py rather than rewriting boundary logic.
Show me the diff before modifying analysis.py's crossing_event_wt_traffic_equipment.
```

Do this one day-section at a time — feeding the whole file at once risks it
jumping ahead to Day 5 evaluation code before Day 3's fusion function exists.
