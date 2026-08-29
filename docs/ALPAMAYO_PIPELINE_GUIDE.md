# VLM BASELINE PIPELINE GUIDE — Technical Investigation & Architecture Reference

---

## 1. Executive Summary

The **VLM Chain-of-Causation (CoC) Full-Video Baseline** represents a major architectural upgrade over the baseline V1 majority-voting system. By combining 5-frame temporal sequence inputs into a single multi-modal request and enforcing a 5-step Chain-of-Causation reasoning prompt, performance on the canonical 39-clip development benchmark jumped from **69.23% to 97.44% Accuracy**.

* **Development Benchmark Results ($N=39$ Clips):**
  - **Accuracy:** **97.44%** (38/39 correct) [Up from 69.23%]
  - **Precision:** **93.75%** (15/16) [Up from 56.52%]
  - **Recall:** **100.00%** (15/15) [Up from 86.67%]
  - **Specificity:** **95.83%** (23/24) [Up from 58.33%]
  - **F1 Score:** **96.77%** [Up from 68.42%]
  - **Confusion Matrix:** TP=15, TN=23, FP=1, FN=0
  - **Latency:** 212.44s total (5.45s / clip)
  - **Remaining Error:** Exactly 1 False Positive (`video_0003.mp4`).

---

## 2. Current Benchmark Results Table ($N=39$)

| Metric | Canonical Baseline V1 (3-Frame Vote) | VLM CoC Baseline (5-Frame Single-Call) | Delta |
|---|:---:|:---:|:---:|
| **Accuracy** | **69.23%** (27/39) | **97.44%** (38/39) | **+28.21%** |
| **Precision** | **56.52%** | **93.75%** | **+37.23%** |
| **Recall** | **86.67%** (13/15) | **100.00%** (15/15) | **+13.33%** |
| **Specificity** | **58.33%** (14/24) | **95.83%** (23/24) | **+37.50%** |
| **F1 Score** | **68.42%** | **96.77%** | **+28.35%** |
| **True Positives (TP)** | 13 | 15 | +2 |
| **True Negatives (TN)** | 14 | 23 | +9 |
| **False Positives (FP)** | 10 | 1 | **-9 (90% FP reduction)** |
| **False Negatives (FN)** | 2 | 0 | **-2 (100% FN reduction)** |

---

## 3. Architecture Diagram

```
                                [Video Clip (.mp4)]
                                         │
                                         ▼
                      [extract_full_video_frames(max_frames=5)]
                   Samples 5 Equidistant Frames Across Video Clip
                                         │
                                         ▼
                             [encode_frame_to_base64()]
                   Encodes 5 BGR Image Frames to JPEG Base64 Strings
                                         │
                                         ▼
                               [Ollama HTTP Client]
                     http://localhost:11434/api/chat (qwen2.5vl:7b)
                Temperature = 0.0, Seed = 42, Max Tokens = 300
              Single Chat Request with 5 Base64 Images + CoC Prompt
                                         │
                                         ▼
                             [Chain-of-Causation Text]
               1. Pedestrian Trajectory & Location
               2. Infrastructure & Right-of-Way
               3. Vehicle Kinematic Response
               4. Causal Analysis
               5. Final Classification: [JAYWALKING / COMPLIANT]
                                         │
                                         ▼
                               [parse_coc_response()]
                   Extracts Decision from Step 5 Text Output
                                         │
                                         ▼
                              [Evaluator / Metrics Output]
                      38/39 Passed (97.44% Accuracy, 100% Recall)
```

---

## 4. Exact Execution Flow & File Reference

### Execution Trace

```
scripts/run_evaluation.py (CLI Entry Point)
  │
  ▼
src/pipeline.py (get_pipeline(mode="alpamayo"))
  │
  ▼
src/vlm/alpamayo_detector.py (FullVideoVLMDetector)
  ├── extract_full_video_frames(video_path, target_fps=5) -> [Frame_0 .. Frame_4]
  ├── src/vlm/client.py: encode_frame_to_base64() -> [Base64_0 .. Base64_4]
  ├── src/vlm/client.py: OllamaClient.generate_chat(prompt=coc_prompt, images=b64_list)
  │     └──> Ollama HTTP API (http://localhost:11434/api/chat) -> qwen2.5vl:7b
  ├── parse_coc_response(raw_text) -> {"prediction": "jaywalking"|"compliant"}
  └── predict(video_path) -> Returns prediction, CoC text, timing
  │
  ▼
evaluation/evaluator.py (Evaluator.run_evaluation())
  │
  ▼
evaluation/metrics.py (compute_metrics()) -> 97.44% Acc Summary
```

### File-by-File Reference

| File | Purpose | Main Classes / Functions | Called By | Calls | Important Parameters |
|---|---|---|---|---|---|
| [`scripts/run_evaluation.py`](scripts/run_evaluation.py) | CLI entry point | `main()` | Terminal CLI | [`src/pipeline.py`](src/pipeline.py), [`evaluation/evaluator.py`](evaluation/evaluator.py) | `--mode alpamayo`, `--gt data/ground_truth.csv` |
| [`src/pipeline.py`](src/pipeline.py) | Pipeline factory | `get_pipeline()` | `run_evaluation.py`, `run_inference.py` | [`src/vlm/alpamayo_detector.py`](src/vlm/alpamayo_detector.py) | `mode="alpamayo"` |
| [`src/vlm/alpamayo_detector.py`](src/vlm/alpamayo_detector.py) | Full-video CoC detector | `FullVideoVLMDetector`, `extract_full_video_frames()`, `parse_coc_response()`, `predict()` | `src/pipeline.py` | [`src/vlm/client.py`](src/vlm/client.py) | `max_frames=5`, `temperature=0.0`, `seed=42` |
| [`src/vlm/client.py`](src/vlm/client.py) | Ollama HTTP API client | `OllamaClient`, `generate_chat()`, `encode_frame_to_base64()` | `alpamayo_detector.py` | `requests`, `cv2`, `base64` | `model="qwen2.5vl:7b"`, `max_tokens=300` |
| [`evaluation/evaluator.py`](evaluation/evaluator.py) | Benchmark execution harness | `Evaluator`, `run_evaluation()` | `run_evaluation.py` | [`src/data_loader.py`](src/data_loader.py), [`evaluation/metrics.py`](evaluation/metrics.py) | `ground_truth_path="data/ground_truth.csv"` |
| [`evaluation/metrics.py`](evaluation/metrics.py) | Metric calculation | `compute_metrics()` | `evaluator.py` | None | Calculates Accuracy, Precision, Recall, Specificity, F1 |

---

## 5. Input Representation & Frame Sampling

1. **What VLM Baseline Receives:** The model receives **5 decoded, sampled video frames** as JPEG base64-encoded strings inside a single multi-modal Ollama chat request message.
2. **Frame Sampling Algorithm:** Frames are sampled by `extract_full_video_frames()`:
   ```python
   step = max(1, int(fps / target_fps))
   indices = list(range(0, total_frames, step))
   if len(indices) > self.max_frames:
       step_idx = len(indices) // self.max_frames
       indices = indices[::step_idx][:self.max_frames]
   ```
3. **Equidistant Sampling:** Frame indices are sampled **equidistantly across the video duration**.
4. **Exact Frame Indices for Examples:**
   - `video_0003.mp4` (210 frames, 29.97 FPS, 7.01s duration): `[0, 40, 80, 120, 160]`
   - `video_0014.mp4` (257 frames, 29.97 FPS, 8.57s duration): `[0, 50, 100, 150, 200]`
   - `video_0028.mp4` (240 frames, 59.94 FPS, 4.00s duration): `[0, 44, 88, 132, 176]`
5. **Image Resolution & Preprocessing:** Original video resolution (e.g. 1920x1080). BGR OpenCV arrays are JPEG-encoded at `quality=85` and converted to base64. No cropping, normalization, or resizing is applied.
6. **Audio & Telemetry:** No audio or vehicle telemetry is used.
7. **Temporal Ordering:** Yes. The 5 base64 image strings are passed in chronological order as a list inside the single chat request payload.

---

## 6. VLM Baseline Model Integration Details

1. **Model Name:** `qwen2.5vl:7b` (running locally via Ollama).
2. **Model Loading:** Managed locally by the background Ollama daemon process (`ollama serve`).
3. **API Path:** Local HTTP REST endpoint `http://localhost:11434/api/chat`.
4. **Python Libraries:** `requests` HTTP library, `cv2`, `base64`, `numpy`, `pandas`.
5. **Inference Function:** `self.client.generate_chat(prompt=self.coc_prompt, base64_images=b64_list)`.
6. **Input Payload Structure:**
   ```json
   {
     "model": "qwen2.5vl:7b",
     "messages": [
       {
         "role": "user",
         "content": "<coc_prompt_text>",
         "images": ["<base64_frame_0>", "<base64_frame_1>", "<base64_frame_2>", "<base64_frame_3>", "<base64_frame_4>"]
       }
     ],
     "stream": false,
     "options": {
       "temperature": 0.0,
       "seed": 42,
       "num_predict": 300
     }
   }
   ```
7. **Generation Options:** `temperature=0.0`, `seed=42`, `max_tokens=300`.
8. **System Prompt:** None.
9. **User Prompt:** `self.coc_prompt` (Chain-of-Causation prompt).
10. **Chain-of-Thought Request:** Explicitly requested in the prompt.
11. **Model Generating Reasoning:** Generated directly by `qwen2.5vl:7b` via Ollama.
12. **Gemma Involvement:** **Gemma is NOT involved in the current active 97.44% pipeline.** The active execution relies entirely on `qwen2.5vl:7b` via Ollama.

---

## 7. Chain-of-Causation (CoC) Prompt & Response Parsing

### "Full-video CoC reasoning" Origin
The phrase `"Full-video CoC reasoning (5 frames)"` is set as the `reason` field in `FullVideoVLMDetector.predict()` to describe the decision mechanism.

### The Exact CoC Prompt (`self.coc_prompt`)
```text
Analyze the full video sequence of pedestrian and vehicle interactions.
Produce Chain-of-Causation (CoC) reasoning steps:
1. Pedestrian Trajectory & Location: [sidewalk / curb / roadway]
2. Infrastructure & Right-of-Way: [marked crosswalk / signal / none]
3. Vehicle Kinematic Response: [yielding / decelerating / accelerating / none]
4. Causal Analysis: Explain why the crossing is legal compliance or an illegal violation.
5. Final Classification: Output EXACTLY either JAYWALKING or COMPLIANT.
```

### Output Parser (`parse_coc_response()`)
Parses the generated CoC text response by normalizing text to uppercase and searching for `"JAYWALKING"` or `"COMPLIANT"` in Step 5 / final lines of the response string.

---

## 8. End-to-End Traces for Sample Videos

### Trace 1: `video_0003.mp4` (The Single False Positive)
* **Metadata:** 210 frames, 29.97 FPS, 7.01s duration. Ground Truth: `compliant`.
* **Sampled Frames:** 5 frames at indices `[0, 40, 80, 120, 160]`.
* **Raw Model CoC Response:**
  ```text
  1. Pedestrian Trajectory & Location: The pedestrian is walking on the roadway, specifically crossing in front of a parked car.
  2. Infrastructure & Right-of-Way: There is no marked crosswalk visible, and there are no traffic signals indicating pedestrian right-of-way.
  3. Vehicle Kinematic Response: The vehicle from which the camera is recording appears to be moving slowly or stopped, observing the pedestrian.
  4. Causal Analysis: The pedestrian is crossing the road where there is no designated crosswalk or signal, which constitutes jaywalking. Although the vehicle may have yielded or slowed down, the action itself is illegal because it occurs outside of a legal crossing zone.
  5. Final Classification: JAYWALKING
  ```

### Trace 2: `video_0028.mp4` (Correctly Classified Jaywalking)
* **Metadata:** 240 frames, 59.94 FPS, 4.00s duration. Ground Truth: `jaywalking`.
* **Sampled Frames:** 5 frames at indices `[0, 44, 88, 132, 176]`.
* **Raw Model CoC Response:**
  ```text
  1. Pedestrian Trajectory & Location: The pedestrian is walking on the roadway, crossing diagonally from the sidewalk towards the middle of the street.
  2. Infrastructure & Right-of-Way: There is no marked crosswalk or traffic signal present in the immediate vicinity where the pedestrian is crossing.
  3. Vehicle Kinematic Response: The vehicle appears to be moving at a consistent speed, observing the pedestrian...
  4. Causal Analysis: The pedestrian is crossing the street outside of a designated crosswalk without any signals granting them right-of-way. This action constitutes jaywalking...
  5. Final Classification: JAYWALKING
  ```

---

## 9. Technical Comparison: Baseline V1 vs. VLM CoC Baseline

| Dimension | Baseline V1 Pipeline | VLM CoC Baseline |
|---|---|---|
| **Input Strategy** | 3 equidistant keyframes | 5 sampled sequence frames |
| **API Requests** | 3 separate single-image requests | **1 multi-image request with 5 frames** |
| **Context Scope** | Isolated single-frame views | **Full multi-frame temporal co-visibility** |
| **Prompt Protocol** | Short binary classification prompt | **5-step Chain-of-Causation (CoC) prompt** |
| **Decision Aggregation** | Majority vote (>= 2/3) | **Parsed from CoC Causal Step 5** |
| **Latency** | ~4.12s total (1.37s/frame) | ~5.45s total (1 single request) |
| **Accuracy ($N=39$)** | **69.23%** (27/39) | **97.44%** (38/39) |
| **False Positives** | **10 FPs** (58.33% Specificity) | **1 FP** (**95.83% Specificity**) |

---

## 10. Failure Analysis of `video_0003.mp4`

* **Scenario:** `video_0003.mp4` is a commercial parking lot driveway / store entrance scene where a pedestrian is walking near parked vehicles.
* **Root Cause of Confusion:** The scene contains ambiguous spatial boundaries (commercial parking lot pavement vs. public street) and a nearby stop sign. In the 5-frame sequence, the model focuses on the pedestrian walking on pavement without marked zebra lines, concluding `"no marked crosswalk visible -> JAYWALKING"`.

---

## 11. What Actually Changed (Root Cause of Accuracy Jump)

The **+28.21% accuracy jump** ($69.23\% 	o 97.44\%$) was driven by two specific architectural changes:

1. **Temporal Co-Visibility (Input Change):** Sending all 5 frames in a **single multi-image request** allows the VLM to observe pedestrian spatial movement and vehicle kinematic response simultaneously across time, rather than judging isolated static frames.
2. **Chain-of-Causation Guardrail (Prompt Change):** Forcing the model to sequentially output pedestrian location, infrastructure presence, vehicle kinematics, and causal analysis **before** declaring a final label prevents premature violation bias and reduces False Positives from 10 down to 1.

---

## 12. Reproducibility Protocol

To reproduce the exact **97.44% Accuracy** result on the canonical 39-clip development benchmark:

```bash
# 1. Ensure local Ollama daemon is running with qwen2.5vl:7b
ollama list

# 2. Run VLM baseline evaluation CLI on canonical ground truth
python3 scripts/run_evaluation.py --mode alpamayo --gt data/ground_truth.csv
```

---

## 13. Mentor-Ready Verbal Explanation

> *"Previously in V1, we sent 3 isolated keyframes into Qwen2.5-VL via separate API calls and majority-voted their outputs. That baseline achieved 69.23% accuracy but suffered from 10 false positives because static single frames lack motion and yielding context.*
>
> *We developed the VLM Baseline Chain-of-Causation (CoC) pipeline. Instead of separate calls, we pass a 5-frame temporal sequence in a single multi-image request and force the VLM through a 5-step Chain-of-Causation prompt analyzing pedestrian trajectory, infrastructure right-of-way, vehicle kinematics, and causal legality before issuing a verdict.*
>
> *This temporal co-visibility and structured reasoning eliminated 9 out of 10 false positives, driving benchmark accuracy from 69.23% up to 97.44% (38/39 clips, 100% recall, 95.83% specificity). The single remaining error is video_0003.mp4 due to commercial driveway spatial ambiguity.*
>
> *Our next immediate step is to freeze this exact VLM CoC Baseline pipeline and evaluate it on an untouched held-out test dataset to verify generalization."*

---

## 14. Recommended Next Research Step

**RECOMMENDATION: FREEZE THE CURRENT ALPAMAYO PIPELINE & EVALUATE ON AN UNTOUCHED HELD-OUT TEST SET.**

* **Action:** Create an untouched test split of unseen JAAD video clips (`data/heldout_ground_truth.csv`), freeze `FullVideoVLMDetector` (`max_frames=5`, `temperature=0.0`, `seed=42`, `coc_prompt`), and evaluate generalization without tuning.

---

## 15. Open Questions & Unknowns

1. *How will 5-frame CoC reasoning generalize to unseen night-time or adverse weather clips?*
2. *Can the remaining 1 FP (`video_0003.mp4`) be resolved by explicit parking lot / shared-space context grounding?*
