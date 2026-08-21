# Experiment Results (BoT-SORT + YOLO26x-Pose)

## Overall
- Videos: 39
- Correct: 25 / 39
- Accuracy: 64.1%
- Precision: 53.33%
- Recall: 53.33%
- Specificity: 70.83%
- F1: 53.33%
- TP: 8
- TN: 17
- FP: 7
- FN: 7
- Average latency: 10.6 s/video

## Per-Video Predictions

| Video | Ground Truth | Prediction | Correct | Responsible Track ID | Entry Frame | Peak Frame |
|---|---|---|:---:|:---:|:---:|:---:|
| video_0003.mp4 | COMPLIANT | COMPLIANT | YES | 2 | 34 | 175 |
| video_0014.mp4 | COMPLIANT | JAYWALKING | NO | 7 | 31 | 253 |
| video_0028.mp4 | JAYWALKING | JAYWALKING | YES | 17 | 55 | 109 |
| video_0030.mp4 | JAYWALKING | COMPLIANT | NO | 33 | 11 | 148 |
| video_0035.mp4 | JAYWALKING | COMPLIANT | NO | 39 | 8 | 132 |
| video_0053.mp4 | JAYWALKING | JAYWALKING | YES | 44 | 12 | 230 |
| video_0054.mp4 | JAYWALKING | JAYWALKING | YES | 53 | 11 | 328 |
| video_0073.mp4 | JAYWALKING | JAYWALKING | YES | 87 | 3 | 23 |
| video_0082.mp4 | COMPLIANT | COMPLIANT | YES | 106 | 11 | 70 |
| video_0083.mp4 | COMPLIANT | COMPLIANT | YES | 130 | 37 | 147 |
| video_0087.mp4 | COMPLIANT | COMPLIANT | YES | 170 | 41 | 191 |
| video_0092.mp4 | JAYWALKING | COMPLIANT | NO | 203 | 9 | 111 |
| video_0099.mp4 | COMPLIANT | COMPLIANT | YES | 222 | 117 | 179 |
| video_0104.mp4 | JAYWALKING | COMPLIANT | NO | 230 | 3 | 141 |
| video_0110.mp4 | JAYWALKING | COMPLIANT | NO | 241 | 100 | 210 |
| video_0122.mp4 | JAYWALKING | JAYWALKING | YES | 248 | 3 | 170 |
| video_0123.mp4 | COMPLIANT | COMPLIANT | YES | 269 | 78 | 138 |
| video_0133.mp4 | JAYWALKING | JAYWALKING | YES | 271 | 3 | 194 |
| video_0138.mp4 | JAYWALKING | COMPLIANT | NO | 277 | 3 | 49 |
| video_0139.mp4 | JAYWALKING | JAYWALKING | YES | 298 | 3 | 98 |
| video_0146.mp4 | COMPLIANT | JAYWALKING | NO | 313 | 7 | 107 |
| video_0150.mp4 | COMPLIANT | JAYWALKING | NO | 340 | 104 | 121 |
| video_0160.mp4 | COMPLIANT | COMPLIANT | YES | 352 | 67 | 84 |
| video_0168.mp4 | COMPLIANT | COMPLIANT | YES | 370 | 54 | 79 |
| video_0190.mp4 | COMPLIANT | COMPLIANT | YES | 384 | 220 | 316 |
| video_0191.mp4 | COMPLIANT | COMPLIANT | YES | 419 | 55 | 148 |
| video_0198.mp4 | COMPLIANT | COMPLIANT | YES | 450 | 8 | 83 |
| video_0212.mp4 | COMPLIANT | COMPLIANT | YES | 472 | 95 | 109 |
| video_0227.mp4 | COMPLIANT | JAYWALKING | NO | 479 | 7 | 33 |
| video_0238.mp4 | COMPLIANT | COMPLIANT | YES | 495 | 72 | 235 |
| video_0240.mp4 | COMPLIANT | JAYWALKING | NO | 519 | 42 | 105 |
| video_0241.mp4 | COMPLIANT | JAYWALKING | NO | 526 | 85 | 101 |
| video_0251.mp4 | COMPLIANT | COMPLIANT | YES | 536 | 51 | 148 |
| video_0297.mp4 | COMPLIANT | JAYWALKING | NO | 566 | 5 | 144 |
| video_0312.mp4 | COMPLIANT | COMPLIANT | YES | 582 | 7 | 120 |
| video_0314.mp4 | COMPLIANT | COMPLIANT | YES | 597 | 24 | 25 |
| video_0322.mp4 | COMPLIANT | COMPLIANT | YES | 602 | 33 | 270 |
| video_0328.mp4 | JAYWALKING | COMPLIANT | NO | 619 | 53 | 119 |
| video_0336.mp4 | JAYWALKING | JAYWALKING | YES | 620 | 3 | 93 |

## Errors

| Video | Ground Truth | Prediction | Responsible Track | Event Interval | Error Reason |
|---|---|---|:---:|:---:|---|
| video_0014.mp4 | COMPLIANT | JAYWALKING | 7 | [30..256] | Pedestrian lateral motion triggered roadway entry |
| video_0030.mp4 | JAYWALKING | COMPLIANT | 33 | [10..149] | Entry motion below sustained threshold or diagonal crossing |
| video_0035.mp4 | JAYWALKING | COMPLIANT | 39 | [7..134] | Entry motion below sustained threshold or diagonal crossing |
| video_0092.mp4 | JAYWALKING | COMPLIANT | 203 | [8..114] | Entry motion below sustained threshold or diagonal crossing |
| video_0104.mp4 | JAYWALKING | COMPLIANT | 230 | [2..143] | Entry motion below sustained threshold or diagonal crossing |
| video_0110.mp4 | JAYWALKING | COMPLIANT | 241 | [99..210] | Entry motion below sustained threshold or diagonal crossing |
| video_0138.mp4 | JAYWALKING | COMPLIANT | 277 | [2..85] | Entry motion below sustained threshold or diagonal crossing |
| video_0146.mp4 | COMPLIANT | JAYWALKING | 313 | [6..108] | Pedestrian lateral motion triggered roadway entry |
| video_0150.mp4 | COMPLIANT | JAYWALKING | 340 | [103..285] | Pedestrian lateral motion triggered roadway entry |
| video_0227.mp4 | COMPLIANT | JAYWALKING | 479 | [6..90] | Pedestrian lateral motion triggered roadway entry |
| video_0240.mp4 | COMPLIANT | JAYWALKING | 519 | [41..210] | Pedestrian lateral motion triggered roadway entry |
| video_0241.mp4 | COMPLIANT | JAYWALKING | 526 | [84..102] | Pedestrian lateral motion triggered roadway entry |
| video_0297.mp4 | COMPLIANT | JAYWALKING | 566 | [4..240] | Pedestrian lateral motion triggered roadway entry |
| video_0328.mp4 | JAYWALKING | COMPLIANT | 619 | [52..120] | Entry motion below sustained threshold or diagonal crossing |
