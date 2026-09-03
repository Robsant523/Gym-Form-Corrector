# Exercise Form Classifier — Starter Pipeline

A 4-stage pipeline: video -> keypoints -> joint angles -> per-rep fault classifier.

## Setup

```bash
pip install -r requirements.txt
```

## Workflow

**1. Extract keypoints from a video**

```bash
python pose_extraction.py --video squat_session.mp4 --out keypoints.csv
```

**2. Compute joint angles**

```bash
python features.py --in keypoints.csv --out angles.csv
```

**3. Label your reps by hand**

Watch the video, count the reps, and create a CSV like:

```csv
rep_index,knee_valgus,insufficient_depth,back_rounding
0,0,0,0
1,1,0,0
2,0,1,0
```

One row per rep, in order, 1 = fault present, 0 = not present. Start with
2-3 fault types you can reliably judge yourself. You need a few dozen
examples of each fault (with more non-fault examples) before the model
is worth trusting at all, and a few hundred before it's reasonably solid.

**4. Train**

```bash
python train_classifier.py --angles angles.csv --labels labels.csv --out model.joblib
```

This trains a separate Random Forest per fault type (multi-label, not
multi-class) and prints a precision/recall report for each so you can see
which faults the model is actually catching.

**5. Predict on a new video**

```bash
python pose_extraction.py --video new_session.mp4 --out new_keypoints.csv
python features.py --in new_keypoints.csv --out new_angles.csv
python predict.py --angles new_angles.csv --model model.joblib
```

Prints per-rep feedback like:

```
Rep 1:
  - Knees are caving inward — focus on pushing them out over your toes.
Rep 2:
  - Looks good, no faults detected.
```

## Notes / where to go next

- **Rep segmentation** (`segment_reps` in `features.py`) uses knee-angle local
  minima as a heuristic. It works reasonably for clean squat/deadlift footage
  but will need tuning (or a smoothing pass) on noisy video — check the
  `prominence` parameter first.
- **Left vs right side**: the pipeline defaults to LEFT landmarks. If the
  camera mostly sees the right side of the body, pass `--side RIGHT` to
  `features.py`, or extend `add_joint_angles` to average both sides.
- **Feature set**: `summarize_rep` currently uses min/max/mean/range of each
  angle plus rep duration. This is deliberately simple — once you have more
  labeled data, this is the first place to expand (e.g. angle at the exact
  bottom frame, velocity/acceleration of the descent).
- **Scaling up**: once classical ML plateaus, the natural next step is
  feeding the full per-frame angle sequence (padded/aligned) into an LSTM or
  small Transformer instead of summary stats — but that needs a much bigger
  labeled dataset to pay off.
