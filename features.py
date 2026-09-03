"""
features.py

Stage 2 of the pipeline: turns raw keypoint CSVs (from pose_extraction.py) into
joint-angle time series, then rolls those into per-rep summary features that a
classical ML model (Random Forest / XGBoost) can consume.

Usage as a library:
    from features import add_joint_angles, segment_reps, summarize_rep

Usage as a script (angles only, no rep segmentation):
    python features.py --in keypoints.csv --out angles.csv
"""

import argparse
import numpy as np
import pandas as pd


def _angle(a, b, c):
    """
    Angle at point b, formed by segments b->a and b->c, in degrees.
    a, b, c are each (x, y) tuples or arrays of them (vectorized over rows).
    """
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cos_angle = np.sum(ba * bc, axis=-1) / (
        np.linalg.norm(ba, axis=-1) * np.linalg.norm(bc, axis=-1) + 1e-8
    )
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))


def add_joint_angles(df: pd.DataFrame, side: str = "LEFT") -> pd.DataFrame:
    """
    Adds columns for the joint angles most useful for lower-body lifts
    (squat, deadlift). `side` picks LEFT or RIGHT landmarks — pick whichever
    side is more visible to the camera, or compute both and average.
    """
    df = df.copy()
    s = side.upper()

    def pt(name):
        return df[[f"{s}_{name}_x", f"{s}_{name}_y"]].values

    shoulder, hip, knee, ankle = pt("SHOULDER"), pt("HIP"), pt("KNEE"), pt("ANKLE")

    # Knee angle: hip-knee-ankle. ~180 = fully extended, smaller = deeper bend.
    df["knee_angle"] = _angle(hip, knee, ankle)

    # Hip angle: shoulder-hip-knee. Tracks torso lean / hip hinge.
    df["hip_angle"] = _angle(shoulder, hip, knee)

    # Back/torso angle relative to vertical (0 = upright, 90 = horizontal).
    # Uses shoulder-hip vector against a straight-up reference point.
    vertical_ref = hip.copy()
    vertical_ref[:, 1] -= 1.0  # a point directly "above" the hip in image space
    df["back_angle_from_vertical"] = _angle(shoulder, hip, vertical_ref)

    # Knee valgus proxy: horizontal (x) distance between knee and ankle,
    # normalized by hip width. Negative/near-zero often indicates knees
    # caving in relative to the foot — tune the threshold on your own data.
    hip_width = np.linalg.norm(
        df[["LEFT_HIP_x", "LEFT_HIP_y"]].values - df[["RIGHT_HIP_x", "RIGHT_HIP_y"]].values,
        axis=-1,
    )
    df["knee_ankle_x_offset_norm"] = (knee[:, 0] - ankle[:, 0]) / (hip_width + 1e-8)

    return df


def segment_reps(df: pd.DataFrame, angle_col: str = "knee_angle", prominence: float = 15.0):
    """
    Splits a continuous set-of-reps time series into individual reps by
    finding local minima in the knee angle (bottom of each rep).
    Returns a list of (start_idx, end_idx) row ranges into df.

    This is a simple heuristic starting point — for noisy footage you may
    want to smooth the signal (e.g. rolling mean) before segmenting.
    """
    from scipy.signal import find_peaks

    # Invert so the bottom of the squat/deadlift becomes a peak.
    inverted = -df[angle_col].values
    peaks, _ = find_peaks(inverted, prominence=prominence)

    if len(peaks) == 0:
        return []

    # Rep boundaries: midpoints between consecutive bottom-of-rep peaks,
    # padded to include the first and last rep.
    boundaries = [0] + [
        (peaks[i] + peaks[i + 1]) // 2 for i in range(len(peaks) - 1)
    ] + [len(df) - 1]

    reps = list(zip(boundaries[:-1], boundaries[1:]))
    return reps


def summarize_rep(df: pd.DataFrame, start: int, end: int) -> dict:
    """
    Collapses one rep's frame-by-frame angles into a fixed-length feature
    vector — this is what actually gets fed to the classifier. Add/remove
    stats here as you find what's predictive.
    """
    rep = df.iloc[start:end + 1]
    feat = {}
    for col in ["knee_angle", "hip_angle", "back_angle_from_vertical", "knee_ankle_x_offset_norm"]:
        feat[f"{col}_min"] = rep[col].min()
        feat[f"{col}_max"] = rep[col].max()
        feat[f"{col}_mean"] = rep[col].mean()
        feat[f"{col}_range"] = rep[col].max() - rep[col].min()
    feat["rep_duration_s"] = rep["time_s"].iloc[-1] - rep["time_s"].iloc[0]
    feat["n_frames"] = len(rep)
    return feat


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute joint angles from a keypoints CSV.")
    parser.add_argument("--in", dest="in_csv", required=True)
    parser.add_argument("--out", dest="out_csv", required=True)
    parser.add_argument("--side", default="LEFT", choices=["LEFT", "RIGHT"])
    args = parser.parse_args()

    keypoints = pd.read_csv(args.in_csv)
    angled = add_joint_angles(keypoints, side=args.side)
    angled.to_csv(args.out_csv, index=False)
    print(f"Wrote {len(angled)} rows with joint angles to {args.out_csv}")
