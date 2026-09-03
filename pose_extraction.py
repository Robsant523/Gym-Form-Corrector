"""
pose_extraction.py

Extracts body keypoints from a video file frame-by-frame using MediaPipe Pose,
and saves them to a CSV. This is stage 1 of the pipeline:

    video -> keypoints (this file) -> joint angles (features.py) -> classifier (train_classifier.py)

Usage:
    python pose_extraction.py --video path/to/clip.mp4 --out keypoints.csv
"""

import argparse
import csv
import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose

# The subset of MediaPipe's 33 landmarks that matter for most lower/upper body
# lifts (squat, deadlift, bench, overhead press). Add more if you need them
# (e.g. wrists for grip width).
LANDMARKS_OF_INTEREST = [
    "LEFT_SHOULDER", "RIGHT_SHOULDER",
    "LEFT_HIP", "RIGHT_HIP",
    "LEFT_KNEE", "RIGHT_KNEE",
    "LEFT_ANKLE", "RIGHT_ANKLE",
    "LEFT_ELBOW", "RIGHT_ELBOW",
    "LEFT_WRIST", "RIGHT_WRIST",
    "NOSE",
]


def extract_keypoints(video_path: str, out_csv: str, min_detection_confidence: float = 0.5):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    fieldnames = ["frame", "time_s"]
    for name in LANDMARKS_OF_INTEREST:
        fieldnames += [f"{name}_x", f"{name}_y", f"{name}_z", f"{name}_visibility"]

    rows_written = 0
    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=0.5,
    ) as pose, open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        frame_idx = 0
        while True:
            success, frame = cap.read()
            if not success:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(frame_rgb)

            if result.pose_landmarks:
                row = {"frame": frame_idx, "time_s": round(frame_idx / fps, 4)}
                lm = result.pose_landmarks.landmark
                for name in LANDMARKS_OF_INTEREST:
                    idx = mp_pose.PoseLandmark[name].value
                    point = lm[idx]
                    row[f"{name}_x"] = point.x
                    row[f"{name}_y"] = point.y
                    row[f"{name}_z"] = point.z
                    row[f"{name}_visibility"] = point.visibility
                writer.writerow(row)
                rows_written += 1

            frame_idx += 1

    cap.release()
    print(f"Wrote {rows_written} frames (of {frame_idx} total) to {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract MediaPipe pose keypoints from a video.")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--out", required=True, help="Path to output CSV file")
    parser.add_argument("--min-confidence", type=float, default=0.5)
    args = parser.parse_args()

    extract_keypoints(args.video, args.out, args.min_confidence)
