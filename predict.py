"""
predict.py

Runs a trained model (from train_classifier.py) on a new, unlabeled angles
CSV and prints a fault report per rep. This is the "give feedback on a new
video" step.

Usage:
    python pose_extraction.py --video new_clip.mp4 --out new_keypoints.csv
    python features.py --in new_keypoints.csv --out new_angles.csv
    python predict.py --angles new_angles.csv --model model.joblib
"""

import argparse
import joblib
import pandas as pd

from features import segment_reps, summarize_rep

FEEDBACK_TEXT = {
    "knee_valgus": "Knees are caving inward — focus on pushing them out over your toes.",
    "insufficient_depth": "Not reaching full depth — try to get your hips below knee level.",
    "back_rounding": "Back is rounding under load — brace your core and keep a neutral spine.",
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict form faults per rep from an angles CSV.")
    parser.add_argument("--angles", required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    bundle = joblib.load(args.model)
    models, feature_cols = bundle["models"], bundle["feature_cols"]

    angles_df = pd.read_csv(args.angles)
    reps = segment_reps(angles_df)

    if not reps:
        print("No reps detected — check the video/segmentation parameters.")
        raise SystemExit

    for rep_i, (start, end) in enumerate(reps):
        feat = summarize_rep(angles_df, start, end)
        X = pd.DataFrame([feat])[feature_cols]

        print(f"\nRep {rep_i + 1}:")
        any_fault = False
        for fault, clf in models.items():
            pred = clf.predict(X)[0]
            if pred == 1:
                any_fault = True
                msg = FEEDBACK_TEXT.get(fault, fault)
                print(f"  - {msg}")
        if not any_fault:
            print("  - Looks good, no faults detected.")
