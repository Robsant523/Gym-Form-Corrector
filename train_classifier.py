"""
train_classifier.py

Stage 3 of the pipeline: takes an angles CSV (from features.py) plus a labels
CSV you create by hand, segments it into reps, builds a feature table, and
trains a multi-label Random Forest (one model per fault type).

Expected labels CSV format (one row per rep, in rep order, matching the video
you extracted features from):

    rep_index,knee_valgus,insufficient_depth,back_rounding
    0,0,0,0
    1,1,0,0
    2,0,1,0
    ...

Usage:
    python train_classifier.py --angles angles.csv --labels labels.csv --out model.joblib
"""

import argparse
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from features import segment_reps, summarize_rep

FAULT_COLUMNS_HINT = ["knee_valgus", "insufficient_depth", "back_rounding"]


def build_feature_table(angles_df: pd.DataFrame, labels_df: pd.DataFrame) -> pd.DataFrame:
    reps = segment_reps(angles_df)
    if len(reps) != len(labels_df):
        raise ValueError(
            f"Found {len(reps)} reps in the video but {len(labels_df)} labeled rows. "
            "They must match 1:1 — check segmentation or re-label."
        )

    rows = []
    for (start, end), (_, label_row) in zip(reps, labels_df.iterrows()):
        feat = summarize_rep(angles_df, start, end)
        feat.update(label_row.to_dict())
        rows.append(feat)

    return pd.DataFrame(rows)


def train(feature_table: pd.DataFrame, fault_columns: list):
    feature_cols = [c for c in feature_table.columns if c not in fault_columns and c != "rep_index"]
    X = feature_table[feature_cols]

    models = {}
    for fault in fault_columns:
        y = feature_table[fault]
        if y.nunique() < 2:
            print(f"Skipping '{fault}': only one class present in this dataset (need both 0 and 1 examples).")
            continue

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
        clf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
        clf.fit(X_train, y_train)

        print(f"\n=== {fault} ===")
        print(classification_report(y_test, clf.predict(X_test), zero_division=0))

        models[fault] = clf

    return models, feature_cols


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train per-fault classifiers from an angles CSV and a labels CSV.")
    parser.add_argument("--angles", required=True, help="CSV from features.py")
    parser.add_argument("--labels", required=True, help="Your hand-labeled CSV (see docstring for format)")
    parser.add_argument("--out", default="model.joblib", help="Where to save the trained models")
    args = parser.parse_args()

    angles_df = pd.read_csv(args.angles)
    labels_df = pd.read_csv(args.labels)

    fault_columns = [c for c in labels_df.columns if c != "rep_index"]
    print(f"Fault types found in labels file: {fault_columns}")

    table = build_feature_table(angles_df, labels_df)
    models, feature_cols = train(table, fault_columns)

    joblib.dump({"models": models, "feature_cols": feature_cols}, args.out)
    print(f"\nSaved {len(models)} fault-specific models to {args.out}")
