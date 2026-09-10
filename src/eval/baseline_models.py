# src/eval/baseline_models.py
"""
M1 and M2 baselines for the ablation study.

M1: logistic-regression / ridge regression over hand-crafted prompt features.
M2: (unimplemented here) DistilBERT over prompt text, no distillation.

M1 is sklearn-only and runs anywhere. M2 needs HF weights; it's left as a
stub that raises NotImplementedError with a pointer to Colab until we can
train it alongside the teacher.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_predict


PROMPT_FEATURE_KEYS = [
    "char_length", "word_count", "sentence_count", "avg_word_length",
    "has_code_snippet", "specificity_marker_count", "ambiguity_marker_count",
    "question_mark_count", "ends_with_question",
]


def load_features_and_labels(csv_path: Path):
    X, y = [], []
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pf = json.loads(row["prompt_features"])
            X.append([float(pf.get(k, 0)) for k in PROMPT_FEATURE_KEYS])
            y.append(float(row["acqp_score"]))
    return np.asarray(X), np.asarray(y)


def run_m1(csv_path: Path, seed: int = 42):
    """Cross-validated M1 predictions. Returns a list aligned with rows."""
    X, y = load_features_and_labels(csv_path)
    if len(X) < 5:
        raise SystemExit(f"M1 needs at least 5 rows, got {len(X)}")
    kf = KFold(n_splits=min(5, len(X)), shuffle=True, random_state=seed)
    preds = cross_val_predict(Ridge(alpha=1.0), X, y, cv=kf)
    return preds.tolist()


def run_m2(csv_path: Path):
    raise NotImplementedError(
        "M2 (DistilBERT over prompt text, no distillation) needs pretrained "
        "HF weights and a GPU. Train it in Colab with src/models/teacher/train.py "
        "minus the code encoder, or wait for the shared teacher run."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--include-m2", action="store_true")
    args = parser.parse_args()

    csv_path = Path(args.data)
    out = {"M1_prompt_features": run_m1(csv_path)}
    if args.include_m2:
        out["M2_distilbert_prompt"] = run_m2(csv_path)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {list(out.keys())} to {args.out}")


if __name__ == "__main__":
    main()
