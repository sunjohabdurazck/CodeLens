"""
Stage 1, Pass 2: lightweight classifier filter.

filter_pairs.py's Pass 1 is a keyword regex — high recall, low precision
(it'll pass "can you fix my code" as readily as an actual code-writing
request that just doesn't mention a trigger word). This is a real
TF-IDF + logistic-regression classifier trained on labeled examples, not
a pretrained model, specifically because pretrained weights aren't
reachable from this environment (no huggingface.co access) and don't need
to be: this is a short-text binary classification problem TF-IDF handles
fine.

It ships with a small seed training set (SEED_EXAMPLES below) so the
pipeline runs end-to-end out of the box, but a seed set this size will
overfit to obvious cases. Before you trust its output for the DP results,
hand-label 150-300 real pairs from your Pass-1 output
(data/samples/classifier_train.csv, columns: prompt,label) and retrain:

    python -m src.mining.filter_classifier --train data/samples/classifier_train.csv \
        --model-out models/filter_classifier.joblib
"""
import argparse
import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config import CLASSIFIER_MIN_CONFIDENCE, RANDOM_SEED

# Minimal bootstrap set: clear positive (genuine code-generation requests)
# and clear negative (code-adjacent but not a generation request) examples.
# Replace with real hand-labeled data before trusting this for results.
SEED_EXAMPLES = [
    ("Write a Python function that merges two sorted lists", 1),
    ("Implement a binary search tree with insert and delete methods", 1),
    ("Can you write me a REST API endpoint in Flask for user login", 1),
    ("Create a React component that fetches and displays a list of posts", 1),
    ("Refactor this function to remove the nested loops", 1),
    ("Debug why my recursive fibonacci function is running so slowly", 1),
    ("Generate a SQL query to find the top 5 customers by revenue", 1),
    ("Fix the off-by-one error in this array indexing", 1),
    ("What does this error message mean?", 0),
    ("Can you explain how garbage collection works in Python?", 0),
    ("What's the difference between a list and a tuple?", 0),
    ("Is this a good approach architecturally?", 0),
    ("Thanks, that worked!", 0),
    ("How do I install Python on Windows?", 0),
    ("What's your opinion on microservices vs monoliths?", 0),
    ("Can you review this code and tell me if it looks okay?", 0),
]


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=5000)),
        ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_SEED, class_weight="balanced")),
    ])


def train(examples=None) -> Pipeline:
    examples = examples or SEED_EXAMPLES
    texts, labels = zip(*examples)
    pipeline = build_pipeline()
    pipeline.fit(list(texts), list(labels))
    return pipeline


def load_labeled_csv(path: Path):
    import csv
    examples = []
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            examples.append((row["prompt"], int(row["label"])))
    return examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", help="CSV with prompt,label columns; defaults to built-in seed set")
    parser.add_argument("--model-out", help="Where to save the trained model (joblib)")
    parser.add_argument("--in", dest="in_path", help="JSONL of pairs to filter")
    parser.add_argument("--out", dest="out_path", help="JSONL of pairs that passed the classifier")
    parser.add_argument("--threshold", type=float, default=CLASSIFIER_MIN_CONFIDENCE)
    args = parser.parse_args()

    examples = load_labeled_csv(Path(args.train)) if args.train else None
    pipeline = train(examples)

    if args.model_out:
        import joblib
        Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, args.model_out)
        print(f"Saved classifier to {args.model_out}")

    if args.in_path:
        in_path, out_path = Path(args.in_path), Path(args.out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        kept, total = 0, 0
        with in_path.open(encoding="utf-8") as in_f, out_path.open("w", encoding="utf-8") as out_f:
            for line in in_f:
                if not line.strip():
                    continue
                pair = json.loads(line)
                total += 1
                proba = pipeline.predict_proba([pair["prompt"]])[0][1]
                if proba >= args.threshold:
                    pair["classifier_confidence"] = round(float(proba), 4)
                    out_f.write(json.dumps(pair) + "\n")
                    kept += 1
        print(f"{total} pairs -> {kept} kept (threshold={args.threshold})")


if __name__ == "__main__":
    main()
