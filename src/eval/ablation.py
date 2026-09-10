"""
Ablation study runner (M1-M5).
"""
import argparse
import csv
import json
from src.eval.metrics import compute_metrics

def load_split(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def run_ablation(test_path: str, predictions: dict):
    examples = load_split(test_path)
    y_true = [float(ex["acqp_score"]) for ex in examples]

    results = {}
    for name, y_pred in predictions.items():
        assert len(y_pred) == len(y_true), f"{name}: prediction/label length mismatch"
        results[name] = compute_metrics(y_true, y_pred)

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", required=True)
    parser.add_argument("--predictions", required=True)
    args = parser.parse_args()

    with open(args.predictions, encoding="utf-8") as f:
        preds = json.load(f)

    results = run_ablation(args.test, preds)
    print(json.dumps(results, indent=2))
