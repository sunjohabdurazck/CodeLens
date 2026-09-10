"""
Stage 2 orchestrator: pairs_raw.csv -> pairs_labeled.csv (13-field schema)

Reads CSV, not JSONL — pairs_raw.csv is a CSV per the data schema in the
report and in mining/build_dataset.py's output. (This module used to read
JSONL, which meant Stage 1's real output could never feed Stage 2 without
a manual conversion step. Fixed here.)
"""
import argparse
import csv
import json
from pathlib import Path

from src.labeling.acqp import compute_acqp
from src.labeling.execution_check import check_python_execution
from src.features.prompt_features import extract_prompt_features
from src.features.code_features import extract_code_features

FIELDNAMES = [
    "conversation_id", "prompt", "code", "language",
    "acqp_score", "acqp_features",
    "prompt_features", "code_features",
    "execution_check", "classifier_confidence", "pii_redactions",
]


def label_pair(pair: dict) -> dict:
    acqp_result = compute_acqp(pair["code"], pair["language"])
    record = {
        **pair,
        "acqp_score": acqp_result["acqp_score"],
        "acqp_features": acqp_result["features"],
        "prompt_features": extract_prompt_features(pair["prompt"]),
        "code_features": extract_code_features(pair["code"], pair["language"]),
    }
    if pair.get("language", "").lower() in ("python", "py"):
        record["execution_check"] = check_python_execution(pair["code"])
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", required=True, help="pairs_raw.csv")
    parser.add_argument("--out", dest="out_path", required=True, help="pairs_labeled.csv")
    args = parser.parse_args()

    in_path, out_path = Path(args.in_path), Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with in_path.open(newline="", encoding="utf-8") as in_f, out_path.open("w", newline="", encoding="utf-8") as out_f:
        reader = csv.DictReader(in_f)
        writer = csv.DictWriter(out_f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()

        for pair in reader:
            record = label_pair(pair)
            if record["acqp_score"] is None:
                continue

            row = dict(record)
            for json_field in ("acqp_features", "prompt_features", "code_features", "execution_check"):
                if json_field in row:
                    row[json_field] = json.dumps(row[json_field])
            writer.writerow(row)
            count += 1

    print(f"Wrote {count} labeled+featurized examples to {out_path}")


if __name__ == "__main__":
    main()
