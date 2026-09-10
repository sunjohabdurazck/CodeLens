"""
Stage 1 orchestrator: raw conversations -> pairs_raw.csv

Wires: filter_pairs (keyword pass) -> filter_classifier (Pass 2) ->
deduplicate -> remove_pii -> pairs_raw.csv

Each intermediate step still writes its own JSONL under data/interim/ so
you can inspect what each stage dropped and why (that's report material
for Section 5.1 — pass-through rates per stage).
"""
import argparse
import csv
import json
from pathlib import Path

from src.config import DEDUP_JACCARD_THRESHOLD, PAIRS_RAW, INTERIM_DIR
from src.mining.filter_pairs import iter_pairs
from src.mining.filter_classifier import train as train_classifier
from src.mining.deduplicate import deduplicate
from src.mining.remove_pii import scrub_pair


def build(in_path: Path, out_path: Path, classifier_threshold: float,
          dedup_threshold: float, classifier_train_csv: Path = None):
    interim = INTERIM_DIR
    interim.mkdir(parents=True, exist_ok=True)

    def read_conversations():
        with in_path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)

    # Pass 1: keyword filter + pair extraction
    stage1 = [pair.__dict__ if hasattr(pair, "__dict__") else pair
              for pair in iter_pairs(read_conversations())]
    stage1_path = interim / f"{in_path.stem}_pass1.jsonl"
    with stage1_path.open("w", encoding="utf-8") as f:
        for p in stage1:
            f.write(json.dumps(p) + "\n")
    print(f"[Pass 1: keyword filter] {len(stage1)} pairs -> {stage1_path}")

    # Pass 2: classifier filter
    from src.mining.filter_classifier import load_labeled_csv
    examples = load_labeled_csv(classifier_train_csv) if classifier_train_csv else None
    clf = train_classifier(examples)
    stage2 = []
    for pair in stage1:
        proba = clf.predict_proba([pair["prompt"]])[0][1]
        if proba >= classifier_threshold:
            pair = {**pair, "classifier_confidence": round(float(proba), 4)}
            stage2.append(pair)
    stage2_path = interim / f"{in_path.stem}_pass2.jsonl"
    with stage2_path.open("w", encoding="utf-8") as f:
        for p in stage2:
            f.write(json.dumps(p) + "\n")
    print(f"[Pass 2: classifier] {len(stage1)} -> {len(stage2)} pairs -> {stage2_path}")

    # Dedup
    stage3 = deduplicate(stage2, threshold=dedup_threshold)
    print(f"[Dedup] {len(stage2)} -> {len(stage3)} pairs")

    # PII scrub
    stage4 = [scrub_pair(p) for p in stage3]
    redacted_count = sum(1 for p in stage4 if "pii_redactions" in p)
    print(f"[PII scrub] {redacted_count}/{len(stage4)} pairs had redactions")

    # Write pairs_raw.csv — flat schema
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["conversation_id", "prompt", "code", "language",
                  "classifier_confidence", "pii_redactions"]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for p in stage4:
            row = dict(p)
            if "pii_redactions" in row:
                row["pii_redactions"] = json.dumps(row["pii_redactions"])
            writer.writerow(row)

    print(f"Wrote {len(stage4)} rows to {out_path}")
    return stage4


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", required=True,
                         help="raw conversations JSONL (one conversation dict per line)")
    parser.add_argument("--out", dest="out_path", default=str(PAIRS_RAW))
    parser.add_argument("--classifier-threshold", type=float, default=0.5)
    parser.add_argument("--dedup-threshold", type=float, default=DEDUP_JACCARD_THRESHOLD)
    parser.add_argument("--classifier-train", help="labeled CSV to pass 2 classifier on")
    args = parser.parse_args()

    build(
        Path(args.in_path), Path(args.out_path),
        args.classifier_threshold, args.dedup_threshold,
        Path(args.classifier_train) if args.classifier_train else None,
    )


if __name__ == "__main__":
    main()