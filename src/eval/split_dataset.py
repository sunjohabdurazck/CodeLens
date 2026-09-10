"""
Split pairs_labeled.csv into train/val/test CSVs.

Referenced by config.py (TRAIN_SPLIT/VAL_SPLIT/TEST_SPLIT), run_eval.sh, and
both evaluate.py scripts, but nothing produced these files before this.

Splits by conversation_id, not by row, so a prompt-code pair from the same
conversation can't end up in both train and test (that would leak
information the teacher/student could memorize rather than generalize
from).
"""
import argparse
import csv
import random
from pathlib import Path

from src.config import TRAIN_FRAC, VAL_FRAC, TEST_FRAC, RANDOM_SEED


def split_rows(rows: list, train_frac: float, val_frac: float, seed: int = RANDOM_SEED):
    ids = sorted({row["conversation_id"] for row in rows})
    rng = random.Random(seed)
    rng.shuffle(ids)

    n = len(ids)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train_ids = set(ids[:n_train])
    val_ids = set(ids[n_train:n_train + n_val])
    test_ids = set(ids[n_train + n_val:])

    train = [r for r in rows if r["conversation_id"] in train_ids]
    val = [r for r in rows if r["conversation_id"] in val_ids]
    test = [r for r in rows if r["conversation_id"] in test_ids]
    return train, val, test


def write_csv(path: Path, rows: list, fieldnames: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", required=True, help="pairs_labeled.csv")
    parser.add_argument("--train-out", required=True)
    parser.add_argument("--val-out", required=True)
    parser.add_argument("--test-out", required=True)
    parser.add_argument("--train-frac", type=float, default=TRAIN_FRAC)
    parser.add_argument("--val-frac", type=float, default=VAL_FRAC)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    in_path = Path(args.in_path)
    with in_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    if len({r["conversation_id"] for r in rows}) < 3:
        raise SystemExit(
            f"Only {len({r['conversation_id'] for r in rows})} distinct conversation_id(s) "
            f"in {in_path} — need at least 3 to form non-empty train/val/test splits. "
            "This will happen on a small smoke-test dataset; run against the real "
            "mined corpus before trusting the split."
        )

    train, val, test = split_rows(rows, args.train_frac, args.val_frac, args.seed)

    write_csv(Path(args.train_out), train, fieldnames)
    write_csv(Path(args.val_out), val, fieldnames)
    write_csv(Path(args.test_out), test, fieldnames)

    print(f"{len(rows)} rows -> train={len(train)}, val={len(val)}, test={len(test)} "
          f"(split by conversation_id, seed={args.seed})")


if __name__ == "__main__":
    main()