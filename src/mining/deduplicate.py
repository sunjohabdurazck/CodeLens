"""
Stage 1: Remove near-duplicate prompt-code pairs.

Uses shingled Jaccard similarity on the code text (5-gram character
shingles) rather than exact-match hashing, because ShareGPT/WildChat
contain many pairs that are the same boilerplate answer with cosmetic
differences (whitespace, comments, added blank lines). Exact dedup would
miss those. O(n^2) comparison is fine at the scale a course-project mining
run produces (thousands, not millions, of pairs); swap in MinHash/LSH if
the corpus grows past that.

Known limitation: character shingles are sensitive to identifiers that
repeat many times in a snippet — renaming a variable used throughout the
function (e.g. "item" -> "product") touches every occurrence and can drop
similarity well below the threshold even though the code is structurally
identical. If your corpus has a lot of that pattern specifically (not just
generic near-duplicates), an AST-normalized comparison (strip identifier
names before shingling) would catch it; this module doesn't do that.
"""
import argparse
import json
from pathlib import Path

from src.config import DEDUP_JACCARD_THRESHOLD


def shingles(text: str, k: int = 5) -> set:
    text = " ".join(text.split())  # normalize whitespace
    if len(text) < k:
        return {text}
    return {text[i:i + k] for i in range(len(text) - k + 1)}


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def deduplicate(pairs: list, threshold: float = DEDUP_JACCARD_THRESHOLD) -> list:
    kept = []
    kept_shingles = []
    for pair in pairs:
        sh = shingles(pair["code"])
        if any(jaccard(sh, existing) >= threshold for existing in kept_shingles):
            continue
        kept.append(pair)
        kept_shingles.append(sh)
    return kept


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", required=True)
    parser.add_argument("--out", dest="out_path", required=True)
    parser.add_argument("--threshold", type=float, default=DEDUP_JACCARD_THRESHOLD)
    args = parser.parse_args()

    in_path, out_path = Path(args.in_path), Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with in_path.open(encoding="utf-8") as f:
        pairs = [json.loads(line) for line in f if line.strip()]

    deduped = deduplicate(pairs, args.threshold)

    with out_path.open("w", encoding="utf-8") as out_f:
        for pair in deduped:
            out_f.write(json.dumps(pair) + "\n")

    print(f"{len(pairs)} pairs -> {len(deduped)} after dedup "
          f"({len(pairs) - len(deduped)} removed, threshold={args.threshold})")


if __name__ == "__main__":
    main()
