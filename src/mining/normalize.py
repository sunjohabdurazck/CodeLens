"""
Normalize raw ShareGPT and WildChat records into the canonical shape
that filter_pairs.iter_pairs expects:

    {"id": str, "turns": [{"role": "user"|"assistant", "text": str}, ...]}

ShareGPT  : id, conversations[{from:"human"/"gpt", value}]
WildChat  : conversation_hash, conversation[{role:"user"/"assistant", content}]
            (turn-level; group by conversation_hash to reassemble)
"""
import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def normalize_sharegpt(rec: dict) -> dict | None:
    convs = rec.get("conversations")
    if not convs:
        return None
    turns = []
    for c in convs:
        role = "user" if c.get("from") == "human" else "assistant"
        text = c.get("value", "")
        if text:
            turns.append({"role": role, "text": text})
    if not turns:
        return None
    return {"id": str(rec.get("id", "unknown")), "turns": turns}


def normalize_wildchat_batch(records: list[dict]) -> list[dict]:
    """WildChat is turn-level: group rows by conversation_hash, sort by turn."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        h = r.get("conversation_hash")
        if h:
            groups[h].append(r)

    out = []
    for h, rows in groups.items():
        rows.sort(key=lambda r: r.get("turn", 0))
        turns = []
        for r in rows:
            for msg in r.get("conversation", []):
                role = msg.get("role")
                text = msg.get("content", "")
                if role in ("user", "assistant") and text:
                    turns.append({"role": role, "text": text})
        if turns:
            out.append({"id": h, "turns": turns})
    return out


def _iter_json_lines(f, in_path: Path):
    """Parse each non-empty line as JSON, skipping malformed lines with a warning
    instead of crashing the whole run."""
    for line_number, line in enumerate(f, start=1):
        if not line.strip():
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError as exc:
            logger.warning("Ligne %d de %s ignorée (JSON invalide) : %s", line_number, in_path, exc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", dest="out_path", required=True)
    ap.add_argument("--format", choices=["sharegpt", "wildchat"], required=True)
    args = ap.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)

    if not in_path.exists():
        raise SystemExit(f"Fichier d'entrée introuvable : {in_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with in_path.open(encoding="utf-8") as f, out_path.open("w", encoding="utf-8") as out:
        if args.format == "sharegpt":
            for rec in _iter_json_lines(f, in_path):
                norm = normalize_sharegpt(rec)
                if norm:
                    out.write(json.dumps(norm, ensure_ascii=False) + "\n")
                    written += 1
        else:  # wildchat
            records = list(_iter_json_lines(f, in_path))
            for norm in normalize_wildchat_batch(records):
                out.write(json.dumps(norm, ensure_ascii=False) + "\n")
                written += 1

    logger.info("Wrote %d normalized conversations to %s", written, out_path)


if __name__ == "__main__":
    main()
