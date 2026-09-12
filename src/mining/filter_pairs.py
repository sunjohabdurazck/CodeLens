"""
Stage 1: Filter conversations into prompt-code pairs.
"""

import argparse
import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Bloc de code délimité par des triples backticks, avec langage optionnel.
CODE_BLOCK_RE = re.compile(r"```(\w*)\n(.*?)```", re.DOTALL)
CODE_REQUEST_RE = re.compile(
    r"\b(write|implement|create|fix|debug|refactor|code|function|script|class|program)\b",
    re.IGNORECASE,
)

DEFAULT_MIN_CODE_LENGTH = 10


@dataclass
class PromptCodePair:
    conversation_id: str
    prompt: str
    code: str
    language: str


def is_code_request(text: str) -> bool:
    return bool(CODE_REQUEST_RE.search(text))


def extract_code_blocks(text: str) -> Iterator[tuple[str, str]]:
    for lang, code in CODE_BLOCK_RE.findall(text):
        cleaned = code.strip()
        if cleaned:
            yield (lang.strip().lower() or "unknown", cleaned)


def iter_pairs(
    conversations: Iterator[dict],
    min_code_length: int = DEFAULT_MIN_CODE_LENGTH,
) -> Iterator[PromptCodePair]:
    for conv in conversations:
        conv_id = conv.get("id", "unknown")
        turns = conv.get("turns", [])

        for i in range(len(turns) - 1):
            user_turn, next_turn = turns[i], turns[i + 1]

            if user_turn.get("role") != "user" or next_turn.get("role") != "assistant":
                continue

            user_text = user_turn.get("text")
            assistant_text = next_turn.get("text")
            if not user_text or not assistant_text:
                # Tour incomplet ou malformé : on l'ignore proprement plutôt
                # que de lever un KeyError.
                continue

            if not is_code_request(user_text):
                continue

            for lang, code in extract_code_blocks(assistant_text):
                if len(code) < min_code_length:
                    continue
                yield PromptCodePair(
                    conversation_id=conv_id,
                    prompt=user_text.strip(),
                    code=code,
                    language=lang,
                )


def read_conversations(in_path: Path) -> Iterator[dict]:
    with in_path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                # Une ligne corrompue ne doit pas interrompre tout le traitement.
                logger.warning("Ligne %d ignorée (JSON invalide) : %s", line_number, exc)


def run(in_path: Path, out_path: Path, min_code_length: int = DEFAULT_MIN_CODE_LENGTH) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with out_path.open("w", encoding="utf-8") as out_f:
        for pair in iter_pairs(read_conversations(in_path), min_code_length=min_code_length):
            out_f.write(json.dumps(asdict(pair), ensure_ascii=False) + "\n")
            count += 1

    return count


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extrait les paires prompt/code d'un fichier de conversations JSONL.")
    parser.add_argument("--in", dest="in_path", required=True, type=Path, help="Fichier JSONL d'entrée")
    parser.add_argument("--out", dest="out_path", required=True, type=Path, help="Fichier JSONL de sortie")
    parser.add_argument(
        "--min-code-length",
        dest="min_code_length",
        type=int,
        default=DEFAULT_MIN_CODE_LENGTH,
        help="Longueur minimale (en caractères) d'un bloc de code pour être conservé",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()

    if not args.in_path.exists():
        raise SystemExit(f"Fichier d'entrée introuvable : {args.in_path}")

    count = run(args.in_path, args.out_path, min_code_length=args.min_code_length)
    logger.info("Wrote %d prompt-code pairs to %s", count, args.out_path)


if __name__ == "__main__":
    main()
