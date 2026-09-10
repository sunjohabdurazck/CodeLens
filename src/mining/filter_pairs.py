"""
Stage 1: Filter conversations into prompt-code pairs.
"""
import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

CODE_BLOCK_RE = re.compile(r"```(\w*)\n(.*?)```", re.DOTALL)
CODE_REQUEST_RE = re.compile(r"\b(write|implement|create|fix|debug|refactor|code|function|script|class|program)\b", re.IGNORECASE)

@dataclass
class PromptCodePair:
    conversation_id: str
    prompt: str
    code: str
    language: str

def is_code_request(text: str) -> bool:
    return bool(CODE_REQUEST_RE.search(text))

def extract_code_blocks(text: str):
    for lang, code in CODE_BLOCK_RE.findall(text):
        yield (lang or "unknown", code.strip())

def iter_pairs(conversations: Iterator[dict]) -> Iterator[PromptCodePair]:
    for conv in conversations:
        turns = conv.get("turns", [])
        for i in range(len(turns) - 1):
            user_turn, next_turn = turns[i], turns[i + 1]
            if user_turn.get("role") != "user" or next_turn.get("role") != "assistant":
                continue
            if not is_code_request(user_turn["text"]):
                continue
            for lang, code in extract_code_blocks(next_turn["text"]):
                if len(code) < 10:
                    continue
                yield PromptCodePair(
                    conversation_id=conv.get("id", "unknown"),
                    prompt=user_turn["text"].strip(),
                    code=code,
                    language=lang,
                )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", required=True)
    parser.add_argument("--out", dest="out_path", required=True)
    args = parser.parse_args()

    in_path, out_path = Path(args.in_path), Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def read_conversations():
        with in_path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)

    count = 0
    with out_path.open("w", encoding="utf-8") as out_f:
        for pair in iter_pairs(read_conversations()):
            out_f.write(json.dumps(asdict(pair)) + "\n")
            count += 1

    print(f"Wrote {count} prompt-code pairs to {out_path}")

if __name__ == "__main__":
    main()
