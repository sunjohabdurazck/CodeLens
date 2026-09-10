"""
Stage 1: Strip PII and secrets from prompt-code pairs before they're persisted.

This runs pattern-based redaction, not a claim of perfect PII removal — it
catches the categories that actually show up in ShareGPT/WildChat dumps
(emails, phone numbers, API keys/tokens, IPs, credit-card-shaped numbers)
and replaces them with typed placeholders so downstream features still see
"there was an email here" without seeing the email itself.
"""
import argparse
import json
import re
from pathlib import Path

PATTERNS = [
    ("EMAIL", re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")),
    ("PHONE", re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}\b")),
    ("CREDIT_CARD", re.compile(r"\b(?:\d[ -]*?){13,16}\b")),
    ("IPV4", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
    # Generic secret-shaped tokens: aws/gcp/openai/github style keys, long
    # base64/hex blobs assigned to a var/env-looking name.
    ("SECRET", re.compile(
        r"(?i)\b(api[_-]?key|secret|token|password|passwd|pwd)\b\s*[:=]\s*['\"]?[A-Za-z0-9_\-/+]{12,}['\"]?"
    )),
    ("AWS_KEY", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("GITHUB_TOKEN", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b")),
]


def scrub(text: str) -> tuple[str, dict]:
    """Redact PII in `text`. Returns (scrubbed_text, counts_by_type)."""
    counts = {}
    for label, pattern in PATTERNS:
        text, n = pattern.subn(f"[REDACTED_{label}]", text)
        if n:
            counts[label] = n
    return text, counts


def scrub_pair(pair: dict) -> dict:
    prompt_clean, prompt_counts = scrub(pair["prompt"])
    code_clean, code_counts = scrub(pair["code"])
    pair = {**pair, "prompt": prompt_clean, "code": code_clean}
    redactions = {**{f"prompt_{k}": v for k, v in prompt_counts.items()},
                  **{f"code_{k}": v for k, v in code_counts.items()}}
    if redactions:
        pair["pii_redactions"] = redactions
    return pair


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", required=True)
    parser.add_argument("--out", dest="out_path", required=True)
    args = parser.parse_args()

    in_path, out_path = Path(args.in_path), Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total, redacted = 0, 0
    with in_path.open(encoding="utf-8") as in_f, out_path.open("w", encoding="utf-8") as out_f:
        for line in in_f:
            if not line.strip():
                continue
            pair = json.loads(line)
            cleaned = scrub_pair(pair)
            if "pii_redactions" in cleaned:
                redacted += 1
            out_f.write(json.dumps(cleaned) + "\n")
            total += 1

    print(f"Scrubbed {total} pairs, {redacted} had at least one redaction. Wrote {out_path}")


if __name__ == "__main__":
    main()
