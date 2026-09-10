#!/usr/bin/env python
"""
Download a small sample of data for testing.

Note: this needs huggingface.co access to actually run. It will not work
in an offline/restricted-network environment -- that's not a bug in this
script, it's the whole point of it. Use tests/fixtures/sample_pairs.csv or
the smoke-test conversations in the README for pipeline testing without
real internet access.
"""

from datasets import load_dataset
import json
from pathlib import Path


def download_sample():
    """Download a small sample of WildChat data."""
    print("Downloading sample data from WildChat...")

    dataset = load_dataset("allenai/WildChat-1M", split="train", streaming=True)

    sample = []
    for i, item in enumerate(dataset):
        if i >= 1000:
            break
        sample.append(item)

    output_path = Path("data/raw/wildchat_sample.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for item in sample:
            f.write(json.dumps(item) + "\n")

    print(f"Saved {len(sample)} samples to {output_path}")
    return output_path


if __name__ == "__main__":
    download_sample()
