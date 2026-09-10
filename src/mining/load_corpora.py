"""
Load raw corpora from Hugging Face.
"""
import argparse
import datetime
import json
import os
from datasets import load_dataset

DATASET_IDS = {
    "wildchat": "allenai/WildChat-1M",
    "sharegpt": "Hwaple/ShareGPT52K",   # working 90K mirror
}

def _json_default(o):
    """Serialize HF's datetime objects (and anything else json can't handle)."""
    if isinstance(o, (datetime.datetime, datetime.date)):
        return o.isoformat()
    return str(o)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASET_IDS.keys(), required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=1000, help="Number of samples to save")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print(f"Loading {args.dataset} from Hugging Face...")

    ds = load_dataset(DATASET_IDS[args.dataset], split="train", streaming=True)

    samples = []
    for i, item in enumerate(ds):
        if i >= args.limit:
            break
        samples.append(item)

    output_path = os.path.join(args.out, "samples.jsonl")
    with open(output_path, "w", encoding="utf-8") as f:
        for item in samples:
            f.write(json.dumps(item, default=_json_default) + "\n")

    print(f"Saved {len(samples)} samples to {output_path}")

if __name__ == "__main__":
    main()