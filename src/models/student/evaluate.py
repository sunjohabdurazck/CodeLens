"""
Evaluate the distilled prompt-only student on a held-out split.

Writes the M5 prediction list that src/eval/ablation.py consumes.

    {"M5_student": [float, float, ...]}

Usage:
    python -m src.models.student.evaluate \
        --data data/processed/test.csv \
        --ckpt src/models/student/checkpoints/student.pt \
        --out results/metrics/predictions_student.json
"""
import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from tqdm import tqdm

from src.config import STUDENT_CHECKPOINT_DIR
from src.models.student.model import PromptOnlyStudentModel, PROMPT_ENCODER


class PromptOnlyDataset(Dataset):
    """Minimal prompt-only dataset for evaluation. No labels needed."""

    def __init__(self, csv_path: Path, tokenizer, max_length: int = 256):
        with csv_path.open(newline="", encoding="utf-8") as f:
            self.rows = list(csv.DictReader(f))
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        prompt = self.rows[idx]["prompt"]
        enc = self.tokenizer(
            prompt, truncation=True, padding="max_length",
            max_length=self.max_length, return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="CSV split to evaluate on")
    parser.add_argument("--ckpt", default=str(STUDENT_CHECKPOINT_DIR / "student.pt"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(PROMPT_ENCODER)
    dataset = PromptOnlyDataset(Path(args.data), tokenizer, max_length=args.max_length)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    model = PromptOnlyStudentModel().to(device)
    state = torch.load(args.ckpt, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded student checkpoint from {args.ckpt} ({len(dataset)} rows)")

    preds = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="predict", leave=False):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            out = model(ids, mask)
            preds.extend(out.cpu().tolist())

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({"M5_student": preds}, f, indent=2)
    print(f"Wrote M5_student to {out_path}")


if __name__ == "__main__":
    main()
