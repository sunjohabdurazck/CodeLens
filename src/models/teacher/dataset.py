"""
PyTorch Dataset for teacher model.

Reads pairs_labeled.csv / train.csv / val.csv / test.csv — CSV, not JSONL,
matching the rest of the pipeline (mining/build_dataset.py and
labeling/build_dataset.py both write CSV). This used to read JSONL and
would crash on real pipeline output.
"""
import csv
import json
import torch
from torch.utils.data import Dataset

class PromptCodeACQPDataset(Dataset):
    def __init__(self, csv_path, prompt_tokenizer, code_tokenizer, max_length: int = 256):
        with open(csv_path, newline="", encoding="utf-8") as f:
            self.examples = list(csv.DictReader(f))

        self.prompt_tokenizer = prompt_tokenizer
        self.code_tokenizer = code_tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]

        prompt_enc = self.prompt_tokenizer(
            ex["prompt"], truncation=True, padding="max_length",
            max_length=self.max_length, return_tensors="pt",
        )
        code_enc = self.code_tokenizer(
            ex["code"], truncation=True, padding="max_length",
            max_length=self.max_length, return_tensors="pt",
        )

        return {
            "prompt_input_ids": prompt_enc["input_ids"].squeeze(0),
            "prompt_attention_mask": prompt_enc["attention_mask"].squeeze(0),
            "code_input_ids": code_enc["input_ids"].squeeze(0),
            "code_attention_mask": code_enc["attention_mask"].squeeze(0),
            "label": torch.tensor(float(ex["acqp_score"]), dtype=torch.float),
        }
