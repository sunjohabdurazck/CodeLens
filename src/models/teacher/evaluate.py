"""
Evaluate the trained joint teacher model on a held-out split.

Writes a predictions JSON that src/eval/ablation.py can consume directly:

    {
        "M4_teacher": [float, float, ...],   # predicted ACQP, aligned with rows
        "M3_code_only": [float, ...]         # optional: same model on code only
    }

Usage:
    python -m src.models.teacher.evaluate \
        --data data/processed/test.csv \
        --ckpt src/models/teacher/checkpoints/teacher.pt \
        --out results/metrics/predictions_teacher.json
"""
import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import TEACHER_CHECKPOINT_DIR
from src.models.teacher.dataset import PromptCodeACQPDataset
from src.models.teacher.model import JointTeacherModel, get_tokenizers


def load_rows(path: Path) -> list:
    """Read pairs_labeled.csv (or a split derived from it) into a list of dicts."""
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def predict(model, loader, device, code_only: bool = False):
    """Run the model over `loader`. If code_only, blank out the prompt side so
    the joint head only sees code — used for the M3 ablation variant.

    NOTE: this does NOT zero the attention mask. A fully-zeroed attention
    mask means every position is masked out, which sends some transformer
    implementations' softmax straight to NaN (nothing left to normalize
    over). Instead we keep the mask's first position (the [CLS]-equivalent
    token) valid and swap the input ids for the tokenizer's pad token, so
    the encoder still runs a real forward pass but sees no prompt content.
    """
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="predict", leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            prompt_ids = batch["prompt_input_ids"]
            prompt_mask = batch["prompt_attention_mask"]
            if code_only:
                pad_id = model.prompt_encoder.config.pad_token_id or 0
                prompt_ids = torch.full_like(prompt_ids, pad_id)
                prompt_mask = torch.zeros_like(prompt_mask)
                prompt_mask[:, 0] = 1  # keep one valid position, avoid all-masked softmax
            out = model(
                prompt_ids, prompt_mask,
                batch["code_input_ids"], batch["code_attention_mask"],
            )
            preds.extend(out.cpu().tolist())
    return preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="CSV split to evaluate on")
    parser.add_argument("--ckpt", default=str(TEACHER_CHECKPOINT_DIR / "teacher.pt"))
    parser.add_argument("--out", required=True, help="Where to write predictions JSON")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--code_only", action="store_true",
                        help="Also produce the M3 code-only variant in the same JSON")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    prompt_tok, code_tok = get_tokenizers()

    rows = load_rows(Path(args.data))
    if not rows:
        raise SystemExit(f"No rows in {args.data}")

    dataset = PromptCodeACQPDataset(args.data, prompt_tok, code_tok,
                                    max_length=args.max_length)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    model = JointTeacherModel().to(device)
    state = torch.load(args.ckpt, map_location=device, weights_only=True)
    model.load_state_dict(state)
    print(f"Loaded teacher checkpoint from {args.ckpt} ({len(rows)} rows)")

    out = {"M4_teacher": predict(model, loader, device, code_only=False)}
    if args.code_only:
        out["M3_code_only"] = predict(model, loader, device, code_only=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {list(out.keys())} to {out_path}")


if __name__ == "__main__":
    main()
