"""
Train the joint teacher model.
"""
import argparse
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.teacher.dataset import PromptCodeACQPDataset
from src.models.teacher.model import JointTeacherModel, get_tokenizers

def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    prompt_tok, code_tok = get_tokenizers()

    dataset = PromptCodeACQPDataset(args.data, prompt_tok, code_tok, max_length=args.max_length)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = JointTeacherModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.MSELoss()

    model.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        for batch in tqdm(loader, desc=f"epoch {epoch + 1}/{args.epochs}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            preds = model(
                batch["prompt_input_ids"], batch["prompt_attention_mask"],
                batch["code_input_ids"], batch["code_attention_mask"],
            )
            loss = loss_fn(preds, batch["label"])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"epoch {epoch + 1}: avg loss = {total_loss / len(loader):.4f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(model.state_dict(), args.out)
    print(f"saved teacher checkpoint to {args.out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=256)
    train(parser.parse_args())
