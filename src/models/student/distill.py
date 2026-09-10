"""
Distill teacher model into prompt-only student.
"""
import argparse
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.teacher.dataset import PromptCodeACQPDataset
from src.models.teacher.model import JointTeacherModel, get_tokenizers
from src.models.student.model import PromptOnlyStudentModel

def distill(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    prompt_tok, code_tok = get_tokenizers()

    dataset = PromptCodeACQPDataset(args.data, prompt_tok, code_tok, max_length=args.max_length)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    teacher = JointTeacherModel().to(device)
    teacher.load_state_dict(torch.load(args.teacher_ckpt, map_location=device))
    teacher.eval()

    student = PromptOnlyStudentModel().to(device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr)
    loss_fn = torch.nn.MSELoss()

    student.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        for batch in tqdm(loader, desc=f"epoch {epoch + 1}/{args.epochs}"):
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.no_grad():
                soft_labels = teacher(
                    batch["prompt_input_ids"], batch["prompt_attention_mask"],
                    batch["code_input_ids"], batch["code_attention_mask"],
                )
            targets = args.alpha * soft_labels + (1 - args.alpha) * batch["label"]

            optimizer.zero_grad()
            preds = student(batch["prompt_input_ids"], batch["prompt_attention_mask"])
            loss = loss_fn(preds, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"epoch {epoch + 1}: avg loss = {total_loss / len(loader):.4f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(student.state_dict(), args.out)
    print(f"saved student checkpoint to {args.out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--teacher_ckpt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--max_length", type=int, default=256)
    distill(parser.parse_args())
