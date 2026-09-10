"""
FastAPI server for student model inference.
"""
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer

from src.models.student.model import PromptOnlyStudentModel, PROMPT_ENCODER

CHECKPOINT_PATH = "src/models/student/checkpoints/student.pt"
MAX_LENGTH = 256

app = FastAPI(title="CodeLens Student Scoring Server")

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(PROMPT_ENCODER)
model = PromptOnlyStudentModel().to(device)

try:
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    print(f"Loaded student checkpoint from {CHECKPOINT_PATH}")
except FileNotFoundError:
    print(f"WARNING: no checkpoint found at {CHECKPOINT_PATH}")

model.eval()

class ScoreRequest(BaseModel):
    prompt: str

class ScoreResponse(BaseModel):
    score: float

@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest):
    enc = tokenizer(
        req.prompt, truncation=True, padding="max_length",
        max_length=MAX_LENGTH, return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        pred = model(enc["input_ids"], enc["attention_mask"])

    return ScoreResponse(score=float(pred.item()))

@app.get("/health")
def health():
    return {"status": "ok"}
