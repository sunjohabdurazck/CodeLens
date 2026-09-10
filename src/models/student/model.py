"""
Prompt-only student model for distillation.
"""
import torch.nn as nn
from transformers import AutoModel

PROMPT_ENCODER = "distilbert-base-uncased"

class PromptOnlyStudentModel(nn.Module):
    def __init__(self, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(PROMPT_ENCODER)
        self.regressor = nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, input_ids, attention_mask):
        pooled = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        return self.regressor(pooled).squeeze(-1)
