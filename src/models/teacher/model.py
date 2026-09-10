"""
Joint teacher model: DistilBERT (prompt) + CodeBERT (code).
"""
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

PROMPT_ENCODER = "distilbert-base-uncased"
CODE_ENCODER = "microsoft/codebert-base"

class JointTeacherModel(nn.Module):
    def __init__(self, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.prompt_encoder = AutoModel.from_pretrained(PROMPT_ENCODER)
        self.code_encoder = AutoModel.from_pretrained(CODE_ENCODER)

        combined_dim = self.prompt_encoder.config.hidden_size + self.code_encoder.config.hidden_size
        self.regressor = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, prompt_input_ids, prompt_attention_mask, code_input_ids, code_attention_mask):
        prompt_out = self.prompt_encoder(
            input_ids=prompt_input_ids, attention_mask=prompt_attention_mask
        ).last_hidden_state[:, 0, :]
        code_out = self.code_encoder(
            input_ids=code_input_ids, attention_mask=code_attention_mask
        ).last_hidden_state[:, 0, :]

        combined = torch.cat([prompt_out, code_out], dim=-1)
        return self.regressor(combined).squeeze(-1)

def get_tokenizers():
    return (
        AutoTokenizer.from_pretrained(PROMPT_ENCODER),
        AutoTokenizer.from_pretrained(CODE_ENCODER),
    )
