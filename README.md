# CodeLens

Quality estimate for coding prompts *before* generation — a joint
DistilBERT+CodeBERT teacher model distilled into a lightweight
prompt-only student, served locally and surfaced live inside VS Code.

## Pipeline stages

| Stage | Location | What it does |
|---|---|---|
| 1. Mining | `src/mining/` | Filters raw ShareGPT/WildChat conversations into `(prompt, code)` pairs |
| 2. Labeling | `src/labeling/` | Computes ACQP (Automated Code Quality Proxy) scores via static analysis |
| 3. Features | `src/features/` | Prompt-side and code-side hand-crafted features |
| 4. Teacher | `src/models/teacher/` | Joint DistilBERT+CodeBERT model, trained on (prompt, code) → ACQP |
| 5. Student | `src/models/student/` | Prompt-only distilled model + FastAPI server for live scoring |
| 6. Eval | `src/eval/` | Ablation (M1–M5) and metrics |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
npm install -g eslint
huggingface-cli login
```

See `SETUP.md` for the full walkthrough (base tools, GPU notes, dataset
gating, extension scaffold, sanity checks).

## Running the pipeline end to end

```bash
# 1. pull raw corpora
python -m src.mining.load_corpora --dataset wildchat --out data/raw/wildchat
python -m src.mining.load_corpora --dataset sharegpt --out data/raw/sharegpt

# 2. filter into prompt-code pairs (adapt the raw schema in filter_pairs.py first)
python -m src.mining.filter_pairs --in data/raw/wildchat.jsonl --out data/interim/filtered_pairs.jsonl

# 3. label with ACQP + attach Stage 3 features
python -m src.labeling.build_dataset --in data/interim/filtered_pairs.jsonl --out data/interim/labeled_pairs.jsonl

# 4. train the joint teacher
python -m src.models.teacher.train --data data/interim/labeled_pairs.jsonl --out src/models/teacher/checkpoints/teacher.pt

# 5. distill into the prompt-only student
python -m src.models.student.distill --data data/interim/labeled_pairs.jsonl \
    --teacher_ckpt src/models/teacher/checkpoints/teacher.pt \
    --out src/models/student/checkpoints/student.pt

# 6. serve the student for the extension
uvicorn src.models.student.serve:app --port 8000 --reload
```

## VS Code extension

```bash
cd extension
npm install
npm run compile
```
Press **F5** in that VS Code window to launch an Extension Development
Host, then run **"CodeLens: Score Selected Prompt"** from the command
palette on a selected block of text.

## Tests

```bash
pytest tests/
```

## Team workflow

Branch per feature (`feature/data-mining`, `feature/teacher-model`,
`feature/extension-scaffold`, ...) so GitHub Insights → Contributors
reflects per-member work for Appendix B.
