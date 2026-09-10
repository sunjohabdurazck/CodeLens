# CodeLens — Full Environment Setup Guide

Starting point: VS Code and Python are already installed. This guide
takes you from there to a working repo with the ML pipeline and the
VS Code extension scaffold both runnable.

## 1. Check your base tools

```bash
python --version
git --version
node --version   # want Node LTS; Python 3.10 or 3.11 recommended
```

- If `node` isn't found: install Node.js LTS (v20.x) from https://nodejs.org — the extension side of CodeLens is TypeScript/Node, not Python.
- If `git` isn't found: install from https://git-scm.com.

## 2. Project structure

```
codelens/
├── data/                  # raw + filtered ShareGPT/WildChat dumps (gitignored)
├── src/
│   ├── mining/            # Stage 1: filtering conversations into prompt-code pairs
│   ├── labeling/          # Stage 2: ACQP computation (static analysis + execution)
│   ├── features/          # Stage 3: prompt-side & code-side feature engineering
│   ├── models/
│   │   ├── teacher/       # Stage 4: joint DistilBERT+CodeBERT model
│   │   └── student/       # Stage 5: prompt-only distilled model
│   └── eval/              # Stage 6: ablation (M1–M5) and metrics
├── extension/             # VS Code extension (TypeScript)
├── notebooks/             # exploratory analysis
├── tests/
├── requirements.txt
├── .gitignore
└── README.md
```

## 3. Python environment

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
```

VS Code should prompt you to select this interpreter — accept it, or
set it manually: `Ctrl+Shift+P` → `Python: Select Interpreter` → pick `.venv`.

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Training DistilBERT/CodeBERT is much faster with a GPU. If your
machine has an NVIDIA GPU, install the CUDA build of PyTorch instead
of the default (check https://pytorch.org/get-started/locally/ for the
exact command). Without a GPU, plan on Colab or a lab machine — CPU
training across the full ablation set (M1–M5) will be slow.

## 4. JavaScript static analysis (for JS/TS samples in ACQP)

```bash
npm install -g eslint
```

(You can also install it locally per-project if you'd rather not
pollute global npm.)

## 5. Getting the ShareGPT and WildChat data

```bash
huggingface-cli login
```

```python
from datasets import load_dataset

# WildChat (gated — you must accept the terms on the dataset page first)
wildchat = load_dataset("allenai/WildChat-1M")

# ShareGPT (community-hosted mirrors vary; check the dataset card for the current one)
sharegpt = load_dataset("anon8231489123/ShareGPT_Vicuna_unfiltered")
```

- **WildChat** requires accepting the dataset's usage terms on
  huggingface.co before `load_dataset` will work — visit the dataset
  page in a browser first, logged in with the same account as your CLI
  token.
- **ShareGPT** mirrors vary in cleanliness — pick one that's already
  deduplicated if possible, since Stage 1 filtering still does a
  second pass anyway.
- Save raw pulls under `data/raw/` — these are large files that don't
  belong in git (already gitignored).

## 6. VS Code extension scaffold

Already scaffolded in `extension/` in this repo. To rebuild from
scratch with Microsoft's official generator instead:

```bash
npm install -g yo generator-code
cd extension
yo code
```

Prompts: New Extension (TypeScript) → name `codelens` → identifier
`codelens` → description "quality estimate for coding prompts before
generation" → bundle with webpack: Yes → package manager: npm.

```bash
cd extension
npm install
npm run compile
```

Press **F5** to launch an Extension Development Host — a second VS
Code window with the extension loaded, where you can test the live
prompt-scoring UI.

The extension calls a local FastAPI server (`src/models/student/serve.py`)
over HTTP rather than running PyTorch inside Node.

## 7. Recommended VS Code extensions for the team

Install inside your **main** window (not the Extension Development Host):

- Python (`ms-python.python`) + Pylance
- Jupyter — for `notebooks/`
- ESLint — lint-as-you-type for the extension's TypeScript
- GitLens — useful for per-member commit tracking (Appendix B)

## 8. Sanity-check the pipeline end to end

```bash
python -c "import torch, transformers, datasets, sklearn, radon; print('python stack OK')"
node -e "console.log('node OK:', process.version)"
eslint --version
pylint --version
bandit --version
pytest tests/
```

If all of those pass, you're ready to start implementing/extending
Stage 1 (`src/mining/`) against the downloaded corpora.

## 9. Team workflow

```bash
git remote add origin <your-github-repo-url>
git add .
git commit -m "Initial project scaffold"
git push -u origin main
```

Since Appendix B tracks per-member commits and PRs, agree early on
branch-per-feature (`feature/data-mining`, `feature/teacher-model`,
`feature/extension-scaffold`) so GitHub Insights → Contributors
cleanly reflects who built what.
