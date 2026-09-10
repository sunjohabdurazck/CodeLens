"""
Central configuration: paths and constants shared by every stage.
Nothing in mining/, labeling/, features/, models/, or eval/ should hardcode
a path — import it from here instead.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
SAMPLES_DIR = DATA_DIR / "samples"

MODELS_DIR = ROOT / "models"
TEACHER_CHECKPOINT_DIR = ROOT / "src" / "models" / "teacher" / "checkpoints"
STUDENT_CHECKPOINT_DIR = ROOT / "src" / "models" / "student" / "checkpoints"

RESULTS_DIR = ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
LOGS_DIR = RESULTS_DIR / "logs"

# Stage 1 outputs
SHAREGPT_RAW = RAW_DIR / "sharegpt"
WILDCHAT_RAW = RAW_DIR / "wildchat"
SHAREGPT_FILTERED = INTERIM_DIR / "sharegpt_filtered.jsonl"
WILDCHAT_FILTERED = INTERIM_DIR / "wildchat_filtered.jsonl"
PAIRS_RAW = PROCESSED_DIR / "pairs_raw.csv"

# Stage 2 outputs
PAIRS_LABELED = PROCESSED_DIR / "pairs_labeled.csv"

# Splits
TRAIN_SPLIT = PROCESSED_DIR / "train.csv"
VAL_SPLIT = PROCESSED_DIR / "val.csv"
TEST_SPLIT = PROCESSED_DIR / "test.csv"
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.70, 0.15, 0.15

# Mining thresholds
MIN_CODE_LENGTH = 10          # chars, matches filter_pairs.py's existing cutoff
DEDUP_JACCARD_THRESHOLD = 0.85  # pairs above this similarity are treated as near-duplicates
CLASSIFIER_MIN_CONFIDENCE = 0.5

# ACQP weights (mirrors the penalty terms already in labeling/acqp.py)
BANDIT_SEVERITY_PENALTY = {"LOW": 0.1, "MEDIUM": 0.3, "HIGH": 0.5}
COMPLEXITY_PENALTY_CAP = 20
COMPLEXITY_PENALTY_WEIGHT = 0.1

RANDOM_SEED = 42

for _dir in (INTERIM_DIR, PROCESSED_DIR, SAMPLES_DIR, METRICS_DIR, FIGURES_DIR,
             TABLES_DIR, LOGS_DIR, TEACHER_CHECKPOINT_DIR, STUDENT_CHECKPOINT_DIR):
    _dir.mkdir(parents=True, exist_ok=True)
