#!/usr/bin/env bash
set -euo pipefail

DATA="${1:-data/processed/test.csv}"
OUT_DIR="${2:-results/metrics}"
mkdir -p "$OUT_DIR"

echo "== M1 baseline (sklearn, runs locally) =="
python -m src.eval.baseline_models --data "$DATA" --out "$OUT_DIR/predictions_baselines.json"

echo "== Teacher evaluation (needs teacher.pt from Colab) =="
if [ -f src/models/teacher/checkpoints/teacher.pt ]; then
  python -m src.models.teacher.evaluate \
    --data "$DATA" --out "$OUT_DIR/predictions_teacher.json" --code_only
else
  echo "  skipped: no teacher.pt"
fi

echo "== Student evaluation (needs student.pt from Colab) =="
if [ -f src/models/student/checkpoints/student.pt ]; then
  python -m src.models.student.evaluate \
    --data "$DATA" --out "$OUT_DIR/predictions_student.json"
else
  echo "  skipped: no student.pt"
fi

echo "== Merge all predictions =="
python - <<'PY'
import json, glob
merged = {}
for path in glob.glob("results/metrics/predictions_*.json"):
    merged.update(json.load(open(path)))
out = "results/metrics/predictions.json"
json.dump(merged, open(out, "w"), indent=2)
print(f"Wrote {list(merged.keys())} to {out}")
PY

echo "== Run ablation =="
python -m src.eval.ablation \
  --test "$DATA" --predictions results/metrics/predictions.json \
  > results/tables/table_02_ablation_results.json
cat results/tables/table_02_ablation_results.json
