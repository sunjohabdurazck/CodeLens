"""
Generate report tables and figures from the pipeline outputs.

Reads:
    data/processed/pairs_raw.csv       (606 mined pairs)
    data/processed/pairs_labeled.csv   (92 labeled Python pairs)
    data/processed/train.csv, val.csv, test.csv
    results/metrics/predictions.json   (M1/M3/M4/M5 predictions)

Writes:
    results/tables/*.csv
    results/figures/*.png
"""
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.config import (PROCESSED_DIR, RESULTS_DIR, METRICS_DIR,
                        TABLES_DIR, FIGURES_DIR)


# ---------------------------------------------------------------- loaders

def read_csv_safe(path):
    p = Path(path)
    if not p.exists():
        print(f"  (skip) {p} not found")
        return None
    return pd.read_csv(p, encoding="utf-8")


def read_json_safe(path):
    p = Path(path)
    if not p.exists():
        print(f"  (skip) {p} not found")
        return None
    with p.open(encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------- tables

def table_pass_through():
    """Pipeline pass-through rates."""
    # Hardcoded from the runs — these are the values the pipeline printed.
    # If you re-run stages 1-4, update these numbers.
    rows = [
        ("Pull (WildChat raw)",      5000,  None, None),
        ("Normalize",                5000,  4969, 4969/5000),
        ("Mine Pass 1 (keyword)",    4969,  867,  867/4969),
        ("Mine Pass 2 (classifier)", 867,   651,  651/867),
        ("Dedup",                    651,   606,  606/651),
        ("Label (Python only)",      606,   92,   92/606),
        ("Split → train",            92,    69,   69/92),
        ("Split → val",              92,    11,   11/92),
        ("Split → test",             92,    12,   12/92),
    ]
    df = pd.DataFrame(rows, columns=["stage", "input", "output", "retention"])
    df["retention_pct"] = (df["retention"] * 100).round(2)
    out = df.drop(columns=["retention"])
    out.to_csv(TABLES_DIR / "pass_through.csv", index=False)
    print(f"  wrote {TABLES_DIR / 'pass_through.csv'}")
    return out


def table_language_distribution():
    """Language counts among the mined pairs."""
    df = read_csv_safe(PROCESSED_DIR / "pairs_raw.csv")
    if df is None:
        return None
    counts = df["language"].fillna("unknown").str.lower().value_counts()
    out = counts.rename_axis("language").reset_index(name="count")
    out.to_csv(TABLES_DIR / "language_distribution.csv", index=False)
    print(f"  wrote {TABLES_DIR / 'language_distribution.csv'} ({len(out)} languages)")
    return out


def table_acqp_distribution():
    """Summary stats of the ACQP target."""
    df = read_csv_safe(PROCESSED_DIR / "pairs_labeled.csv")
    if df is None:
        return None
    s = df["acqp_score"].astype(float)
    out = pd.DataFrame([{
        "n": len(s),
        "zeros": int((s == 0.0).sum()),
        "min": round(float(s.min()), 3),
        "median": round(float(s.median()), 3),
        "mean": round(float(s.mean()), 3),
        "max": round(float(s.max()), 3),
        "stdev": round(float(s.std()), 3),
    }])
    out.to_csv(TABLES_DIR / "acqp_distribution.csv", index=False)
    print(f"  wrote {TABLES_DIR / 'acqp_distribution.csv'}")
    return out


def table_ablation():
    """The M1-M5 ablation comparison table."""
    from src.eval.metrics import compute_metrics

    preds = read_json_safe(METRICS_DIR / "predictions.json")
    test = read_csv_safe(PROCESSED_DIR / "test.csv")
    if preds is None or test is None:
        return None

    y_true = test["acqp_score"].astype(float).tolist()
    rows = []
    for name, y_pred in preds.items():
        m = compute_metrics(y_true, [float(v) for v in y_pred])
        rows.append({
            "model": name,
            "mae": round(m["mae"], 4),
            "rmse": round(m["rmse"], 4),
            "pearson_r": round(m["pearson_r"], 4),
            "spearman_r": round(m["spearman_r"], 4),
        })

    # Sort M1..M5 by name
    out = pd.DataFrame(rows).sort_values("model").reset_index(drop=True)
    out.to_csv(TABLES_DIR / "ablation.csv", index=False)
    print(f"  wrote {TABLES_DIR / 'ablation.csv'}")
    return out


# --------------------------------------------------------------- figures

def figure_acqp_histogram():
    """Histogram of ACQP scores across the labeled pairs."""
    df = read_csv_safe(PROCESSED_DIR / "pairs_labeled.csv")
    if df is None:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(df["acqp_score"].astype(float), bins=20,
            color="#3b6ea5", edgecolor="black", linewidth=0.5)
    ax.axvline(0.0, color="red", linestyle="--", linewidth=1, alpha=0.7,
               label=f"zeros: {(df['acqp_score'] == 0).sum()}")
    ax.set_xlabel("ACQP score")
    ax.set_ylabel("count")
    ax.set_title(f"ACQP distribution (n={len(df)})")
    ax.legend()
    fig.tight_layout()
    out = FIGURES_DIR / "acqp_histogram.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  wrote {out}")


def figure_language_distribution(top_n=15):
    """Bar chart of the top languages in the mined pairs."""
    df = read_csv_safe(PROCESSED_DIR / "pairs_raw.csv")
    if df is None:
        return
    counts = df["language"].fillna("unknown").str.lower().value_counts().head(top_n)
    fig, ax = plt.subplots(figsize=(8, 5))
    counts[::-1].plot(kind="barh", ax=ax, color="#3b6ea5")
    ax.set_xlabel("count")
    ax.set_title(f"Top {top_n} languages in mined pairs (n={len(df)})")
    fig.tight_layout()
    out = FIGURES_DIR / "language_distribution.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  wrote {out}")


def figure_pred_vs_true():
    """Scatter of predicted vs. true ACQP for each ablation model."""
    preds = read_json_safe(METRICS_DIR / "predictions.json")
    test = read_csv_safe(PROCESSED_DIR / "test.csv")
    if preds is None or test is None:
        return

    y_true = test["acqp_score"].astype(float).values
    models = list(preds.keys())
    n = len(models)
    cols = 2
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(10, 4.5 * rows),
                             squeeze=False)
    lo = float(min(y_true.min(), min(min(p) for p in preds.values())))
    hi = float(max(y_true.max(), max(max(p) for p in preds.values())))
    pad = 0.05 * (hi - lo + 1e-6)

    for i, name in enumerate(models):
        ax = axes[i // cols][i % cols]
        y_pred = [float(v) for v in preds[name]]
        ax.scatter(y_true, y_pred, s=45, alpha=0.75,
                   color="#3b6ea5", edgecolor="black", linewidth=0.5)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad],
                "r--", linewidth=1, alpha=0.6, label="y = x")
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlabel("true ACQP")
        ax.set_ylabel("predicted ACQP")
        ax.set_title(f"{name}  (n={len(y_true)})")
        ax.legend(loc="upper left", fontsize=8)

    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")

    fig.tight_layout()
    out = FIGURES_DIR / "pred_vs_true.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  wrote {out}")


def figure_retention_funnel():
    """Funnel chart of pipeline retention."""
    stages = ["raw", "normalized", "pass1", "pass2", "dedup", "labeled"]
    counts = [5000, 4969, 867, 651, 606, 92]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(stages, counts, color="#3b6ea5", edgecolor="black", linewidth=0.5)
    for i, c in enumerate(counts):
        ax.text(i, c + max(counts) * 0.015, str(c), ha="center", fontsize=9)
    ax.set_ylabel("count")
    ax.set_yscale("log")
    ax.set_title("Pipeline retention (log scale)")
    fig.tight_layout()
    out = FIGURES_DIR / "retention_funnel.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  wrote {out}")


# --------------------------------------------------------------- driver

def main():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Tables:")
    table_pass_through()
    table_language_distribution()
    table_acqp_distribution()
    table_ablation()

    print("Figures:")
    figure_acqp_histogram()
    figure_language_distribution()
    figure_pred_vs_true()
    figure_retention_funnel()

    print("\nDone.")


if __name__ == "__main__":
    main()