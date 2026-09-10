"""
ACQP - Automated Code Quality Proxy computation.
"""
from src.labeling.static_analysis import analyze_python

def compute_acqp(code: str, language: str) -> dict:
    if language.lower() in ("python", "py"):
        features = analyze_python(code)
        pylint = features.get("pylint_score") or 0.0
        bandit_penalty = features.get("bandit_total", 0) * 0.5
        complexity_penalty = min(features.get("avg_cyclomatic_complexity", 0), 20) * 0.1
        acqp = max(0.0, pylint - bandit_penalty - complexity_penalty)
    else:
        features = {"note": f"no analyzer configured for language={language}"}
        acqp = None

    return {"acqp_score": acqp, "features": features}
