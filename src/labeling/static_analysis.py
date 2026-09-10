"""
Python static analysis for ACQP.
"""
import json
import subprocess
import tempfile
from pathlib import Path

def _write_temp_py(code: str) -> Path:
    tmp = tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w")
    tmp.write(code)
    tmp.close()
    return Path(tmp.name)

def pylint_score(code: str) -> dict:
    path = _write_temp_py(code)
    try:
        result = subprocess.run(
            ["pylint", str(path), "--output-format=json2"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        data = json.loads(result.stdout or "{}")
        messages = data.get("messages", [])

        # pylint's json2 has no usable numeric score, so compute one the way
        # pylint does: start at 10, subtract weighted message penalties.
        severity_weight = {
            "fatal": 5.0, "error": 5.0, "warning": 1.0,
            "refactor": 0.5, "convention": 0.1, "info": 0.0,
        }
        penalty = sum(severity_weight.get(m.get("type", "info"), 0.0)
                      for m in messages)
        score = max(0.0, 10.0 - penalty)

        return {"pylint_score": score, "pylint_message_count": len(messages)}
    finally:
        path.unlink(missing_ok=True)

def bandit_findings(code: str) -> dict:
    path = _write_temp_py(code)
    try:
        result = subprocess.run(
            ["bandit", "-f", "json", "-q", str(path)],
            capture_output=True, text=True,
        )
        data = json.loads(result.stdout or "{}")
        results = data.get("results", [])
        counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
        for r in results:
            sev = r.get("issue_severity", "LOW")
            counts[sev] = counts.get(sev, 0) + 1
        return {"bandit_total": len(results), **{f"bandit_{k.lower()}": v for k, v in counts.items()}}
    finally:
        path.unlink(missing_ok=True)

def radon_complexity(code: str) -> dict:
    path = _write_temp_py(code)
    try:
        cc_result = subprocess.run(
            ["radon", "cc", str(path), "-j"],
            capture_output=True, text=True,
        )
        cc_data = json.loads(cc_result.stdout or "{}")
        blocks = next(iter(cc_data.values()), [])
        complexities = [b["complexity"] for b in blocks]
        avg_cc = sum(complexities) / len(complexities) if complexities else 0.0

        return {
            "avg_cyclomatic_complexity": avg_cc,
            "max_cyclomatic_complexity": max(complexities) if complexities else 0,
        }
    finally:
        path.unlink(missing_ok=True)

def analyze_python(code: str) -> dict:
    features = {}
    for fn in (pylint_score, bandit_findings, radon_complexity):
        try:
            features.update(fn(code))
        except Exception:
            pass
    return features
