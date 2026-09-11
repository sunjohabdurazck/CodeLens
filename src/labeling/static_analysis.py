"""
Python static analysis for ACQP.
"""
import json
import logging
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)

# Timeout (en secondes) appliqué à chaque outil externe, pour éviter
# qu'un process bloqué ne fasse pendre tout le pipeline d'évaluation.
_SUBPROCESS_TIMEOUT = 60


@contextmanager
def _temp_py_file(code: str) -> Iterator[Path]:
    """Crée un fichier .py temporaire contenant `code`, le supprime à la sortie."""
    tmp = tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w", encoding="utf-8")
    try:
        tmp.write(code)
        tmp.close()
        yield Path(tmp.name)
    finally:
        Path(tmp.name).unlink(missing_ok=True)


def _run_tool(cmd: list, **kwargs) -> subprocess.CompletedProcess:
    """Wrapper autour de subprocess.run avec timeout et encodage cohérents."""
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=_SUBPROCESS_TIMEOUT,
        **kwargs,
    )


def pylint_score(code: str) -> dict:
    with _temp_py_file(code) as path:
        try:
            result = _run_tool(["pylint", str(path), "--output-format=json2"])
        except FileNotFoundError:
            logger.warning("pylint introuvable : vérifie qu'il est installé et dans le PATH.")
            return {}
        except subprocess.TimeoutExpired:
            logger.warning("pylint a dépassé le timeout de %ss.", _SUBPROCESS_TIMEOUT)
            return {}

        try:
            data = json.loads(result.stdout or "{}")
        except json.JSONDecodeError:
            logger.warning("Sortie pylint non-JSON (stderr: %s)", result.stderr[:500])
            return {}

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


def bandit_findings(code: str) -> dict:
    with _temp_py_file(code) as path:
        try:
            result = _run_tool(["bandit", "-f", "json", "-q", str(path)])
        except FileNotFoundError:
            logger.warning("bandit introuvable : vérifie qu'il est installé et dans le PATH.")
            return {}
        except subprocess.TimeoutExpired:
            logger.warning("bandit a dépassé le timeout de %ss.", _SUBPROCESS_TIMEOUT)
            return {}

        try:
            data = json.loads(result.stdout or "{}")
        except json.JSONDecodeError:
            logger.warning("Sortie bandit non-JSON (stderr: %s)", result.stderr[:500])
            return {}

        results = data.get("results", [])
        counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
        for r in results:
            sev = r.get("issue_severity", "LOW")
            counts[sev] = counts.get(sev, 0) + 1
        return {"bandit_total": len(results), **{f"bandit_{k.lower()}": v for k, v in counts.items()}}


def radon_complexity(code: str) -> dict:
    with _temp_py_file(code) as path:
        try:
            cc_result = _run_tool(["radon", "cc", str(path), "-j"])
        except FileNotFoundError:
            logger.warning("radon introuvable : vérifie qu'il est installé et dans le PATH.")
            return {}
        except subprocess.TimeoutExpired:
            logger.warning("radon a dépassé le timeout de %ss.", _SUBPROCESS_TIMEOUT)
            return {}

        try:
            cc_data = json.loads(cc_result.stdout or "{}")
        except json.JSONDecodeError:
            logger.warning("Sortie radon non-JSON (stderr: %s)", cc_result.stderr[:500])
            return {}

        blocks = next(iter(cc_data.values()), [])
        complexities = [b["complexity"] for b in blocks if "complexity" in b]
        avg_cc = sum(complexities) / len(complexities) if complexities else 0.0

        return {
            "avg_cyclomatic_complexity": avg_cc,
            "max_cyclomatic_complexity": max(complexities) if complexities else 0,
        }


def analyze_python(code: str) -> dict:
    features = {}
    for fn in (pylint_score, bandit_findings, radon_complexity):
        try:
            features.update(fn(code))
        except Exception:
            logger.exception("Échec inattendu dans %s", fn.__name__)
    return features
