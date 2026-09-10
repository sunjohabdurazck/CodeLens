"""
ESLint-based static analysis for JavaScript/TypeScript samples in the
ACQP pipeline. Requires `npm install -g eslint` (or a local project
install — adjust the command below if you go that route).
"""
import json
import subprocess
import tempfile
from pathlib import Path


def _write_temp_js(code: str, ext: str = ".js") -> Path:
    tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False, mode="w")
    tmp.write(code)
    tmp.close()
    return Path(tmp.name)


def eslint_findings(code: str, ext: str = ".js") -> dict:
    path = _write_temp_js(code, ext)
    try:
        result = subprocess.run(
            ["eslint", str(path), "--format=json", "--no-eslintrc", "--env", "es2021,node"],
            capture_output=True, text=True,
        )
        data = json.loads(result.stdout or "[]")
        messages = data[0]["messages"] if data else []
        errors = sum(1 for m in messages if m.get("severity") == 2)
        warnings = sum(1 for m in messages if m.get("severity") == 1)
        return {"eslint_errors": errors, "eslint_warnings": warnings, "eslint_total": len(messages)}
    finally:
        path.unlink(missing_ok=True)
