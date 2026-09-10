"""
Stage 2: Sandboxed execution check for the ACQP score.

Runs the candidate code in a separate subprocess (not exec()/eval() in this
process) with a wall-clock timeout, a memory cap, and network access is
irrelevant here since this is a subprocess of the pipeline itself, not a
container — that's a known limitation, see note below.

Currently supports Python only, matching static_analysis.py's scope.
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    import resource
    HAS_RESOURCE = True
except ImportError:
    HAS_RESOURCE = False

DEFAULT_TIMEOUT_SEC = 5
DEFAULT_MEMORY_LIMIT_MB = 256

# NOTE: this limits the *subprocess's* CPU/memory/address space and kills it
# on timeout, which stops runaway loops and memory bombs. It is NOT a
# security sandbox against malicious code doing filesystem or network I/O —
# that needs a container (e.g. run this inside a locked-down Docker image
# with --network=none) before pointing it at untrusted internet-scraped
# code at scale. Fine for a DP-scale corpus you're mining and inspecting
# yourself; flag this limitation explicitly if you cite it in the report.
#
# On Windows, `resource` is unavailable and `preexec_fn` is unsupported, so
# the memory/CPU rlimits are skipped and only the wall-clock timeout applies.
def _limit_resources(memory_mb: int):
    if not HAS_RESOURCE:
        return
    mem_bytes = memory_mb * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
    resource.setrlimit(resource.RLIMIT_CPU, (DEFAULT_TIMEOUT_SEC, DEFAULT_TIMEOUT_SEC))


def check_python_execution(code: str, timeout: int = DEFAULT_TIMEOUT_SEC,
                            memory_mb: int = DEFAULT_MEMORY_LIMIT_MB) -> dict:
    """Attempt to run `code` in a fresh Python subprocess. Returns a dict with
    executes (bool), error_type, error_message, and timed_out."""
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w",
                                     encoding="utf-8") as tmp:
        tmp.write(code)
        path = Path(tmp.name)

    try:
        kwargs = dict(capture_output=True, text=True, timeout=timeout)
        if HAS_RESOURCE:
            kwargs["preexec_fn"] = lambda: _limit_resources(memory_mb)
        result = subprocess.run([sys.executable, str(path)], **kwargs)
        if result.returncode == 0:
            return {"executes": True, "error_type": None, "error_message": None, "timed_out": False}

        stderr_lines = result.stderr.strip().splitlines()
        error_type = stderr_lines[-1].split(":")[0] if stderr_lines else "UnknownError"
        return {
            "executes": False,
            "error_type": error_type,
            "error_message": stderr_lines[-1] if stderr_lines else "",
            "timed_out": False,
        }
    except subprocess.TimeoutExpired:
        return {"executes": False, "error_type": "TimeoutExpired",
                "error_message": f"exceeded {timeout}s", "timed_out": True}
    finally:
        path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", required=True)
    parser.add_argument("--out", dest="out_path", required=True)
    args = parser.parse_args()

    in_path, out_path = Path(args.in_path), Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total, executed = 0, 0
    with in_path.open(encoding="utf-8") as in_f, out_path.open("w", encoding="utf-8") as out_f:
        for line in in_f:
            if not line.strip():
                continue
            pair = json.loads(line)
            total += 1
            if pair.get("language", "").lower() in ("python", "py"):
                pair["execution_check"] = check_python_execution(pair["code"])
                if pair["execution_check"]["executes"]:
                    executed += 1
            out_f.write(json.dumps(pair) + "\n")

    print(f"Checked {total} pairs, {executed} Python snippets executed cleanly. Wrote {out_path}")


if __name__ == "__main__":
    main()