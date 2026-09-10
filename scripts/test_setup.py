#!/usr/bin/env python
"""
Comprehensive setup sanity check.
"""

import sys
import subprocess
import importlib
from pathlib import Path


def test_python_packages():
    """Test all required Python packages."""
    packages = [
        'torch', 'transformers', 'datasets', 'sklearn', 'pandas', 'numpy',
        'pylint', 'bandit', 'radon', 'huggingface_hub', 'tqdm', 'pytest',
        'jupyter', 'fastapi', 'uvicorn',
    ]

    print("Testing Python packages...")
    failed = []
    for pkg in packages:
        import_name = 'sklearn' if pkg == 'sklearn' else pkg.replace('-', '_')
        try:
            importlib.import_module(import_name)
            print(f"  OK   {pkg}")
        except ImportError as e:
            print(f"  FAIL {pkg} - {e}")
            failed.append(pkg)

    return failed


def test_node_tools():
    """Test Node.js tools (for the VS Code extension and ESLint-based JS analysis)."""
    print("\nTesting Node.js tools...")
    tools = ['node', 'npm', 'eslint']
    failed = []

    for tool in tools:
        try:
            result = subprocess.run([tool, '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                print(f"  OK   {tool} {result.stdout.strip()}")
            else:
                print(f"  FAIL {tool} (return code {result.returncode})")
                failed.append(tool)
        except (subprocess.SubprocessError, FileNotFoundError) as e:
            print(f"  FAIL {tool} (not found: {e})")
            failed.append(tool)

    return failed


def test_directory_structure():
    """Test project directory structure."""
    print("\nTesting directory structure...")
    required_dirs = [
        'data/raw', 'data/interim', 'src/mining', 'src/labeling',
        'src/features', 'src/models/teacher', 'src/models/student',
        'src/eval', 'extension', 'notebooks', 'tests', 'scripts',
    ]

    missing = []
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"  OK   {dir_path}")
        else:
            print(f"  FAIL {dir_path} (missing)")
            missing.append(dir_path)

    return missing


def test_huggingface_login():
    """Test Hugging Face login status."""
    print("\nTesting Hugging Face login...")
    try:
        result = subprocess.run(['huggingface-cli', 'whoami'], capture_output=True, text=True)
        if result.returncode == 0:
            print("  OK   Logged in to Hugging Face")
            return True
        print("  FAIL Not logged in (run: huggingface-cli login)")
        return False
    except FileNotFoundError:
        print("  FAIL huggingface-cli not found (run: pip install huggingface_hub)")
        return False


def main():
    print("=" * 60)
    print("CodeLens Setup Sanity Check")
    print("=" * 60)
    print(f"\nPython version: {sys.version}")

    failed_packages = test_python_packages()
    failed_tools = test_node_tools()
    missing_dirs = test_directory_structure()
    hf_login = test_huggingface_login()

    print("\n" + "=" * 60)
    print("Summary:")

    if not (failed_packages or failed_tools or missing_dirs):
        print("All checks passed. Environment is ready.")
        if not hf_login:
            print("Note: not logged in to Hugging Face -- run huggingface-cli login "
                  "before training or downloading corpora.")
    else:
        if failed_packages:
            print(f"Missing Python packages: {', '.join(failed_packages)}")
            print("  Run: pip install " + ' '.join(failed_packages))
        if failed_tools:
            print(f"Missing Node.js tools: {', '.join(failed_tools)}")
        if missing_dirs:
            print(f"Missing directories: {', '.join(missing_dirs)}")

    print("=" * 60)


if __name__ == "__main__":
    main()
