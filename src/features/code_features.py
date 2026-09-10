"""
Code-side feature extraction.
"""
import ast

def extract_code_features(code: str, language: str = "python") -> dict:
    features = {
        "char_length": len(code),
        "line_count": code.count("\n") + 1,
        "has_comments": ("#" in code) if language == "python" else ("//" in code or "/*" in code),
    }

    if language.lower() in ("python", "py"):
        try:
            tree = ast.parse(code)
            features["function_count"] = sum(isinstance(n, ast.FunctionDef) for n in ast.walk(tree))
            features["class_count"] = sum(isinstance(n, ast.ClassDef) for n in ast.walk(tree))
            features["import_count"] = sum(isinstance(n, (ast.Import, ast.ImportFrom)) for n in ast.walk(tree))
            features["parses_cleanly"] = True
        except SyntaxError:
            features["parses_cleanly"] = False

    return features
