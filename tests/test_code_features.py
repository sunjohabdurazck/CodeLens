from src.features.code_features import extract_code_features


def test_python_parses_cleanly():
    code = "def add(a, b):\n    return a + b\n"
    feats = extract_code_features(code, language="python")
    assert feats["parses_cleanly"] is True
    assert feats["function_count"] == 1


def test_python_syntax_error_detected():
    code = "def add(a, b:\n    return a + b\n"
    feats = extract_code_features(code, language="python")
    assert feats["parses_cleanly"] is False
