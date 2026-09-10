from src.features.prompt_features import extract_prompt_features


def test_basic_counts():
    feats = extract_prompt_features("Write a function to sort a list. It should handle edge cases?")
    assert feats["word_count"] > 0
    assert feats["sentence_count"] == 2
    assert feats["ends_with_question"] is True
    assert feats["specificity_marker_count"] >= 1
