"""
Prompt-side feature extraction.
"""
import re

SPECIFICITY_MARKERS = re.compile(r"\b(specifically|must|should|input|output|example|edge case|constraint|complexity|O\(|test)\b", re.IGNORECASE)
AMBIGUITY_MARKERS = re.compile(r"\b(something|somehow|maybe|kind of|sort of|etc\.?|whatever)\b", re.IGNORECASE)

def extract_prompt_features(prompt: str) -> dict:
    words = prompt.split()
    sentences = [s for s in re.split(r"[.!?]+", prompt) if s.strip()]

    return {
        "char_length": len(prompt),
        "word_count": len(words),
        "sentence_count": len(sentences),
        "avg_word_length": sum(len(w) for w in words) / len(words) if words else 0,
        "has_code_snippet": "```" in prompt,
        "specificity_marker_count": len(SPECIFICITY_MARKERS.findall(prompt)),
        "ambiguity_marker_count": len(AMBIGUITY_MARKERS.findall(prompt)),
        "question_mark_count": prompt.count("?"),
        "ends_with_question": prompt.strip().endswith("?"),
    }
