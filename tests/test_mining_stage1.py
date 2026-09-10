from src.mining.remove_pii import scrub
from src.mining.deduplicate import deduplicate, jaccard, shingles
from src.mining.filter_classifier import train, SEED_EXAMPLES


def test_scrub_redacts_email():
    text, counts = scrub("reach me at jane.doe@example.com please")
    assert "[REDACTED_EMAIL]" in text
    assert counts["EMAIL"] == 1


def test_scrub_redacts_api_key():
    text, counts = scrub('api_key = "sk-abcdef1234567890abcdef"')
    assert "[REDACTED_SECRET]" in text
    assert counts["SECRET"] == 1


def test_scrub_leaves_clean_text_alone():
    text, counts = scrub("just a normal sentence about code")
    assert counts == {}
    assert text == "just a normal sentence about code"


def test_jaccard_identical_sets():
    a = shingles("def foo(x): return x + 1")
    assert jaccard(a, a) == 1.0


def test_deduplicate_drops_near_identical():
    # Character-shingle Jaccard is sensitive to identifiers that repeat many
    # times in a snippet (renaming "item" -> "product" touches every
    # occurrence), so it catches cosmetic diffs like whitespace/comments
    # better than pervasive renames. That's a real limitation -- see
    # deduplicate.py's docstring -- not something this test should paper
    # over, so it's exercised here with the kind of near-duplicate it
    # actually catches well.
    original = (
        "def calculate_total(items):\n"
        "    total = 0\n"
        "    for item in items:\n"
        "        total += item.price * item.quantity\n"
        "    return total"
    )
    cosmetic_variant = original + "\n\n# helper for cart totals"
    pairs = [
        {"code": original},
        {"code": cosmetic_variant},
        {"code": "class CompletelyDifferentThing:\n    def unrelated_method(self):\n        pass"},
    ]
    result = deduplicate(pairs, threshold=0.8)
    assert len(result) == 2


def test_classifier_separates_seed_examples():
    clf = train(SEED_EXAMPLES)
    pos_proba = clf.predict_proba(["Write a function that sorts an array"])[0][1]
    neg_proba = clf.predict_proba(["Thanks, that worked!"])[0][1]
    assert pos_proba > neg_proba
