from src.mining.filter_pairs import iter_pairs, is_code_request, extract_code_blocks


def test_is_code_request():
    assert is_code_request("Can you write a function that reverses a string?")
    assert not is_code_request("What's the weather like today?")


def test_extract_code_blocks():
    text = "Here you go:\n```python\nprint('hi')\n```\nLet me know if that works."
    blocks = list(extract_code_blocks(text))
    assert len(blocks) == 1
    assert blocks[0][0] == "python"
    assert "print" in blocks[0][1]


def test_iter_pairs():
    conversations = [
        {
            "id": "conv1",
            "turns": [
                {"role": "user", "text": "Write a function to add two numbers"},
                {"role": "assistant", "text": "```python\ndef add(a, b):\n    return a + b\n```"},
            ],
        }
    ]
    pairs = list(iter_pairs(conversations))
    assert len(pairs) == 1
    assert pairs[0].conversation_id == "conv1"
    assert pairs[0].language == "python"
