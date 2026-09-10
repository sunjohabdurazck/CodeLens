"""
Smoke-test the real student scoring server (src/models/student/serve.py).

This needs: a trained student.pt checkpoint AND huggingface.co access to
pull the DistilBERT tokenizer — so it can't run in an offline sandbox,
only on a machine with real internet access. Start the server first:

    uvicorn src.models.student.serve:app --port 8000

then run this in another terminal:

    python scripts/test_server.py
"""
import requests

def test_server():
    print("Testing CodeLens student server...")
    print("-" * 40)

    try:
        response = requests.get("http://localhost:8000/health", timeout=2)
        print("Server is running:", response.json())
        print()

        test_prompts = [
            "Write a Python function to reverse a string",
            "Create a React component for a todo list",
            "What is the weather like today?",
            "Implement a binary search algorithm in JavaScript",
            "How do I sort a list?",
        ]

        print("Test Prompts and Scores:")
        print("-" * 40)
        for prompt in test_prompts:
            response = requests.post(
                "http://localhost:8000/score",
                json={"prompt": prompt},
                timeout=5,
            )
            score = response.json()["score"]
            print(f"  {prompt[:45]:45}... {score:.2f}")

        print()
        print("All requests succeeded.")
        return True

    except requests.exceptions.ConnectionError:
        print("Server not running.")
        print("  Start with: uvicorn src.models.student.serve:app --port 8000")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False


if __name__ == "__main__":
    test_server()
