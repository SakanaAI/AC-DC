"""
Quick test script for the embedding server.
"""

import requests
import time

def test_server():
    """Test if the embedding server is running and working."""
    url = "http://localhost:8010"

    # Check if server is running
    try:
        response = requests.get(url)
        print(f"✓ Server is running!")
        print(f"  {response.json()}")
    except requests.exceptions.ConnectionError:
        print("✗ Server is not running. Please start it first:")
        print("  python benchmark_question_db/embedding_server.py")
        return False

    # Test embeddings endpoint
    print("\nTesting embeddings endpoint...")
    test_text = "What is the capital of France?"

    response = requests.post(
        f"{url}/v1/embeddings",
        json={
            "input": test_text,
            "model": "intfloat/e5-mistral-7b-instruct"
        }
    )

    if response.status_code == 200:
        data = response.json()
        embedding = data['data'][0]['embedding']
        print(f"✓ Embeddings endpoint working!")
        print(f"  Input: '{test_text}'")
        print(f"  Embedding dimension: {len(embedding)}")
        print(f"  First 5 values: {embedding[:5]}")
        return True
    else:
        print(f"✗ Error: {response.status_code}")
        print(response.text)
        return False

if __name__ == "__main__":
    test_server()
