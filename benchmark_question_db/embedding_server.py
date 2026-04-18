"""
Simple OpenAI-compatible embedding server for sentence-transformers models.
Optimized for M2 MacBook Pro with MPS support.
"""

from fastapi import FastAPI
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel
from typing import List, Union
import uvicorn
import torch

app = FastAPI()

# Initialize model with MPS support if available
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# Load the model - this will take a minute on first run
print("Loading e5-mistral-7b-instruct model (this may take a minute)...")
model = SentenceTransformer('intfloat/e5-mistral-7b-instruct', device=device)
print(f"Model loaded successfully on {device}!")
print(f"Embedding dimension: {model.get_sentence_embedding_dimension()}")


class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]]
    model: str


class EmbeddingResponse(BaseModel):
    data: List[dict]
    model: str
    usage: dict


def add_instruction(text: str, is_query: bool = True) -> str:
    """
    E5-Mistral requires special formatting:
    - Queries: prefix with "Instruct: <instruction>\nQuery: "
    - Passages: no prefix needed

    For our use case, we're embedding questions/passages, so we don't need
    the "Instruct" prefix.
    """
    # For embeddings in the database, we're storing passages/questions
    # so we don't need special formatting
    return text


@app.post("/v1/embeddings")
async def create_embedding(request: EmbeddingRequest):
    """OpenAI-compatible embeddings endpoint."""
    # Handle both single string and list of strings
    texts = [request.input] if isinstance(request.input, str) else request.input

    # E5-Mistral doesn't need special formatting for passage encoding
    # (only for query encoding in retrieval tasks)

    # Encode the texts
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,  # E5 models benefit from normalization
        show_progress_bar=False,
    )

    # Format response to match OpenAI API
    return {
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "embedding": emb.tolist(),
                "index": i,
            }
            for i, emb in enumerate(embeddings)
        ],
        "model": request.model,
        "usage": {
            "prompt_tokens": sum(len(t.split()) for t in texts),
            "total_tokens": sum(len(t.split()) for t in texts),
        },
    }


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "running",
        "model": "intfloat/e5-mistral-7b-instruct",
        "device": device,
        "dimension": model.get_sentence_embedding_dimension(),
    }


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("E5-Mistral-7B Embedding Server")
    print("=" * 80)
    print(f"Model: intfloat/e5-mistral-7b-instruct")
    print(f"Device: {device}")
    print(f"Endpoint: http://localhost:8010/v1/embeddings")
    print("=" * 80 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8010, log_level="info")
