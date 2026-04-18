"""
Fast OpenAI-compatible embedding server using BGE-base-en-v1.5
Optimized for M2 MacBook Pro with MPS support.

BGE-base-en-v1.5:
- Dimension: 768
- Speed: ~10 embeddings/sec on M2
- Quality: Excellent (top-ranked on MTEB)
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

# Load the model - BGE is much faster than E5-Mistral
print("Loading BAAI/bge-base-en-v1.5 model...")
model = SentenceTransformer('BAAI/bge-base-en-v1.5', device=device)
print(f"Model loaded successfully on {device}!")
print(f"Embedding dimension: {model.get_sentence_embedding_dimension()}")


class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]]
    model: str


class EmbeddingResponse(BaseModel):
    data: List[dict]
    model: str
    usage: dict


@app.post("/v1/embeddings")
async def create_embedding(request: EmbeddingRequest):
    """OpenAI-compatible embeddings endpoint."""
    # Handle both single string and list of strings
    texts = [request.input] if isinstance(request.input, str) else request.input

    # BGE models benefit from adding instruction for queries in retrieval tasks
    # For general embeddings (like ours), we don't need special formatting

    # Encode the texts
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,  # BGE models benefit from normalization
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
        "model": "BAAI/bge-base-en-v1.5",
        "device": device,
        "dimension": model.get_sentence_embedding_dimension(),
    }


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("BGE-Base-EN-v1.5 Fast Embedding Server")
    print("=" * 80)
    print(f"Model: BAAI/bge-base-en-v1.5")
    print(f"Device: {device}")
    print(f"Endpoint: http://localhost:8010/v1/embeddings")
    print(f"Expected speed: ~10 embeddings/sec on M2")
    print("=" * 80 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8010, log_level="info")
