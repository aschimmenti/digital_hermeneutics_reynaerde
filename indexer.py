import faiss
import numpy as np
from typing import Optional


def create_hybrid_index(embeddings: np.ndarray) -> faiss.Index:
    """Create an optimized FAISS index for similarity search.

    - Uses IVF for large collections, FlatIP for smaller ones.
    - Assumes Voyage embeddings are normalized; Inner Product is appropriate.
    """
    if embeddings is None or len(embeddings) == 0:
        raise ValueError("Embeddings are empty; cannot create index")

    dim = embeddings.shape[1]

    if len(embeddings) > 1000:
        nlist = min(100, len(embeddings) // 10)
        index = faiss.IndexIVFFlat(
            faiss.IndexFlatL2(dim), dim, nlist
        )
        index.train(embeddings)
    else:
        index = faiss.IndexFlatIP(dim)

    index.add(embeddings)
    return index


def save_index(index: faiss.Index, path: str) -> None:
    """Save the FAISS index to disk."""
    if index is None:
        raise ValueError("Index is None; nothing to save")
    faiss.write_index(index, path)


def load_index(index_path: str) -> faiss.Index:
    """Load a FAISS index from disk and return it."""
    return faiss.read_index(index_path)
