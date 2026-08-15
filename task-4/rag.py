"""A proper indexed RAG system over the OpenStax Prealgebra excerpts,
replacing Task 1's keyword search with real dense retrieval: local Ollama
embeddings (`nomic-embed-text`) in a persistent Chroma collection.

Originally built against sentence-transformers' `all-MiniLM-L6-v2`
downloaded from the HF Hub, but the HF CDN was severely bandwidth-throttled
in this environment (observed ~10-30KB/s and degrading, vs. ~1.8MB/s to
other hosts, so not a general connectivity problem), which would have made
a one-time ~87MB model fetch take upwards of an hour with no guarantee of
finishing. Swapped to `nomic-embed-text` via the local Ollama server
instead: it was already pulled (used nowhere else in this task, but present
from earlier roadmap work), needs no network fetch at all, and is arguably
a better fit anyway since every other task in this repo is Ollama-first.
"""

from __future__ import annotations

import json
from pathlib import Path

import ollama

CURRICULUM_DIR = Path(__file__).parent / "curriculum"
CHROMA_DIR = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "curriculum"
EMBED_MODEL = "nomic-embed-text"


def _embed(text: str) -> list[float]:
    return ollama.embeddings(model=EMBED_MODEL, prompt=text)["embedding"]


def _embed_many(texts: list[str]) -> list[list[float]]:
    return [_embed(t) for t in texts]


def load_sections() -> list[dict]:
    """Flatten every chapter's sections into one list of indexable chunks.

    Pure stdlib, no embedding model or Chroma involved, so this alone is
    fast and offline enough to unit test in CI.
    """
    sections = []
    for path in sorted(CURRICULUM_DIR.glob("*.json")):
        chapter = json.loads(path.read_text())
        for i, section in enumerate(chapter["sections"]):
            sections.append(
                {
                    "id": f"{path.stem}-{i}",
                    "chapter": chapter["chapter"],
                    "title": section["title"],
                    "text": section["text"],
                    "source": chapter["source"],
                }
            )
    return sections


def build_index(persist_dir=CHROMA_DIR):
    import chromadb

    client = chromadb.PersistentClient(path=str(persist_dir))
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME)

    sections = load_sections()
    texts_to_embed = [f"{s['chapter']}: {s['title']}. {s['text']}" for s in sections]
    embeddings = _embed_many(texts_to_embed)

    collection.add(
        ids=[s["id"] for s in sections],
        embeddings=embeddings,
        documents=[s["text"] for s in sections],
        metadatas=[
            {"chapter": s["chapter"], "title": s["title"], "source": s["source"]}
            for s in sections
        ],
    )
    return collection


def retrieve(query: str, k: int = 2, persist_dir=CHROMA_DIR) -> list[dict]:
    import chromadb

    client = chromadb.PersistentClient(path=str(persist_dir))
    collection = client.get_collection(COLLECTION_NAME)
    query_embedding = [_embed(query)]
    results = collection.query(query_embeddings=query_embedding, n_results=k)

    out = []
    for doc, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        out.append(
            {
                "text": doc,
                "chapter": meta["chapter"],
                "title": meta["title"],
                "source": meta["source"],
                "distance": dist,
            }
        )
    return out


if __name__ == "__main__":
    print("Building Chroma index over curriculum/*.json ...")
    build_index()
    print(f"Indexed. Collection persisted at {CHROMA_DIR}")
