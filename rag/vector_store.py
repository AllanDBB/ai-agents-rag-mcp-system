from pathlib import Path

import chromadb

import config

_EMBEDDING_FN = None


def _get_embedding_fn():
    global _EMBEDDING_FN
    if _EMBEDDING_FN is None:
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
        _EMBEDDING_FN = DefaultEmbeddingFunction()
    return _EMBEDDING_FN


def _client() -> chromadb.PersistentClient:
    Path(config.CHROMA_PATH).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=config.CHROMA_PATH)


def get_collection(name: str) -> chromadb.Collection:
    return _client().get_or_create_collection(
        name=name,
        embedding_function=_get_embedding_fn(),
        metadata={"hnsw:space": "cosine"},
    )


def add_documents(chunks: list[dict], collection_name: str) -> int:
    col = get_collection(collection_name)
    ids = [
        f"{c['metadata']['source']}_{c['metadata']['config']}_{c['metadata']['chunk_index']}"
        for c in chunks
    ]
    col.add(
        ids=ids,
        documents=[c["text"] for c in chunks],
        metadatas=[c["metadata"] for c in chunks],
    )
    return len(chunks)


def search(query: str, collection_name: str, n_results: int = None) -> dict:
    k = n_results or config.RAG_TOP_K
    col = get_collection(collection_name)
    return col.query(query_texts=[query], n_results=k)


def collection_count(collection_name: str) -> int:
    try:
        return _client().get_collection(collection_name).count()
    except Exception:
        return 0
