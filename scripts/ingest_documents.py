"""
Indexa los PDFs de DOCUMENTS_PATH en ChromaDB (configuraciones A y B).

Uso:
    python scripts/ingest_documents.py
    python scripts/ingest_documents.py --path ./Apuntes
    python scripts/ingest_documents.py --reset   # borra y reindexa todo
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from rag.document_processor import process_config_a, process_config_b
from rag.vector_store import add_documents, collection_count, _client


def reset_collections():
    client = _client()
    for name in [config.COLLECTION_A, config.COLLECTION_B]:
        try:
            client.delete_collection(name)
            print(f"  Colección '{name}' eliminada.")
        except Exception:
            pass


def ingest(documents_path: str = None, reset: bool = False):
    docs_dir = Path(documents_path or config.DOCUMENTS_PATH)

    if not docs_dir.exists():
        print(f"[ERROR] Directorio no encontrado: {docs_dir}")
        return

    pdfs = sorted(docs_dir.glob("*.pdf"))
    if not pdfs:
        print(f"[ERROR] No se encontraron PDFs en {docs_dir}")
        return

    print(f"Directorio: {docs_dir}")
    print(f"PDFs encontrados: {len(pdfs)}")

    if reset:
        print("Reseteando colecciones...")
        reset_collections()

    total_a = total_b = 0
    errors = []

    for pdf in pdfs:
        print(f"  → {pdf.name}", end=" ", flush=True)
        try:
            chunks_a = process_config_a(str(pdf))
            add_documents(chunks_a, config.COLLECTION_A)
            total_a += len(chunks_a)

            chunks_b = process_config_b(str(pdf))
            add_documents(chunks_b, config.COLLECTION_B)
            total_b += len(chunks_b)

            print(f"[A:{len(chunks_a)} B:{len(chunks_b)}]")
        except Exception as e:
            print(f"[ERROR: {e}]")
            errors.append((pdf.name, str(e)))

    print(f"\n{'='*50}")
    print(f"Config A — chunks totales : {total_a:,}  |  en DB: {collection_count(config.COLLECTION_A):,}")
    print(f"Config B — chunks totales : {total_b:,}  |  en DB: {collection_count(config.COLLECTION_B):,}")
    if errors:
        print(f"\n⚠ Errores ({len(errors)}):")
        for name, err in errors:
            print(f"  {name}: {err}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path",  default=None, help="Ruta a los documentos")
    parser.add_argument("--reset", action="store_true", help="Borrar y reindexa todo")
    args = parser.parse_args()
    ingest(args.path, args.reset)
