"""
Compara las dos configuraciones RAG (A=512 fijo, B=1024 semántico)
sobre un conjunto de preguntas factuales del curso.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from rag.vector_store import search

QUESTIONS = [
    "¿Qué es el descenso del gradiente?",
    "¿Cuál es la diferencia entre aprendizaje supervisado y no supervisado?",
    "¿Qué es una red neuronal convolucional?",
    "¿Qué es el overfitting y cómo se previene?",
    "¿Qué es la función de activación ReLU?",
    "¿En qué consiste la retropropagación?",
    "¿Qué es el aprendizaje por refuerzo?",
    "¿Qué diferencia hay entre LSTM y RNN simple?",
    "¿Qué es K-Nearest Neighbors (KNN)?",
    "¿Qué es la regularización L1 y L2?",
]

def run_comparison():
    results = []
    for q in QUESTIONS:
        row = {"question": q}
        for config_name, collection in [("A", config.COLLECTION_A), ("B", config.COLLECTION_B)]:
            t0 = time.time()
            r = search(q, collection_name=collection)
            elapsed = round(time.time() - t0, 3)
            docs = r.get("documents", [[]])[0]
            metas = r.get("metadatas", [[]])[0]
            distances = r.get("distances", [[]])[0]
            row[f"config_{config_name}"] = {
                "chunks_retrieved": len(docs),
                "latency_s": elapsed,
                "avg_distance": round(sum(distances) / len(distances), 4) if distances else None,
                "min_distance": round(min(distances), 4) if distances else None,
                "sources": list({m.get("source", "?") for m in metas}),
                "avg_chunk_len": round(sum(len(d) for d in docs) / len(docs)) if docs else 0,
                "top_chunk_preview": docs[0][:150] if docs else "",
            }
        results.append(row)
        print(f"  ✓ {q[:60]}...")

    out = Path(__file__).parent / "rag_comparison_results.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nGuardado en {out}")
    return results

if __name__ == "__main__":
    print("Comparando Config A vs Config B...\n")
    results = run_comparison()

    print("\n=== RESUMEN ===")
    lat_a = [r["config_A"]["latency_s"] for r in results]
    lat_b = [r["config_B"]["latency_s"] for r in results]
    dist_a = [r["config_A"]["avg_distance"] for r in results if r["config_A"]["avg_distance"]]
    dist_b = [r["config_B"]["avg_distance"] for r in results if r["config_B"]["avg_distance"]]
    len_a = [r["config_A"]["avg_chunk_len"] for r in results]
    len_b = [r["config_B"]["avg_chunk_len"] for r in results]

    print(f"Config A — latencia promedio: {sum(lat_a)/len(lat_a):.3f}s | distancia coseno promedio: {sum(dist_a)/len(dist_a):.4f} | longitud chunk promedio: {sum(len_a)//len(len_a)} chars")
    print(f"Config B — latencia promedio: {sum(lat_b)/len(lat_b):.3f}s | distancia coseno promedio: {sum(dist_b)/len(dist_b):.4f} | longitud chunk promedio: {sum(len_b)//len(len_b)} chars")
