---
name: project-ai-agents-rag-mcp
description: Tarea 04 y 05 — Sistema Multi-Agente con RAG, A2A, MCP y Memoria para curso de IA del TEC
metadata:
  type: project
---

Sistema multi-agente académico para el curso de Inteligencia Artificial del Instituto Tecnológico de Costa Rica.

**Why:** Tarea universitaria (Tarea 04 y 05) — primera entrega vale 30 pts, segunda 70 pts (total 100 pts).

**How to apply:** Priorizar que todo lo del enunciado esté implementado. La primera entrega se centra en orquestador + RAG + Langfuse + Streamlit. La segunda agrega MCP completo, memoria histórica, seguridad y evaluación experimental.

## Stack
- Orquestador: claude-sonnet-4-6
- Agentes secundarios: claude-haiku-4-5-20251001
- Vector DB: ChromaDB local (paraphrase-multilingual-MiniLM-L12-v2)
- Observabilidad: Langfuse Cloud
- Frontend: Streamlit
- MCP Server: FastMCP con SSE transport (puerto 8001)
- BD ficticia: SQLite en `data/transactions.db`
- Memoria histórica: SQLite en `data/memory.db`

## Apuntes
51 PDFs en `./Apuntes/` — naming: `{semana}_SEMANA_AI_{fecha}_{num}.pdf`
Semanas 1–14 del curso.

## Comandos clave
```bash
pip install -r requirements.txt
python scripts/ingest_documents.py     # indexar apuntes en ChromaDB
python scripts/seed_database.py        # poblar BD ficticia
python mcp_server/server.py            # MCP Server (terminal aparte)
streamlit run app/streamlit_app.py     # UI
python eval/test_questions.py          # evaluación experimental
```

## .env
Rellenar ANTHROPIC_API_KEY, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY.
DOCUMENTS_PATH ya apunta a ./Apuntes.
