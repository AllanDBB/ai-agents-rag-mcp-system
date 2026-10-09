# Sistema Multi-Agente con RAG, A2A, MCP y Memoria

Chatbot académico sobre los apuntes del curso de Inteligencia Artificial del TEC. En lugar de responder solo con un LLM, tiene 5 agentes especializados coordinados por un orquestador que decide dinámicamente a quién delegar cada pregunta.

---

## Arquitectura

```
Browser (8501)
    ↓
Streamlit  →  OrchestratorAgent (Sonnet 4.6)
                ├── RAGAgent         →  ChromaDB  →  Haiku 4.5
                ├── WebSearchAgent   →  Tavily API
                ├── SummarizerAgent  →  Haiku 4.5
                ├── TransactionalAgent → MCP Server (8000) → SQLite
                └── search_history   →  memory.db
                ↓
           Langfuse Cloud (trazas, tokens, latencia)
```

---

## Componentes

### Agentes (`agents/`)

| Archivo | Agente | Modelo | Qué hace |
|---|---|---|---|
| `orchestrator.py` | Orquestador | Sonnet 4.6 | Recibe la pregunta, decide qué herramienta invocar con tool use, construye la respuesta final |
| `rag_agent.py` | RAG | Haiku 4.5 | Busca en los PDFs indexados del curso, responde solo con lo que encuentra |
| `web_search_agent.py` | Web Search | — | Llama a Tavily API, solo cuando el usuario lo pide explícitamente |
| `summarizer_agent.py` | Resumidor | Haiku 4.5 | Resume conversaciones, respuestas largas o historiales |
| `transactional_agent.py` | Transaccional | — | Consulta la BD ficticia de transacciones vía MCP Server |

El orquestador funciona con un **agentic loop**: llama al modelo → el modelo pide una herramienta → se ejecuta → se devuelve el resultado → el modelo decide si terminar o pedir otra herramienta.

### RAG (`rag/`)

**Flujo de indexación** (`scripts/ingest_documents.py`):
1. PyMuPDF extrae texto de cada PDF
2. Se limpia (guiones de fin de línea, espacios dobles, saltos excesivos)
3. Se corta en chunks con LangChain
4. Se generan embeddings con `all-MiniLM-L6-v2` (local, sin API)
5. Se guardan en ChromaDB con metadatos (semana, fecha, documento)

Dos configuraciones comparadas:

| Config | Chunk size | Overlap | Chunks totales | Mejor para |
|---|---|---|---|---|
| A | 512 chars | 64 | 1,238 | Preguntas cortas y factuales |
| B | 1,024 chars + separadores naturales | 128 | 643 | Preguntas que necesitan más contexto |

Al buscar: el query se convierte en vector → ChromaDB devuelve los 5 chunks más similares → Haiku genera respuesta basada solo en esos chunks.

### MCP Server (`mcp_server/`)

Servidor independiente en `localhost:8000`. Expone 5 herramientas controladas sobre una BD SQLite ficticia de transacciones bancarias:

- `get_transaction_by_id` — busca transacción por ID
- `search_transactions` — filtra con al menos un filtro (status, país, monto, fecha)
- `get_customer_risk_summary` — perfil de riesgo de un cliente
- `get_recent_flagged_transactions` — sospechosas en los últimos N días (máx 365)
- `create_fraud_case` — crea caso de fraude (solo inserta, no modifica)

Reglas de seguridad: toda llamada requiere justificación, no expone números de cuenta completos, límite de 100 registros por consulta.

La BD tiene: 10 clientes ficticios, 10 cuentas, 164 transacciones (52 sospechosas según criterios de monto, horario y país).

### Memoria (`memory/`)

**Sesión** (`session_memory.py`): lista de mensajes en RAM. Se pasa completa a la API en cada llamada para mantener contexto. Se borra al cerrar sesión.

**Histórica** (`historical_memory.py`): SQLite en `data/memory.db`. Persiste todos los mensajes de todas las sesiones. El orquestador puede buscar en ella con `search_history`.

```
Tablas: sessions | messages | summaries
```

### Observabilidad (`observability/`)

Langfuse Cloud registra cada consulta como traza anidada:

```
assistant-question
  ├── orchestrator_llm      (tokens entrada/salida, latencia)
  ├── tool_rag_search
  │     ├── rag_retrieval   (chunks recuperados)
  │     └── rag_generation  (tokens, costo)
  └── orchestrator_llm      (respuesta final)
```

---

## Configuración (`config.py` / `.env`)

```env
ANTHROPIC_API_KEY=sk-ant-...
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://us.cloud.langfuse.com
TAVILY_API_KEY=tvly-...
```

Modelos configurables en `config.py`:
```python
ORCHESTRATOR_MODEL = "claude-sonnet-4-6"
RAG_MODEL          = "claude-haiku-4-5-20251001"
SEARCH_MODEL       = "claude-haiku-4-5-20251001"
SUMMARIZER_MODEL   = "claude-haiku-4-5-20251001"
TRANSACTIONAL_MODEL= "claude-haiku-4-5-20251001"
```

---

## Cómo correrlo

### 1. Instalar dependencias
```bash
python -m venv .venv
source .venv/bin/activate   # fish: source .venv/bin/activate.fish
pip install -r requirements.txt
```

### 2. Configurar variables de entorno
```bash
cp .env.example .env
# Editar .env con las API keys
```

### 3. Indexar los PDFs (solo la primera vez)
```bash
python scripts/ingest_documents.py --path ./Apuntes
# Con --reset para reindexa desde cero
```

### 4. Poblar la BD transaccional (solo si se borró)
```bash
python scripts/seed_database.py
```

### 5. Iniciar el MCP Server (terminal separada)
```bash
python mcp_server/server.py
# Queda corriendo en localhost:8000
```

### 6. Lanzar la app
```bash
python -m streamlit run app/streamlit_app.py
# Abre http://localhost:8501
```

O con Docker para el MCP Server:
```bash
docker-compose up -d   # inicia el MCP Server en contenedor
python -m streamlit run app/streamlit_app.py
```

---

## Preguntas de ejemplo

| Pregunta | Agente activado |
|---|---|
| `¿Qué es el descenso del gradiente?` | RAG → ChromaDB |
| `Compara overfitting y underfitting` | RAG → ChromaDB |
| `Busca en internet qué es GPT-4o` | Web Search → Tavily |
| `¿Cuántas transacciones sospechosas hay esta semana?` | Transaccional → MCP → SQLite |
| `Resume la conversación` | Resumidor → Haiku |
| `¿Qué pregunté antes sobre redes neuronales?` | search_history → memory.db |

---

## Estructura del proyecto

```
.
├── agents/
│   ├── orchestrator.py
│   ├── rag_agent.py
│   ├── web_search_agent.py
│   ├── summarizer_agent.py
│   └── transactional_agent.py
├── app/
│   └── streamlit_app.py
├── mcp_server/
│   ├── server.py
│   ├── database.py
│   └── Dockerfile
├── rag/
│   ├── document_processor.py
│   └── vector_store.py
├── memory/
│   ├── session_memory.py
│   └── historical_memory.py
├── observability/
│   └── langfuse_client.py
├── scripts/
│   ├── ingest_documents.py
│   └── seed_database.py
├── eval/
│   ├── test_questions.py
│   └── rag_comparison.py
├── data/
│   ├── chroma_db/      # vectores indexados
│   ├── memory.db       # historial de sesiones
│   └── transactions.db # BD ficticia
├── Apuntes/            # 51 PDFs del curso
├── config.py
├── requirements.txt
└── docker-compose.yml
```
