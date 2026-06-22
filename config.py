import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

# Paths
CHROMA_PATH = os.getenv("CHROMA_PATH", "./data/chroma_db")
DOCUMENTS_PATH = os.getenv("DOCUMENTS_PATH", "./data/documents")
DATABASE_PATH = os.getenv("DATABASE_PATH", "./data/transactions.db")
MEMORY_DB_PATH = os.getenv("MEMORY_DB_PATH", "./data/memory.db")

# MCP Server
MCP_SERVER_HOST = os.getenv("MCP_SERVER_HOST", "localhost")
MCP_SERVER_PORT = int(os.getenv("MCP_SERVER_PORT", "8000"))
MCP_SERVER_URL = f"http://{MCP_SERVER_HOST}:{MCP_SERVER_PORT}"

# Models
ORCHESTRATOR_MODEL = "claude-sonnet-4-6"
RAG_MODEL = "claude-haiku-4-5-20251001"
SEARCH_MODEL = "claude-haiku-4-5-20251001"
SUMMARIZER_MODEL = "claude-haiku-4-5-20251001"
TRANSACTIONAL_MODEL = "claude-haiku-4-5-20251001"

# RAG - Config A: chunks fijos
RAG_CHUNK_SIZE_A = 50
RAG_CHUNK_OVERLAP_A = 5

# RAG - Config B: chunks semánticos
RAG_CHUNK_SIZE_B = 1024
RAG_CHUNK_OVERLAP_B = 128

# Retrieval
RAG_FETCH_K = 30   # chunks que se traen de ChromaDB
RAG_TOP_K = 5      # chunks que llegan al LLM tras el rerank

# Early stopping en reranking: si el mejor score supera este umbral se retorna antes
RAG_EARLY_STOP_THRESHOLD = 0.85

# Collections ChromaDB
COLLECTION_A = "course_notes_a"
COLLECTION_B = "course_notes_b"
