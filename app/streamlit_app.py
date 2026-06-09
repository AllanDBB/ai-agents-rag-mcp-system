import streamlit as st
from agents.orchestrator import OrchestratorAgent
from memory.session_memory import SessionMemory
from memory import historical_memory as hist
from rag.vector_store import collection_count
import config

st.set_page_config(
    page_title="Sistema Multi-Agente — Curso IA TEC",
    page_icon="🤖",
    layout="wide",
)

# ── Estado de sesión ──────────────────────────────────────────────────────────
if "session" not in st.session_state:
    st.session_state.session = SessionMemory()
if "orchestrator" not in st.session_state:
    st.session_state.orchestrator = OrchestratorAgent()
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🤖 Sistema Multi-Agente")
    st.caption("Curso IA — TEC")

    st.divider()
    st.subheader("Base vectorial")
    count_a = collection_count(config.COLLECTION_A)
    count_b = collection_count(config.COLLECTION_B)
    st.metric("Config A (chunks fijos)", count_a)
    st.metric("Config B (semántico)", count_b)
    if count_a == 0:
        st.warning("Sin documentos indexados.\nEjecuta: `python scripts/ingest_documents.py`")

    st.divider()
    st.subheader("Sesión actual")
    st.code(st.session_state.session.session_id[:8] + "...", language=None)
    st.metric("Mensajes en sesión", len(st.session_state.session))

    if st.button("🗑️ Nueva sesión"):
        st.session_state.session.clear()
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.subheader("Agentes disponibles")
    agents = [
        ("🎯", "Orquestador", config.ORCHESTRATOR_MODEL),
        ("📚", "RAG", config.RAG_MODEL),
        ("🌐", "Web Search", config.SEARCH_MODEL),
        ("📝", "Resumidor", config.SUMMARIZER_MODEL),
        ("💳", "Transaccional", config.TRANSACTIONAL_MODEL),
    ]
    for icon, name, model in agents:
        st.caption(f"{icon} **{name}** — `{model.split('-')[1]}`")

    st.divider()
    st.caption("Observabilidad: [Langfuse](https://cloud.langfuse.com)")

# ── Chat principal ────────────────────────────────────────────────────────────
st.title("💬 Asistente del Curso de IA")

# Preguntas de ejemplo
with st.expander("💡 Preguntas de ejemplo"):
    examples = [
        "¿Qué es el descenso del gradiente?",
        "Explica las redes neuronales convolucionales",
        "¿Cuál es la diferencia entre overfitting y underfitting?",
        "Busca en internet las últimas noticias sobre IA (búsqueda web explícita)",
        "¿Cuántas transacciones sospechosas hay en los últimos 7 días?",
        "¿Qué pregunté anteriormente?",
        "Resume la conversación de esta sesión",
    ]
    for ex in examples:
        if st.button(ex, key=ex):
            st.session_state._pending_input = ex
            st.rerun()

# Mostrar historial
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("tools_used"):
            with st.expander(f"🔧 Agentes utilizados ({len(msg['tools_used'])})"):
                for tool in msg["tools_used"]:
                    st.json(tool)

# Input del usuario
pending = st.session_state.pop("_pending_input", None)
prompt = st.chat_input("Escribe tu pregunta sobre el curso...") or pending

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Procesando con los agentes..."):
            result = st.session_state.orchestrator.run(
                prompt, st.session_state.session
            )
        st.markdown(result["answer"])
        if result["tools_used"]:
            with st.expander(f"🔧 Agentes utilizados ({len(result['tools_used'])})"):
                for tool in result["tools_used"]:
                    st.json(tool)

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "tools_used": result["tools_used"],
    })
