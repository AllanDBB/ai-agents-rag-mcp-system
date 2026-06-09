import anthropic
import config
from rag.vector_store import search as vector_search
from observability.langfuse_client import get_langfuse

CONTRACT = {
    "agent_name": "RAGAgent",
    "description": "Agente especializado en recuperar información desde los apuntes del curso de IA.",
    "role": "Recuperador de información documental",
    "skills": ["search_notes", "answer_from_notes"],
    "allowed_tools": ["rag_tool"],
    "input_schema": {"query": "str", "collection": "str (opcional)"},
    "output_schema": {"answer": "str", "sources": "list[str]", "chunks_retrieved": "int"},
    "restrictions": [
        "Debe citar documentos y autores.",
        "No debe inventar fuentes.",
        "No debe usar búsqueda web.",
        "Si la info no está en los apuntes, indicarlo explícitamente.",
    ],
}

SYSTEM_PROMPT = """Eres un agente especializado en responder preguntas basadas en apuntes del curso de Inteligencia Artificial del TEC.

REGLAS ESTRICTAS:
- Responde ÚNICAMENTE con información del contexto proporcionado.
- Cita la fuente usando el formato [Fuente: nombre_archivo].
- Si la información no está en el contexto, responde: "Esta información no se encuentra en los apuntes disponibles."
- NO uses conocimiento externo ni inventes datos.
- Responde en español."""


class RAGAgent:
    def __init__(self):
        self.client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        self.model = config.RAG_MODEL

    def search(self, query: str, collection: str = config.COLLECTION_A) -> dict:
        lf = get_langfuse()

        def _do_search():
            results = vector_search(query, collection_name=collection)
            docs  = results.get("documents", [[]])[0]
            metas = results.get("metadatas",  [[]])[0]

            if not docs:
                return {
                    "answer": "No encontré información relevante en los apuntes para esta consulta.",
                    "sources": [],
                    "chunks_retrieved": 0,
                }

            context = "\n\n---\n\n".join(
                f"[Fuente: {m.get('source', '?')} — Semana {m.get('semana', '?')}]\n{doc}"
                for doc, m in zip(docs, metas)
            )
            sources = list({m.get("source", "?") for m in metas})

            response = self.client.messages.create(
                model=self.model,
                max_tokens=2048,
                system=SYSTEM_PROMPT,
                messages=[{
                    "role": "user",
                    "content": f"Contexto de los apuntes:\n{context}\n\nPregunta: {query}",
                }],
            )
            answer = response.content[0].text
            return {"answer": answer, "sources": sources, "chunks_retrieved": len(docs)}

        if not lf:
            return _do_search()

        with lf.start_as_current_observation(
            name="rag_retrieval",
            as_type="retriever",
            input={"query": query, "collection": collection},
        ):
            results = vector_search(query, collection_name=collection)
            docs  = results.get("documents", [[]])[0]
            metas = results.get("metadatas",  [[]])[0]

            if not docs:
                result = {
                    "answer": "No encontré información relevante en los apuntes para esta consulta.",
                    "sources": [],
                    "chunks_retrieved": 0,
                }
                lf.update_current_span(output=result)
                return result

            context = "\n\n---\n\n".join(
                f"[Fuente: {m.get('source', '?')} — Semana {m.get('semana', '?')}]\n{doc}"
                for doc, m in zip(docs, metas)
            )
            sources = list({m.get("source", "?") for m in metas})

            with lf.start_as_current_observation(
                name="rag_generation",
                as_type="generation",
                model=self.model,
                input=[{
                    "role": "user",
                    "content": f"Contexto:\n{context}\n\nPregunta: {query}",
                }],
            ):
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=2048,
                    system=SYSTEM_PROMPT,
                    messages=[{
                        "role": "user",
                        "content": f"Contexto de los apuntes:\n{context}\n\nPregunta: {query}",
                    }],
                )
                answer = response.content[0].text
                lf.update_current_generation(
                    output=answer,
                    usage_details={
                        "input": response.usage.input_tokens,
                        "output": response.usage.output_tokens,
                    },
                )

            result = {"answer": answer, "sources": sources, "chunks_retrieved": len(docs)}
            lf.update_current_span(output=result)
            return result
