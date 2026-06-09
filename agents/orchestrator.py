import json

import anthropic

import config
from agents.rag_agent import RAGAgent
from agents.web_search_agent import WebSearchAgent
from agents.summarizer_agent import SummarizerAgent
from agents.transactional_agent import TransactionalAgent
from memory.session_memory import SessionMemory
from memory import historical_memory as hist
from observability.langfuse_client import get_langfuse, flush

SYSTEM_PROMPT = """Eres el Agente Orquestador de un sistema multi-agente del curso de Inteligencia Artificial del TEC.
Tu función es recibir preguntas, decidir qué agente o herramienta utilizar, y construir la respuesta final.

REGLAS:
1. Para preguntas sobre el contenido del curso de IA, usa `rag_search`.
2. Para búsquedas en internet, usa `web_search` SOLO si el usuario lo solicita explícitamente.
3. Para resumir contenido largo o el historial de la conversación, usa `summarize`.
4. Para consultar datos de transacciones ficticias, usa `mcp_query`.
5. Para buscar en el historial de conversaciones anteriores, usa `search_history`.
6. Justifica siempre cada llamada a una herramienta.
7. Responde siempre en español.
8. Cuando uses RAG, incluye las fuentes en tu respuesta final.
9. Si no tienes suficiente información para responder, indícalo claramente."""

TOOLS = [
    {
        "name": "rag_search",
        "description": (
            "Consulta la base vectorial de apuntes del curso de IA del TEC. "
            "Usar para preguntas sobre temas vistos en clase."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Consulta a buscar en los apuntes"},
                "justification": {"type": "string", "description": "Por qué se invoca este agente"},
            },
            "required": ["query", "justification"],
        },
    },
    {
        "name": "web_search",
        "description": "Realiza una búsqueda en internet. SOLO cuando el usuario lo pide explícitamente.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "justification": {"type": "string"},
            },
            "required": ["query", "justification"],
        },
    },
    {
        "name": "summarize",
        "description": "Resume contenido largo, historial de conversación o respuestas extensas.",
        "input_schema": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "Contenido a resumir"},
                "summary_type": {
                    "type": "string",
                    "enum": ["conversation", "documents", "response"],
                },
            },
            "required": ["content", "summary_type"],
        },
    },
    {
        "name": "mcp_query",
        "description": "Consulta la base de datos transaccional ficticia vía MCP Server.",
        "input_schema": {
            "type": "object",
            "properties": {
                "tool": {
                    "type": "string",
                    "enum": [
                        "get_transaction_by_id",
                        "search_transactions",
                        "get_customer_risk_summary",
                        "get_recent_flagged_transactions",
                        "create_fraud_case",
                    ],
                },
                "params": {"type": "object"},
                "justification": {"type": "string"},
            },
            "required": ["tool", "params", "justification"],
        },
    },
    {
        "name": "search_history",
        "description": "Busca en el historial de conversaciones anteriores (memoria histórica persistente).",
        "input_schema": {
            "type": "object",
            "properties": {
                "keyword": {"type": "string", "description": "Palabra clave a buscar"},
            },
            "required": ["keyword"],
        },
    },
]


class OrchestratorAgent:
    def __init__(self):
        self.client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        self.model = config.ORCHESTRATOR_MODEL
        self.rag = RAGAgent()
        self.web = WebSearchAgent()
        self.summarizer = SummarizerAgent()
        self.transactional = TransactionalAgent()

    def _execute_tool(self, name: str, inputs: dict) -> str:
        if name == "rag_search":
            result = self.rag.search(inputs["query"])
            return json.dumps(result, ensure_ascii=False)
        if name == "web_search":
            result = self.web.search(inputs["query"])
            return json.dumps(result, ensure_ascii=False)
        if name == "summarize":
            summary = self.summarizer.summarize(inputs["content"], inputs.get("summary_type", "response"))
            return json.dumps({"summary": summary}, ensure_ascii=False)
        if name == "mcp_query":
            result = self.transactional.query(inputs["tool"], inputs["params"], inputs["justification"])
            return json.dumps(result, ensure_ascii=False)
        if name == "search_history":
            results = hist.search_history(inputs["keyword"])
            return json.dumps({"history": results}, ensure_ascii=False)
        return json.dumps({"error": f"Herramienta desconocida: {name}"})

    def _run_loop(self, api_messages: list, tools_used: list) -> str:
        """Core agentic loop — returns final answer text."""
        lf = get_langfuse()

        while True:
            if lf:
                with lf.start_as_current_observation(
                    name="orchestrator_llm",
                    as_type="generation",
                    model=self.model,
                    input=api_messages,
                ):
                    response = self.client.messages.create(
                        model=self.model,
                        max_tokens=4096,
                        system=SYSTEM_PROMPT,
                        tools=TOOLS,
                        messages=api_messages,
                    )
                    out_text = next((b.text for b in response.content if hasattr(b, "text")), "")
                    lf.update_current_generation(
                        output=out_text or str(response.content),
                        usage_details={
                            "input": response.usage.input_tokens,
                            "output": response.usage.output_tokens,
                        },
                    )
            else:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=4096,
                    system=SYSTEM_PROMPT,
                    tools=TOOLS,
                    messages=api_messages,
                )
                out_text = next((b.text for b in response.content if hasattr(b, "text")), "")

            if response.stop_reason == "end_turn":
                return out_text

            if response.stop_reason == "tool_use":
                api_messages.append({"role": "assistant", "content": response.content})
                tool_results = []

                for block in response.content:
                    if block.type == "tool_use":
                        tools_used.append({"tool": block.name, "input": block.input})
                        if lf:
                            with lf.start_as_current_observation(
                                name=f"tool_{block.name}",
                                as_type="tool",
                                input=block.input,
                            ):
                                result_str = self._execute_tool(block.name, block.input)
                                lf.update_current_span(output=result_str)
                        else:
                            result_str = self._execute_tool(block.name, block.input)

                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result_str,
                        })

                api_messages.append({"role": "user", "content": tool_results})

    def run(self, user_message: str, session: SessionMemory) -> dict:
        lf = get_langfuse()
        api_messages = session.get_messages_for_api()
        api_messages.append({"role": "user", "content": user_message})
        tools_used: list[dict] = []

        if lf:
            from langfuse.types import TraceContext
            trace_ctx = TraceContext(
                session_id=session.session_id,
                trace_name="assistant-question",
            )
            with lf.start_as_current_observation(
                name="assistant-question",
                as_type="span",
                trace_context=trace_ctx,
                input={"message": user_message},
            ):
                answer = self._run_loop(api_messages, tools_used)
                lf.update_current_span(output={"answer": answer, "tools_used": tools_used})
        else:
            answer = self._run_loop(api_messages, tools_used)

        session.add_message("user", user_message)
        session.add_message("assistant", answer)
        hist.save_message(session.session_id, "user", user_message)
        hist.save_message(session.session_id, "assistant", answer)
        flush()

        return {"answer": answer, "tools_used": tools_used}
