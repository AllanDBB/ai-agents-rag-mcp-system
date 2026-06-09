import config
from observability.langfuse_client import get_langfuse

CONTRACT = {
    "agent_name": "WebSearchAgent",
    "description": "Realiza búsquedas en internet cuando el usuario lo solicita explícitamente.",
    "role": "Buscador web externo",
    "allowed_tools": ["tavily_search"],
    "input_schema": {"query": "str"},
    "output_schema": {"results": "list[dict]", "error": "str (opcional)"},
    "restrictions": [
        "Solo cuando el usuario lo pide explícitamente.",
        "No reemplaza al RAG para preguntas del curso.",
    ],
}


class WebSearchAgent:
    def search(self, query: str) -> dict:
        lf = get_langfuse()

        if not config.TAVILY_API_KEY:
            return {
                "results": [],
                "error": "Búsqueda web no disponible. Configure TAVILY_API_KEY en .env",
            }

        def _do():
            from tavily import TavilyClient
            client = TavilyClient(api_key=config.TAVILY_API_KEY)
            resp = client.search(query=query, max_results=5)
            return [
                {"title": r.get("title", ""), "url": r.get("url", ""), "content": r.get("content", "")}
                for r in resp.get("results", [])
            ]

        if not lf:
            try:
                return {"results": _do()}
            except Exception as e:
                return {"results": [], "error": str(e)}

        with lf.start_as_current_observation(
            name="web_search",
            as_type="tool",
            input={"query": query},
        ):
            try:
                results = _do()
                lf.update_current_span(output={"results_count": len(results)})
                return {"results": results}
            except Exception as e:
                lf.update_current_span(level="ERROR", status_message=str(e))
                return {"results": [], "error": str(e)}
