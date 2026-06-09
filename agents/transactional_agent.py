import asyncio
import json

import config
from observability.langfuse_client import get_langfuse

CONTRACT = {
    "agent_name": "TransactionalAgent",
    "description": "Consulta la base de datos transaccional ficticia mediante el MCP Server.",
    "role": "Agente de consulta transaccional",
    "allowed_tools": [
        "get_transaction_by_id",
        "search_transactions",
        "get_customer_risk_summary",
        "get_recent_flagged_transactions",
        "create_fraud_case",
    ],
    "input_schema": {"tool": "str", "params": "dict", "justification": "str"},
    "output_schema": {"result": "dict", "justification": "str"},
    "restrictions": [
        "No accede directamente a la base de datos.",
        "Toda llamada debe incluir justificación descriptiva.",
        "No muestra números de cuenta completos.",
        "No permite consultas masivas sin filtros.",
        "No permite modificar transacciones existentes.",
    ],
}

ALLOWED_TOOLS = set(CONTRACT["allowed_tools"])


class TransactionalAgent:
    def query(self, tool: str, params: dict, justification: str) -> dict:
        if tool not in ALLOWED_TOOLS:
            return {"error": f"Herramienta '{tool}' no permitida.", "allowed": list(ALLOWED_TOOLS)}
        if not justification or len(justification.strip()) < 10:
            return {"error": "Justificación requerida (mínimo 10 caracteres)."}

        lf = get_langfuse()

        full_params = {**params, "justification": justification}

        def _do():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, self._call_mcp(tool, full_params))
                try:
                    return future.result(timeout=30)
                except Exception as e:
                    return {
                        "error": f"MCP Server no disponible: {e}",
                        "hint": "Ejecute: python mcp_server/server.py",
                    }

        if not lf:
            return _do()

        with lf.start_as_current_observation(
            name=f"mcp_{tool}",
            as_type="tool",
            input={"tool": tool, "params": params, "justification": justification},
        ):
            result = _do()
            level = "ERROR" if "error" in result else "DEFAULT"
            lf.update_current_span(output=result, level=level)
            return result

    async def _call_mcp(self, tool: str, params: dict) -> dict:
        try:
            from mcp.client.sse import sse_client
            from mcp import ClientSession

            url = f"{config.MCP_SERVER_URL}/sse"
            async with sse_client(url) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool(tool, params)
                    content = result.content
                    if content and hasattr(content[0], "text"):
                        return json.loads(content[0].text)
                    return {"result": str(content)}
        except Exception as e:
            return {
                "error": f"MCP Server no disponible: {e}",
                "hint": "Ejecute: python mcp_server/server.py",
            }
