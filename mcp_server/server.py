"""MCP Server — transacciones ficticias."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp.server.fastmcp import FastMCP
from mcp_server import database as db

mcp = FastMCP("transactional-server")


# ── Herramientas ─────────────────────────────────────────────────────────────

@mcp.tool()
def get_transaction_by_id(transaction_id: str, justification: str) -> str:
    """Obtiene una transacción por su ID. Muestra solo los últimos 4 dígitos de cuenta."""
    db.log_tool_call("get_transaction_by_id", json.dumps({"transaction_id": transaction_id}), justification)
    result = db.get_transaction_by_id(transaction_id)
    if not result:
        return json.dumps({"error": f"Transacción {transaction_id} no encontrada."})
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def search_transactions(
    justification: str,
    status: str = None,
    country: str = None,
    min_amount: float = None,
    max_amount: float = None,
    date_from: str = None,
    date_to: str = None,
    limit: int = 20,
) -> str:
    """Busca transacciones con filtros. Requiere al menos un filtro para evitar consultas masivas."""
    filters = {k: v for k, v in {
        "status": status, "country": country,
        "min_amount": min_amount, "max_amount": max_amount,
        "date_from": date_from, "date_to": date_to,
    }.items() if v is not None}

    if not filters:
        return json.dumps({"error": "Se requiere al menos un filtro para evitar consultas masivas."})

    db.log_tool_call("search_transactions", json.dumps(filters), justification)

    try:
        results = db.search_transactions(
            status=status, country=country,
            min_amount=min_amount, max_amount=max_amount,
            date_from=date_from, date_to=date_to,
            limit=min(limit, 100),
        )
        return json.dumps({"count": len(results), "transactions": results}, ensure_ascii=False)
    except ValueError as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
def get_customer_risk_summary(customer_id: str, justification: str) -> str:
    """Devuelve el resumen de riesgo de un cliente."""
    db.log_tool_call("get_customer_risk_summary", json.dumps({"customer_id": customer_id}), justification)
    result = db.get_customer_risk_summary(customer_id)
    if not result:
        return json.dumps({"error": f"Cliente {customer_id} no encontrado."})
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def get_recent_flagged_transactions(days: int, justification: str) -> str:
    """Obtiene transacciones sospechosas o marcadas en los últimos N días."""
    if days > 365:
        return json.dumps({"error": "El rango máximo es 365 días."})
    db.log_tool_call("get_recent_flagged_transactions", json.dumps({"days": days}), justification)
    results = db.get_recent_flagged_transactions(days)
    return json.dumps({"count": len(results), "transactions": results}, ensure_ascii=False)


@mcp.tool()
def create_fraud_case(
    transaction_id: str,
    reason: str,
    severity: str,
    justification: str,
) -> str:
    """Crea un caso de fraude. Severity: low | medium | high | critical."""
    db.log_tool_call(
        "create_fraud_case",
        json.dumps({"transaction_id": transaction_id, "severity": severity}),
        justification,
    )
    try:
        result = db.create_fraud_case(transaction_id, reason, severity)
        return json.dumps(result, ensure_ascii=False)
    except ValueError as e:
        return json.dumps({"error": str(e)})


if __name__ == "__main__":
    import config
    import os
    os.environ.setdefault("FASTMCP_PORT", str(config.MCP_SERVER_PORT))
    mcp.run(transport="sse")
