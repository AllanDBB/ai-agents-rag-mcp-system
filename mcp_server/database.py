"""Base de datos transaccional ficticia."""
import json
import re
import sqlite3
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import config


def _conn() -> sqlite3.Connection:
    Path(config.DATABASE_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(config.DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with _conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS customers (
                customer_id   TEXT PRIMARY KEY,
                name          TEXT NOT NULL,
                email         TEXT NOT NULL,
                risk_level    TEXT NOT NULL DEFAULT 'low'
            );

            CREATE TABLE IF NOT EXISTS accounts (
                account_id      TEXT PRIMARY KEY,
                customer_id     TEXT NOT NULL,
                last_four       TEXT NOT NULL,
                balance         REAL NOT NULL,
                FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
            );

            CREATE TABLE IF NOT EXISTS transactions (
                transaction_id   TEXT PRIMARY KEY,
                account_id       TEXT NOT NULL,
                amount           REAL NOT NULL,
                transaction_date TEXT NOT NULL,
                country          TEXT NOT NULL,
                transaction_type TEXT NOT NULL,
                status           TEXT NOT NULL DEFAULT 'normal',
                hour             INTEGER NOT NULL,
                FOREIGN KEY (account_id) REFERENCES accounts(account_id)
            );

            CREATE TABLE IF NOT EXISTS fraud_cases (
                case_id        TEXT PRIMARY KEY,
                transaction_id TEXT NOT NULL,
                reason         TEXT NOT NULL,
                severity       TEXT NOT NULL,
                created_at     TEXT NOT NULL,
                FOREIGN KEY (transaction_id) REFERENCES transactions(transaction_id)
            );

            CREATE TABLE IF NOT EXISTS tool_usage_log (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                tool_name     TEXT NOT NULL,
                params        TEXT NOT NULL,
                justification TEXT NOT NULL,
                called_at     TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS tool_usage_rules (
                rule_id     INTEGER PRIMARY KEY AUTOINCREMENT,
                tool_name   TEXT NOT NULL,
                description TEXT NOT NULL,
                requires_justification INTEGER NOT NULL DEFAULT 1,
                max_results INTEGER,
                allows_bulk INTEGER NOT NULL DEFAULT 0,
                sensitive   INTEGER NOT NULL DEFAULT 0
            );
        """)
        _seed_rules(conn)


def _seed_rules(conn: sqlite3.Connection):
    existing = conn.execute("SELECT COUNT(*) FROM tool_usage_rules").fetchone()[0]
    if existing > 0:
        return
    rules = [
        ("get_transaction_by_id",          "Obtiene una transacción por ID. Solo últimos 4 dígitos de cuenta.", 1, 1,   0, 0),
        ("search_transactions",            "Busca transacciones con filtros obligatorios. Máx 100 resultados.",  1, 100, 0, 0),
        ("get_customer_risk_summary",      "Resumen de riesgo de un cliente. Email anonimizado.",               1, 1,   0, 1),
        ("get_recent_flagged_transactions","Transacciones sospechosas. Máx 365 días hacia atrás.",              1, None,0, 0),
        ("create_fraud_case",              "Crea caso de fraude. Requiere justificación detallada.",            1, None,0, 0),
    ]
    conn.executemany(
        "INSERT INTO tool_usage_rules (tool_name, description, requires_justification, max_results, allows_bulk, sensitive) VALUES (?,?,?,?,?,?)",
        rules,
    )


def _anonymize(row: dict) -> dict:
    """Anonimiza datos sensibles: email y datos de identificación directa."""
    if "email" in row:
        local, _, domain = row["email"].partition("@")
        row["email"] = local[:2] + "***@" + domain
    return row


def log_tool_call(tool_name: str, params: str, justification: str):
    with _conn() as conn:
        conn.execute(
            "INSERT INTO tool_usage_log (tool_name, params, justification, called_at) VALUES (?,?,?,?)",
            (tool_name, params, justification, datetime.now().isoformat()),
        )


# ── Queries ───────────────────────────────────────────────────────────────────

def get_transaction_by_id(transaction_id: str) -> dict | None:
    with _conn() as conn:
        row = conn.execute(
            """SELECT t.transaction_id, t.amount, t.transaction_date,
                      t.country, t.transaction_type, t.status, t.hour,
                      a.last_four AS account_last_four,
                      c.name AS customer_name, c.risk_level
               FROM transactions t
               JOIN accounts a ON t.account_id = a.account_id
               JOIN customers c ON a.customer_id = c.customer_id
               WHERE t.transaction_id = ?""",
            (transaction_id,),
        ).fetchone()
    return dict(row) if row else None


def search_transactions(
    status: str = None,
    country: str = None,
    min_amount: float = None,
    max_amount: float = None,
    date_from: str = None,
    date_to: str = None,
    limit: int = 20,
) -> list[dict]:
    if limit > 100:
        raise ValueError("El límite máximo es 100 registros.")

    clauses, params = [], []
    if status:
        clauses.append("t.status = ?"); params.append(status)
    if country:
        clauses.append("t.country = ?"); params.append(country)
    if min_amount is not None:
        clauses.append("t.amount >= ?"); params.append(min_amount)
    if max_amount is not None:
        clauses.append("t.amount <= ?"); params.append(max_amount)
    if date_from:
        clauses.append("t.transaction_date >= ?"); params.append(date_from)
    if date_to:
        clauses.append("t.transaction_date <= ?"); params.append(date_to)

    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    params.append(limit)

    with _conn() as conn:
        rows = conn.execute(
            f"""SELECT t.transaction_id, t.amount, t.transaction_date,
                       t.country, t.transaction_type, t.status, t.hour,
                       a.last_four AS account_last_four, c.name AS customer_name
                FROM transactions t
                JOIN accounts a ON t.account_id = a.account_id
                JOIN customers c ON a.customer_id = c.customer_id
                {where}
                ORDER BY t.transaction_date DESC LIMIT ?""",
            params,
        ).fetchall()
    return [dict(r) for r in rows]


def get_customer_risk_summary(customer_id: str) -> dict | None:
    with _conn() as conn:
        customer = conn.execute(
            "SELECT * FROM customers WHERE customer_id = ?", (customer_id,)
        ).fetchone()
        if not customer:
            return None
        stats = conn.execute(
            """SELECT COUNT(*) AS total, SUM(amount) AS total_amount,
                      SUM(CASE WHEN status='suspicious' THEN 1 ELSE 0 END) AS suspicious_count,
                      SUM(CASE WHEN status='flagged'    THEN 1 ELSE 0 END) AS flagged_count
               FROM transactions t
               JOIN accounts a ON t.account_id = a.account_id
               WHERE a.customer_id = ?""",
            (customer_id,),
        ).fetchone()

    result = {
        "customer_id": customer_id,
        "name": dict(customer)["name"],
        "email": dict(customer)["email"],
        "risk_level": dict(customer)["risk_level"],
        "total_transactions": dict(stats)["total"],
        "total_amount": round(dict(stats)["total_amount"] or 0, 2),
        "suspicious_count": dict(stats)["suspicious_count"],
        "flagged_count": dict(stats)["flagged_count"],
    }
    return _anonymize(result)


def get_recent_flagged_transactions(days: int = 7) -> list[dict]:
    if days > 365:
        raise ValueError("El rango máximo permitido es 365 días.")
    cutoff = (datetime.now() - timedelta(days=days)).isoformat()
    with _conn() as conn:
        rows = conn.execute(
            """SELECT t.transaction_id, t.amount, t.transaction_date,
                      t.country, t.status, t.hour,
                      a.last_four AS account_last_four, c.name AS customer_name
               FROM transactions t
               JOIN accounts a ON t.account_id = a.account_id
               JOIN customers c ON a.customer_id = c.customer_id
               WHERE t.status IN ('suspicious', 'flagged')
                 AND t.transaction_date >= ?
               ORDER BY t.transaction_date DESC""",
            (cutoff,),
        ).fetchall()
    return [dict(r) for r in rows]


def create_fraud_case(transaction_id: str, reason: str, severity: str) -> dict:
    valid = {"low", "medium", "high", "critical"}
    if severity not in valid:
        raise ValueError(f"Severidad inválida. Opciones: {valid}")
    if len(reason.strip()) < 15:
        raise ValueError("La razón debe ser descriptiva (mínimo 15 caracteres).")

    case_id = f"CASE-{uuid.uuid4().hex[:8].upper()}"
    with _conn() as conn:
        conn.execute(
            "INSERT INTO fraud_cases (case_id, transaction_id, reason, severity, created_at) VALUES (?,?,?,?,?)",
            (case_id, transaction_id, reason, severity, datetime.now().isoformat()),
        )
        conn.execute(
            "UPDATE transactions SET status = 'flagged' WHERE transaction_id = ?",
            (transaction_id,),
        )
    return {"case_id": case_id, "transaction_id": transaction_id, "severity": severity}


init_db()
