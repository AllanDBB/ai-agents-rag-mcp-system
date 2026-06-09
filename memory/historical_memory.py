import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

import config


def _get_conn() -> sqlite3.Connection:
    Path(config.MEMORY_DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(config.MEMORY_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with _get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id  TEXT PRIMARY KEY,
                created_at  TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS messages (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  TEXT NOT NULL,
                role        TEXT NOT NULL,
                content     TEXT NOT NULL,
                created_at  TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            );
            CREATE TABLE IF NOT EXISTS summaries (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  TEXT NOT NULL,
                summary     TEXT NOT NULL,
                created_at  TEXT NOT NULL
            );
        """)


def save_session(session_id: str):
    with _get_conn() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO sessions (session_id, created_at) VALUES (?, ?)",
            (session_id, datetime.now().isoformat()),
        )


def save_message(session_id: str, role: str, content: str):
    save_session(session_id)
    with _get_conn() as conn:
        conn.execute(
            "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (session_id, role, content, datetime.now().isoformat()),
        )


def save_summary(session_id: str, summary: str):
    with _get_conn() as conn:
        conn.execute(
            "INSERT INTO summaries (session_id, summary, created_at) VALUES (?, ?, ?)",
            (session_id, summary, datetime.now().isoformat()),
        )


def get_recent_sessions(limit: int = 10) -> list[dict]:
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM sessions ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def get_session_messages(session_id: str) -> list[dict]:
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT role, content, created_at FROM messages WHERE session_id = ? ORDER BY id",
            (session_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def search_history(keyword: str, limit: int = 10) -> list[dict]:
    with _get_conn() as conn:
        rows = conn.execute(
            """SELECT m.session_id, m.role, m.content, m.created_at
               FROM messages m
               WHERE m.content LIKE ?
               ORDER BY m.created_at DESC LIMIT ?""",
            (f"%{keyword}%", limit),
        ).fetchall()
    return [dict(r) for r in rows]


def get_all_summaries(limit: int = 20) -> list[dict]:
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM summaries ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


init_db()
