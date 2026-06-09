"""Puebla la BD ficticia con clientes, cuentas y transacciones de prueba."""
import sys
import random
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from mcp_server.database import init_db

COUNTRIES = ["CR", "US", "MX", "BR", "CO", "CN", "RU", "NG", "DE", "ES"]
TX_TYPES  = ["compra", "transferencia", "retiro", "deposito", "pago_servicio"]
NAMES = [
    "Ana Torres", "Carlos Mora", "Lucia Jiménez", "Diego Vargas", "Sofia Rojas",
    "Miguel Cruz", "Valentina Pérez", "Andrés Méndez", "Isabella Chaves", "Sebastián Álvarez",
]
RISK_LEVELS = ["low", "low", "low", "low", "low", "low", "medium", "medium", "high", "high"]


def _conn():
    return sqlite3.connect(config.DATABASE_PATH)


def seed(random_seed: int = 42):
    random.seed(random_seed)
    init_db()
    conn = _conn()

    customers, accounts, transactions = [], [], []

    for i, name in enumerate(NAMES):
        # IDs predecibles para facilitar evaluación experimental
        cid = f"CUST-{i+1:08d}"
        customers.append((cid, name, f"{name.replace(' ', '.').lower()}@correo.com", RISK_LEVELS[i]))

        aid = f"ACC-{i+1:08d}"
        balance = round(random.uniform(500, 50000), 2)
        last_four = str(random.randint(1000, 9999))
        accounts.append((aid, cid, last_four, balance))

        n_tx = random.randint(8, 25)
        for j in range(n_tx):
            tid = f"TX-{i+1:04d}{j+1:06d}"
            amount = round(random.uniform(10, 15000), 2)
            days_ago = random.randint(0, 90)
            tx_date = (datetime.now() - timedelta(days=days_ago)).isoformat()
            country = random.choice(COUNTRIES)
            tx_type = random.choice(TX_TYPES)
            hour = random.randint(0, 23)

            status = "normal"
            if amount > 10000 and hour < 5:
                status = "suspicious"
            elif country in ["CN", "RU", "NG"] and amount > 5000:
                status = "suspicious"
            elif RISK_LEVELS[i] == "high" and random.random() < 0.35:
                status = "suspicious"

            transactions.append((tid, aid, amount, tx_date, country, tx_type, status, hour))

    with conn:
        conn.executemany("INSERT OR IGNORE INTO customers VALUES (?,?,?,?)", customers)
        conn.executemany("INSERT OR IGNORE INTO accounts  VALUES (?,?,?,?)", accounts)
        conn.executemany(
            "INSERT OR IGNORE INTO transactions VALUES (?,?,?,?,?,?,?,?)", transactions
        )

    conn.close()

    total     = len(transactions)
    suspicious = sum(1 for t in transactions if t[6] == "suspicious")
    print(f"✓ {len(customers)} clientes | {len(accounts)} cuentas | {total} transacciones")
    print(f"  Sospechosas: {suspicious} ({100*suspicious//total}%)")
    print(f"  IDs de ejemplo: CUST-00000001 … CUST-0000000{len(customers)}")


if __name__ == "__main__":
    seed()
