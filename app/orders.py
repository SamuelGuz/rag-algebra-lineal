"""Pedidos confirmados, guardados en SQLite."""
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime

from app import config

STATES = ["recibido", "preparando", "listo", "entregado"]


class OrderStore:
    def __init__(self, path=None):
        self.path = str(path or config.ORDERS_DB)
        with self._conn() as c:
            c.execute(
                """CREATE TABLE IF NOT EXISTS pedidos (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cliente TEXT NOT NULL,
                    items TEXT NOT NULL,
                    total INTEGER NOT NULL,
                    estado TEXT NOT NULL,
                    creado TEXT NOT NULL
                )"""
            )

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def create(self, cliente: str, cart: dict) -> dict:
        creado = datetime.now().isoformat(timespec="seconds")
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO pedidos (cliente, items, total, estado, creado) VALUES (?, ?, ?, ?, ?)",
                (cliente, json.dumps(cart["items"], ensure_ascii=False), cart["total"], STATES[0], creado),
            )
            return self.get(cur.lastrowid, c)

    def get(self, order_id: int, conn=None) -> dict | None:
        if conn is None:
            with self._conn() as c:
                return self.get(order_id, c)
        row = conn.execute("SELECT * FROM pedidos WHERE id = ?", (order_id,)).fetchone()
        return self._row(row) if row else None

    def list(self) -> list[dict]:
        with self._conn() as c:
            return [self._row(r) for r in c.execute("SELECT * FROM pedidos ORDER BY id DESC")]

    def set_status(self, order_id: int, estado: str) -> dict | None:
        if estado not in STATES:
            raise ValueError(f"Estado inválido: {estado}")
        with self._conn() as c:
            c.execute("UPDATE pedidos SET estado = ? WHERE id = ?", (estado, order_id))
            return self.get(order_id, c)

    @staticmethod
    def _row(row) -> dict:
        d = dict(row)
        d["items"] = json.loads(d["items"])
        return d
