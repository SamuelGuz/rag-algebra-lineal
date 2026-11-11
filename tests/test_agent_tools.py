from types import SimpleNamespace

import pytest
from google.genai import types

from app import agent, vector_store
from app.orders import OrderStore


@pytest.fixture
def session(col, tmp_path, monkeypatch):
    # Usa la colección temporal en todas las funciones de vector_store
    monkeypatch.setattr(vector_store, "collection", lambda: col)
    caches = [vector_store.menu_vectors, vector_store.menu_pca, vector_store.menu_hnsw]
    for c in caches:
        c.cache_clear()
    yield agent.Session(OrderStore(tmp_path / "p.sqlite3"))
    for c in caches:
        c.cache_clear()


class FakeClient:
    """Imita client.models.generate_content devolviendo respuestas guionadas."""

    def __init__(self, script):
        self.script = list(script)
        self.models = self

    def generate_content(self, model, contents, config):
        step = self.script.pop(0)
        if isinstance(step, str):
            content = types.Content(role="model", parts=[types.Part(text=step)])
            return SimpleNamespace(candidates=[SimpleNamespace(content=content)], function_calls=None, text=step)
        calls = [types.FunctionCall(name=n, args=a) for n, a in step]
        content = types.Content(role="model", parts=[types.Part(function_call=c) for c in calls])
        return SimpleNamespace(candidates=[SimpleNamespace(content=content)], function_calls=calls, text=None)


def test_buscar_genera_traza(session):
    out = session.run_tool("buscar_platos", {"consulta": "sopa de pollo"})
    assert len(out["resultados"]) == 5
    t = session.traces[0]
    assert t["dimensiones"] == 768 and abs(t["norma"] - 1) < 1e-4
    assert len(t["vector"]) == 768 and len(t["consulta_xy"]) == 2 and len(t["consulta_xyz"]) == 3
    assert t["norma_cruda"] > 0 and len(t["scores"]) == 41
    assert all(len(r["xy"]) == 2 for r in t["resultados"])
    # el top-1 de Chroma es el mayor de s = X·q
    assert t["resultados"][0]["id"] == max(t["scores"], key=t["scores"].get)
    assert t["hnsw"]["resultado"][0]["id"] == t["resultados"][0]["id"]


def test_agregar_id_invalido_devuelve_error(session):
    assert "error" in session.run_tool("agregar_al_carrito", {"id": "X99"})
    assert session.cart.total == 0


def test_confirmar_carrito_vacio(session):
    assert "error" in session.run_tool("confirmar_pedido", {"nombre_cliente": "Ana"})


def test_turno_completo_con_herramientas(session):
    client = FakeClient([
        [("buscar_platos", {"consulta": "ajiaco"})],
        [("agregar_al_carrito", {"id": "S01", "cantidad": 2})],
        "Listo, agregué 2 ajiacos 🍲",
    ])
    reply = session.chat("dos ajiacos porfa", client)
    assert reply.startswith("Listo")
    assert session.cart.items["S01"]["cantidad"] == 2
    assert session.traces and session.traces[0]["consulta"] == "ajiaco"

    client = FakeClient([[("confirmar_pedido", {"nombre_cliente": "Ana"})], "Pedido confirmado"])
    session.chat("confirmo, soy Ana", client)
    assert session.last_order["cliente"] == "Ana" and session.cart.total == 0


def test_error_de_gemini_no_deja_historial_a_medias(session):
    class Broken:
        models = None

        def __init__(self):
            self.models = self

        def generate_content(self, **kw):
            raise RuntimeError("sin red")

    with pytest.raises(RuntimeError):
        session.chat("hola", Broken())
    assert session.history == []
