import numpy as np

from app.embeddings import embed
from app.hnsw_sim import HNSW
from app.menu import dish_to_document, load_menu


def build():
    menu = list(load_menu().values())
    X = embed([dish_to_document(d) for d in menu], kind="passage")
    return HNSW(X, [d["id"] for d in menu]), X, [d["id"] for d in menu]


def test_grafo_por_capas():
    h, X, ids = build()
    g = h.to_dict()
    assert len(g["capas"][0]["nodos"]) == len(ids)  # la capa 0 tiene todos los nodos
    sizes = [len(c["nodos"]) for c in g["capas"]]
    assert sizes == sorted(sizes, reverse=True) and len(sizes) >= 2  # capas superiores más chicas
    assert g["entrada"] in g["capas"][-1]["nodos"]


def test_hnsw_coincide_con_busqueda_exacta():
    h, X, ids = build()
    for query in ["ajiaco", "algo del caribe con pescado", "un postre con arequipe", "jugo refrescante"]:
        q = embed([query])[0]
        exact = [ids[i] for i in np.argsort(-(X @ q))[:3]]
        res = h.search(q, k=5)
        assert res["resultado"][0]["id"] == exact[0]
        assert len(res["capas"]) == h.top + 1 and res["capas"][-1]["capa"] == 0
        assert res["calculos_distancia"] > 0
