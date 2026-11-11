from app import vector_store
from app.menu import load_menu


def top_ids(col, query, k=3, **filters):
    return [r["id"] for r in vector_store.search(query, k=k, col=col, **filters)["resultados"]]


def test_ajiaco_aparece_para_sopa_de_pollo_con_papa(col):
    assert "S01" in top_ids(col, "sopa de pollo con papa y mazorca")


def test_algo_dulce_devuelve_postres(col):
    res = vector_store.search("quiero algo dulce de postre", k=3, col=col)["resultados"]
    assert sum(r["plato"]["categoria"] == "postres" for r in res) >= 2


def test_pescado_del_caribe(col):
    assert "P05" in top_ids(col, "pescado frito de la costa", k=3)


def test_filtro_vegetariano(col):
    res = vector_store.search("algo para comer con carne", k=5, vegetariano=True, col=col)["resultados"]
    assert res and all(r["plato"]["vegetariano"] for r in res)


def test_filtro_precio_y_categoria(col):
    res = vector_store.search("bebida", k=5, categoria="bebidas", precio_max=7000, col=col)["resultados"]
    assert res and all(r["plato"]["precio"] <= 7000 and r["plato"]["categoria"] == "bebidas" for r in res)


def test_reindexar_no_duplica(col):
    assert vector_store.index_menu(col) == len(load_menu())


def test_resultados_ordenados_por_distancia(col):
    d = [r["distancia_chroma"] for r in vector_store.search("arepa", k=5, col=col)["resultados"]]
    assert d == sorted(d)


def test_nombres_propios_quedan_primeros(col):
    for query, dish_id in [("ajiaco", "S01"), ("bandeja paisa", "P01"), ("limonada de coco", "B02")]:
        assert top_ids(col, query, k=1) == [dish_id]
