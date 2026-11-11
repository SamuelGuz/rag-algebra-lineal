"""Base de datos vectorial (ChromaDB): indexación del menú y búsqueda semántica.

Cada plato se guarda como (id, vector, documento, metadata). Al buscar, la
consulta se convierte en vector q y Chroma devuelve los k platos cuyo vector
está más cerca de q usando la distancia coseno: d(q, v) = 1 - cos(q, v).
"""
from functools import lru_cache

import chromadb
import numpy as np
from chromadb.config import Settings

from app import config
from app.embeddings import embed, embed_raw
from app.menu import dish_to_document, load_menu
from app.hnsw_sim import HNSW
from app.vector_math import PCA, cosine_similarity, fit_pca, norm


def get_collection(path=None):
    client = chromadb.PersistentClient(
        path=str(path or config.CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )
    metadata = {"hnsw:space": "cosine", "modelo": config.EMBEDDING_MODEL}  # métrica: distancia coseno
    col = client.get_or_create_collection(name=config.COLLECTION_NAME, metadata=metadata)
    if (col.metadata or {}).get("modelo") != config.EMBEDDING_MODEL:
        # Vectores de otro modelo (otra dimensión): se reconstruye la colección
        client.delete_collection(config.COLLECTION_NAME)
        col = client.create_collection(name=config.COLLECTION_NAME, metadata=metadata)
    return col


@lru_cache(maxsize=1)
def collection():
    return get_collection()


def index_menu(col=None) -> int:
    """Vectoriza todos los platos y los guarda en Chroma. Es idempotente (upsert)."""
    col = col or collection()
    menu = list(load_menu().values())
    ids = [d["id"] for d in menu]
    documents = [dish_to_document(d) for d in menu]
    vectors = embed(documents, kind="passage")
    col.upsert(
        ids=ids,
        embeddings=vectors.tolist(),
        documents=documents,
        metadatas=[
            {"nombre": d["nombre"], "categoria": d["categoria"],
             "vegetariano": d["vegetariano"], "precio": d["precio"]}
            for d in menu
        ],
    )
    # Elimina platos que ya no están en el menú
    stale = set(col.get(include=[])["ids"]) - set(ids)
    if stale:
        col.delete(ids=list(stale))
    return col.count()


def build_where(categoria=None, vegetariano=None, precio_max=None) -> dict | None:
    """Filtros por metadata (se aplican además de la similitud semántica)."""
    conditions = []
    if categoria:
        conditions.append({"categoria": categoria})
    if vegetariano is not None:
        conditions.append({"vegetariano": bool(vegetariano)})
    if precio_max:
        conditions.append({"precio": {"$lte": int(precio_max)}})
    if not conditions:
        return None
    return conditions[0] if len(conditions) == 1 else {"$and": conditions}


def search(query: str, k: int = 5, categoria=None, vegetariano=None, precio_max=None, col=None) -> dict:
    """Búsqueda semántica. Devuelve los platos y todos los datos matemáticos del proceso."""
    col = col or collection()
    q_raw = embed_raw([query], kind="query")[0]
    q = q_raw / norm(q_raw)  # normalización: q / ||q||
    where = build_where(categoria, vegetariano, precio_max)
    res = col.query(
        query_embeddings=[q.tolist()],
        n_results=k,
        where=where,
        include=["metadatas", "distances", "embeddings"],
    )
    menu = load_menu()
    results = []
    for dish_id, dist, vec in zip(res["ids"][0], res["distances"][0], res["embeddings"][0]):
        v = np.asarray(vec, dtype=np.float32)
        results.append({
            "id": dish_id,
            "plato": menu[dish_id],
            "distancia_chroma": float(dist),
            "coseno_manual": cosine_similarity(q, v),  # verificación: ≈ 1 - distancia
            "producto_punto": float(np.dot(q, v)),
        })
    candidatos = [d["id"] for d in menu.values() if matches_filters(d, categoria, vegetariano, precio_max)]
    return {"consulta": query, "filtros": where, "vector": q, "norma": norm(q), "norma_cruda": norm(q_raw),
            "candidatos": candidatos, "resultados": results}


def matches_filters(dish: dict, categoria=None, vegetariano=None, precio_max=None) -> bool:
    """Lo mismo que el filtro `where` de Chroma, en Python (para dibujar qué filas se descartan)."""
    return ((not categoria or dish["categoria"] == categoria)
            and (vegetariano is None or dish["vegetariano"] == bool(vegetariano))
            and (not precio_max or dish["precio"] <= int(precio_max)))


@lru_cache(maxsize=1)
def menu_vectors() -> tuple[list[str], np.ndarray]:
    """Lee de Chroma los vectores guardados de todos los platos."""
    data = collection().get(include=["embeddings"])
    return data["ids"], np.asarray(data["embeddings"], dtype=np.float32)


@lru_cache(maxsize=1)
def menu_pca() -> PCA:
    return fit_pca(menu_vectors()[1], k=3)  # 3 componentes: mapa 3D (y las 2 primeras para 2D)


@lru_cache(maxsize=1)
def menu_hnsw() -> HNSW:
    ids, X = menu_vectors()
    return HNSW(X, ids)
