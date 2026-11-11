import numpy as np

from app import vector_store
from app.vector_math import cosine_similarity, fit_pca, norm


def test_coseno_ejemplo_a_mano():
    a, b = np.array([1.0, 0.0]), np.array([1.0, 1.0])
    assert abs(cosine_similarity(a, b) - 1 / np.sqrt(2)) < 1e-9


def test_vector_consulta_es_unitario(col):
    assert abs(vector_store.search("ajiaco", col=col)["norma"] - 1) < 1e-4


def test_coseno_manual_coincide_con_chroma(col):
    for r in vector_store.search("algo calientito para el frío", k=5, col=col)["resultados"]:
        assert abs(r["coseno_manual"] - (1 - r["distancia_chroma"])) < 1e-4
        assert abs(r["coseno_manual"] - r["producto_punto"]) < 1e-4  # vectores unitarios


def test_pca():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 768))
    pca = fit_pca(X)
    assert pca.project(X).shape == (40, 2)
    assert all(0 < v < 1 for v in pca.explained_variance_ratio)
    # componentes principales ortonormales
    assert np.allclose(pca.components @ pca.components.T, np.eye(2), atol=1e-6)
    assert abs(norm(pca.components[0]) - 1) < 1e-6


def test_normalizacion_manual_igual_a_la_libreria():
    from app.embeddings import embed, embed_raw, get_model
    texts = ["algo calientito con pollo", "ajiaco"]
    raw = embed_raw(texts)
    assert (np.linalg.norm(raw, axis=1) > 2).all()  # antes de normalizar NO mide 1
    ref = get_model().encode(["query: " + t for t in texts], normalize_embeddings=True)
    assert np.abs(embed(texts) - ref).max() < 1e-5
