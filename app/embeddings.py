"""Convierte texto en vectores (embeddings) con un sentence-transformer.

Cada texto se transforma en un vector v de R^768. Los vectores se normalizan
(v / ||v||) para que todos tengan norma 1: así la similitud coseno entre dos
vectores es simplemente su producto punto.

El modelo E5 está entrenado para búsqueda: la consulta del cliente lleva el
prefijo "query: " y la descripción de cada plato el prefijo "passage: ".
"""
from functools import lru_cache

import numpy as np

from app import config


@lru_cache(maxsize=1)
def get_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(config.EMBEDDING_MODEL)


def _prefix(kind: str) -> str:
    return f"{kind}: " if "e5" in config.EMBEDDING_MODEL.lower() else ""


def embed(texts: list[str], kind: str = "query") -> np.ndarray:
    """Devuelve una matriz (len(texts), dim) con un vector unitario por fila.

    kind = "query" para consultas del cliente, "passage" para documentos del menú.
    """
    raw = embed_raw(texts, kind)
    return raw / np.linalg.norm(raw, axis=1, keepdims=True)  # v / ||v||


def embed_raw(texts: list[str], kind: str = "query") -> np.ndarray:
    """Vectores SIN normalizar: la salida del modelo antes de su capa final `Normalize`.

    Así se puede mostrar ||q|| antes de dividir entre la norma.
    """
    import torch

    model = get_model()
    layers = [m for m in model if type(m).__name__ != "Normalize"]
    with torch.no_grad():
        prepare = getattr(model, "preprocess", None) or model.tokenize
        features = prepare([_prefix(kind) + t for t in texts])
        features = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in features.items()}
        for layer in layers:
            features = layer(features)
    return features["sentence_embedding"].cpu().numpy().astype(np.float32)
