"""Álgebra lineal hecha "a mano" con numpy, para verificar y visualizar lo que hace ChromaDB.

- cosine_similarity: cos(θ) = (a · b) / (||a|| ||b||)
- PCA: proyecta los vectores de R^768 a R^k (k = 2 o 3) usando la descomposición
  SVD de la matriz centrada X - μ = U Σ Vᵀ. Las k primeras filas de Vᵀ son las
  direcciones de mayor varianza (componentes principales).
"""
from dataclasses import dataclass

import numpy as np


def norm(v: np.ndarray) -> float:
    return float(np.sqrt(np.dot(v, v)))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (norm(a) * norm(b)))


@dataclass
class PCA:
    mean: np.ndarray            # μ, vector promedio (768,)
    components: np.ndarray      # (k, 768): las k direcciones principales
    explained_variance_ratio: np.ndarray  # fracción de la varianza total en cada componente

    def project(self, vectors: np.ndarray) -> np.ndarray:
        """Coordenadas en k dimensiones: (v - μ) · componentesᵀ."""
        return (np.atleast_2d(vectors) - self.mean) @ self.components.T


def fit_pca(X: np.ndarray, k: int = 2) -> PCA:
    mean = X.mean(axis=0)
    centered = X - mean
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    variance = singular_values**2
    return PCA(
        mean=mean,
        components=vt[:k],
        explained_variance_ratio=variance[:k] / variance.sum(),
    )
