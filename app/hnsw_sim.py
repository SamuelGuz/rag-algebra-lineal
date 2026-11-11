"""Simulación educativa de HNSW (Hierarchical Navigable Small World).

ChromaDB usa HNSW internamente para no comparar la consulta contra TODOS los
vectores. Aquí construimos un HNSW pequeño con los mismos vectores del menú
para poder dibujarlo y registrar paso a paso cómo navega la búsqueda.

Idea:
- Cada vector es un nodo. Cada nodo recibe un nivel aleatorio
  L = floor(-ln(U) * mL): casi todos quedan solo en la capa 0 y pocos suben.
- En cada capa los nodos se conectan con sus vecinos más cercanos.
- Buscar = entrar por la capa más alta (pocos nodos, saltos largos) y avanzar
  "codiciosamente" hacia el vecino más parecido a q; al no poder mejorar, se
  baja a la capa siguiente. En la capa 0 se hace una búsqueda con lista de
  candidatos (ef) para obtener los k mejores.
"""
import heapq
import math
import random

import numpy as np


class HNSW:
    def __init__(self, vectors: np.ndarray, ids: list[str], M: int = 4, ef_construction: int = 12,
                 max_level: int = 3, seed: int = 7):
        self.X = vectors
        self.ids = ids
        self.M = M
        self.M0 = 2 * M  # la capa 0 admite más vecinos
        self.mL = 1 / math.log(M)
        self.levels: list[int] = []
        self.graph: list[dict[int, set[int]]] = [dict() for _ in range(max_level + 1)]
        self.entry: int | None = None
        rng = random.Random(seed)
        for i in range(len(ids)):
            level = min(max_level, int(-math.log(1 - rng.random()) * self.mL))
            self._insert(i, level, ef_construction)
        self.top = max(self.levels)
        self.graph = self.graph[: self.top + 1]

    # distancia coseno entre vectores unitarios: 1 - q·v
    def _dist(self, q: np.ndarray, i: int) -> float:
        return float(1 - np.dot(q, self.X[i]))

    def _search_layer(self, q, entry_points: list[int], ef: int, layer: int, log=None):
        """Búsqueda en una capa con lista de candidatos de tamaño ef."""
        counter = {"n": 0}

        def dist(i):
            counter["n"] += 1
            return self._dist(q, i)

        visited = set(entry_points)
        candidates = [(dist(e), e) for e in entry_points]
        heapq.heapify(candidates)
        best = [(-d, e) for d, e in candidates]  # max-heap de los ef mejores
        heapq.heapify(best)
        while candidates:
            d, c = heapq.heappop(candidates)
            if d > -best[0][0] and len(best) >= ef:
                break
            evaluated = []
            for nb in self.graph[layer].get(c, ()):
                if nb in visited:
                    continue
                visited.add(nb)
                dn = dist(nb)
                evaluated.append((nb, dn))
                if len(best) < ef or dn < -best[0][0]:
                    heapq.heappush(candidates, (dn, nb))
                    heapq.heappush(best, (-dn, nb))
                    if len(best) > ef:
                        heapq.heappop(best)
            if log is not None:
                log.append({"nodo": c, "dist": d, "evaluados": evaluated})
        result = sorted((-d, e) for d, e in best)
        return result, counter["n"]

    def _insert(self, i: int, level: int, ef: int):
        self.levels.append(level)
        for layer in range(level + 1):
            self.graph[layer][i] = set()
        if self.entry is None:
            self.entry = i
            return
        q = self.X[i]
        ep = [self.entry]
        top = self.levels[self.entry]
        for layer in range(top, level, -1):
            ep = [self._search_layer(q, ep, 1, layer)[0][0][1]]
        for layer in range(min(level, top), -1, -1):
            found, _ = self._search_layer(q, ep, ef, layer)
            neighbors = [e for _, e in found[: self.M]]
            cap = self.M0 if layer == 0 else self.M
            for nb in neighbors:
                self.graph[layer][i].add(nb)
                self.graph[layer][nb].add(i)
                if len(self.graph[layer][nb]) > cap:  # poda: conserva los más cercanos
                    keep = sorted(self.graph[layer][nb], key=lambda x: self._dist(self.X[nb], x))[:cap]
                    for dropped in self.graph[layer][nb] - set(keep):
                        self.graph[layer][dropped].discard(nb)
                    self.graph[layer][nb] = set(keep)
            ep = [e for _, e in found]
        if level > top:
            self.entry = i

    def search(self, q: np.ndarray, k: int = 5, ef: int = 5) -> dict:
        """Busca y devuelve la traza paso a paso (para animarla)."""
        layers = []
        total = 0
        ep = [self.entry]
        for layer in range(self.top, 0, -1):
            log = []
            found, n = self._search_layer(q, ep, 1, layer, log)
            total += n
            ep = [found[0][1]]
            layers.append({"capa": layer, "pasos": self._steps(log), "sale": self.ids[ep[0]]})
        log = []
        found, n = self._search_layer(q, ep, max(ef, k), 0, log)
        total += n
        layers.append({"capa": 0, "pasos": self._steps(log), "sale": self.ids[found[0][1]]})
        return {
            "entrada": self.ids[self.entry],
            "capas": layers,
            "resultado": [{"id": self.ids[e], "similitud": round(1 - d, 6)} for d, e in found[:k]],
            "calculos_distancia": total,
            "total_vectores": len(self.ids),
        }

    def _steps(self, log):
        return [
            {"nodo": self.ids[s["nodo"]], "similitud": round(1 - s["dist"], 6),
             "evaluados": [[self.ids[e], round(1 - d, 6)] for e, d in s["evaluados"]]}
            for s in log
        ]

    def to_dict(self) -> dict:
        return {
            "entrada": self.ids[self.entry],
            "niveles": {self.ids[i]: lvl for i, lvl in enumerate(self.levels)},
            "capas": [
                {"capa": layer, "nodos": [self.ids[i] for i in sorted(adj)],
                 "aristas": sorted({tuple(sorted((self.ids[a], self.ids[b]))) for a, nbs in adj.items() for b in nbs})}
                for layer, adj in enumerate(self.graph)
            ],
            "M": self.M,
        }
