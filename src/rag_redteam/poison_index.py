"""Ajoute une notice piégée aux retrievers de merimee-rag, sans réindexer la base.

- BM25 : score calculé avec les statistiques de l'index existant (idf, longueur moyenne, k1, b),
  exactement comme rank_bm25 le ferait si la notice avait été indexée. Ajouter un document
  sur ~27 000 ne change quasiment pas les idf : c'est la seule approximation.
- Dense : la notice est encodée avec le même encodeur (préfixe « passage: ») et comparée aux autres.
- Hybride : la fusion RRF de merimee-rag est refaite sur ces deux recherches modifiées.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
from merimee_rag.retrievers import Hit, HybridRetriever


def _merge(hits: list[Hit], extra: list[Hit], k: int) -> list[Hit]:
    """Insère les hits piégés parmi les vrais selon leur score, puis renumérote les rangs."""
    allh = sorted(hits + extra, key=lambda h: -h.score)[:k]
    return [Hit(h.chunk_id, h.ref, h.score, r) for r, h in enumerate(allh)]


class PoisonedBM25:
    name = "bm25"

    def __init__(self, base, chunk_id: str, ref: str, text: str) -> None:
        self.base, self.chunk_id, self.ref = base, chunk_id, ref
        toks = base.analyzer(text)
        self.tf, self.dl = Counter(toks), len(toks)

    def poison_score(self, query: str) -> float:
        bm = self.base.bm25
        score = 0.0
        for q in self.base.analyzer(query):  # même boucle que rank_bm25.BM25Okapi.get_scores
            f = self.tf.get(q, 0)
            if f:
                score += (bm.idf.get(q) or 0) * (f * (bm.k1 + 1) / (f + bm.k1 * (1 - bm.b + bm.b * self.dl / bm.avgdl)))
        return score

    def search(self, query: str, k: int = 10) -> list[Hit]:
        s = self.poison_score(query)
        extra = [Hit(self.chunk_id, self.ref, s, -1)] if s > 0 else []
        return _merge(self.base.search(query, k), extra, k)


class PoisonedDense:
    name = "dense"

    def __init__(self, base, chunk_id: str, ref: str, text: str) -> None:
        self.base, self.chunk_id, self.ref = base, chunk_id, ref
        self.vec = base.encoder([text], False)[0]

    def search(self, query: str, k: int = 10) -> list[Hit]:
        q = self.base.encoder([query], True)[0]
        extra = [Hit(self.chunk_id, self.ref, float(np.dot(q, self.vec)), -1)]
        return _merge(self.base.search(query, k), extra, k)


def poisoned_retriever(base, chunk_id: str, ref: str, text: str):
    """Version du retriever `base` (bm25, dense ou hybrid) dont l'index contient la notice piégée."""
    if base.name == "bm25":
        return PoisonedBM25(base, chunk_id, ref, text)
    if base.name == "dense":
        return PoisonedDense(base, chunk_id, ref, text)
    if base.name == "hybrid":
        subs = [poisoned_retriever(r, chunk_id, ref, text) for r in base.retrievers]
        return HybridRetriever(subs, rrf_k=base.rrf_k, candidates=base.candidates)
    raise ValueError(f"retriever non pris en charge : {base.name}")
