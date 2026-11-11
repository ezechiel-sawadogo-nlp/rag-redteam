"""Choisir le seuil de la défense « length » sur les données (sans LLM).

Compare la longueur (mots d'historique) des notices piégées à celle des vrais passages, et
calcule pour plusieurs seuils : la part des attaques bloquées, la part des vrais passages
écartés, la part des bonnes notices du jeu de test écartées (coût direct sur les réponses).

    python src\\rag_redteam\\scan_length.py
"""
from __future__ import annotations

import json
import sys

import numpy as np

from defenses import body_words
from run_injection import make_poison
from target import MERIMEE_ROOT, load_rag

THRESHOLDS = [10, 20, 30, 40, 50, 60, 80, 100]

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

rag = load_rag("hybrid")
real = np.array([body_words(c.text) for c in rag.chunks.values()])

testset = [json.loads(line) for line in (MERIMEE_ROOT / "eval" / "testset.jsonl")
           .read_text(encoding="utf-8").splitlines() if line.strip()]
pool = [q for q in testset if q.get("type") == "synthetic" and not q.get("ambiguous")]
gold = np.array([body_words(rag.chunks[f"{q['gold_refs'][0]}#0"].text) for q in pool])

poisons: dict[str, np.ndarray] = {}
for t in ("A1c", "A1b", "A2"):
    lens = [body_words(p.text) for q in pool if (p := make_poison(rag, q, t)[0]) is not None]
    poisons[t] = np.array(lens)


def describe(name: str, a: np.ndarray) -> None:
    p5, p25, p50 = np.percentile(a, [5, 25, 50])
    print(f"  {name:38s} n={len(a):6d}   min {a.min():4d}   5 % {p5:5.0f}   25 % {p25:5.0f}   médiane {p50:5.0f}")


print("Longueur de l'historique (mots)\n")
describe("vrais passages (toute la base)", real)
describe("bonnes notices du jeu de test (1er passage)", gold)
for t, a in poisons.items():
    describe(f"notices piégées {t}", a)

print(f"\nSi on écarte les extraits de moins de N mots :\n")
print(f"  {'N':>4s}   {'A1c bloquées':>13s}   {'A2 bloquées':>12s}   {'vrais passages écartés':>23s}   {'bonnes notices écartées':>24s}")
for n in THRESHOLDS:
    print(f"  {n:4d}   {np.mean(poisons['A1c'] < n):13.0%}   {np.mean(poisons['A2'] < n):12.0%}"
          f"   {np.mean(real < n):23.1%}   {np.mean(gold < n):24.1%}")
