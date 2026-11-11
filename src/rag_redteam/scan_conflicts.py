"""Coût de la défense « conflict » sans attaque : combien de vrais extraits écarte-t-elle ?

Pour chaque question du jeu de test de merimee-rag, on récupère les 5 extraits (sans LLM)
et on regarde si la défense en écarterait, en particulier la bonne notice.

    python src\\rag_redteam\\scan_conflicts.py
"""
from __future__ import annotations

import json
import sys

from defenses import find_conflicts
from target import MERIMEE_ROOT, load_rag

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

questions = [json.loads(line) for line in (MERIMEE_ROOT / "eval" / "testset.jsonl")
             .read_text(encoding="utf-8").splitlines() if line.strip()]
rag = load_rag("hybrid")

n = n_alert = n_gold_lost = 0
examples = []
for q in questions:
    ctxs = rag.retrieve(q["question"])
    bad = find_conflicts(ctxs)
    n += 1
    if not bad:
        continue
    n_alert += 1
    gold = set(q.get("gold_refs", []))
    lost = gold & {ctxs[i].ref for i in bad}
    n_gold_lost += bool(lost)
    if len(examples) < 6:
        examples.append(f"  {q['id']}  écartés : " + " | ".join(f"{ctxs[i].ref} {ctxs[i].title}" for i in sorted(bad))
                        + ("   ← BONNE NOTICE ÉCARTÉE" if lost else ""))

print(f"Questions où « conflict » écarte des extraits sans attaque : {n_alert}/{n} = {n_alert / n:.1%}")
print(f"  dont la bonne notice écartée : {n_gold_lost}/{n} = {n_gold_lost / n:.1%}")
print("\nExemples :")
print("\n".join(examples) or "  (aucun)")
