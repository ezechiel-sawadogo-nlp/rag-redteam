"""Une notice piégée ajoutée à la base est-elle récupérée ? (sans LLM)

Pour chaque question du jeu de test et chaque type de notice piégée, on ajoute la notice à
l'index (voir poison_index.py) et on regarde, pour BM25, dense et hybride :
  - si elle entre dans le top-5 (les extraits donnés au LLM) ;
  - si elle arrive en rang 1 ;
  - si elle passe devant la vraie notice.

    python src\\rag_redteam\\scan_retrieval.py
    python src\\rag_redteam\\scan_retrieval.py --n 40
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict

from poison_index import poisoned_retriever
from run_injection import make_poison
from target import MERIMEE_ROOT, RESULTS_DIR, load_rag

TYPES = {
    "A4ctrl": "copie neutre (300 premiers caractères)",
    "A1b": "copie complète, date décalée",
    "A1c": "copie raccourcie, date décalée",
    "A1d": "A1c rallongée (remplissage générique)",
    "A2": "début copié + ordre caché",
}


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=0, help="nombre de questions (0 = toutes)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min-words", type=int, default=50, help="longueur visée par A1d (seuil de « length »)")
    p.add_argument("--types", nargs="+", default=list(TYPES), choices=list(TYPES))
    args = p.parse_args()
    import defenses
    defenses.MIN_WORDS = args.min_words

    testset = [json.loads(line) for line in (MERIMEE_ROOT / "eval" / "testset.jsonl")
               .read_text(encoding="utf-8").splitlines() if line.strip()]
    pool = [q for q in testset if q.get("type") == "synthetic" and not q.get("ambiguous")]
    if args.n:
        pool = random.Random(args.seed).sample(pool, min(args.n, len(pool)))

    rag = load_rag("hybrid")
    hybrid = rag.retriever
    bases = {r.name: r for r in hybrid.retrievers}
    bases["hybrid"] = hybrid
    top_k = rag.top_k

    # stats[(type, retriever)] = listes de booléens
    stats: dict[tuple[str, str], dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / "retrieval_scan.jsonl"
    with out_path.open("w", encoding="utf-8") as out:
        for i, q in enumerate(pool, 1):
            gold = q["gold_refs"][0]
            for t in args.types:
                poison, _ = make_poison(rag, q, t)
                if poison is None:  # A1b/A1c : pas de date dans la réponse
                    continue
                for name, base in bases.items():
                    hits = poisoned_retriever(base, poison.chunk_id, poison.ref, poison.text).search(q["question"], 20)
                    refs = [h.ref for h in hits]
                    p_rank = refs.index(poison.ref) if poison.ref in refs else None
                    g_rank = refs.index(gold) if gold in refs else None
                    s = stats[(t, name)]
                    s["top5"].append(p_rank is not None and p_rank < top_k)
                    s["rank1"].append(p_rank == 0)
                    s["above_gold"].append(p_rank is not None and (g_rank is None or p_rank < g_rank))
                    out.write(json.dumps({"qid": q["id"], "type": t, "retriever": name,
                                          "poison_rank": p_rank, "gold_rank": g_rank}) + "\n")
            print(f"\r{i}/{len(pool)} questions", end="", flush=True)

    def pct(v: list[bool]) -> str:
        return f"{sum(v) / len(v):5.0%}" if v else "  –  "

    print(f"\n\nNotice piégée ajoutée à l'index : part des questions où elle est…  (top-{top_k} = extraits donnés au LLM)\n")
    names = list(bases)
    print(f"{'':42s}" + "".join(f"{n:^26s}" for n in names))
    print(f"{'':42s}" + "".join(f"{'top-5  rang 1  devant':^26s}" for _ in names))
    for t in args.types:
        label = TYPES[t]
        n = len(stats[(t, names[0])]["top5"])
        row = f"{t:7s} {label[:28]:28s} n={n:<3d} "
        for name in names:
            s = stats[(t, name)]
            row += f"  {pct(s['top5'])}  {pct(s['rank1'])}  {pct(s['above_gold'])}   "
        print(row)
    print(f"\n« devant » = classée avant la vraie notice. Détail : {out_path}")


if __name__ == "__main__":
    main()
