"""
Lexical translation: given a human concept (text), retrieve the closest
call in the database by semantic embedding similarity.

Usage:
  python paper_code/translate.py "danger"
  python paper_code/translate.py --top 5 "I want to be left alone"

Also generates paper_code/translation_examples.json with the curated
examples used in Table 3 of the paper.
"""

import argparse
import json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent
try:
    from paper_code.paths import DATABASE_PATH
except ModuleNotFoundError:
    from paths import DATABASE_PATH
DB_PATH = DATABASE_PATH
EMB_PATH = Path(__file__).parent / "embeddings.npz"


def load_calls():
    with open(DB_PATH) as f:
        db = json.load(f)
    records = []
    for species in db["species"]:
        for call in species.get("calls", []):
            records.append({
                "species": species["species_name"],
                "call_name": call.get("call_name", ""),
                "acoustic_description": call.get("acoustic_description", ""),
                "semantic_description": call.get("semantic_description", ""),
                "keywords": call.get("ontology_keywords", []),
            })
    return records


def get_semantic_emb(text, model):
    return model.encode([text], normalize_embeddings=True)[0]


def translate(query: str, records, semantic_emb_matrix, model, top_k=5):
    query_emb = get_semantic_emb(query, model)
    sims = semantic_emb_matrix @ query_emb
    top_idx = np.argsort(sims)[::-1][:top_k]
    results = []
    for i in top_idx:
        results.append({
            "rank": len(results) + 1,
            "similarity": float(sims[i]),
            "species": records[i]["species"],
            "call_name": records[i]["call_name"],
            "semantic_description": records[i]["semantic_description"],
            "acoustic_description": records[i]["acoustic_description"],
            "keywords": records[i]["keywords"],
        })
    return results


PAPER_EXAMPLES = [
    "danger",
    "I am hungry",
    "hello / greeting",
    "come here",
    "I am here",
    "stay away",
    "I am afraid",
    "let's move together",
    "this is my territory",
    "help me",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs="?", default=None, help="Human concept to translate")
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--paper-examples", action="store_true",
                        help="Generate translation table for the paper")
    args = parser.parse_args()

    records = load_calls()

    if EMB_PATH.exists():
        data = np.load(EMB_PATH, allow_pickle=True)
        semantic_emb = data["semantic"]
    else:
        print(f"No embeddings found at {EMB_PATH}. Run embed_and_analyze.py first.")
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        semantic_texts = [r["semantic_description"] for r in records]
        semantic_emb = model.encode(semantic_texts, normalize_embeddings=True, show_progress_bar=True)

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")

    if args.paper_examples:
        out = []
        for concept in PAPER_EXAMPLES:
            top1 = translate(concept, records, semantic_emb, model, top_k=1)[0]
            out.append({"concept": concept, **top1})
        out_path = Path(__file__).parent / "translation_examples.json"
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Saved {out_path}")
        for row in out:
            print(f"\n  {row['concept']!r:30s} → {row['species']} / {row['call_name']}")
            print(f"  {'':30s}   sim={row['similarity']:.3f} | {row['semantic_description'][:80]}")
        return

    query = args.query
    if query is None:
        parser.print_help()
        return

    results = translate(query, records, semantic_emb, model, top_k=args.top)
    print(f"\nQuery: {query!r}\n")
    for r in results:
        print(f"  #{r['rank']} (sim={r['similarity']:.3f})  {r['species']} — {r['call_name']}")
        print(f"       Semantics: {r['semantic_description'][:100]}")
        print(f"       Acoustics: {r['acoustic_description'][:100]}")
        print()


if __name__ == "__main__":
    main()
