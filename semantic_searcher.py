#!/usr/bin/env python3
"""
Search NeurIPS 2025 San Diego papers from a JSONL file in two modes:

1) Keyword mode  (--mode keyword)
   - Exact (case-insensitive) keyword matching over title + abstract
   - Keywords = whitespace- or comma-separated terms

2) Semantic mode (--mode semantic)
   - SentenceTransformers embeddings (default: all-mpnet-base-v2)
   - Cosine similarity ranking (optionally via FAISS)

Prereq: run the scraper first, e.g.:
    python scrape_neurips_2025.py --output neurips_2025_sandiego.jsonl

Examples:
    # Keyword mode (exact keyword match on title+abstract)
    python semantic_search_neurips.py neurips_2025_sandiego.jsonl \
        "cybersecurity adversarial RL" --mode keyword --top-k 30

    # Semantic mode (embedding-based search)
    python semantic_search_neurips.py neurips_2025_sandiego.jsonl \
        "cybersecurity adversarial reinforcement learning" \
        --mode semantic --top-k 30 --use-faiss
"""

import argparse
import json
import os
import sys
from typing import List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer  # type: ignore

# Try FAISS optionally
try:
    import faiss  # type: ignore
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False


# ---------------------------------------------------------------------
# Common helpers
# ---------------------------------------------------------------------

def load_papers(jsonl_path: str) -> List[dict]:
    """Load papers from JSONL file (one JSON object per line)."""
    papers: List[dict] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            papers.append(json.loads(line))
    return papers


def build_texts(papers: List[dict]) -> List[str]:
    """Build a representative text per paper (title + abstract)."""
    texts: List[str] = []
    for p in papers:
        title = p.get("title", "") or ""
        abstract = p.get("abstract", "") or ""
        if abstract:
            text = f"{title}. {abstract}"
        else:
            text = title
        texts.append(text)
    return texts


def normalize_embeddings(emb: np.ndarray) -> np.ndarray:
    """L2-normalize embeddings so dot product = cosine similarity."""
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return emb / norms


def get_embedding_cache_path(jsonl_path: str) -> str:
    base = os.path.splitext(jsonl_path)[0]
    return base + "_embeddings.npz"


# ---------------------------------------------------------------------
# Semantic mode
# ---------------------------------------------------------------------

def build_or_load_embeddings(
    jsonl_path: str,
    model_name: str = "sentence-transformers/all-mpnet-base-v2",
) -> Tuple[np.ndarray, List[dict]]:
    """
    Load papers and embeddings from cache if present; otherwise compute and cache.
    """
    cache_path = get_embedding_cache_path(jsonl_path)

    papers = load_papers(jsonl_path)
    print(f"[info] Loaded {len(papers)} papers from {jsonl_path}", file=sys.stderr)

    if os.path.exists(cache_path):
        print(f"[info] Loading cached embeddings from {cache_path}", file=sys.stderr)
        data = np.load(cache_path)
        embeddings = data["embeddings"]
        if embeddings.shape[0] != len(papers):
            print(
                "[warn] Cached embeddings count does not match number of papers; "
                "recomputing.",
                file=sys.stderr,
            )
        else:
            return embeddings, papers

    print(f"[info] Computing embeddings with {model_name}", file=sys.stderr)
    model = SentenceTransformer(model_name)
    texts = build_texts(papers)
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    embeddings = normalize_embeddings(embeddings)
    np.savez(cache_path, embeddings=embeddings)
    print(f"[info] Saved embeddings to {cache_path}", file=sys.stderr)

    return embeddings, papers


def semantic_search_numpy(
    embeddings: np.ndarray,
    query_vec: np.ndarray,
    top_k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Brute-force cosine similarity search via NumPy (fine for ~6k docs)."""
    sims = embeddings @ query_vec  # shape (N,)
    top_k = min(top_k, embeddings.shape[0])
    idx = np.argpartition(-sims, top_k - 1)[:top_k]
    idx = idx[np.argsort(-sims[idx])]
    return idx, sims[idx]


def semantic_search_faiss(
    embeddings: np.ndarray,
    query_vec: np.ndarray,
    top_k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """FAISS-based cosine-sim (inner product) search."""
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings.astype(np.float32))

    query_vec = query_vec.astype(np.float32)[None, :]  # shape (1, d)
    top_k = min(top_k, embeddings.shape[0])
    sims, idx = index.search(query_vec, top_k)
    return idx[0], sims[0]


# ---------------------------------------------------------------------
# Keyword mode
# ---------------------------------------------------------------------

def parse_keywords(query: str) -> List[str]:
    """
    Split query string into keywords.

    - If query contains commas, treat them as separators: "cyber, rl, adversarial"
    - Else split on whitespace: "cyber rl adversarial"
    """
    if "," in query:
        kws = [q.strip() for q in query.split(",")]
    else:
        kws = query.split()

    # Drop empties
    return [k for k in kws if k]


def keyword_search(
    papers: List[dict],
    query: str,
    top_k: int,
    match_all: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple keyword search over title + abstract (case-insensitive).

    - Keywords are exact substrings (no stemming/lemmatization).
    - match_all = require all keywords to appear, otherwise any.
    - Returns indices and a simple 'score' = number of matched keywords.
    """
    keywords = parse_keywords(query)
    if not keywords:
        # No keywords -> return nothing
        return np.array([], dtype=int), np.array([], dtype=float)

    keywords_lower = [k.lower() for k in keywords]

    scores = []
    indices = []

    for i, p in enumerate(papers):
        text = (p.get("title", "") or "") + " " + (p.get("abstract", "") or "")
        text_lower = text.lower()

        hits = [kw in text_lower for kw in keywords_lower]
        if match_all and not all(hits):
            continue
        if not match_all and not any(hits):
            continue

        score = sum(hits)  # simple: number of matched keywords
        indices.append(i)
        scores.append(float(score))

    if not indices:
        return np.array([], dtype=int), np.array([], dtype=float)

    indices = np.array(indices, dtype=int)
    scores = np.array(scores, dtype=float)

    # Sort by score descending, then by title lexicographically (stable-ish)
    sort_order = np.lexsort(
        (np.array([papers[i]["title"] for i in indices], dtype="object"), -scores)
    )
    indices = indices[sort_order]
    scores = scores[sort_order]

    if top_k is not None:
        top_k = min(top_k, len(indices))
        indices = indices[:top_k]
        scores = scores[:top_k]

    return indices, scores


# ---------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------

def pretty_print_results(
    indices: np.ndarray,
    scores: np.ndarray,
    papers: List[dict],
    query: str,
    mode: str,
):
    print()
    print("=" * 80)
    print(f"Top {len(indices)} results for query ({mode}): {query!r}")
    print("=" * 80)
    print()

    for rank, (i, s) in enumerate(zip(indices, scores), start=1):
        p = papers[int(i)]
        title = p.get("title", "<no title>")
        url = p.get("url", "")
        authors = p.get("authors", "")
        kind = p.get("kind", "")
        abstract = (p.get("abstract") or "").replace("\n", " ")

        if len(abstract) > 500:
            abstract = abstract[:500].rstrip() + "..."

        print(f"[{rank}]  score={s:.4f}")
        print(f"Title   : {title}")
        if kind:
            print(f"Kind    : {kind}")
        if authors:
            print(f"Authors : {authors}")
        print(f"URL     : {url}")
        if abstract:
            print(f"Abstract: {abstract}")
        print("-" * 80)


# ---------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Search NeurIPS 2025 San Diego papers (keyword or semantic)."
    )
    ap.add_argument(
        "jsonl_path",
        help="Path to JSONL file produced by scrape_neurips_2025.py",
    )
    ap.add_argument(
        "query",
        help="Search query (keywords or natural language).",
    )
    ap.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of results to show (default: 20).",
    )
    ap.add_argument(
        "--mode",
        choices=["keyword", "semantic"],
        default="semantic",
        help="Search mode: 'keyword' (exact substring) or 'semantic' (embeddings).",
    )
    ap.add_argument(
        "--match-all",
        action="store_true",
        help="(Keyword mode only) Require ALL keywords to appear. Default: any.",
    )
    ap.add_argument(
        "--model",
        default="sentence-transformers/all-mpnet-base-v2",
        help="(Semantic mode) SentenceTransformers model name.",
    )
    ap.add_argument(
        "--use-faiss",
        action="store_true",
        help="(Semantic mode) Use FAISS for search (requires faiss-cpu).",
    )
    args = ap.parse_args()

    if args.mode == "keyword":
        # Pure keyword mode: just load papers, no embeddings/model
        papers = load_papers(args.jsonl_path)
        print(
            f"[info] Keyword mode over {len(papers)} papers; "
            f"query={args.query!r}",
            file=sys.stderr,
        )
        idx, scores = keyword_search(
            papers, args.query, top_k=args.top_k, match_all=args.match_all
        )
        if len(idx) == 0:
            print("[info] No matches found.", file=sys.stderr)
        pretty_print_results(idx, scores, papers, args.query, mode="keyword")
        return

    # Semantic mode
    embeddings, papers = build_or_load_embeddings(
        args.jsonl_path, model_name=args.model
    )

    # Encode query
    model = SentenceTransformer(args.model)
    query_vec = model.encode([args.query], convert_to_numpy=True)[0:1, :]  # (1, d)
    query_vec = normalize_embeddings(query_vec)[0]

    # Perform search
    if args.use_faiss and HAS_FAISS:
        print("[info] Using FAISS for semantic search", file=sys.stderr)
        idx, sims = semantic_search_faiss(embeddings, query_vec, args.top_k)
    else:
        if args.use_faiss and not HAS_FAISS:
            print(
                "[warn] --use-faiss requested but faiss is not installed; "
                "falling back to NumPy.",
                file=sys.stderr,
            )
        idx, sims = semantic_search_numpy(embeddings, query_vec, args.top_k)

    pretty_print_results(idx, sims, papers, args.query, mode="semantic")


if __name__ == "__main__":
    main()
