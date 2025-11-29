#!/usr/bin/env python3
import os
import json
import sys
from typing import List, Tuple

from flask import Flask, render_template, request

import numpy as np
from sentence_transformers import SentenceTransformer  # type: ignore

# ------------------------
# Config
# ------------------------
JSONL_PATH = "neurips_2025_sandiego.jsonl"
EMBED_CACHE_PATH = "neurips_2025_sandiego_embeddings.npz"
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

app = Flask(__name__)

PAPERS: List[dict] = []
EMBEDDINGS: np.ndarray | None = None
MODEL: SentenceTransformer | None = None
TIME_SLOTS: List[str] = []


# ------------------------
# Data + embedding helpers
# ------------------------

def load_papers(jsonl_path: str) -> List[dict]:
    papers: List[dict] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            papers.append(json.loads(line))
    return papers


def build_texts(papers: List[dict]) -> List[str]:
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
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return emb / norms


def load_or_build_embeddings() -> Tuple[np.ndarray, List[dict]]:
    """Load papers and cached embeddings; compute if needed."""
    if not os.path.exists(JSONL_PATH):
        print(f"[error] JSONL not found at {JSONL_PATH}", file=sys.stderr)
        raise SystemExit(1)

    papers = load_papers(JSONL_PATH)
    print(f"[info] Loaded {len(papers)} papers", file=sys.stderr)

    if os.path.exists(EMBED_CACHE_PATH):
        print(f"[info] Loading cached embeddings from {EMBED_CACHE_PATH}", file=sys.stderr)
        data = np.load(EMBED_CACHE_PATH)
        emb = data["embeddings"]
        if emb.shape[0] == len(papers):
            return emb, papers
        else:
            print("[warn] Embedding count mismatch; recomputing.", file=sys.stderr)

    print(f"[info] Computing embeddings with {MODEL_NAME}", file=sys.stderr)
    model = SentenceTransformer(MODEL_NAME)
    texts = build_texts(papers)
    emb = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    emb = normalize_embeddings(emb)
    np.savez(EMBED_CACHE_PATH, embeddings=emb)
    print(f"[info] Saved embeddings to {EMBED_CACHE_PATH}", file=sys.stderr)
    return emb, papers


# ------------------------
# Search logic
# ------------------------

def parse_keywords(query: str) -> List[str]:
    if "," in query:
        parts = [p.strip() for p in query.split(",")]
    else:
        parts = query.split()
    return [p for p in parts if p]


def keyword_search(
    papers: List[dict],
    query: str,
    top_k: int,
    match_all: bool = False,
    allowed_times: List[str] | None = None,
) -> List[dict]:
    """Exact (case-insensitive) keyword substring search over title+abstract."""
    keywords = parse_keywords(query)
    if not keywords:
        return []

    key_lower = [k.lower() for k in keywords]

    hits_indices: List[int] = []
    scores: List[float] = []

    for i, p in enumerate(papers):
        if allowed_times and p.get("time") not in allowed_times:
            continue

        text = (p.get("title") or "") + " " + (p.get("abstract") or "")
        text_lower = text.lower()
        flags = [kw in text_lower for kw in key_lower]

        if match_all and not all(flags):
            continue
        if not match_all and not any(flags):
            continue

        score = float(sum(flags))
        hits_indices.append(i)
        scores.append(score)

    if not hits_indices:
        return []

    idx_arr = np.array(hits_indices, dtype=int)
    score_arr = np.array(scores, dtype=float)

    titles = np.array([papers[i]["title"] for i in idx_arr], dtype=object)
    order = np.lexsort((titles, -score_arr))
    idx_arr = idx_arr[order]
    score_arr = score_arr[order]

    top_k = min(top_k, len(idx_arr))
    idx_arr = idx_arr[:top_k]
    score_arr = score_arr[:top_k]

    results: List[dict] = []
    for rank, (i, s) in enumerate(zip(idx_arr, score_arr), start=1):
        p = papers[int(i)].copy()
        p["score"] = float(s)
        p["rank"] = rank
        results.append(p)
    return results


def semantic_search(
    embeddings: np.ndarray,
    papers: List[dict],
    model: SentenceTransformer,
    query: str,
    top_k: int,
    allowed_times: List[str] | None = None,
) -> List[dict]:
    """Cosine similarity search using sentence-transformer embeddings."""
    if not query.strip():
        return []

    q_vec = model.encode([query], convert_to_numpy=True)[0:1, :]
    q_vec = normalize_embeddings(q_vec)[0]

    sims = embeddings @ q_vec
    idx_unsorted = np.argsort(-sims)  # all docs sorted by similarity desc

    results: List[dict] = []
    for i in idx_unsorted:
        p = papers[int(i)]

        if allowed_times and p.get("time") not in allowed_times:
            continue

        score = float(sims[i])
        p_copy = p.copy()
        p_copy["score"] = score
        p_copy["rank"] = len(results) + 1
        results.append(p_copy)

        if len(results) >= top_k:
            break

    return results


# ------------------------
# Flask routes
# ------------------------

@app.route("/", methods=["GET", "POST"])
def index():
    query = ""
    mode = "semantic"
    top_k = 20
    match_all = False
    results: List[dict] = []
    selected_times: List[str] = []

    if request.method == "POST":
        query = request.form.get("query", "").strip()
        mode = request.form.get("mode", "semantic")
        top_k_str = request.form.get("top_k", "20")
        match_all = request.form.get("match_all") == "on"
        selected_times = request.form.getlist("time_filter")

        try:
            top_k = max(1, min(100, int(top_k_str)))
        except ValueError:
            top_k = 20

        allowed_times = selected_times or None

        if mode == "keyword":
            results = keyword_search(
                PAPERS,
                query,
                top_k=top_k,
                match_all=match_all,
                allowed_times=allowed_times,
            )
        else:
            if MODEL is None or EMBEDDINGS is None:
                results = []
            else:
                results = semantic_search(
                    EMBEDDINGS,
                    PAPERS,
                    MODEL,
                    query,
                    top_k=top_k,
                    allowed_times=allowed_times,
                )

    return render_template(
        "index.html",
        query=query,
        mode=mode,
        top_k=top_k,
        match_all=match_all,
        results=results,
        total_papers=len(PAPERS),
        time_slots=TIME_SLOTS,
        selected_times=selected_times,
    )


# ------------------------
# App startup
# ------------------------

def init_app():
    global PAPERS, EMBEDDINGS, MODEL, TIME_SLOTS
    EMBEDDINGS, PAPERS = load_or_build_embeddings()
    MODEL = SentenceTransformer(MODEL_NAME)
    TIME_SLOTS = sorted({p.get("time") for p in PAPERS if p.get("time")})
    print("[info] App initialized", file=sys.stderr)


if __name__ == "__main__":
    init_app()
    port = int(os.environ.get("PORT", 5000))
    # Railway sets PORT env var; we bind to it
    app.run(host="0.0.0.0", port=port)
