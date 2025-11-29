#!/usr/bin/env python3
import json
import os
import sys
from typing import List, Tuple, Dict, Any

import gradio as gr
import html
import numpy as np
from sentence_transformers import SentenceTransformer

# ----------------------------------------------------
# Config
# ----------------------------------------------------
JSONL_PATH = "neurips_2025_sandiego.jsonl"
EMBED_CACHE_PATH = "neurips_2025_sandiego_embeddings.npz"

# Smaller + fast semantic model (great quality)
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

PAPERS: List[Dict[str, Any]] = []
EMBEDDINGS: np.ndarray | None = None
MODEL: SentenceTransformer | None = None
TIME_SLOTS: List[str] = []


# ----------------------------------------------------
# Data + embedding helpers
# ----------------------------------------------------

def load_papers(path: str) -> List[dict]:
    papers: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
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


def load_or_build_embeddings(
    jsonl_path: str,
    cache_path: str,
    model_name: str,
) -> Tuple[np.ndarray, List[dict]]:
    if not os.path.exists(jsonl_path):
        print(f"[error] JSONL not found at {jsonl_path}", file=sys.stderr)
        raise SystemExit(1)

    papers = load_papers(jsonl_path)
    print(f"[info] Loaded {len(papers)} papers", file=sys.stderr)

    if os.path.exists(cache_path):
        print(f"[info] Loading cached embeddings from {cache_path}", file=sys.stderr)
        data = np.load(cache_path)
        emb = data["embeddings"]
        if emb.shape[0] == len(papers):
            return emb, papers
        else:
            print(
                "[warn] Cached embeddings count mismatch; recomputing.",
                file=sys.stderr,
            )

    print(f"[info] Computing embeddings with {model_name}", file=sys.stderr)
    model = SentenceTransformer(model_name)
    texts = build_texts(papers)
    emb = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    emb = normalize_embeddings(emb)
    np.savez(cache_path, embeddings=emb)
    print(f"[info] Saved embeddings to {cache_path}", file=sys.stderr)
    return emb, papers


# ----------------------------------------------------
# Search logic
# ----------------------------------------------------

def parse_keywords(query: str) -> List[str]:
    if "," in query:
        kws = [p.strip() for p in query.split(",")]
    else:
        kws = query.split()
    return [k for k in kws if k]


def keyword_search(
    papers: List[dict],
    query: str,
    top_k: int,
    match_all: bool,
    allowed_times: List[str] | None,
) -> List[dict]:
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
    allowed_times: List[str] | None,
) -> List[dict]:
    if not query.strip():
        return []

    q_vec = model.encode([query], convert_to_numpy=True)[0:1, :]
    q_vec = normalize_embeddings(q_vec)[0]

    sims = embeddings @ q_vec
    idx_unsorted = np.argsort(-sims)

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


# ----------------------------------------------------
# Gradio callback
# ----------------------------------------------------

def run_search(
    query: str,
    mode: str,
    top_k: int,
    match_all: bool,
    selected_times: List[str],
):
    query = (query or "").strip()
    if not query:
        return "<p>Enter a query to search.</p>", "Enter a query to search."

    allowed_times = selected_times or None

    if mode == "Keyword":
        results = keyword_search(
            PAPERS, query, top_k=top_k, match_all=match_all, allowed_times=allowed_times
        )
    else:
        if MODEL is None or EMBEDDINGS is None:
            return "<p>Model not loaded.</p>", "Model not loaded."
        results = semantic_search(
            EMBEDDINGS,
            PAPERS,
            MODEL,
            query,
            top_k=top_k,
            allowed_times=allowed_times,
        )

    if not results:
        return "<p>No results found.</p>", "No results found."

    cards = []

    for r in results:
        rank = r.get("rank", "")
        score = r.get("score", 0.0)
        title = r.get("title", "") or ""
        kind = r.get("kind", "") or ""
        authors = r.get("authors", "") or ""
        time_ = r.get("time", "") or ""
        loc = r.get("location", "") or ""
        url = r.get("url", "") or ""
        abstract = (r.get("abstract") or "").replace("\n", " ")

        # if len(abstract) > 1200:
        #     abstract = abstract[:1200].rstrip() + "..."

        # Escape text for HTML safety
        esc_title = html.escape(title)
        esc_authors = html.escape(authors)
        esc_time = html.escape(time_)
        esc_loc = html.escape(loc)
        esc_kind = html.escape(kind)
        esc_abs = html.escape(abstract)

        # Build badges only if present
        meta_bits = []
        if esc_kind:
            meta_bits.append(f'<span class="badge badge-kind">{esc_kind}</span>')
        if esc_time:
            meta_bits.append(f'<span class="badge badge-time">{esc_time}</span>')
        if esc_loc:
            meta_bits.append(f'<span class="badge badge-loc">{esc_loc}</span>')

        meta_line = " ".join(meta_bits)

        card_html = f"""
        <div class="result-card">
          <div class="result-header">
            <span class="rank">#{rank}</span>
            {"<span class='score-pill'>score: %.4f</span>" % score}
          </div>
          <div class="title-line">
            {"<a href='%s' target='_blank' rel='noopener noreferrer'>%s</a>" % (html.escape(url), esc_title) if url else esc_title}
          </div>
          <div class="meta-line">
            {meta_line}
          </div>
          <div class="authors-line">
            {esc_authors}
          </div>
          <div class="abstract-line">
            {esc_abs}
          </div>
        </div>
        """
        cards.append(card_html)

    # Wrap in a container + inline CSS for nicer styling
    all_cards_html = """
    <style>
      .results-container {
        display: flex;
        flex-direction: column;
        gap: 12px;
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      }
      .result-card {
        background: #ffffff;
        border-radius: 10px;
        padding: 10px 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
      }
      .result-header {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 0.85rem;
        color: #555;
        margin-bottom: 4px;
      }
      .rank {
        font-weight: 600;
      }
      .score-pill {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 999px;
        background: #e5f0ff;
        color: #1d4ed8;
        font-size: 0.75rem;
      }
      .title-line a {
        font-size: 1.0rem;
        font-weight: 600;
        color: #1d4ed8;
        text-decoration: none;
      }
      .title-line a:hover {
        text-decoration: underline;
      }
      .meta-line {
        margin-top: 4px;
        margin-bottom: 4px;
        font-size: 0.78rem;
        color: #444;
      }
      .badge {
        display: inline-block;
        padding: 2px 6px;
        border-radius: 999px;
        margin-right: 4px;
        margin-bottom: 2px;
        font-size: 0.7rem;
      }
      .badge-kind { background: #fee2e2; color: #b91c1c; }
      .badge-time { background: #e0f2fe; color: #0369a1; }
      .badge-loc  { background: #ecfdf3; color: #166534; }
      .authors-line {
        font-size: 0.8rem;
        color: #555;
        margin-bottom: 4px;
      }
      .abstract-line {
        font-size: 0.85rem;
        color: #222;
      }
    </style>
    <div class="results-container">
    """ + "\n".join(cards) + "</div>"

    status = f"Showing {len(results)} results (mode: {mode})."
    return all_cards_html, status


# ----------------------------------------------------
# App init (runs once when Space starts)
# ----------------------------------------------------

print("[info] Initializing app...", file=sys.stderr)
EMBEDDINGS, PAPERS = load_or_build_embeddings(JSONL_PATH, EMBED_CACHE_PATH, MODEL_NAME)
MODEL = SentenceTransformer(MODEL_NAME)
TIME_SLOTS = sorted({p.get("time") for p in PAPERS if p.get("time")})
print("[info] Ready.", file=sys.stderr)


# ----------------------------------------------------
# Gradio UI
# ----------------------------------------------------

with gr.Blocks(title="NeurIPS 2025 Paper Search") as demo:
    gr.Markdown(
        "# NeurIPS 2025 Paper Search\n"
        "Semantic + keyword search over NeurIPS 2025 San Diego papers.\n"
        "Type a query, choose a mode, and optionally filter by time slots."
    )

    with gr.Row():
        query_box = gr.Textbox(
            label="Query",
            placeholder="e.g. cybersecurity adversarial RL, 3D gaussian splatting, offline reinforcement learning",
            lines=3,
        )

    with gr.Row():
        mode_radio = gr.Radio(
            ["Semantic", "Keyword"],
            value="Semantic",
            label="Search mode",
            info="Semantic uses sentence-transformer embeddings; Keyword does exact substring search.",
        )
        topk_slider = gr.Slider(
            minimum=5,
            maximum=100,
            value=25,
            step=1,
            label="Top K results",
        )
        match_all_checkbox = gr.Checkbox(
            label="Require ALL keywords (keyword mode)",
            value=False,
        )

    time_filter = gr.CheckboxGroup(
        choices=TIME_SLOTS,
        label="Filter by time slot(s)",
        info="Select one or more times to include. Leave blank to include all.",
    )

    search_button = gr.Button("Search")

    results_html = gr.HTML(label="Results")

    status_text = gr.Markdown("")

    search_button.click(
        fn=run_search,
        inputs=[query_box, mode_radio, topk_slider, match_all_checkbox, time_filter],
        outputs=[results_html, status_text],
    )

if __name__ == "__main__":
    demo.launch()
