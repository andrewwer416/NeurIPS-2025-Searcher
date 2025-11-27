#!/usr/bin/env python3
"""
Scrape NeurIPS 2025 San Diego papers (titles + abstracts) into a JSONL file.
RESUMABLE: If you stop it, just run again and it continues from where it left.
"""

import argparse
import json
import sys
import time
import os
from typing import Dict, List, Optional
from urllib.parse import urljoin, urlparse
import re

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

BASE_LIST_URL = "https://neurips.cc/virtual/2025/loc/san-diego/papers.html"
SESSION = requests.Session()
SESSION.headers.update(
    {"User-Agent": "NeurIPS-Semantic-Search-Scraper (academic use)"}
)


def fetch_listing() -> List[Dict]:
    print(f"[info] Fetching listing page {BASE_LIST_URL}", file=sys.stderr)
    resp = SESSION.get(BASE_LIST_URL)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    papers: List[Dict] = []
    for a in soup.find_all("a"):
        href = a.get("href") or ""
        title = a.get_text(strip=True)
        if not title or not href:
            continue

        if "/virtual/2025/" in href and (
            "/poster/" in href
            or "/oral/" in href
            or "/spotlight/" in href
            or "/paper/" in href
        ):
            url = urljoin(BASE_LIST_URL, href)
            papers.append({"title": title, "url": url})

    seen = set()
    unique = []
    for p in papers:
        if p["url"] not in seen:
            seen.add(p["url"])
            unique.append(p)

    print(f"[info] Found {len(unique)} paper links", file=sys.stderr)
    return unique


TIME_PATTERN = re.compile(
    r"(Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+\d{1,2}\s+.*?PST",
    re.IGNORECASE,
)

def parse_kind_and_meta(soup: BeautifulSoup) -> Dict[str, Optional[str]]:
    """
    Parse kind (Poster/Oral/Spotlight), authors, location, and time.

    Layout (typical):
      <h3>San Diego Poster</h3>    -> kind
      <h2>Title</h2>
      <h3>Author1 · Author2 · ...</h3> -> authors
      <h5>Exhibit Hall C,D,E #1501</h5> -> location
      ... "Wed 3 Dec 11 a.m. PST — 2 p.m. PST" ... -> time (text node)
    """

    kind = None
    authors = None
    location = None
    time_slot = None

    # ----- kind & authors -----
    # First h3 with "Poster/Oral/Spotlight" -> kind
    # First other h3 -> authors
    for h3 in soup.find_all("h3"):
        txt = h3.get_text(" ", strip=True)
        if not txt:
            continue
        lower = txt.lower()
        if "san diego poster" in lower:
            kind = "Poster"
            continue
        if "san diego oral" in lower:
            kind = "Oral"
            continue
        if "san diego spotlight" in lower:
            kind = "Spotlight"
            continue

        # If we get here, it's an h3 that is *not* the "San Diego ..." header.
        # Treat the first such h3 as authors.
        if authors is None:
            authors = txt

    # ----- location -----
    # First h5 is the exhibit / location info
    h5 = soup.find("h5")
    if h5:
        loc_txt = h5.get_text(" ", strip=True)
        if loc_txt:
            location = loc_txt

    # ----- time -----
    # Search all stripped text nodes for something that looks like "Wed 3 Dec ... PST"
    for s in soup.stripped_strings:
        m = TIME_PATTERN.search(s)
        if m:
            time_slot = m.group(0).strip()
            break

    return {
        "kind": kind,
        "authors": authors,
        "location": location,
        "time": time_slot,
    }


def extract_abstract(soup: BeautifulSoup) -> str:
    for p in soup.find_all("p"):
        txt = p.get_text(" ", strip=True)
        if txt.lower().startswith("abstract:"):
            return txt[len("abstract:") :].strip()

    abstract_label = soup.find(string=lambda s: isinstance(s, str) and "Abstract:" in s)
    if abstract_label:
        parent = abstract_label.parent
        next_p = parent.find_next("p")
        if next_p:
            return next_p.get_text(" ", strip=True)

    return ""


def scrape_paper_detail(paper: Dict, sleep: float = 0.0) -> Dict:
    url = paper["url"]
    if sleep > 0:
        time.sleep(sleep)

    try:
        resp = SESSION.get(url)
        resp.raise_for_status()
    except Exception as e:
        print(f"[warn] Failed to fetch {url}: {e}", file=sys.stderr)
        return {
            **paper,
            "abstract": "",
            "kind": None,
            "location": None,
            "time": None,
            "authors": None,
        }

    soup = BeautifulSoup(resp.text, "html.parser")

    meta = parse_kind_and_meta(soup)
    abstract = extract_abstract(soup)

    parsed = urlparse(url)
    paper_id = parsed.path.rstrip("/").split("/")[-1]

    return {
        "id": paper_id,
        "title": paper["title"],
        "url": url,
        "kind": meta["kind"],
        "authors": meta["authors"],
        "location": meta["location"],
        "time": meta["time"],
        "abstract": abstract,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", "-o", required=True)
    ap.add_argument("--max-papers", type=int, default=None)
    ap.add_argument("--sleep", type=float, default=0.0)
    args = ap.parse_args()

    papers = fetch_listing()
    if args.max_papers is not None:
        papers = papers[: args.max_papers]

    # RESUME SUPPORT
    existing_urls = set()
    if os.path.exists(args.output):
        print(f"[info] Resuming from existing file {args.output}", file=sys.stderr)
        with open(args.output, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    if "url" in obj:
                        existing_urls.add(obj["url"])
                except:
                    pass

        print(f"[info] Found {len(existing_urls)} already-scraped papers", file=sys.stderr)

    # Open file in append mode
    out = open(args.output, "a", encoding="utf-8")

    to_scrape = [p for p in papers if p["url"] not in existing_urls]
    print(f"[info] Still need to scrape {len(to_scrape)} papers", file=sys.stderr)

    for p in tqdm(to_scrape, desc="Scraping"):
        detail = scrape_paper_detail(p, sleep=args.sleep)
        out.write(json.dumps(detail, ensure_ascii=False) + "\n")
        out.flush()

    out.close()
    print(f"[info] Done. File = {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
