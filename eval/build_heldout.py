"""Build the held-out Italian corpus for bits-per-byte: Wikipedia articles created after a cutoff.

Steps (each cached in ``data/heldout/``, so the script is resumable):

1. ``creations.jsonl``: page-creation log of it.wikipedia (main namespace) between ``--start``
   and ``--end`` (MediaWiki API, ``list=logevents``, ``letype=create``).
2. ``candidates.jsonl``: pages visited in a seeded random order; redirects, deleted pages,
   disambiguations and lists are dropped; the plain text of the current revision
   (``prop=extracts``) is cleaned (reference sections removed) and kept if long enough.
3. 13-gram overlap of every candidate against the Wikipedia dump used to pretrain
   BoccaccioAI (``wikimedia/wikipedia`` config ``20231101.it``): documents above
   ``--max-overlap`` are discarded (e.g. articles split from an older article).
4. ``wiki_it_2025.jsonl`` (``title, pageid, created, overlap, text``) and
   ``wiki_it_2025.stats.json``.

The cutoff (default 2025-06-01) is later than the release of every evaluated model
(the most recent, Qwen3, was released in April 2025).
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import requests
from huggingface_hub import snapshot_download

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
API = "https://it.wikipedia.org/w/api.php"
USER_AGENT = "BoccaccioAI-validation/1.0 (research evaluation; https://github.com/lollix91/boccaccioAI-validation)"
SEED = 1234
NGRAM = 13
DUMP_REPO = "wikimedia/wikipedia"
DUMP_CONFIG = "20231101.it"
CUT_SECTIONS = ("Note", "Bibliografia", "Collegamenti esterni", "Voci correlate",
                "Altri progetti", "Riferimenti", "Fonti", "Filmografia", "Discografia",
                "Opere", "Pubblicazioni", "Premi e riconoscimenti", "Palmarès",
                "Onorificenze")
SKIP_TITLE = re.compile(r"^(Lista|Liste|Elenco|Calendario|Discografia|Filmografia)\b")
WORD = re.compile(r"\w+")
_HELDOUT: np.ndarray | None = None


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out-dir", type=Path, default=ROOT / "data" / "heldout")
    p.add_argument("--start", default="2025-06-01T00:00:00Z")
    p.add_argument("--end", default="2026-09-30T23:59:59Z")
    p.add_argument("--target-docs", type=int, default=2000)
    p.add_argument("--oversample", type=float, default=1.2,
                   help="candidates collected = target * oversample (room for overlap drops)")
    p.add_argument("--min-chars", type=int, default=1500)
    p.add_argument("--max-chars", type=int, default=20000)
    p.add_argument("--max-overlap", type=float, default=0.10)
    p.add_argument("--workers", type=int, default=20)
    return p.parse_args()


def api_get(session: requests.Session, params: dict) -> dict:
    """Call the MediaWiki API with retries and maxlag handling.

    Parameters
    ----------
    session : requests.Session
        HTTP session with the user agent set.
    params : dict
        Query parameters (``format=json`` and ``formatversion=2`` are added).

    Returns
    -------
    dict
        Decoded JSON response.

    Raises
    ------
    ValueError
        If the request keeps failing.
    """
    params = {**params, "format": "json", "formatversion": 2, "maxlag": 5}
    for attempt in range(8):
        try:
            r = session.get(API, params=params, timeout=60)
            if r.status_code == 200:
                data = r.json()
                if data.get("error", {}).get("code") != "maxlag":
                    return data
        except (requests.RequestException, ValueError) as exc:
            log.warning("request failed (%s), retrying", exc)
        time.sleep(2 ** attempt)
    raise ValueError(f"API request failed repeatedly: {params}")


def collect_creations(session: requests.Session, args: argparse.Namespace, path: Path) -> list[dict]:
    """Download (or load from cache) the page-creation log in the date window."""
    if path.exists():
        return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
    events, cont = [], {}
    params = {"action": "query", "list": "logevents", "letype": "create", "lenamespace": 0,
              "lestart": args.start, "leend": args.end, "ledir": "newer", "lelimit": "max",
              "leprop": "title|ids|timestamp"}
    while True:
        data = api_get(session, {**params, **cont})
        for e in data["query"]["logevents"]:
            if e.get("pageid"):
                events.append({"pageid": e["pageid"], "title": e["title"],
                               "created": e["timestamp"]})
        log.info("creation log: %d events (last %s)", len(events),
                 events[-1]["created"] if events else "-")
        if "continue" not in data:
            break
        cont = data["continue"]
    path.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in events),
                    encoding="utf-8")
    return events


def clean_extract(text: str, max_chars: int) -> str:
    """Drop reference sections and headings, normalise blank lines, cap the length."""
    out = []
    for line in text.splitlines():
        m = re.match(r"^(=+)\s*(.*?)\s*=+$", line.strip())
        if m:
            if m.group(2) in CUT_SECTIONS:
                break
            continue
        if line.strip():
            out.append(line.strip())
    text = "\n".join(out)
    if len(text) > max_chars:
        text = text[:max_chars].rsplit("\n", 1)[0]
    return text


def collect_candidates(session: requests.Session, events: list[dict], args: argparse.Namespace,
                       path: Path) -> list[dict]:
    """Visit pages in seeded random order and keep clean, long-enough articles."""
    done = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            done[row["pageid"]] = row
    want = int(args.target_docs * args.oversample)
    kept = [r for r in done.values() if r.get("text")]
    order = list({e["pageid"]: e for e in events}.values())
    random.Random(SEED).shuffle(order)
    order = [e for e in order if e["pageid"] not in done and not SKIP_TITLE.match(e["title"])]
    with path.open("a", encoding="utf-8") as f:
        for i in range(0, len(order), 50):
            if len(kept) >= want:
                break
            batch = order[i:i + 50]
            info = api_get(session, {"action": "query", "prop": "info|pageprops",
                                     "ppprop": "disambiguation",
                                     "pageids": "|".join(str(e["pageid"]) for e in batch)})
            pages = {p["pageid"]: p for p in info["query"]["pages"] if "pageid" in p}
            for e in batch:
                p = pages.get(e["pageid"])
                row = {**e, "text": ""}
                if p and not p.get("missing") and not p.get("redirect") and p.get("ns") == 0 \
                        and "disambiguation" not in p.get("pageprops", {}):
                    ext = api_get(session, {"action": "query", "prop": "extracts",
                                            "explaintext": 1, "exsectionformat": "wiki",
                                            "pageids": e["pageid"]})
                    raw = ext["query"]["pages"][0].get("extract", "")
                    text = clean_extract(raw, args.max_chars)
                    if len(text) >= args.min_chars:
                        row["text"] = text
                        row["title"] = ext["query"]["pages"][0].get("title", e["title"])
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                if row["text"]:
                    kept.append(row)
                    if len(kept) >= want:
                        break
            f.flush()
            log.info("candidates kept %d / %d", len(kept), want)
    return kept


# Fixed weights for the linear 13-gram fingerprint below (FNV prime powers mod 2^64).
_GRAM_WEIGHTS = np.array([pow(1099511628211, j, 1 << 64) for j in range(NGRAM)],
                         dtype=np.uint64)


def ngram_hashes(text: str) -> np.ndarray:
    """Unique 64-bit fingerprints of the lower-cased word 13-grams of a text.

    Each word is mapped to an integer code and every 13-word window gets the linear
    fingerprint ``sum(code[i+j] * prime**j)`` mod 2^64, vectorized with a sliding
    window view. Collisions are astronomically unlikely and would only slightly
    inflate the measured overlap.
    """
    words = WORD.findall(text.lower())
    if len(words) < NGRAM:
        return np.empty(0, dtype=np.uint64)
    codes = pd.factorize(words)[0].astype(np.uint64)
    win = np.lib.stride_tricks.sliding_window_view(codes, NGRAM)
    return np.unique((win * _GRAM_WEIGHTS).sum(axis=1))


def _init_worker(heldout: np.ndarray) -> None:
    global _HELDOUT
    _HELDOUT = heldout


def _scan_row_group(job: tuple[str, int]) -> np.ndarray:
    """Return the held-out hashes that occur in one row group of the dump."""
    path, rg = job
    texts = pq.ParquetFile(path).read_row_group(rg, columns=["text"]).column(0).to_pylist()
    found = []
    for t in texts:
        h = ngram_hashes(t)
        if len(h):
            hit = h[np.isin(h, _HELDOUT, assume_unique=True)]
            if len(hit):
                found.append(hit)
    return np.unique(np.concatenate(found)) if found else np.empty(0, dtype=np.uint64)


def overlap_with_dump(docs: list[dict], workers: int) -> list[float]:
    """Fraction of each document's 13-grams that also occur in the 2023 Wikipedia dump."""
    local = snapshot_download(DUMP_REPO, repo_type="dataset",
                              allow_patterns=[f"{DUMP_CONFIG}/*.parquet"])
    files = sorted(Path(local, DUMP_CONFIG).glob("*.parquet"))
    if not files:
        raise ValueError("Wikipedia dump not found after download")
    doc_hashes = [ngram_hashes(d["text"]) for d in docs]
    heldout = np.unique(np.concatenate(doc_hashes))
    jobs = [(str(f), rg) for f in files for rg in range(pq.ParquetFile(f).num_row_groups)]
    log.info("scanning %d row groups of %s against %d held-out 13-grams", len(jobs),
             DUMP_CONFIG, len(heldout))
    seen = []
    with Pool(workers, initializer=_init_worker, initargs=(heldout,)) as pool:
        for i, hit in enumerate(pool.imap_unordered(_scan_row_group, jobs), 1):
            seen.append(hit)
            if i % 20 == 0 or i == len(jobs):
                log.info("row groups %d / %d", i, len(jobs))
    seen_all = np.unique(np.concatenate(seen)) if seen else np.empty(0, dtype=np.uint64)
    return [float(np.isin(h, seen_all).mean()) if len(h) else 1.0 for h in doc_hashes]


def main() -> None:
    """Entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args = parse_args()
    if (args.out_dir / "wiki_it_2025.jsonl").exists():
        raise ValueError(f"{args.out_dir / 'wiki_it_2025.jsonl'} already exists; the corpus is "
                         "frozen once built (delete it by hand to rebuild)")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    events = collect_creations(session, args, args.out_dir / "creations.jsonl")
    cands = collect_candidates(session, events, args, args.out_dir / "candidates.jsonl")
    overlaps = overlap_with_dump(cands, args.workers)
    clean = [{**d, "overlap": o} for d, o in zip(cands, overlaps) if o <= args.max_overlap]
    final = clean[:args.target_docs]
    if len(final) < args.target_docs:
        log.warning("only %d documents after overlap filtering (target %d): "
                    "re-run with a larger --oversample", len(final), args.target_docs)
    out = args.out_dir / "wiki_it_2025.jsonl"
    keys = ("title", "pageid", "created", "overlap", "text")
    out.write_text("".join(json.dumps({k: d[k] for k in keys}, ensure_ascii=False) + "\n"
                           for d in final), encoding="utf-8")
    ov = np.array(overlaps)
    stats = {
        "source": "it.wikipedia.org, articles created in the window, current revision",
        "window": [args.start, args.end], "seed": SEED, "creation_events": len(events),
        "candidates": len(cands), "dropped_overlap": int((ov > args.max_overlap).sum()),
        "documents": len(final), "bytes": sum(len(d["text"].encode()) for d in final),
        "overlap_reference": f"{DUMP_REPO} {DUMP_CONFIG} ({NGRAM}-grams)",
        "max_overlap": args.max_overlap,
        "candidate_overlap_quantiles": {q: float(np.quantile(ov, q))
                                        for q in (0.5, 0.9, 0.99)} if len(ov) else {},
        "created_range": [min(d["created"] for d in final), max(d["created"] for d in final)]
        if final else [],
        "built": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (args.out_dir / "wiki_it_2025.stats.json").write_text(json.dumps(stats, indent=2),
                                                          encoding="utf-8")
    print(f"{len(final)} documents, {stats['bytes'] / 1e6:.1f} MB -> {out}")


if __name__ == "__main__":
    main()
