"""Run the benchmark-v2 questions through the deployed BoccaccioAI RAG endpoint.

Produces ``benchmark/answers/prod-rag.jsonl`` and ``benchmark/answers/prod-norag.jsonl``
in the same record format as ``judge.py gen`` outputs, so the standard
``judge.py judge`` / ``report`` pipeline scores them like two extra "models".

Resumable: each answered question is appended+flushed immediately; restarting
skips ids already present. Endpoint credentials come from .env
(BOCCACCIO_MODAL_URL, BOCCACCIO_API_KEY). Cold start is ~1-3 min; the script
keeps the instance warm with sequential calls.

    python eval/judge/run_production.py [--limit N]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from judge import append_record, done_ids, load_env, load_jsonl  # noqa: E402

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
PAYLOAD_BASE = {"top_k": 3, "top_n": 20, "max_new_tokens": 200,
                "temperature": 0.0, "repetition_penalty": 1.0, "abstain": True}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--only", choices=["rag", "norag"], default=None,
                 help="run only one condition (default: both)")
    p.add_argument("--workers", type=int, default=5,
                 help="concurrent endpoint calls (Modal autoscales replicas)")
    return p.parse_args()


def query(url: str, key: str, question: str, use_rag: bool) -> dict:
    """POST one question to the Modal endpoint with cold-start retries."""
    payload = {"question": question, "use_rag": use_rag, **PAYLOAD_BASE}
    last = "no attempt"
    for attempt in range(4):
        try:
            # (connect, read): a dead endpoint fails in ~30 s, a cold-start generation
            # gets up to 300 s of read time
            r = requests.post(url, json=payload, timeout=(30, 300),
                              headers={"X-API-Key": key})
            if r.status_code == 200:
                return r.json()
            last = f"HTTP {r.status_code}: {r.text[:150]}"
        except requests.RequestException as exc:
            last = str(exc)[:150]
        time.sleep(min(10 * (attempt + 1), 60))
    return {"response": "", "error": last}


def main() -> None:
    """Entry point."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    load_env()
    args = parse_args()
    url = os.environ.get("BOCCACCIO_MODAL_URL")
    key = os.environ.get("BOCCACCIO_API_KEY")
    if not url or not key:
        raise ValueError("BOCCACCIO_MODAL_URL / BOCCACCIO_API_KEY missing in .env")
    items = load_jsonl(ROOT / "benchmark" / "benchmark_v2.jsonl")
    if args.limit:
        items = items[: args.limit]
    conditions = [("rag", True), ("norag", False)] if not args.only else \
        [(args.only, args.only == "rag")]
    lockfile = ROOT / "benchmark" / "answers" / ".prod.lock"
    if lockfile.exists():
        log.warning("lock file %s exists: another run may be in progress "
                    "(delete it to force)", lockfile)
    lockfile.parent.mkdir(parents=True, exist_ok=True)
    lockfile.write_text(str(os.getpid()), encoding="utf-8")
    for cond, use_rag in conditions:
        out_path = ROOT / "benchmark" / "answers" / f"prod-{cond}.jsonl"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.touch(exist_ok=True)  # file visible immediately, before the first answer
        seen = done_ids(out_path)
        todo = [it for it in items if it["id"] not in seen]
        log.info("prod-%s: %d to do (have %d)", cond, len(todo), len(seen))
        lock = threading.Lock()
        done = [0]

        def one(it: dict) -> None:
            data = query(url, key, it["question"], use_rag)
            with lock:
                append_record(out_path, {
                    "id": it["id"], "question": it["question"],
                    "answerable": it["answerable"],
                    "answer": data.get("response", ""),
                    "abstained": data.get("abstained"), "used_rag": data.get("used_rag"),
                    "sources": [s.get("title") for s in data.get("sources", [])][:3],
                    "error": data.get("error")})
                done[0] += 1
                if done[0] % 25 == 0 or done[0] == len(todo):
                    log.info("  prod-%s: %d/%d", cond, done[0], len(todo))

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            list(ex.map(one, todo))
        log.info("prod-%s: done -> %s", cond, out_path)
    lockfile.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
