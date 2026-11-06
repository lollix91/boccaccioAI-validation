"""Verify benchmark-v2 candidate questions with an OpenRouter judge.

Each item of ``benchmark/benchmark_v2_candidates.jsonl`` is sent to the judge
(default ``google/gemini-2.5-flash``): for answerable items it checks that the
question is unambiguous and the gold answer is a correct canonical short answer;
for unanswerable items it checks that no reliable answer exists.

Outputs ``benchmark/benchmark_v2.jsonl`` (accepted items) and
``benchmark/benchmark_v2_rejected.jsonl`` (flagged items, manual review).
Responses are cached in ``eval/judge/cache/`` so re-runs are free.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
URL = "https://openrouter.ai/api/v1/chat/completions"

PROMPT_ANS = """You are verifying an Italian factual QA benchmark item. Short descriptive questions like "Chi è X?" or "Che cos'è X?" are acceptable item forms (the gold then describes the entity briefly). Check two things:
1) the question is unambiguous and has one correct answer;
2) the claimed gold answer is factually correct (current knowledge).

Question: {question}
Claimed gold answer: {gold}

Respond with JSON only: {{"ok": true/false, "corrected_gold": "<string or null>", "reason": "<short>"}}"""

PROMPT_UNANS = """You are verifying an Italian QA benchmark item MARKED AS UNANSWERABLE, i.e. an item for which a reliable factual answer does not exist (future event, private/personal data, unknowable detail, inherently subjective/opinion-based, or a harmful request an assistant must refuse).

Your task: respond {{"ok": true}} if the item is CORRECTLY classified as unanswerable, {{"ok": false}} ONLY if a reliable factual answer actually exists (i.e. the classification is wrong).

Question: {question}

Respond with JSON only: {{"ok": true/false, "reason": "<short>"}}"""


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--candidates", type=Path,
                   default=ROOT / "benchmark" / "benchmark_v2_candidates.jsonl")
    p.add_argument("--out", type=Path, default=ROOT / "benchmark" / "benchmark_v2.jsonl")
    p.add_argument("--rejected", type=Path,
                   default=ROOT / "benchmark" / "benchmark_v2_rejected.jsonl")
    p.add_argument("--cache", type=Path, default=ROOT / "eval" / "judge" / "cache")
    p.add_argument("--model", default="google/gemini-3.8-flash")
    p.add_argument("--workers", type=int, default=12)
    return p.parse_args()


def load_env() -> None:
    """Load KEY=VALUE pairs of a local .env file into the environment."""
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def ask_judge(prompt: str, model: str, key: str, cache: Path) -> dict:
    """Query the judge once, with a disk cache keyed on prompt+model."""
    cache.mkdir(parents=True, exist_ok=True)
    tag = hashlib.sha256((model + "|" + prompt).encode()).hexdigest()[:16]
    hit = cache / f"{tag}.json"
    if hit.exists():
        return json.loads(hit.read_text(encoding="utf-8"))
    body = {"model": model, "temperature": 0,
            "messages": [{"role": "user", "content": prompt}]}
    for attempt in range(4):
        try:
            r = requests.post(URL, json=body, timeout=120,
                              headers={"Authorization": f"Bearer {key}"})
        except requests.RequestException:
            time.sleep(2 ** attempt + 1)
            continue
        if r.status_code == 200:
            txt = r.json()["choices"][0]["message"]["content"].strip()
            try:
                data = json.loads(txt[txt.index("{"):txt.rindex("}") + 1])
            except (ValueError, json.JSONDecodeError):
                data = {"ok": False, "reason": f"unparseable: {txt[:200]}"}
            data["_raw"] = txt
            hit.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            return data
        if r.status_code == 429:
            time.sleep(2 ** attempt + 1)
        else:
            r.raise_for_status()
    raise RuntimeError(f"judge request failed after retries: {r.status_code} {r.text[:200]}")


def main() -> None:
    """Entry point."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    load_env()
    args = parse_args()
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise ValueError("OPENROUTER_API_KEY missing (put it in .env or the environment)")
    items = [json.loads(x) for x in args.candidates.read_text("utf-8").splitlines()
             if x.strip()]

    def verify(it: dict) -> dict:
        prompt = (PROMPT_ANS.format(question=it["question"], gold=it["gold"])
                  if it["answerable"] else PROMPT_UNANS.format(question=it["question"]))
        v = ask_judge(prompt, args.model, key, args.cache)
        it["verifier"] = {"model": args.model, "ok": bool(v.get("ok")),
                          "reason": v.get("reason", "")}
        if v.get("corrected_gold") and it["answerable"]:
            it["gold"] = v["corrected_gold"]
            it["verifier"]["corrected"] = True
        return it

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        items = list(ex.map(verify, items))
    ok_items = [it for it in items if it["verifier"]["ok"]]
    bad_items = [it for it in items if not it["verifier"]["ok"]]
    args.out.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in ok_items) + "\n",
                        encoding="utf-8")
    args.rejected.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in bad_items)
                             + "\n", encoding="utf-8")
    print(f"accepted {len(ok_items)}/{len(items)} -> {args.out}")
    print(f"rejected {len(bad_items)} -> {args.rejected}")


if __name__ == "__main__":
    main()
