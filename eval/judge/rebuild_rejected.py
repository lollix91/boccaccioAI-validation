"""Rebuild ``benchmark/benchmark_v2_rejected.jsonl`` from the verifier cache.

The rejected-items file written by ``verify_questions.py`` was lost. Re-running the
verifier would overwrite ``benchmark_v2.jsonl``, so this script reconstructs the
rejected file offline: every candidate not present in the accepted set is looked up
in the judge cache (same prompt template and key as ``verify_questions.py``) with
each verifier model that appears in the accepted set.

    python eval/judge/rebuild_rejected.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from judge import load_jsonl  # noqa: E402
from verify_questions import PROMPT_ANS, PROMPT_UNANS  # noqa: E402

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "benchmark"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--cache", type=Path, default=ROOT / "eval" / "judge" / "cache")
    return p.parse_args()


def main() -> None:
    """Entry point."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    accepted = load_jsonl(BENCH / "benchmark_v2.jsonl")
    ids = {it["id"] for it in accepted}
    verifiers = sorted({it["verifier"]["model"] for it in accepted})
    rejected = []
    for it in load_jsonl(BENCH / "benchmark_v2_candidates.jsonl"):
        if it["id"] in ids:
            continue
        prompt = (PROMPT_ANS.format(question=it["question"], gold=it["gold"])
                  if it["answerable"] else PROMPT_UNANS.format(question=it["question"]))
        found = None
        for model in verifiers:
            tag = hashlib.sha256((model + "|" + prompt).encode()).hexdigest()[:16]
            hit = args.cache / f"{tag}.json"
            if hit.exists():
                v = json.loads(hit.read_text("utf-8"))
                found = {"model": model, "ok": bool(v.get("ok")), "reason": v.get("reason", "")}
                break
        it["verifier"] = found or {"model": None, "ok": False,
                                   "reason": "verifier response not found in cache"}
        rejected.append(it)
    out = BENCH / "benchmark_v2_rejected.jsonl"
    out.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rejected),
                   encoding="utf-8")
    for x in rejected:
        print(f"{x['id']}: {x['question']} -> {x['verifier']['model']}: "
              f"{x['verifier']['reason'][:120]}")
    log.info("written %d rejected items -> %s", len(rejected), out)


if __name__ == "__main__":
    main()
