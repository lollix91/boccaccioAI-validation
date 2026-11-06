"""Equivalence checks between the local benchmark pipeline and the production endpoint.

1. weights: SHA-256 of the production weights (Modal volume ``boccaccio-rag-data``,
   downloaded to ``--rag-dir``) vs the distilled checkpoint exported for evaluation;
2. retrieval: abstention decisions and top-k source titles of
   ``benchmark/rag_contexts.jsonl`` (eval/judge/retrieve_contexts.py) vs the production
   answers ``benchmark/answers/prod-rag.jsonl``;
3. generation: answers of the distilled model generated locally in the production chat
   format (``judge.py gen --models boccaccio-distill-chat --out-dir benchmark/checks``)
   vs the production parametric answers ``benchmark/answers/prod-norag.jsonl``.

Output: ``benchmark/checks/checks.json``.

    python eval/judge/checks.py --rag-dir E:/boccaccio-rag
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from judge import (N_RESAMPLES, REPORT_SEED, bootstrap_ci, excluded_ids,  # noqa: E402
                   load_jsonl, refused, score_of)

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "benchmark"
CHECKS = BENCH / "checks"
# sensitivity variant -> scored benchmark system with the same weights
SENSITIVITY_SCORED = {"boccaccio-sft-v1": "boccaccio-sft-v1-chat"}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--rag-dir", type=Path, default=Path("E:/boccaccio-rag"))
    p.add_argument("--local-weights", type=Path,
                   default=ROOT / "models" / "boccaccio-distill-hf" / "model.safetensors")
    return p.parse_args()


def sha256(path: Path) -> str:
    """SHA-256 of a file, streamed."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    """Entry point."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    out: dict = {}
    prod_w = args.rag_dir / "model.safetensors"
    if prod_w.exists() and args.local_weights.exists():
        a, b = sha256(prod_w), sha256(args.local_weights)
        out["weights"] = {"production_sha256": a, "local_sha256": b, "identical": a == b}

    prod_rag = {r["id"]: r for r in load_jsonl(BENCH / "answers" / "prod-rag.jsonl")}
    ctx = load_jsonl(BENCH / "rag_contexts.jsonl")
    if ctx:
        diffs = [r["id"] for r in ctx
                 if prod_rag[r["id"]]["abstained"] != r["abstained"]
                 or prod_rag[r["id"]]["sources"] != [s["title"] for s in r["sources"]]]
        out["retrieval"] = {"n": len(ctx), "n_abstained": sum(r["abstained"] for r in ctx),
                            "identical": len(ctx) - len(diffs), "differing_ids": diffs}

    local = {r["id"]: r for r in load_jsonl(CHECKS / "boccaccio-distill-chat.jsonl")}
    if local:
        prod = {r["id"]: r for r in load_jsonl(BENCH / "answers" / "prod-norag.jsonl")}
        diffs = [i for i in local if local[i]["answer"].strip() != prod[i]["answer"].strip()]
        out["generation"] = {"n": len(local), "identical": len(local) - len(diffs),
                             "differing_ids": diffs}

    # prompt-sensitivity variants (outside the scored benchmark)
    sens = BENCH / "sensitivity"
    if sens.exists():
        import numpy as np  # noqa: E402
        items = {it["id"]: it for it in load_jsonl(BENCH / "benchmark_v2.jsonl")}
        drop = excluded_ids()
        ans_ids = [i for i in items if items[i]["answerable"] and i not in drop]
        una_ids = [i for i in items if not items[i]["answerable"] and i not in drop]
        rng = np.random.default_rng(REPORT_SEED)
        for ans_path in sens.glob("*.jsonl"):
            if "__" in ans_path.name:
                continue
            system = ans_path.stem
            answers = {a["id"]: a for a in load_jsonl(ans_path)}
            jd = {f.name.split("__")[0]: {r["id"]: r for r in load_jsonl(f)}
                  for f in sens.glob(f"*__{system}.jsonl")
                  if not f.name.startswith("human")}
            if not jd:
                continue
            acc = np.array([np.mean([score_of(j[i]) for j in jd.values()])
                            for i in ans_ids])
            abst = np.array([np.mean([score_of(j[i]) for j in jd.values()])
                             for i in una_ids])
            answered = [i for i in ans_ids if not refused(answers[i]["answer"])]
            acc_answered = np.array([np.mean([score_of(j[i]) for j in jd.values()])
                                     for i in answered])
            rec = {
                "note": "same weights as the scored benchmark system, generated with "
                        "the generic prompt instead of the training chat format "
                        "(prompt-sensitivity check, not a scored benchmark row)",
                "accuracy": float(acc.mean()), "accuracy_ci": bootstrap_ci(acc, rng),
                "accuracy_on_answered": float(acc_answered.mean()),
                "n_answered": len(answered),
                "refusals_on_answerable": len(ans_ids) - len(answered),
                "abstention_rate": float(abst.mean()),
            }
            # split by what the scored chat-format system does on the same items: is the
            # over-refusal targeted at questions the model would get wrong anyway?
            scored = SENSITIVITY_SCORED.get(system)
            scored_path = BENCH / "answers" / f"{scored}.jsonl" if scored else None
            if scored_path and scored_path.exists():
                chat = {a["id"]: a for a in load_jsonl(scored_path)}
                by_id = dict(zip(ans_ids, acc))
                chat_ans = [i for i in ans_ids if not refused(chat[i]["answer"])]
                chat_ref = [i for i in ans_ids if refused(chat[i]["answer"])]
                rec["scored_system"] = scored
                rec["accuracy_on_scored_answered"] = float(np.mean([by_id[i] for i in chat_ans]))
                rec["accuracy_on_scored_refused"] = float(np.mean([by_id[i] for i in chat_ref]))
                rec["n_scored_answered"] = len(chat_ans)
                rec["n_scored_refused"] = len(chat_ref)
            out.setdefault("sensitivity", {})[system] = rec

    CHECKS.mkdir(parents=True, exist_ok=True)
    (CHECKS / "checks.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
