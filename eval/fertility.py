"""Tokenizer fertility (tokens per whitespace word) on the held-out Italian corpus.

Runs every model tokenizer of ``eval/model_meta.json`` (the three BoccaccioAI stages share
one tokenizer and are merged) plus the legacy tokenizers of the previous small-passage
table on ``data/heldout/wiki_it_2025.jsonl``. CPU only, deterministic.

Output: ``eval/results/fertility_wiki-it-2025.json`` with, per tokenizer: vocabulary size,
total tokens/words, pooled fertility, per-document mean +- std, 95% CI, UTF-8 bytes/token
and the ratio against the BoccaccioAI tokenizer.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
Z95 = 1.959964

# Legacy tokenizers kept for continuity with the previous 204-word table.
EXTRA_TOKENIZERS = [
    ("GPT-2", "gpt2"),
    ("XLM-R", "xlm-roberta-base"),
    ("mBERT", "bert-base-multilingual-cased"),
    ("mT5", "google/mt5-small"),
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--meta", type=Path, default=ROOT / "eval" / "model_meta.json")
    p.add_argument("--corpus", type=Path,
                   default=ROOT / "data" / "heldout" / "wiki_it_2025.jsonl")
    p.add_argument("--out", type=Path,
                   default=ROOT / "eval" / "results" / "fertility_wiki-it-2025.json")
    return p.parse_args()


def load_env() -> None:
    """Load KEY=VALUE pairs of a local .env file into the environment (HF_TOKEN)."""
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def main() -> None:
    """Entry point."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    load_env()
    args = parse_args()
    if not args.corpus.exists():
        raise ValueError(f"held-out corpus {args.corpus} not found; run eval/build_heldout.py")
    from transformers import AutoTokenizer

    docs = [json.loads(x)["text"]
            for x in args.corpus.read_text(encoding="utf-8").splitlines() if x.strip()]
    meta = json.loads(args.meta.read_text(encoding="utf-8"))
    toks: list[tuple[str, str]] = []
    seen_ours = False
    for m in meta["models"]:
        if m["group"] == "ours":
            if seen_ours:
                continue
            seen_ours = True
            toks.append(("BoccaccioAI (IT BPE)", m["hf"]))
        else:
            toks.append((m["display"], m["hf"]))
    toks += EXTRA_TOKENIZERS

    entries = []
    for label, hf in toks:
        path = str(ROOT / hf) if (ROOT / hf).exists() else hf
        tok = AutoTokenizer.from_pretrained(path)
        n_words = n_tokens = n_bytes = 0
        fert_docs = []
        for text in docs:
            words = len(text.split())
            if words < 30:
                continue
            ids = tok(text, add_special_tokens=False)["input_ids"]
            fert_docs.append(len(ids) / words)
            n_words += words
            n_tokens += len(ids)
            n_bytes += len(text.encode("utf-8"))
        fert = np.array(fert_docs)
        entries.append({
            "label": label, "hf": hf, "vocab": len(tok), "docs": len(fert),
            "tokens": n_tokens, "words": n_words,
            "fertility": n_tokens / n_words,
            "fertility_mean": float(fert.mean()), "fertility_std": float(fert.std()),
            "fertility_ci95": float(Z95 * fert.std() / math.sqrt(len(fert))),
            "bytes_per_token": n_bytes / n_tokens,
        })
        log.info("%-22s vocab=%6d  phi=%.3f  B/tok=%.3f  docs=%d",
                 label, len(tok), entries[-1]["fertility"],
                 entries[-1]["bytes_per_token"], len(fert))
    ref = entries[0]["fertility"]
    for e in entries:
        e["relative"] = e["fertility"] / ref
    args.out.write_text(json.dumps({
        "corpus": str(args.corpus), "docs": len(docs),
        "definition": "tokens per whitespace word, pooled over the corpus",
        "entries": entries,
    }, indent=2), encoding="utf-8")
    print(f"Written {args.out}")


if __name__ == "__main__":
    main()
