"""Tokenizer-agnostic language-modelling score: bits per UTF-8 byte on held-out text.

Perplexity per token is not comparable across tokenizers, so the total negative
log-likelihood (nats) is divided by the number of UTF-8 bytes of the raw text:
``BPB = NLL / (ln 2 * n_bytes)``. Each document is split into non-overlapping chunks of
``--max-len`` tokens; every chunk is preceded by the tokenizer's BOS token (EOS when the
tokenizer has no BOS, as lm-eval-harness does) so that the first real token is scored too.

Data sources: a plain-text/JSONL file (``--text-file``, one document per line, JSONL needs
``--field``) or a Hugging Face dataset (``--hf-dataset``).

Usage
-----
python eval/bpb.py --model models/boccaccio-distill-hf --name boccaccio-distill \
    --text-file data/heldout.jsonl --field text --dataset-name heldout
python eval/bpb.py --model Qwen/Qwen2.5-0.5B --name qwen2.5-0.5b \
    --hf-dataset crux82/squad_it --split test --field context --max-docs 500 --dataset-name smoke
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path

from datasets import load_dataset  # noqa: I001  (import before torch: DLL order crash on Windows)

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--dataset-name", required=True, help="label used in the output file name")
    p.add_argument("--text-file", type=Path)
    p.add_argument("--hf-dataset")
    p.add_argument("--hf-config")
    p.add_argument("--split", default="test")
    p.add_argument("--field", default="text")
    p.add_argument("--max-docs", type=int, default=None)
    p.add_argument("--max-len", type=int, default=2048, help="tokens per chunk (incl. prefix)")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def load_docs(args: argparse.Namespace) -> list[str]:
    """Load the evaluation documents.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments.

    Returns
    -------
    list[str]
        Non-empty documents.

    Raises
    ------
    ValueError
        If no data source is given.
    """
    if args.text_file:
        docs = []
        for line in args.text_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            docs.append(json.loads(line)[args.field] if args.text_file.suffix == ".jsonl" else line)
    elif args.hf_dataset:
        ds = load_dataset(args.hf_dataset, args.hf_config, split=args.split)
        docs = [r[args.field] for r in ds]
    else:
        raise ValueError("give --text-file or --hf-dataset")
    docs = [d for d in docs if d and d.strip()]
    return docs[: args.max_docs] if args.max_docs else docs


@torch.no_grad()
def main() -> None:
    """Compute bits per byte for one model on one dataset and save a JSON file."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    out = ROOT / "results" / args.name
    out_file = out / f"bpb_{args.dataset_name}.json"
    if out_file.exists():
        raise ValueError(f"{out_file} already exists; delete it by hand to recompute")
    docs = load_docs(args)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=getattr(torch, args.dtype))
    model = model.to(args.device).eval()
    prefix = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    if prefix is None:
        raise ValueError("tokenizer has neither BOS nor EOS token")
    body = args.max_len - 1

    chunks: list[list[int]] = []
    n_bytes = n_tokens = 0
    for doc in docs:
        ids = tok(doc, add_special_tokens=False)["input_ids"]
        n_bytes += len(doc.encode("utf-8"))
        n_tokens += len(ids)
        chunks += [[prefix] + ids[i:i + body] for i in range(0, len(ids), body)]
    chunks.sort(key=len)
    log.info("docs=%d bytes=%d tokens=%d chunks=%d", len(docs), n_bytes, n_tokens, len(chunks))

    total_nll = 0.0
    t0 = time.time()
    for i in range(0, len(chunks), args.batch_size):
        batch = chunks[i:i + args.batch_size]
        width = max(len(c) for c in batch)
        pad = tok.pad_token_id if tok.pad_token_id is not None else prefix
        ids = torch.tensor([c + [pad] * (width - len(c)) for c in batch], device=args.device)
        mask = torch.tensor([[1] * len(c) + [0] * (width - len(c)) for c in batch], device=args.device)
        logits = model(input_ids=ids, attention_mask=mask).logits[:, :-1].float()
        nll = F.cross_entropy(logits.transpose(1, 2), ids[:, 1:], reduction="none")
        total_nll += (nll * mask[:, 1:]).sum().item()
    bpb = total_nll / (math.log(2) * n_bytes)
    result = {
        "model": args.model, "name": args.name, "dataset": args.dataset_name,
        "bits_per_byte": bpb, "nll_nats": total_nll, "bytes": n_bytes, "tokens": n_tokens,
        "bytes_per_token": n_bytes / n_tokens, "token_ppl": math.exp(total_nll / n_tokens),
        "docs": len(docs), "max_len": args.max_len, "seconds": round(time.time() - t0, 1),
    }
    out.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
    log.info("BPB=%.4f  bytes/token=%.3f  token-ppl=%.2f", bpb, result["bytes_per_token"], result["token_ppl"])


if __name__ == "__main__":
    main()
