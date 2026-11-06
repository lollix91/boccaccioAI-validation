"""Check that the exported HF model reproduces the original BoccaccioAI logits.

The original source tree is imported read-only (bytecode writing disabled).

Usage
-----
python scripts/verify_equivalence.py \
    --hf-dir models/boccaccio-distill-hf --source-repo E:/boccaccioAI
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file
from tokenizers import Tokenizer
from transformers import AutoModelForCausalLM

log = logging.getLogger(__name__)

PROMPTS = [
    "### Utente:\nQual e' la capitale d'Italia?\n### Assistente:\n",
    "La Divina Commedia e' un poema scritto da Dante Alighieri nel XIV secolo.",
    "Il Decameron di Giovanni Boccaccio e' una raccolta di cento novelle, " * 20,
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--hf-dir", type=Path, required=True)
    p.add_argument("--source-repo", type=Path, required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    """Compare logits of the original and exported models in fp32."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(args.source_repo))
    from src.model.config import BoccaccioConfig
    from src.model.transformer import BoccaccioForCausalLM

    orig = BoccaccioForCausalLM(BoccaccioConfig())
    state = load_file(str(args.hf_dir / "model.safetensors"))
    orig.load_state_dict(state, strict=False)
    orig = orig.float().to(args.device).eval()
    hf = AutoModelForCausalLM.from_pretrained(args.hf_dir, dtype=torch.float32)
    hf = hf.to(args.device).eval()
    log.info("params: orig=%d hf=%d", orig.count_parameters(), sum(p.numel() for p in hf.parameters()))

    tok = Tokenizer.from_file(str(args.hf_dir / "tokenizer.json"))
    worst = 0.0
    for prompt in PROMPTS:
        ids = torch.tensor([tok.encode(prompt).ids[:512]], device=args.device)
        with torch.no_grad():
            a = orig(ids)["logits"]
            b = hf(ids).logits
        diff = (a - b).abs().max().item()
        same_top1 = bool((a.argmax(-1) == b.argmax(-1)).all())
        log.info("len=%d max|dlogit|=%.3e top1_equal=%s", ids.shape[1], diff, same_top1)
        worst = max(worst, diff)
        if not same_top1:
            raise SystemExit("top-1 predictions differ")
    if worst > 1e-2:
        raise SystemExit(f"max logit difference too large: {worst}")
    log.info("OK: exported model is equivalent (worst %.3e)", worst)


if __name__ == "__main__":
    main()
