"""Export a BoccaccioAI checkpoint as a standard Hugging Face Llama model.

The BoccaccioAI architecture (RMSNorm, RoPE with rotate_half, GQA, SwiGLU, tied
embeddings) is numerically identical to ``LlamaForCausalLM``, and the parameter
names already follow the Llama convention, so the export is a config + tokenizer
wrapper around the original safetensors weights. The source repository is only
read, never modified.

Usage
-----
python scripts/convert_to_hf.py \
    --weights E:/boccaccioAI/checkpoints/finetune_distill/model.safetensors \
    --tokenizer-dir E:/boccaccioAI/tokenizer/boccaccio-32k-hf \
    --output-dir models/boccaccio-distill-hf
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

log = logging.getLogger(__name__)

CONFIG = {
    "architectures": ["LlamaForCausalLM"],
    "model_type": "llama",
    "vocab_size": 32000,
    "hidden_size": 1536,
    "num_hidden_layers": 26,
    "num_attention_heads": 12,
    "num_key_value_heads": 4,
    "head_dim": 128,
    "intermediate_size": 4096,
    "hidden_act": "silu",
    "max_position_embeddings": 2048,
    "rms_norm_eps": 1e-5,
    "rope_theta": 10000.0,
    "attention_bias": False,
    "mlp_bias": False,
    "tie_word_embeddings": True,
    "initializer_range": 0.02,
    "bos_token_id": 1,
    "eos_token_id": 1,
    "pad_token_id": 2,
    "torch_dtype": "bfloat16",
}


def pt_to_safetensors(src: Path, dst: Path) -> None:
    """Convert a raw ``model.pt`` state dict (fp32) into a BF16 safetensors file.

    Strips a ``_orig_mod.`` prefix (torch.compile) and stores the tied embedding once.

    Parameters
    ----------
    src : Path
        Path to the ``.pt`` state dict.
    dst : Path
        Output ``.safetensors`` path.

    Raises
    ------
    ValueError
        If tied embedding and LM head differ.
    """
    sd = torch.load(src, map_location="cpu", weights_only=True)
    sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
    if not torch.equal(sd["lm_head.weight"], sd["model.embed_tokens.weight"]):
        raise ValueError("lm_head and embed_tokens are not tied in this checkpoint")
    sd = {k: v.to(torch.bfloat16).contiguous() for k, v in sd.items()}
    sd["lm_head.weight"] = sd["model.embed_tokens.weight"].clone()
    save_file(sd, str(dst), metadata={"format": "pt"})


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--tokenizer-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    """Build the HF model directory."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    if not args.weights.is_file():
        raise ValueError(f"weights not found: {args.weights}")

    tok_path = args.tokenizer_dir / "tokenizer.json"
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(tok_path))
    ids = {t: tok.token_to_id(t) for t in ("<s>", "</s>", "<pad>", "<unk>")}
    log.info("special token ids: %s", ids)

    cfg = dict(CONFIG)
    cfg["eos_token_id"] = ids["</s>"]
    cfg["bos_token_id"] = ids["</s>"]
    cfg["pad_token_id"] = ids["<pad>"]

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if args.weights.suffix == ".pt":
        pt_to_safetensors(args.weights, out / "model.safetensors")
    else:
        shutil.copyfile(args.weights, out / "model.safetensors")
    with safe_open(str(out / "model.safetensors"), "pt") as f:
        shape = f.get_slice("model.embed_tokens.weight").get_shape()
    if shape != [cfg["vocab_size"], cfg["hidden_size"]]:
        raise ValueError(f"unexpected embedding shape {shape}")
    shutil.copyfile(tok_path, out / "tokenizer.json")
    (out / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    # Pretraining separated documents with </s> only (no <s>), so </s> is used as the
    # prefix token (what lm-eval-harness prepends for empty contexts).
    tok_cfg = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "bos_token": "</s>",
        "eos_token": "</s>",
        "pad_token": "<pad>",
        "unk_token": "<unk>",
        "add_bos_token": False,
        "model_max_length": 2048,
        "clean_up_tokenization_spaces": False,
    }
    (out / "tokenizer_config.json").write_text(json.dumps(tok_cfg, indent=2), encoding="utf-8")
    (out / "special_tokens_map.json").write_text(
        json.dumps({k: v for k, v in tok_cfg.items() if k.endswith("_token")}, indent=2),
        encoding="utf-8",
    )
    gen = {"bos_token_id": cfg["bos_token_id"], "eos_token_id": cfg["eos_token_id"],
           "pad_token_id": cfg["pad_token_id"], "do_sample": False}
    (out / "generation_config.json").write_text(json.dumps(gen, indent=2), encoding="utf-8")
    log.info("exported to %s", out)


if __name__ == "__main__":
    main()
