"""Run the paper task suite on one model with lm-evaluation-harness.

Every model gets the same tasks, few-shot settings, seed and precision. Results and
run metadata (library versions, GPU, timing) are written to ``eval/results/<name>/``.

Usage
-----
python eval/run_all.py --model models/boccaccio-distill-hf --name boccaccio-distill
python eval/run_all.py --model Qwen/Qwen2.5-0.5B --name qwen2.5-0.5b --limit 50
"""
from __future__ import annotations

import argparse
import json
import logging
import platform
import time
from pathlib import Path

import lm_eval  # noqa: I001  (must be imported before torch: DLL load-order crash on Windows)
from lm_eval import simple_evaluate
from lm_eval.tasks import TaskManager
from lm_eval.utils import handle_non_serializable

import transformers
import torch

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
SEED = 1234


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", required=True, help="local path or Hugging Face model id")
    p.add_argument("--name", required=True, help="short name used for the results folder")
    p.add_argument("--tasks-file", type=Path, default=ROOT / "tasks.json")
    p.add_argument("--only", nargs="*", default=None, help="subset of task names to run")
    p.add_argument("--limit", type=int, default=None, help="max examples per task (debug)")
    p.add_argument("--batch-size", default="auto")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--trust-remote-code", action="store_true")
    return p.parse_args()


def main() -> None:
    """Evaluate one model on every task of the suite."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    config = json.loads(args.tasks_file.read_text(encoding="utf-8"))
    suite = config["suite"]
    include_path = str(ROOT / "tasks") if config.get("include_path") else None
    if args.only:
        suite = [t for t in suite if t["task"] in args.only]
    if not suite:
        raise ValueError("no tasks selected")

    # Debug runs with --limit go to a separate folder so they can never overwrite full results.
    out = ROOT / "results" / (args.name if args.limit is None else f"{args.name}__limit{args.limit}")
    out.mkdir(parents=True, exist_ok=True)
    model_args = f"pretrained={args.model},dtype={args.dtype}"
    if args.trust_remote_code:
        model_args += ",trust_remote_code=True"

    meta = {
        "model": args.model,
        "name": args.name,
        "seed": SEED,
        "limit": args.limit,
        "lm_eval": lm_eval.__version__,
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "platform": platform.platform(),
        "tasks": {},
    }
    meta_path = out / "run_meta.json"
    if meta_path.is_file():
        meta["tasks"] = json.loads(meta_path.read_text(encoding="utf-8")).get("tasks", {})
    for entry in suite:
        task, shots = entry["task"], entry["num_fewshot"]
        if args.limit is None and (out / f"{task}.json").is_file():
            log.info("== %s already done, skipping", task)
            continue
        log.info("== %s (%d-shot) on %s", task, shots, args.name)
        t0 = time.time()
        task_manager = TaskManager(include_path=include_path) if include_path else None
        res = simple_evaluate(
            model="hf",
            model_args=model_args,
            tasks=[task],
            task_manager=task_manager,
            num_fewshot=shots,
            limit=args.limit,
            batch_size=entry.get("batch_size", args.batch_size),
            device=args.device,
            random_seed=SEED,
            numpy_random_seed=SEED,
            torch_random_seed=SEED,
            fewshot_random_seed=SEED,
            log_samples=True,
        )
        meta["tasks"][task] = {"num_fewshot": shots, "seconds": round(time.time() - t0, 1)}
        samples = res.pop("samples")
        (out / f"{task}.json").write_text(
            json.dumps(res, indent=2, default=handle_non_serializable, ensure_ascii=False),
            encoding="utf-8",
        )
        with (out / f"{task}.samples.jsonl").open("w", encoding="utf-8") as f:
            for name, rows in samples.items():
                for row in rows:
                    f.write(json.dumps({"task": name, **row}, default=handle_non_serializable,
                                       ensure_ascii=False) + "\n")
        log.info("%s done in %.1fs", task, meta["tasks"][task]["seconds"])
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
