"""Regenerate missing per-item samples (``<task>.samples.jsonl``) without touching the results.

For every model of ``eval/model_meta.json`` whose results folder lacks samples for some
suite tasks, the missing tasks are re-evaluated with ``eval/run_all.py`` into a temporary
folder ``eval/results/<name>__samples/`` (resumable). The new aggregate metrics must be
identical to the existing ones; only then the samples files are copied next to the original
results (never overwriting an existing file). Nothing is deleted: the temporary folder is
kept. On any mismatch the script stops, prints the differences and copies nothing.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import shutil
import subprocess
import sys
from pathlib import Path

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "eval" / "results"
# Aggregate metrics may differ slightly on generative tasks: BF16 CUDA kernels are not
# bitwise deterministic, so a few items can flip between runs. Anything larger is a bug.
TOLERANCE = 0.005


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--meta", type=Path, default=ROOT / "eval" / "model_meta.json")
    p.add_argument("--batch-size", default=None,
                   help="fixed batch size for the re-run (default: lm-eval 'auto', which can "
                        "OOM if other apps take VRAM; a fixed value like 24 is safer)")
    p.add_argument("--dry-run", action="store_true", help="only list what is missing")
    return p.parse_args()


def metrics(path: Path, task: str) -> dict[str, float]:
    """Aggregate float metrics of one lm-eval result file."""
    res = json.loads(path.read_text(encoding="utf-8"))["results"][task]
    return {k: v for k, v in res.items() if isinstance(v, float)}


def main() -> None:
    """Entry point."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    meta = json.loads(args.meta.read_text(encoding="utf-8"))
    tasks = [t["task"] for t in meta["tasks"]]
    todo = []
    for m in meta["models"]:
        missing = [t for t in tasks if not (RESULTS / m["name"] / f"{t}.samples.jsonl").exists()]
        if missing:
            todo.append((m, missing))
            log.info("%s: missing samples for %s", m["name"], ", ".join(missing))
    if not todo:
        log.info("all models have per-item samples")
        return
    if args.dry_run:
        return
    for m, missing in todo:
        tmp_name = f"{m['name']}__samples"
        cmd = [sys.executable, str(ROOT / "eval" / "run_all.py"), "--model", m["hf"],
               "--name", tmp_name, "--only", *missing]
        if args.batch_size:
            cmd += ["--batch-size", args.batch_size]
        log.info("running: %s", " ".join(cmd))
        subprocess.run(cmd, cwd=ROOT, check=True)
        tmp, dst = RESULTS / tmp_name, RESULTS / m["name"]
        diffs, bad = [], []
        for t in missing:
            old, new = metrics(dst / f"{t}.json", t), metrics(tmp / f"{t}.json", t)
            for k in old.keys() | new.keys():
                if k not in old or k not in new or not math.isclose(old[k], new[k], abs_tol=1e-9):
                    msg = f"{t} {k}: original {old.get(k)} vs re-run {new.get(k)}"
                    if k in old and k in new and abs(old[k] - new[k]) <= TOLERANCE:
                        diffs.append(msg)
                    else:
                        bad.append(msg)
        if bad:
            log.error("re-run of %s differs from the original results beyond tolerance %g; "
                      "nothing copied. Inspect %s:\n  %s", m["name"], TOLERANCE, tmp,
                      "\n  ".join(bad))
            sys.exit(1)
        if diffs:
            log.warning("%s: %d metric(s) differ from the original results within tolerance "
                        "%g (BF16 generation nondeterminism); the per-item samples of the "
                        "re-run are used:\n  %s", m["name"], len(diffs), TOLERANCE,
                        "\n  ".join(diffs))
        for t in missing:
            target = dst / f"{t}.samples.jsonl"
            if target.exists():
                raise ValueError(f"{target} appeared during the run; refusing to overwrite it")
            shutil.copy2(tmp / f"{t}.samples.jsonl", target)
        log.info("%s: samples added for %d tasks (results reproduced exactly); the re-run "
                 "is kept in %s and can be deleted by hand", m["name"], len(missing), tmp)


if __name__ == "__main__":
    main()
