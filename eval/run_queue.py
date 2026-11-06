"""Run the evaluation suite for the first unfinished model of the queue, then stop.

Reads ``eval/models.json``; a model counts as done when ``eval/results/<name>/``
contains ``run_meta.json`` plus one result file per task of ``eval/tasks.json``.
Invocation is intentionally one model at a time: run this script again (or schedule it)
to proceed to the next model. Models marked ``"gated"`` are skipped when ``HF_TOKEN``
is not set; skips are recorded in ``eval/results/<name>/SKIPPED``.

Usage
-----
python eval/run_queue.py              # evaluate the next model and stop
python eval/run_queue.py --status     # only print the queue status
python eval/run_all.py ...            # extra args are forwarded to run_all.py
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def queue() -> list[dict]:
    """Read the model queue.

    Returns
    -------
    list[dict]
        Queue entries in evaluation order.
    """
    return json.loads((ROOT / "models.json").read_text(encoding="utf-8"))["queue"]


def is_done(entry: dict, n_tasks: int) -> bool:
    """Check whether a model already has complete results.

    Parameters
    ----------
    entry : dict
        Queue entry.
    n_tasks : int
        Number of tasks in the suite.

    Returns
    -------
    bool
        True if all result files exist.
    """
    out = RESULTS / entry["name"]
    if not (out / "run_meta.json").is_file():
        return False
    return len(list(out.glob("*.json"))) - 1 >= n_tasks


def main() -> None:
    """Evaluate the first unfinished model, or print the queue status."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--status", action="store_true")
    p.add_argument("--extra-args", nargs=argparse.REMAINDER, default=[])
    args = p.parse_args()

    entries = queue()
    n_tasks = len(json.loads((ROOT / "tasks.json").read_text(encoding="utf-8"))["suite"])
    for e in entries:
        skipped = (RESULTS / e["name"] / "SKIPPED").is_file()
        failed = (RESULTS / e["name"] / "FAILED").is_file()
        state = ("DONE" if is_done(e, n_tasks) else "SKIPPED" if skipped
                 else "FAILED" if failed else "TODO")
        log.info("%-22s %s", e["name"], state)
    if args.status:
        return

    nxt = next((e for e in entries if not is_done(e, n_tasks)
                and not (RESULTS / e["name"] / "SKIPPED").is_file()
                and not (RESULTS / e["name"] / "FAILED").is_file()), None)
    if nxt is None:
        log.info("queue complete")
        raise SystemExit(7)
    if nxt.get("gated") and not os.environ.get("HF_TOKEN"):
        out = RESULTS / nxt["name"]
        out.mkdir(parents=True, exist_ok=True)
        (out / "SKIPPED").write_text(
            "gated model: HF_TOKEN not set\n", encoding="utf-8")
        log.warning("skipping %s (gated, HF_TOKEN missing)", nxt["name"])
        return

    cmd = [sys.executable, str(ROOT / "run_all.py"),
           "--model", nxt["model"], "--name", nxt["name"]]
    if nxt.get("trust_remote_code"):
        cmd.append("--trust-remote-code")
    cmd += args.extra_args
    log.info("running: %s", " ".join(cmd))
    t0 = time.time()
    rc = subprocess.call(cmd)
    log.info("finished %s (rc=%d) in %.0fs", nxt["name"], rc, time.time() - t0)
    if rc != 0:
        out = RESULTS / nxt["name"]
        out.mkdir(parents=True, exist_ok=True)
        (out / "FAILED").write_text(f"run_all.py exited {rc}\n", encoding="utf-8")
        log.warning("marked %s as FAILED; continuing with the next model on re-run",
                    nxt["name"])
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
