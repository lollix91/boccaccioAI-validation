"""Paired significance tests between a reference model and every other model.

For each task the per-item scores are read from ``eval/results/<model>/<task>.samples.jsonl``
and paired by (sub-task, doc_id). Tests:

* binary metrics (accuracy, containment): exact McNemar test (binomial on discordant pairs);
* continuous metrics (TruthfulQA mc2) and the 9-task average: paired sign-flip permutation
  test (each item's difference is multiplied by a random +-1);
* 95% paired bootstrap confidence interval of the difference for every comparison;
* Holm-Bonferroni correction over all comparisons.

Output: ``eval/results/significance.json``.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
SEED = 1234
CHUNK = 500


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--results-dir", type=Path, default=ROOT / "eval" / "results")
    p.add_argument("--meta", type=Path, default=ROOT / "eval" / "model_meta.json")
    p.add_argument("--ref", default="boccaccio-distill", help="reference model name")
    p.add_argument("--models", nargs="*", default=None, help="restrict the comparisons")
    p.add_argument("--n-resamples", type=int, default=10000)
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def load_items(results_dir: Path, model: str, task: str, metric: str) -> dict[tuple, float]:
    """Read per-item scores of one model on one task.

    Parameters
    ----------
    results_dir : Path
        Results root.
    model : str
        Model folder name.
    task : str
        Suite task name (file stem).
    metric : str
        Per-item metric key.

    Returns
    -------
    dict[tuple, float]
        ``{(subtask, doc_id): score}``.

    Raises
    ------
    ValueError
        If the samples file is missing or its mean disagrees with the aggregate JSON.
    """
    path = results_dir / model / f"{task}.samples.jsonl"
    if not path.exists():
        raise ValueError(f"missing per-item samples {path}")
    items = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            items[(row["task"], row["doc_id"])] = float(row[metric])
    agg = json.loads((results_dir / model / f"{task}.json").read_text(encoding="utf-8"))
    expected = agg["results"][task][f"{metric},none"]
    observed = float(np.mean(list(items.values())))
    if not math.isclose(observed, expected, abs_tol=1e-9):
        if abs(observed - expected) > 0.005:
            raise ValueError(f"{path}: per-item mean {observed} differs from aggregate "
                             f"{expected} beyond tolerance")
        log.warning("%s/%s: per-item mean %.6f differs from aggregate %.6f "
                    "(BF16 nondeterminism across runs)", model, task, observed, expected)
    return items


def resampled_means(d: np.ndarray, n_resamples: int, rng: np.random.Generator,
                    kind: str) -> np.ndarray:
    """Means of ``d`` under bootstrap resampling or sign-flip permutation.

    Parameters
    ----------
    d : np.ndarray
        Paired per-item differences.
    n_resamples : int
        Number of resamples.
    rng : np.random.Generator
        Random generator.
    kind : str
        ``"bootstrap"`` or ``"permutation"``.

    Returns
    -------
    np.ndarray
        Array of ``n_resamples`` resampled means.
    """
    out = np.empty(n_resamples)
    n = len(d)
    for s in range(0, n_resamples, CHUNK):
        b = min(CHUNK, n_resamples - s)
        if kind == "bootstrap":
            out[s:s + b] = d[rng.integers(0, n, size=(b, n))].mean(axis=1)
        else:
            out[s:s + b] = (d * rng.choice([-1.0, 1.0], size=(b, n))).mean(axis=1)
    return out


def perm_pvalue(observed: float, null: np.ndarray) -> float:
    """Two-sided permutation p-value with the +1 correction."""
    return (1 + np.sum(np.abs(null) >= abs(observed) - 1e-12)) / (len(null) + 1)


def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (same order as input)."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj, running = [0.0] * m, 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj


def main() -> None:
    """Entry point."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    meta = json.loads(args.meta.read_text(encoding="utf-8"))
    tasks = meta["tasks"]
    others = [m["name"] for m in meta["models"]
              if m["name"] != args.ref and (not args.models or m["name"] in args.models)]
    rng = np.random.default_rng(SEED)
    ref_items = {t["task"]: load_items(args.results_dir, args.ref, t["task"], t["metric"])
                 for t in tasks}
    comparisons: dict[str, dict[str, dict]] = {}
    flat: list[tuple[str, str]] = []
    for other in others:
        comparisons[other] = {}
        boot_avg = np.zeros(args.n_resamples)
        perm_avg = np.zeros(args.n_resamples)
        diffs = []
        for t in tasks:
            a = ref_items[t["task"]]
            b = load_items(args.results_dir, other, t["task"], t["metric"])
            if a.keys() != b.keys():
                raise ValueError(f"{other}/{t['task']}: items do not match the reference")
            keys = sorted(a)
            x = np.array([a[k] for k in keys])
            y = np.array([b[k] for k in keys])
            d = y - x
            diff = float(d.mean())
            boot = resampled_means(d, args.n_resamples, rng, "bootstrap")
            perm = resampled_means(d, args.n_resamples, rng, "permutation")
            binary = set(np.unique(np.concatenate([x, y]))) <= {0.0, 1.0}
            if binary:
                n01 = int(np.sum((x == 1) & (y == 0)))
                n10 = int(np.sum((x == 0) & (y == 1)))
                p = binomtest(min(n01, n10), n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
                test = "mcnemar_exact"
                extra = {"ref_only": n01, "other_only": n10}
            else:
                p, test, extra = perm_pvalue(diff, perm), "paired_permutation", {}
            comparisons[other][t["task"]] = {
                "diff": diff, "ci_low": float(np.percentile(boot, 2.5)),
                "ci_high": float(np.percentile(boot, 97.5)), "p": float(p), "test": test,
                "n": len(keys), **extra,
            }
            boot_avg += boot / len(tasks)
            perm_avg += perm / len(tasks)
            diffs.append(diff)
            flat.append((other, t["task"]))
        avg = sum(diffs) / len(diffs)
        comparisons[other]["avg"] = {
            "diff": avg, "ci_low": float(np.percentile(boot_avg, 2.5)),
            "ci_high": float(np.percentile(boot_avg, 97.5)),
            "p": float(perm_pvalue(avg, perm_avg)), "test": "paired_permutation",
        }
        flat.append((other, "avg"))
        log.info("%-20s avg diff %+.4f [%+.4f, %+.4f] p=%.4g", other, avg,
                 comparisons[other]["avg"]["ci_low"], comparisons[other]["avg"]["ci_high"],
                 comparisons[other]["avg"]["p"])
    adjusted = holm([comparisons[o][k]["p"] for o, k in flat])
    for (o, k), p in zip(flat, adjusted):
        comparisons[o][k]["p_holm"] = p
    out = args.out or args.results_dir / "significance.json"
    if out.exists():
        backup = out.with_name(f"{out.stem}.{time.strftime('%Y%m%d-%H%M%S')}.json")
        out.rename(backup)
        log.info("previous results kept as %s", backup)
    out.write_text(json.dumps({
        "reference": args.ref, "n_resamples": args.n_resamples, "seed": SEED,
        "correction": "holm over all comparisons", "comparisons": comparisons,
    }, indent=2), encoding="utf-8")
    print(f"Written {out}")


if __name__ == "__main__":
    main()
