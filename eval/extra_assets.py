"""Extra paper macros: 8-task average without SQuAD-IT, RAG gains and benchmark category ranges.

Writes ``results_macros_extra.tex`` (``\\bres{<model>}{avg8*}``, ``\\bres{rag}{gain-*}``,
``\\bres{bench}{n-{ans,unans}-{min,max}}``) next to the macros of ``paper_assets.py``.

The 8-task average is the unweighted mean of the nine-task suite without SQuAD-IT (in-domain
for the distilled model). Differences from the reference are tested with the paired sign-flip
permutation test on the per-item differences averaged over tasks, 95% paired bootstrap CI,
Holm correction over all comparisons.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from significance import SEED, holm, load_items, perm_pvalue, resampled_means

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = Path(r"D:\Repository\Papers E Presentazioni\Paper-BoccaccioAI\generated")
EXCLUDED_TASK = "squad_it"
RAG_PAIRS = [("gemma-3-1b-pt", "gemma-3-1b-pt-rag"), ("llama-3.2-1b", "llama-3.2-1b-rag"),
             ("minerva-1b", "minerva-1b-rag"), ("qwen3-0.6b", "qwen3-0.6b-rag"),
             ("qwen2.5-0.5b", "qwen2.5-0.5b-rag")]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--results-dir", type=Path, default=ROOT / "eval" / "results")
    p.add_argument("--meta", type=Path, default=ROOT / "eval" / "model_meta.json")
    p.add_argument("--bench-dir", type=Path, default=ROOT / "benchmark")
    p.add_argument("--cost-model", type=Path, default=ROOT / "eval" / "cost_model.json")
    p.add_argument("--total-usd", type=float, default=97.0,
                   help="approximate total bill of the author (rentals + teacher API)")
    p.add_argument("--ref", default="boccaccio-distill")
    p.add_argument("--n-resamples", type=int, default=10000)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return p.parse_args()


def fmt_signed(x: float) -> str:
    """Format a difference in points with an explicit sign."""
    return f"{x:+.1f}"


def fmt_p(p: float) -> str:
    """Format a p-value as in paper_assets."""
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def avg8_macros(args: argparse.Namespace) -> list[str]:
    """Macros for the average without SQuAD-IT and its paired tests against the reference."""
    meta = json.loads(args.meta.read_text(encoding="utf-8"))
    tasks = [t for t in meta["tasks"] if t["task"] != EXCLUDED_TASK]
    names = [m["name"] for m in meta["models"]]
    rng = np.random.default_rng(SEED)
    items = {m: {t["task"]: load_items(args.results_dir, m, t["task"], t["metric"])
                 for t in tasks} for m in names}
    out, comp = [], {}
    means = {m: 100 * float(np.mean([np.mean(list(items[m][t["task"]].values()))
                                     for t in tasks])) for m in names}
    for m in names:
        out.append(f"\\expandafter\\def\\csname bres@{m}@avg8\\endcsname{{{means[m]:.1f}}}")
    for o in (m for m in names if m != args.ref):
        boot = np.zeros(args.n_resamples)
        perm = np.zeros(args.n_resamples)
        for t in tasks:
            a, b = items[args.ref][t["task"]], items[o][t["task"]]
            if a.keys() != b.keys():
                raise ValueError(f"{o}/{t['task']}: items do not match the reference")
            keys = sorted(a)
            d = np.array([b[k] - a[k] for k in keys])
            boot += resampled_means(d, args.n_resamples, rng, "bootstrap") / len(tasks)
            perm += resampled_means(d, args.n_resamples, rng, "permutation") / len(tasks)
        diff = 100 * (means[o] - means[args.ref]) / 100
        comp[o] = (diff, 100 * np.percentile(boot, 2.5), 100 * np.percentile(boot, 97.5),
                   float(perm_pvalue(diff / 100, perm)))
        log.info("%-20s avg8 %.2f diff %+.2f [%+.2f, %+.2f] p=%.4g", o, means[o], *comp[o])
    adj = dict(zip(comp, holm([v[3] for v in comp.values()])))
    for o, (diff, lo, hi, _) in comp.items():
        out += [
            f"\\expandafter\\def\\csname bres@{o}@avg8-diff\\endcsname{{{fmt_signed(diff)}}}",
            f"\\expandafter\\def\\csname bres@{o}@avg8-ci\\endcsname"
            f"{{[{fmt_signed(lo)}, {fmt_signed(hi)}]}}",
            f"\\expandafter\\def\\csname bres@{o}@avg8-gap\\endcsname{{{abs(diff):.1f}}}",
            f"\\expandafter\\def\\csname bres@{o}@avg8-p\\endcsname{{{fmt_p(adj[o])}}}",
        ]
    return out


def rag_macros(bench_dir: Path) -> list[str]:
    """Range of the accuracy gain given by the shared retrieval to the five baselines."""
    rep = json.loads((bench_dir / "judge_report.json").read_text(encoding="utf-8"))["models"]
    gains = [100 * (rep[r]["accuracy"] - rep[p]["accuracy"]) for p, r in RAG_PAIRS]
    return [f"\\expandafter\\def\\csname bres@rag@gain-min\\endcsname{{{round(min(gains))}}}",
            f"\\expandafter\\def\\csname bres@rag@gain-max\\endcsname{{{round(max(gains))}}}"]


def category_macros(bench_dir: Path) -> list[str]:
    """Min and max items per category of the scored benchmark, answerable and not."""
    acc = [json.loads(x) for x in (bench_dir / "benchmark_v2.jsonl").read_text("utf-8")
           .splitlines() if x.strip()]
    excl = {e["id"] for e in json.loads((bench_dir / "excluded.json").read_text("utf-8"))["items"]}
    out = []
    for answerable, tag in ((True, "ans"), (False, "unans")):
        cats: dict[str, int] = {}
        for r in acc:
            if r["id"] not in excl and r["answerable"] == answerable:
                cats[r["category"]] = cats.get(r["category"], 0) + 1
        out += [f"\\expandafter\\def\\csname bres@bench@n-{tag}-min\\endcsname{{{min(cats.values())}}}",
                f"\\expandafter\\def\\csname bres@bench@n-{tag}-max\\endcsname{{{max(cats.values())}}}"]
    return out


def cost_rate_macros(cost_path: Path, total_usd: float) -> list[str]:
    """Blended hourly rate implied by the author's total bill (rentals = total - teacher API)."""
    cm = json.loads(cost_path.read_text(encoding="utf-8"))
    by_key = {s["key"]: s for s in cm["stages"]}
    p1 = by_key["p1"]
    hours, api = 0.0, 0.0
    for s in cm["stages"]:
        if s.get("usd") is not None:
            api += s["usd"]
        elif s.get("gpu", True):
            if "hours" in s:
                hours += s["hours"]
            elif "hours_from_throughput_of" in s:
                hours += s["steps"] * p1["hours"] / p1["steps"]
            else:
                hours += s["hours_full"] * s["steps"] / s["steps_full"]
    rental = total_usd - api
    return [f"\\expandafter\\def\\csname bres@cost@total-usd\\endcsname{{{total_usd:.0f}}}",
            f"\\expandafter\\def\\csname bres@cost@rental-usd\\endcsname{{{rental:.0f}}}",
            f"\\expandafter\\def\\csname bres@cost@rate-usd\\endcsname{{{rental / hours:.1f}}}"]


def main() -> None:
    """Entry point."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    lines = ["% Generated by boccaccioAI-validation/eval/extra_assets.py -- do not edit by hand."]
    lines += avg8_macros(args) + rag_macros(args.bench_dir) + category_macros(args.bench_dir)
    lines += cost_rate_macros(args.cost_model, args.total_usd)
    out = args.out_dir / "results_macros_extra.tex"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Written {out}")


if __name__ == "__main__":
    main()
