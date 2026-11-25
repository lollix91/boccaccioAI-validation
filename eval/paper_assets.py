"""Generate the paper tables, figures and LaTeX macros from the evaluation results.

Every number that ends up in the manuscript is read from ``eval/results/<model>/*.json``
(lm-eval-harness output) or computed from the model configs; static metadata (display
names, declared pretraining tokens and their sources) lives in ``eval/model_meta.json``.

Outputs (in ``--out-dir``):
    tab_benchmarks.tex   12 models x 9 tasks matrix with standard errors
    tab_cost_tokens.tex  parameters, pretraining tokens, 6ND compute, average score
    tab_fertility.tex    tokenizer fertility on the held-out corpus
    tab_rag.tex          benchmark v2: accuracy, refusals, abstention, RAG (all 18 systems)
    tab_cost.tex         estimated training cost per stage (eval/cost_model.json)
    results_macros.tex   ``\\bres{<model>}{<key>}`` macros for numbers quoted in the text
    fig_stage_ablation.pdf, fig_score_vs_tokens.pdf
    summary.json         machine-readable dump of everything above
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = Path(r"D:\Repository\Papers E Presentazioni\Paper-BoccaccioAI\generated")
Z95 = 1.959964
BPB_DATASET = "wiki-it-2025"
SIG_ALPHA = 0.05
GROUP_STYLE = {
    "ours": {"color": "#d62728", "marker": "*", "size": 220},
    "italian": {"color": "#2ca02c", "marker": "s", "size": 70},
    "multilingual": {"color": "#1f77b4", "marker": "o", "size": 70},
    "english": {"color": "#7f7f7f", "marker": "^", "size": 70},
}
STAGE_COLORS = ["#c7c7c7", "#6baed6", "#d62728"]
# (row label, parametric system, RAG system) for tab:rag; BoccaccioAI rows are the
# production endpoint, whose parametric answers equal boccaccio-distill-chat 368/368
RAG_TABLE = [
    ("\\textbf{BoccaccioAI (distilled)}", "prod-norag", "prod-rag"),
    ("Gemma-3-1B", "gemma-3-1b-pt", "gemma-3-1b-pt-rag"),
    ("Llama-3.2-1B", "llama-3.2-1b", "llama-3.2-1b-rag"),
    ("Minerva-1B", "minerva-1b", "minerva-1b-rag"),
    ("Qwen3-0.6B$^{\\ast}$", "qwen3-0.6b", "qwen3-0.6b-rag"),
    ("Qwen2.5-0.5B", "qwen2.5-0.5b", "qwen2.5-0.5b-rag"),
]
# parametric-only rows of tab:rag (no RAG counterpart was run)
PARAM_ONLY_TABLE = [
    ("BoccaccioAI base (stage 2)", "boccaccio-base-p2"),
    ("BoccaccioAI SFT V1", "boccaccio-sft-v1-chat"),
    ("Minerva-350M", "minerva-350m"),
    ("SmolLM2-360M", "smollm2-360m"),
    ("Pythia-1B", "pythia-1b"),
    ("Pythia-410M", "pythia-410m"),
]
# BoccaccioAI stages on benchmark v2 (each in its training format)
QA_STAGES = [("Base", "boccaccio-base-p2"), ("SFT V1", "boccaccio-sft-v1-chat"),
             ("Distilled", "prod-norag")]
COST_MODEL = ROOT / "eval" / "cost_model.json"
LABEL_OFFSET = {
    "qwen2.5-0.5b": (-6, 4, "right"),
    "qwen3-0.6b": (-6, 4, "right"),
    "llama-3.2-1b": (6, -9, "left"),
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--results-dir", type=Path, default=ROOT / "eval" / "results")
    p.add_argument("--meta", type=Path, default=ROOT / "eval" / "model_meta.json")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return p.parse_args()


def load_scores(results_dir: Path, meta: dict) -> dict[str, dict[str, dict[str, float]]]:
    """Read every ``<metric>,none`` value and stderr for the suite tasks.

    Parameters
    ----------
    results_dir : Path
        Directory with one sub-directory per model.
    meta : dict
        Parsed ``model_meta.json``.

    Returns
    -------
    dict
        ``scores[model][task][metric]`` and ``scores[model][task][metric + '_se']``.

    Raises
    ------
    ValueError
        If a result file or metric is missing.
    """
    scores: dict[str, dict[str, dict[str, float]]] = {}
    for m in meta["models"]:
        scores[m["name"]] = {}
        for t in meta["tasks"]:
            path = results_dir / m["name"] / f"{t['task']}.json"
            if not path.exists():
                raise ValueError(f"missing result file {path}")
            res = json.loads(path.read_text(encoding="utf-8"))["results"][t["task"]]
            vals = {}
            for key, v in res.items():
                if not isinstance(v, float) or "," not in key:
                    continue
                metric = key.split(",")[0]
                if metric.endswith("_stderr"):
                    vals[metric.removesuffix("_stderr") + "_se"] = v
                else:
                    vals[metric] = v
            if t["metric"] not in vals or t["metric"] + "_se" not in vals:
                raise ValueError(f"{path}: metric {t['metric']} or its stderr not found")
            scores[m["name"]][t["task"]] = vals
    return scores


def count_params(hf_id: str) -> tuple[int, int]:
    """Count total and embedding parameters by instantiating the config on the meta device.

    Parameters
    ----------
    hf_id : str
        Hub id or path relative to the repository root.

    Returns
    -------
    tuple[int, int]
        (unique parameters, input + untied output embedding parameters).
    """
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    local = ROOT / hf_id
    cfg = AutoConfig.from_pretrained(str(local) if local.exists() else hf_id)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg)
    total = sum(p.numel() for p in model.parameters())
    inp = model.get_input_embeddings().weight
    emb = inp.numel()
    out = model.get_output_embeddings()
    if out is not None and out.weight is not inp:
        emb += out.weight.numel()
    return total, emb


def fmt_pct(x: float, digits: int = 1) -> str:
    """Format a fraction as a percentage string."""
    return f"{100 * x:.{digits}f}"


def fmt_count(n: float) -> str:
    """Format a large count as e.g. ``10.8B`` / ``18T`` / ``703M``."""
    for div, suf in ((1e12, "T"), (1e9, "B"), (1e6, "M")):
        if n >= 0.9995 * div:
            v = n / div
            return f"{v:.0f}{suf}" if v >= 100 or v == int(v) else f"{v:.1f}{suf}"
    return f"{n:.0f}"


def fmt_ratio(x: float) -> str:
    """Format a token ratio with a LaTeX-safe thousands separator."""
    return f"{x:,.0f}".replace(",", "{,}") if x >= 10 else f"{x:.1f}"


def fmt_p(p: float) -> str:
    """Format a p-value for the text (``<0.001`` below one in a thousand)."""
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def fmt_sci(x: float) -> str:
    """Format a float in LaTeX scientific notation with two significant digits."""
    e = int(math.floor(math.log10(x)))
    return f"${x / 10**e:.1f}\\times10^{{{e}}}$"


def build_summary(meta: dict, scores: dict, results_dir: Path) -> dict:
    """Compute averages, ranks, token ratios, parameter counts; attach tests and BPB if present."""
    tasks = meta["tasks"]
    ref = next(m for m in meta["models"] if m["name"] == "boccaccio-distill")
    rows = []
    for m in meta["models"]:
        s = scores[m["name"]]
        vals = [s[t["task"]][t["metric"]] for t in tasks]
        ses = [s[t["task"]][t["metric"] + "_se"] for t in tasks]
        total, emb = count_params(m["hf"])
        rows.append({
            **m,
            "scores": {t["task"]: v for t, v in zip(tasks, vals)},
            "stderr": {t["task"]: v for t, v in zip(tasks, ses)},
            "avg": sum(vals) / len(vals),
            "avg_se": math.sqrt(sum(x * x for x in ses)) / len(ses),
            "params": total,
            "params_nonemb": total - emb,
            "compute_6nd": 6 * total * m["pretrain_tokens"],
            "token_ratio": m["pretrain_tokens"] / ref["pretrain_tokens"],
            "extra": {
                "squad_it_em": s["squad_it"]["exact_match"],
                "squad_it_f1": s["squad_it"]["f1"],
                "lambada_ppl": s["lambada_openai_mt_it"]["perplexity"],
            },
        })
    order = sorted(rows, key=lambda r: -r["avg"])
    for i, r in enumerate(order, 1):
        r["rank"] = i
    sig_path = results_dir / "significance.json"
    sig = json.loads(sig_path.read_text(encoding="utf-8")) if sig_path.exists() else None
    if sig and sig["reference"] != ref["name"]:
        raise ValueError(f"{sig_path} has reference {sig['reference']}, expected {ref['name']}")
    for r in rows:
        if sig:
            r["sig"] = sig["comparisons"].get(r["name"])
        bpb = results_dir / r["name"] / f"bpb_{BPB_DATASET}.json"
        if bpb.exists():
            r["bpb"] = json.loads(bpb.read_text(encoding="utf-8"))
    fert_path = results_dir / "fertility_wiki-it-2025.json"
    fert = json.loads(fert_path.read_text(encoding="utf-8")) if fert_path.exists() else None
    return {"tasks": tasks, "rows": rows, "groups": meta["groups"],
            "ours_cost_usd": meta["ours_cost_usd"], "significance": sig is not None,
            "bpb": all("bpb" in r for r in rows), "fertility": fert}


def table_fertility(results_dir: Path) -> str | None:
    """Render the fertility table on the held-out corpus, or None if not computed yet."""
    path = results_dir / "fertility_wiki-it-2025.json"
    if not path.exists():
        return None
    fert = json.loads(path.read_text(encoding="utf-8"))
    lines = [
        "% Generated by boccaccioAI-validation/eval/paper_assets.py -- do not edit by hand.",
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{Tokenizer fertility $\\phi$ (tokens per whitespace word, lower is better) "
        f"and UTF-8 bytes per token on {fert['docs']:,} held-out Italian Wikipedia articles "
        f"created between June 2025 and September 2026 (Section~\\ref{{sec:eval}}). "
        f"The column ``vs.\\ ours'' gives the ratio of tokens per word relative to the BoccaccioAI tokenizer (1 = same density).}}",
        "\\label{tab:fertility}",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Tokenizer & Vocab & $\\phi$ & B/token & vs.\\ ours \\\\",
        "\\midrule",
    ]
    merged: list[dict] = []
    for e in fert["entries"]:
        if merged and math.isclose(e["fertility"], merged[-1]["fertility"], abs_tol=1e-9):
            merged[-1]["label"] += " / " + e["label"]
            continue
        merged.append(dict(e))
    for e in merged:
        name = f"\\textbf{{{e['label']}}}" if e["label"].startswith("Boccaccio") else e["label"]
        rel = "1" if e["label"].startswith("Boccaccio") else f"{e['relative']:.2f}$\\times$"
        lines.append(" & ".join([
            name, f"{e['vocab']:,}".replace(",", "{,}"), f"{e['fertility']:.3f}",
            f"{e['bytes_per_token']:.3f}", rel,
        ]) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


def table_benchmarks(summary: dict) -> str:
    """Render the model x task matrix as a LaTeX table."""
    tasks, rows = summary["tasks"], summary["rows"]
    keys = [t["task"] for t in tasks] + ["avg"]

    def val(r: dict, k: str) -> float:
        return r["avg"] if k == "avg" else r["scores"][k]

    def se(r: dict, k: str) -> float:
        return r["avg_se"] if k == "avg" else r["stderr"][k]

    best = {k: max(val(r, k) for r in rows) for k in keys}
    native = [r for r in rows if r["group"] in ("ours", "italian")]
    best_nat = {k: max(val(r, k) for r in native) for k in keys}
    head = " & ".join(["Model"] + [t["label"] for t in tasks] + ["Avg."])
    lines = [
        "% Generated by boccaccioAI-validation/eval/paper_assets.py -- do not edit by hand.",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Accuracy (\\%) on the Italian benchmark suite (lm-evaluation-harness 0.4.13, "
        "zero-shot except MMLU 5-shot; ARC, HellaSwag and Belebele use length-normalised "
        "accuracy, SQuAD-IT gold-answer containment on a fixed 2000-item test subset). "
        "Subscripts are standard errors; Avg.\\ is the unweighted mean of the nine columns. "
        "\\textbf{Bold}: best overall; \\underline{underlined}: best Italian-native model."
        + (" $^{\\dagger}$: not significantly different from the distilled BoccaccioAI "
           f"(paired exact McNemar or permutation test, Holm-corrected $p\\geq{SIG_ALPHA}$)."
           if summary["significance"] else "") + "}",
        "\\label{tab:benchmarks}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{l" + "c" * len(keys) + "}",
        "\\toprule",
        head + " \\\\",
    ]
    for g, gname in summary["groups"].items():
        grp = [r for r in rows if r["group"] == g]
        if not grp:
            continue
        lines.append("\\midrule")
        lines.append(f"\\multicolumn{{{len(keys) + 1}}}{{l}}{{\\emph{{{gname}}}}} \\\\")
        for r in grp:
            cells = []
            for k in keys:
                v = val(r, k)
                c = fmt_pct(v)
                if r["group"] in ("ours", "italian") and math.isclose(v, best_nat[k]):
                    c = f"\\underline{{{c}}}"
                if math.isclose(v, best[k]):
                    c = f"\\textbf{{{c}}}"
                dag = "^{\\dagger}" if r.get("sig") and r["sig"][k]["p_holm"] >= SIG_ALPHA else ""
                cells.append(f"{c}${dag}_{{\\pm{fmt_pct(se(r, k))}}}$")
            label = r.get("table_label", r["display"])
            lines.append(" & ".join([label] + cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}}", "\\end{table}", ""]
    return "\n".join(lines)


def table_cost_tokens(summary: dict) -> str:
    """Render parameters, pretraining tokens, compute and average score as a LaTeX table."""
    rows = [r for r in summary["rows"] if r["group"] != "ours" or r["name"] == "boccaccio-distill"]
    rows.sort(key=lambda r: r["pretrain_tokens"])
    lines = [
        "% Generated by boccaccioAI-validation/eval/paper_assets.py -- do not edit by hand.",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Model size, declared pretraining tokens $D$ (post-training excluded), "
        "pretraining compute $C\\approx6ND$ with $N$ the total parameter count, and average "
        "score of Table~\\ref{tab:benchmarks}. Token counts are taken from the model cards "
        "and technical reports; parameters are counted from the released configurations."
        + (" BPB: bits per UTF-8 byte on Italian Wikipedia articles created after the release "
           "of every model (lower is better); B/tok: UTF-8 bytes per token of each tokenizer "
           "on the same text." if summary["bpb"] else "") + "}",
        "\\label{tab:cost_tokens}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{lrrrrrc" + ("cc" if summary["bpb"] else "") + "}",
        "\\toprule",
        "Model & $N$ & $N_{\\text{non-emb}}$ & $D$ & $D/D_{\\text{ours}}$ & $C$ (FLOPs) "
        "& Avg.\\ (\\%)" + (" & BPB & B/tok" if summary["bpb"] else "") + " \\\\",
        "\\midrule",
    ]
    best_bpb = min(r["bpb"]["bits_per_byte"] for r in rows) if summary["bpb"] else None
    for r in rows:
        name = f"\\textbf{{{r['display']}}}" if r["group"] == "ours" else r["display"]
        ratio = "1" if r["group"] == "ours" else f"{fmt_ratio(r['token_ratio'])}$\\times$"
        cells = [
            name, fmt_count(r["params"]), fmt_count(r["params_nonemb"]),
            fmt_count(r["pretrain_tokens"]), ratio, fmt_sci(r["compute_6nd"]),
            f"{fmt_pct(r['avg'])}$\\pm${fmt_pct(r['avg_se'])}",
        ]
        if summary["bpb"]:
            b = f"{r['bpb']['bits_per_byte']:.3f}"
            if math.isclose(r["bpb"]["bits_per_byte"], best_bpb):
                b = f"\\textbf{{{b}}}"
            cells += [b, f"{r['bpb']['bytes_per_token']:.2f}"]
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}}", "\\end{table}", ""]
    return "\n".join(lines)


def macros(summary: dict) -> str:
    """Render ``\\bres{model}{key}`` macros for every number quoted in the text."""
    out = [
        "% Generated by boccaccioAI-validation/eval/paper_assets.py -- do not edit by hand.",
        "% Usage: \\bres{boccaccio-distill}{squad_it}, \\bres{minerva-1b}{avg}, ...",
        "\\providecommand{\\bres}[2]{\\csname bres@#1@#2\\endcsname}",
    ]

    def d(model: str, key: str, value: str) -> None:
        out.append(f"\\expandafter\\def\\csname bres@{model}@{key}\\endcsname{{{value}}}")

    for r in summary["rows"]:
        n = r["name"]
        for t in summary["tasks"]:
            d(n, t["task"], fmt_pct(r["scores"][t["task"]]))
            d(n, t["task"] + "-se", fmt_pct(r["stderr"][t["task"]]))
        d(n, "avg", fmt_pct(r["avg"]))
        d(n, "avg-se", fmt_pct(r["avg_se"]))
        d(n, "rank", str(r["rank"]))
        d(n, "tokens", fmt_count(r["pretrain_tokens"]))
        d(n, "tokratio", fmt_ratio(r["token_ratio"]))
        d(n, "params", fmt_count(r["params"]))
        d(n, "squad_it-em", fmt_pct(r["extra"]["squad_it_em"]))
        d(n, "squad_it-f1", fmt_pct(r["extra"]["squad_it_f1"]))
        d(n, "lambada-ppl", f"{r['extra']['lambada_ppl']:.0f}")
        for k, c in (r.get("sig") or {}).items():
            d(n, f"{k}-diff", f"{100 * c['diff']:+.1f}")
            d(n, f"{k}-ci", f"[{100 * c['ci_low']:+.1f}, {100 * c['ci_high']:+.1f}]")
            d(n, f"{k}-p", fmt_p(c["p_holm"]))
        if "bpb" in r:
            d(n, "bpb", f"{r['bpb']['bits_per_byte']:.3f}")
            d(n, "bytes-per-token", f"{r['bpb']['bytes_per_token']:.2f}")
    fert = summary.get("fertility")
    if fert:
        d("fert", "docs", f"{fert['docs']:,}".replace(",", "{,}"))
        for e in fert["entries"]:
            slug = re.sub(r"[^a-z0-9]+", "-", e["label"].lower()).strip("-")
            for key in ("fertility", "relative", "bytes_per_token"):
                d("fert", f"{slug}-{key.replace('_', '-')}",
                  f"{e[key]:.3f}" if key != "relative" else f"{e[key]:.2f}")
    d("global", "nmodels", str(len(summary["rows"])))
    d("global", "ntasks", str(len(summary["tasks"])))
    return "\n".join(out) + "\n"


def load_judge(report_path: Path, human_path: Path) -> dict | None:
    """Read the LLM-judge report and the human-validation summary, if present."""
    if not report_path.exists():
        return None
    rep = json.loads(report_path.read_text(encoding="utf-8"))
    if "scoring" not in rep:
        raise ValueError(f"{report_path} predates the fixed report: rerun judge.py report")
    rep["human"] = (json.loads(human_path.read_text(encoding="utf-8"))
                    if human_path.exists() else None)
    checks = ROOT / "benchmark" / "checks" / "checks.json"
    rep["sensitivity"] = (json.loads(checks.read_text(encoding="utf-8")).get("sensitivity", {})
                          if checks.exists() else {})
    return rep


def table_rag(judge: dict) -> str | None:
    """Render benchmark v2 for every scored system: parametric accuracy, refusals on
    answerable items, abstention on unanswerable items and, where run, the RAG score with
    the shared production retrieval."""
    models = judge["models"]
    rows = [r for r in RAG_TABLE if r[1] in models and r[2] in models]
    solo = [r for r in PARAM_ONLY_TABLE if r[1] in models]
    if len(rows) < len(RAG_TABLE) or len(solo) < len(PARAM_ONLY_TABLE):
        log.warning("RAG table: missing systems, skipped")
        return None
    sc = judge["scoring"]
    gate = models[RAG_TABLE[0][2]]["abstention_rate"]

    def acc(s: str, best: float) -> str:
        m = models[s]
        v = fmt_pct(m["accuracy"])
        if math.isclose(m["accuracy"], best):
            v = f"\\textbf{{{v}}}"
        return (f"{v}$_{{[{fmt_pct(m['accuracy_ci'][0])},"
                f"{fmt_pct(m['accuracy_ci'][1])}]}}$")

    def best_of(xs: list[float]) -> float:
        return max(xs)

    par_all = [r[1] for r in rows] + [r[1] for r in solo]
    best_par = best_of([models[s]["accuracy"] for s in par_all])
    best_rag = best_of([models[r[2]]["accuracy"] for r in rows])
    best_abst = best_of([models[s]["abstention_rate"] for s in par_all])

    def abst(s: str) -> str:
        v = fmt_pct(models[s]["abstention_rate"])
        return f"\\textbf{{{v}}}" if math.isclose(models[s]["abstention_rate"], best_abst) else v

    lines = [
        "% Generated by boccaccioAI-validation/eval/paper_assets.py -- do not edit by hand.",
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{Factual QA on benchmark v2 ({sc['n_answerable']} answerable and "
        f"{sc['n_unanswerable']} unanswerable Italian questions), judged by two LLMs "
        "(answerable items: correct 1, partial 0.5, wrong 0; mean of the two judges). "
        "Acc.: accuracy on answerable items with 95\\% bootstrap confidence intervals. "
        "Ref.: share of answerable items the generator declines to answer (lexical refusal "
        "detector). Abst.: share of unanswerable items on which it abstains (judges' label). "
        "RAG: the production retrieval pipeline (exact search, cross-encoder reranking, "
        "abstention gate) with identical passages and gate decisions for every generator; "
        f"the gate alone abstains on {fmt_pct(gate)}\\% of the unanswerable items for all "
        "RAG rows. $^{\\ast}$Post-trained release. \\textbf{Bold}: best in column.}",
        "\\label{tab:rag}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        " & \\multicolumn{3}{c}{Parametric (question only)} & \\multicolumn{2}{c}{With RAG} \\\\",
        "\\cmidrule(lr){2-4}\\cmidrule(lr){5-6}",
        "Generator & Acc.\\ (\\%) & Ref.\\ (\\%) & Abst.\\ (\\%) & Acc.\\ (\\%) & $\\Delta$ (pp) \\\\",
        "\\midrule",
    ]
    for label, par, rag in rows:
        p, r = models[par], models[rag]
        lines.append(" & ".join([
            label, acc(par, best_par), fmt_pct(p["refusal_rate_answerable"]), abst(par),
            acc(rag, best_rag), f"{100 * (r['accuracy'] - p['accuracy']):+.1f}",
        ]) + " \\\\")
    lines.append("\\midrule")
    for label, par in solo:
        p = models[par]
        lines.append(" & ".join([
            label, acc(par, best_par), fmt_pct(p["refusal_rate_answerable"]), abst(par),
            "--", "--"]) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}}", "\\end{table}", ""]
    return "\n".join(lines)


def bench_macros(bench_dir: Path) -> list[str]:
    """``\\bres{bench}{<key>}``: construction counts of benchmark v2 and items per category."""
    def rows(name: str) -> list[dict]:
        p = bench_dir / name
        return ([json.loads(x) for x in p.read_text("utf-8").splitlines() if x.strip()]
                if p.exists() else [])

    cand, acc, rej = (rows("benchmark_v2_candidates.jsonl"), rows("benchmark_v2.jsonl"),
                      rows("benchmark_v2_rejected.jsonl"))
    if not acc:
        return []
    excl_path = bench_dir / "excluded.json"
    excl = ({e["id"] for e in json.loads(excl_path.read_text("utf-8"))["items"]}
            if excl_path.exists() else set())
    out = []

    def d(key: str, value: str) -> None:
        out.append(f"\\expandafter\\def\\csname bres@bench@{key}\\endcsname{{{value}}}")

    d("candidates", str(len(cand)))
    d("candidates-v1", str(sum(1 for r in cand if r.get("source") == "v1")))
    d("candidates-new", str(sum(1 for r in cand if r.get("source") != "v1")))
    d("accepted", str(len(acc)))
    d("rejected", str(len(rej)))
    d("excluded", str(len(excl)))
    scored = [r for r in acc if r["id"] not in excl]
    for answerable, tag in ((True, "ans"), (False, "unans")):
        cats: dict[str, int] = {}
        for r in scored:
            if r["answerable"] == answerable:
                cats[r["category"]] = cats.get(r["category"], 0) + 1
        d(f"ncat-{tag}", str(len(cats)))
        for c, n in sorted(cats.items()):
            d(f"n-{tag}-{c}", str(n))
    verifier = {r.get("verifier", {}).get("model") for r in acc} - {None}
    d("verifier", ", ".join(sorted(verifier)).replace("_", "\\_"))
    return out


def load_cost(path: Path) -> dict | None:
    """Read eval/cost_model.json and derive hours and totals for every run."""
    if not path.exists():
        return None
    cm = json.loads(path.read_text(encoding="utf-8"))
    by_key = {s["key"]: s for s in cm["stages"]}
    for s in cm["stages"]:  # direct hours first: throughput-derived stages refer to them
        if "hours" in s:
            s["hours_est"] = s["hours"]
        elif "hours_full" in s:
            s["hours_est"] = s["hours_full"] * s["steps"] / s["steps_full"]
        else:
            s["hours_est"] = None
    for s in cm["stages"]:
        if "hours_from_throughput_of" in s:
            ref = by_key[s["hours_from_throughput_of"]]
            s["hours_est"] = s["steps"] / (ref["steps"] / ref["hours_est"])

    def gpu_h(kind: str | None, hardware: str | None = None) -> float:
        return sum(s["hours_est"] for s in cm["stages"]
                   if s["hours_est"] is not None and s.get("gpu", True)
                   and (kind is None or s["kind"] == kind)
                   and (hardware is None or s["hardware"] == hardware))

    dep = [s for s in cm["stages"] if s["kind"] == "deployed"]
    pre = [s for s in dep if s.get("pretrain")]
    p1 = by_key["p1"]
    tps = cm["tokens_per_step_pretrain"]
    p1_flops = 6 * cm["n_params"] * p1["steps"] * tps
    cm["totals"] = {
        "gpu_hours_deployed": gpu_h("deployed"),
        "gpu_hours_other": gpu_h("other"),
        "gpu_hours_all": gpu_h(None),
        "h100_hours_deployed": gpu_h("deployed", "H100"),
        "pretrain_hours": sum(s["hours_est"] for s in pre),
        "pretrain_tokens": sum(s["steps"] for s in pre if not s.get("warm_start")) * tps,
        "processed_tokens": sum(s["steps"] for s in pre) * tps,
        "interrupted_tokens": by_key["p1_interrupted"]["steps"] * tps,
        "p1_tokens_per_s": p1["steps"] * tps / (p1["hours_est"] * 3600),
        "p1_mfu": p1_flops / (p1["hours_est"] * 3600 * cm["h100_bf16_dense_peak_flops"]),
    }
    return cm


def table_cost(cm: dict) -> str:
    """Render the compute of every run (deployed chain, then other runs) and the bill."""
    t = cm["totals"]

    def row(s: dict) -> str:
        if s["hours_est"] is None:
            amount = f"\\${s['usd']:.2f} (API)" if "usd" in s else "--"
        else:
            amount = f"{s['hours_est']:.1f}"
        return f"{s['label']} & {s['hardware']} & {amount} \\\\"

    lines = [
        "% Generated by boccaccioAI-validation/eval/paper_assets.py -- do not edit by hand.",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Compute of every run. Hours are those recorded when the runs were made, or derived "
        "from the measured stage-1 throughput where the duration was not logged reliably "
        "(stage~2 and the interrupted first attempt). \\emph{Deployed chain}: the runs that "
        "produced the deployed model, including the interrupted first attempt, whose weights "
        "(but not optimiser state) warm-started stage~1; \\emph{other runs}: the "
        "instruction-tuning runs that were evaluated but not deployed. The total bill covers "
        "every run listed, "
        "including the teacher API.}",
        "\\label{tab:cost}",
        "\\small",
        "\\begin{tabular}{p{0.52\\textwidth}lr}",
        "\\toprule",
        "Run & Hardware & Hours \\\\",
        "\\midrule",
        "\\multicolumn{3}{l}{\\emph{Deployed chain}} \\\\",
    ]
    lines += [row(s) for s in cm["stages"] if s["kind"] == "deployed"]
    lines += [f"Subtotal (GPU hours) & & {t['gpu_hours_deployed']:.1f} \\\\",
              "\\midrule",
              "\\multicolumn{3}{l}{\\emph{Other runs}} \\\\"]
    lines += [row(s) for s in cm["stages"] if s["kind"] == "other"]
    lines += [f"Subtotal (GPU hours) & & {t['gpu_hours_other']:.1f} \\\\",
              "\\midrule",
              f"\\textbf{{Total bill, all runs}} & & \\textbf{{under \\${cm['invoiced_total_usd_below']}}} \\\\",
              "\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


def cost_macros(cm: dict) -> list[str]:
    """``\\bres{cost}{<key>}``, ``\\bres{distill}{...}``, ``\\bres{rag}{...}`` macros."""
    out = []

    def d(model: str, key: str, value: str) -> None:
        out.append(f"\\expandafter\\def\\csname bres@{model}@{key}\\endcsname{{{value}}}")

    def cnt(n: float) -> str:
        return f"{n:,.0f}".replace(",", "{,}")

    t = cm["totals"]
    d("cost", "bill-below", str(cm["invoiced_total_usd_below"]))
    d("cost", "pretrain-share", f"{100 * t['pretrain_hours'] / t['gpu_hours_deployed']:.0f}")
    d("cost", "pretrain-tokens", fmt_count(t["pretrain_tokens"]))
    d("cost", "interrupted-tokens", fmt_count(t["interrupted_tokens"]))
    d("cost", "p1-tokps", cnt(t["p1_tokens_per_s"]))
    d("cost", "p1-mfu", f"{100 * t['p1_mfu']:.0f}")
    d("cost", "gpuh-deployed", f"{t['gpu_hours_deployed']:.0f}")
    d("cost", "gpuh-other", f"{t['gpu_hours_other']:.0f}")
    d("cost", "gpuh-all", f"{t['gpu_hours_all']:.0f}")
    d("cost", "h100h-deployed", f"{t['h100_hours_deployed']:.0f}")
    for s in cm["stages"]:
        k = s["key"].replace("_", "-")
        if "usd" in s:
            d("cost", f"{k}-usd", f"{s['usd']:.2f}")
        if s["hours_est"] is not None:
            d("cost", f"{k}-h", f"{s['hours_est']:.1f}")
        if "steps" in s:
            d("cost", f"{k}-steps", cnt(s["steps"]))
    for k, v in cm["distill_data"].items():
        if isinstance(v, int):
            d("distill", k.replace("_", "-"), cnt(v))
    dd, sd = cm["distill_data"], cm["sft_data"]
    d("distill", "refusal-share", f"{100 * dd['refusal'] / dd['filtered']:.1f}")
    d("sft", "examples", cnt(sd["examples"]))
    d("sft", "refusal", cnt(sd["refusal"]))
    d("sft", "refusal-share", f"{100 * sd['refusal'] / sd['examples']:.1f}")
    r = cm["rag"]
    d("rag", "passages", cnt(r["passages"]))
    d("rag", "pages", cnt(r["pages"]))
    d("rag", "passages-m", f"{r['passages'] / 1e6:.1f}")
    d("rag", "pages-m", f"{r['pages'] / 1e6:.1f}")
    d("rag", "threshold", f"{r['threshold']:.2f}")
    inf = cm["inference"]
    d("infer", "l4-price", f"{inf['l4_usd_per_hour']:.2f}")
    d("infer", "usd-warm", f"{inf['usd_per_request_warm']:.3f}")
    d("infer", "usd-cold", f"{inf['usd_per_request_cold']:.3f}")
    d("infer", "cold-start", str(inf["cold_start_s"]))
    return out


def judge_macros(judge: dict) -> list[str]:
    """``\\bres{judge-<system>}{<key>}`` and agreement macros for the LLM-judged benchmark."""
    out = []

    def d(model: str, key: str, value: str) -> None:
        out.append(f"\\expandafter\\def\\csname bres@{model}@{key}\\endcsname{{{value}}}")

    sc = judge["scoring"]
    for k in ("n_items", "n_answerable", "n_unanswerable"):
        d("judge", k.replace("_", "-"), str(sc[k]))
    for s, m in judge["models"].items():
        n = f"judge-{s}"
        d(n, "acc", fmt_pct(m["accuracy"]))
        d(n, "acc-ci", f"[{fmt_pct(m['accuracy_ci'][0])}, {fmt_pct(m['accuracy_ci'][1])}]")
        d(n, "strict", fmt_pct(m["strict_accuracy"]))
        d(n, "acc-norecent", fmt_pct(m["accuracy_excl_recent"]))
        d(n, "abst", fmt_pct(m["abstention_rate"]))
        d(n, "abst-ci", f"[{fmt_pct(m['abstention_ci'][0])}, {fmt_pct(m['abstention_ci'][1])}]")
        if m.get("false_abstentions") is not None:
            d(n, "false-abst", str(m["false_abstentions"]))
        if "refusals_on_answerable" in m:
            d(n, "refusals", str(m["refusals_on_answerable"]))
            d(n, "refusal-rate", fmt_pct(m["refusal_rate_answerable"]))
            d(n, "acc-answered", fmt_pct(m["accuracy_on_answered"]))
            d(n, "n-answered", str(m["n_answered"]))
        for cat, v in m["per_category"].items():
            d(n, f"cat-{cat}", fmt_pct(v))
    for key, c in judge["comparisons"].items():
        ref, other = key.split(" vs ")
        n = f"judgecmp-{ref}-{other}"
        d(n, "diff", f"{100 * c['diff']:+.1f}")
        d(n, "ci", f"[{100 * c['ci_low']:+.1f}, {100 * c['ci_high']:+.1f}]")
        d(n, "p", fmt_p(c["p_holm"]))
    (pair, ij), = judge["inter_judge"].items()
    d("judge", "agree", fmt_pct(ij["agreement"]))
    d("judge", "kappa-ans", f"{ij['answerable']['cohen_kappa']:.2f}")
    d("judge", "wkappa-ans", f"{ij['answerable']['weighted_kappa_linear']:.2f}")
    d("judge", "kappa-unans", f"{ij['unanswerable']['cohen_kappa']:.2f}")
    d("judge", "n-pairs", f"{ij['n']:,}".replace(",", "{,}"))
    for s, m in judge.get("sensitivity", {}).items():
        n = f"judge-sens-{s}"
        d(n, "acc", fmt_pct(m["accuracy"]))
        d(n, "acc-ci", f"[{fmt_pct(m['accuracy_ci'][0])}, {fmt_pct(m['accuracy_ci'][1])}]")
        d(n, "acc-answered", fmt_pct(m["accuracy_on_answered"]))
        d(n, "refusals", str(m["refusals_on_answerable"]))
        d(n, "n-answered", str(m["n_answered"]))
        d(n, "abst", fmt_pct(m["abstention_rate"]))
        if "accuracy_on_scored_answered" in m:
            d(n, "acc-on-scored-answered", fmt_pct(m["accuracy_on_scored_answered"]))
            d(n, "acc-on-scored-refused", fmt_pct(m["accuracy_on_scored_refused"]))
            d(n, "n-scored-answered", str(m["n_scored_answered"]))
            d(n, "n-scored-refused", str(m["n_scored_refused"]))
    hum = judge.get("human")
    if hum:
        d("human", "n", str(hum["n_labeled"]))
        for j, st in hum["judges"].items():
            slug = "gemini" if "gemini" in j else "deepseek" if "deepseek" in j else j
            d(f"human-{slug}", "kappa-ans", f"{st['answerable']['cohen_kappa']:.2f}")
            d(f"human-{slug}", "wkappa-ans", f"{st['answerable']['weighted_kappa_linear']:.2f}")
            d(f"human-{slug}", "kappa-unans", f"{st['unanswerable']['cohen_kappa']:.2f}")
            d(f"human-{slug}", "agree", fmt_pct(st["agreement"]))
            d(f"human-{slug}", "acc-human", fmt_pct(st["answerable_accuracy_human"]))
            d(f"human-{slug}", "acc-judge", fmt_pct(st["answerable_accuracy_judge"]))
            d(f"human-{slug}", "gap", f"{100 * (st['answerable_accuracy_judge'] - st['answerable_accuracy_human']):.1f}")
            d(f"human-{slug}", "n-ans", str(st["answerable"]["n"]))
            d(f"human-{slug}", "n-unans", str(st["unanswerable"]["n"]))
            for g, gs in st.get("answerable_by_group", {}).items():
                d(f"human-{slug}-{g}", "gap", f"{100 * gs['gap']:.1f}")
                d(f"human-{slug}-{g}", "n", str(gs["n"]))
                d(f"human-{slug}-{g}", "acc-human", fmt_pct(gs["accuracy_human"]))
        d("human", "per-system", str(min(hum["per_model"].values())))
    return out


def fig_stage_ablation(summary: dict, path: Path, judge: dict | None = None) -> None:
    """Grouped bar chart base -> SFT -> distill with 95% confidence intervals; with the
    judge report, a second panel shows the same stages on benchmark v2 (parametric)."""
    stages = [r for name in ("boccaccio-base-p2", "boccaccio-sft-v1", "boccaccio-distill")
              for r in summary["rows"] if r["name"] == name]
    labels = [t["label"] for t in summary["tasks"]] + ["Avg."]
    keys = [t["task"] for t in summary["tasks"]] + ["avg"]
    qa = judge is not None and all(s in judge["models"] for _, s in QA_STAGES)
    if qa:
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.5, 2.9),
                                      gridspec_kw={"width_ratios": [3.2, 1]})
    else:
        fig, ax = plt.subplots(figsize=(7.5, 2.8))
    width = 0.27
    for i, (r, color) in enumerate(zip(stages, STAGE_COLORS)):
        v = [100 * (r["avg"] if k == "avg" else r["scores"][k]) for k in keys]
        e = [100 * Z95 * (r["avg_se"] if k == "avg" else r["stderr"][k]) for k in keys]
        x = [j + (i - 1) * width for j in range(len(keys))]
        ax.bar(x, v, width, yerr=e, color=color, label=r["short"], capsize=1.5,
               error_kw={"elinewidth": 0.6}, edgecolor="black", linewidth=0.3)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("Score (%)")
    ax.axvline(len(keys) - 1.5, color="black", linewidth=0.5, linestyle=":")
    ax.legend(fontsize=8, ncol=3, frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=8)
    if qa:
        ax.set_title("(a) Italian benchmark suite", fontsize=8.5, loc="left")
        metrics = [("Acc.", "accuracy"), ("Refusal\n(answ.)", "refusal_rate_answerable"),
                   ("Abstention\n(unansw.)", "abstention_rate")]
        for i, ((_, sys_name), color) in enumerate(zip(QA_STAGES, STAGE_COLORS)):
            m = judge["models"][sys_name]
            v = [100 * m[k] for _, k in metrics]
            x = [j + (i - 1) * width for j in range(len(metrics))]
            err = [[100 * (m["accuracy"] - m["accuracy_ci"][0]), 0, 100 * (m["abstention_rate"]
                    - m["abstention_ci"][0])],
                   [100 * (m["accuracy_ci"][1] - m["accuracy"]), 0, 100 * (m["abstention_ci"][1]
                    - m["abstention_rate"])]]
            ax2.bar(x, v, width, yerr=err, color=color, capsize=1.5,
                    error_kw={"elinewidth": 0.6}, edgecolor="black", linewidth=0.3)
        ax2.set_xticks(range(len(metrics)))
        ax2.set_xticklabels([m[0] for m in metrics], fontsize=7.5)
        ax2.set_ylabel("Rate (%)")
        ax2.set_ylim(0, 100)
        ax2.set_title("(b) Benchmark v2, parametric", fontsize=8.5, loc="left")
        ax2.spines[["top", "right"]].set_visible(False)
        ax2.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_score_vs_tokens(summary: dict, path: Path, cost_label: str | None = None) -> None:
    """Average score vs declared pretraining tokens (log x) with the token-efficiency frontier."""
    rows = [r for r in summary["rows"] if r["group"] != "ours" or r["name"] == "boccaccio-distill"]
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    for g, st in GROUP_STYLE.items():
        grp = [r for r in rows if r["group"] == g]
        if not grp:
            continue
        ax.errorbar([r["pretrain_tokens"] for r in grp], [100 * r["avg"] for r in grp],
                    yerr=[100 * Z95 * r["avg_se"] for r in grp], fmt="none",
                    ecolor=st["color"], elinewidth=0.6, capsize=2, zorder=2)
        ax.scatter([r["pretrain_tokens"] for r in grp], [100 * r["avg"] for r in grp],
                   s=st["size"], marker=st["marker"], color=st["color"],
                   edgecolor="black", linewidth=0.4, label=summary["groups"][g], zorder=3)
    front, best = [], -1.0
    for r in sorted(rows, key=lambda r: r["pretrain_tokens"]):
        if r["avg"] > best:
            front.append(r)
            best = r["avg"]
    ax.step([r["pretrain_tokens"] for r in front], [100 * r["avg"] for r in front],
            where="post", color="black", linewidth=0.7, linestyle="--", zorder=1,
            label="Token-efficiency frontier")
    for r in rows:
        txt = r["display"]
        if r["group"] == "ours":
            txt = f"BoccaccioAI ({cost_label})" if cost_label else "BoccaccioAI"
        dx, dy, ha = LABEL_OFFSET.get(r["name"], (5, 4, "left"))
        ax.annotate(txt, (r["pretrain_tokens"], 100 * r["avg"]), textcoords="offset points",
                    xytext=(dx, dy), fontsize=7, ha=ha)
    ax.set_xscale("log")
    ax.set_xlabel("Declared pretraining tokens")
    ax.set_ylabel("Average score (%)")
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    xs = [r["pretrain_tokens"] for r in rows]
    ax.set_xlim(min(xs) / 2, max(xs) * 4)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    """Entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    args = parse_args()
    meta = json.loads(args.meta.read_text(encoding="utf-8"))
    scores = load_scores(args.results_dir, meta)
    summary = build_summary(meta, scores, args.results_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "serif", "pdf.fonttype": 42})
    (args.out_dir / "tab_benchmarks.tex").write_text(table_benchmarks(summary), encoding="utf-8")
    (args.out_dir / "tab_cost_tokens.tex").write_text(table_cost_tokens(summary), encoding="utf-8")
    fert_tex = table_fertility(args.results_dir)
    if fert_tex:
        (args.out_dir / "tab_fertility.tex").write_text(fert_tex, encoding="utf-8")
    judge = load_judge(ROOT / "benchmark" / "judge_report.json",
                       ROOT / "benchmark" / "human_review" / "summary.json")
    tex_macros = macros(summary)
    if judge:
        tex_macros += "\n".join(judge_macros(judge) + bench_macros(ROOT / "benchmark")) + "\n"
        rag_tex = table_rag(judge)
        if rag_tex:
            (args.out_dir / "tab_rag.tex").write_text(rag_tex, encoding="utf-8")
        summary["judge"] = judge
    cost = load_cost(COST_MODEL)
    cost_label = None
    if cost:
        tex_macros += "\n".join(cost_macros(cost)) + "\n"
        (args.out_dir / "tab_cost.tex").write_text(table_cost(cost), encoding="utf-8")
        cost_label = f"under \\${cost['invoiced_total_usd_below']}"
        summary["cost"] = cost
    (args.out_dir / "results_macros.tex").write_text(tex_macros, encoding="utf-8")
    fig_stage_ablation(summary, args.out_dir / "fig_stage_ablation.pdf", judge)
    fig_score_vs_tokens(summary, args.out_dir / "fig_score_vs_tokens.pdf", cost_label)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for r in sorted(summary["rows"], key=lambda r: r["rank"]):
        log.info("%2d  %-20s avg=%.4f+-%.4f  N=%d (non-emb %d)  D=%s", r["rank"], r["name"],
                 r["avg"], r["avg_se"], r["params"], r["params_nonemb"],
                 fmt_count(r["pretrain_tokens"]))
    print(f"Assets written to {args.out_dir}")


if __name__ == "__main__":
    main()
