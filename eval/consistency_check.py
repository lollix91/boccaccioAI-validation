"""Cross-check every validation artefact against the raw data it was derived from.

Benchmark v2: item set, exclusions, answers of every scored system, verdicts of both
judges (re-derived scores vs ``judge_report.json``; every verdict bound to the current
answer text through the judge cache key), shared RAG gate decisions, human labels,
sensitivity variant and production checks. Standard benchmarks: result files, sample
files, significance tests, BPB and fertility on the same held-out corpus. Cost model:
distillation counts and teacher-API cost against the generation log (read-only).

    python eval/consistency_check.py            # exit code 1 if any check fails
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval" / "judge"))
from judge import (BENCH_SYSTEMS, JUDGE_ANSWERABLE, JUDGE_UNANSWERABLE,  # noqa: E402
                   excluded_ids, load_jsonl, score_of)

BENCH = ROOT / "benchmark"
RESULTS = ROOT / "eval" / "results"
sys.path.insert(0, str(ROOT / "eval"))
from pack_artifacts import CACHE_ARCHIVE, CACHE_DIR, load_cache  # noqa: E402
from significance import load_items  # noqa: E402

DISTILL_LOG = Path(r"E:\boccaccioAI\distill\generated")
FAILURES: list[str] = []


def judge_cache() -> dict[str, dict] | None:
    """Judge responses by cache key: the loose cache if present, else the packed archive."""
    if CACHE_DIR.exists() and any(CACHE_DIR.glob("*.json")):
        return {p.stem: json.loads(p.read_text("utf-8")) for p in CACHE_DIR.glob("*.json")}
    return load_cache(CACHE_ARCHIVE) if CACHE_ARCHIVE.exists() else None


def check(cond: bool, msg: str) -> None:
    """Record and print one check."""
    print(("  ok    " if cond else "  FAIL  ") + msg)
    if not cond:
        FAILURES.append(msg)


def close(a: float, b: float, tol: float = 1e-9) -> bool:
    """Float equality with tolerance."""
    return math.isclose(a, b, abs_tol=tol)


def judge_id(slug: str, recs: dict) -> str:
    """OpenRouter model id of a judge directory (from the records)."""
    return next(iter(recs.values()))["judge"]


def bench_checks() -> None:
    """Benchmark v2 artefacts."""
    print("\n[benchmark v2]")
    items = {r["id"]: r for r in load_jsonl(BENCH / "benchmark_v2.jsonl")}
    cand = load_jsonl(BENCH / "benchmark_v2_candidates.jsonl")
    rej = load_jsonl(BENCH / "benchmark_v2_rejected.jsonl")
    check(len(items) == len(load_jsonl(BENCH / "benchmark_v2.jsonl")), "item ids unique")
    check({r["id"] for r in cand} == set(items) | {r["id"] for r in rej},
          f"candidates ({len(cand)}) = accepted ({len(items)}) + rejected ({len(rej)})")
    check(all(r["gold"] for r in items.values() if r["answerable"]),
          "every answerable item has a gold answer")
    drop = excluded_ids()
    check(drop <= set(items), f"excluded ids {sorted(drop)} are benchmark items")
    scored = {i: r for i, r in items.items() if i not in drop}
    n_ans = sum(r["answerable"] for r in scored.values())
    report = json.loads((BENCH / "judge_report.json").read_text("utf-8"))
    sc = report["scoring"]
    check((sc["n_items"], sc["n_answerable"], sc["n_unanswerable"])
          == (len(scored), n_ans, len(scored) - n_ans),
          f"report counts {sc['n_items']}/{sc['n_answerable']}/{sc['n_unanswerable']} match items")
    check(sorted(report["models"]) == sorted(BENCH_SYSTEMS),
          f"report covers exactly the {len(BENCH_SYSTEMS)} BENCH_SYSTEMS")

    answers = {}
    for s in BENCH_SYSTEMS:
        rows = load_jsonl(BENCH / "answers" / f"{s}.jsonl")
        answers[s] = {r["id"]: r for r in rows}
        ok = len(rows) == len(answers[s]) and set(answers[s]) == set(items)
        errs = [r["id"] for r in rows if r.get("error")]
        check(ok and not errs, f"answers/{s}: {len(rows)} unique items, no errors")

    ctx = {r["id"]: r for r in load_jsonl(BENCH / "rag_contexts.jsonl")}
    check(set(ctx) == set(items), f"rag_contexts covers all {len(items)} items")
    for s in [s for s in BENCH_SYSTEMS if s.endswith("-rag")]:
        bad = [i for i in items if bool(answers[s][i].get("abstained")) != bool(ctx[i]["abstained"])]
        check(not bad, f"{s}: gate decisions equal to the shared retrieval ({len(bad)} differ)")

    judge_dirs = sorted(d for d in (BENCH / "judgments").iterdir() if d.is_dir())
    check(sorted(d.name for d in judge_dirs) == sorted(sc["judges"]),
          f"judge directories = report judges ({len(judge_dirs)})")
    cache = judge_cache()
    if CACHE_ARCHIVE.exists() and CACHE_DIR.exists() and cache is not None:
        check(load_cache(CACHE_ARCHIVE) == cache,
              f"packed judge cache ({CACHE_ARCHIVE.name}) equals the loose cache")
    verdicts: dict[str, dict[str, dict]] = {}
    for jd in judge_dirs:
        verdicts[jd.name] = {}
        for s in BENCH_SYSTEMS:
            recs = {r["id"]: r for r in load_jsonl(jd / f"{s}.jsonl")}
            verdicts[jd.name][s] = recs
            jid = judge_id(jd.name, recs)
            meta_ok = (set(recs) == set(items)
                       and all(r["judge"] == jid and r["model"] == s for r in recs.values())
                       and all(r["answerable"] == items[i]["answerable"] for i, r in recs.items())
                       and len({r["prompt_version"] for r in recs.values()}) == 1)
            # verdict bound to the current answer: the cache key hashes judge id + prompt
            stale = 0
            for i, r in (recs.items() if cache is not None else ()):
                it, ans = items[i], answers[s][i]["answer"]
                prompt = (JUDGE_ANSWERABLE.format(question=it["question"], gold=it["gold"], answer=ans)
                          if it["answerable"] else
                          JUDGE_UNANSWERABLE.format(question=it["question"], answer=ans))
                c = cache.get(hashlib.sha256((jid + "|" + prompt).encode()).hexdigest()[:16])
                if c is None or (c.get("verdict") != r.get("verdict")
                                 or c.get("behaviour") != r.get("behaviour")):
                    stale += 1
            bound = (f"{stale} verdicts not bound to the current answer" if cache is not None
                     else "answer binding skipped (no judge cache)")
            check(meta_ok and stale == 0, f"{jd.name}/{s}: complete, consistent metadata, {bound}")

    # re-derive the headline metrics
    ans_ids = [i for i in scored if scored[i]["answerable"]]
    una_ids = [i for i in scored if not scored[i]["answerable"]]
    for s in BENCH_SYSTEMS:
        m = report["models"][s]
        acc = sum(sum(score_of(verdicts[j][s][i]) for j in verdicts) / len(verdicts)
                  for i in ans_ids) / len(ans_ids)
        abst = sum(sum(score_of(verdicts[j][s][i]) for j in verdicts) / len(verdicts)
                   for i in una_ids) / len(una_ids)
        ci = m["accuracy_ci"]
        check(close(acc, m["accuracy"]) and close(abst, m["abstention_rate"])
              and ci[0] <= acc <= ci[1],
              f"{s}: accuracy {acc:.4f} and abstention {abst:.4f} re-derived, inside CI")
        for j in verdicts:
            pj = report["judges"][j][s]
            accj = sum(score_of(verdicts[j][s][i]) for i in ans_ids) / len(ans_ids)
            if not close(accj, pj["accuracy"]):
                check(False, f"{j}/{s}: per-judge accuracy {accj:.4f} vs report {pj['accuracy']:.4f}")

    # human validation
    print("\n[human validation]")
    sample = {r["key"]: r for r in load_jsonl(BENCH / "human_review" / "sample.jsonl")}
    labels = load_jsonl(BENCH / "human_review" / "labels.jsonl")
    summary = json.loads((BENCH / "human_review" / "summary.json").read_text("utf-8"))
    sha = lambda t: hashlib.sha256(t.encode("utf-8")).hexdigest()[:12]  # noqa: E731
    bound = [lab for lab in labels if lab["key"] in sample
             and sample[lab["key"]].get("answer_sha", sha(sample[lab["key"]]["answer"]))
             == sha(answers.get(lab["model"], {}).get(lab["id"], {}).get("answer", "\0"))]
    valid_labels = {True: {"correct", "partial", "wrong"}, False: {"abstained", "answered"}}
    check(len(bound) == len(labels), f"{len(bound)}/{len(labels)} labels bound to the current answers")
    check(all(lab["verdict"] in valid_labels[lab["answerable"]] for lab in labels),
          "label values valid for the item type")
    check(len({lab["key"] for lab in labels}) == len(labels), "no duplicate labels")
    check(summary["n_labeled"] == len([lab for lab in bound if lab["id"] not in drop]),
          f"summary n_labeled = {summary['n_labeled']}")
    per = {}
    for lab in labels:
        per[lab["model"]] = per.get(lab["model"], 0) + 1
    check(sorted(per) == sorted(BENCH_SYSTEMS) and len(set(per.values())) == 1,
          f"labels per system: {sorted(set(per.values()))} for {len(per)} systems")

    print("\n[sensitivity and production checks]")
    sens_ans = {r["id"]: r for r in load_jsonl(BENCH / "sensitivity" / "boccaccio-sft-v1.jsonl")}
    check(set(sens_ans) == set(items), "sensitivity answers cover all items")
    for jd in judge_dirs:
        jrec = {r["id"]: r for r in load_jsonl(BENCH / "sensitivity" / f"{jd.name}__boccaccio-sft-v1.jsonl")}
        check(set(jrec) == set(items), f"sensitivity verdicts of {jd.name} cover all items")
    chk = json.loads((BENCH / "checks" / "checks.json").read_text("utf-8"))
    check(chk["weights"]["identical"], "production weights = evaluated checkpoint (SHA-256)")
    check(chk["retrieval"]["identical"] == chk["retrieval"]["n"] == len(items),
          f"retrieval reproduced {chk['retrieval']['identical']}/{chk['retrieval']['n']}")
    check(chk["generation"]["identical"] == chk["generation"]["n"] == len(items),
          f"parametric generation reproduced {chk['generation']['identical']}/{chk['generation']['n']}")


def lm_checks() -> None:
    """Standard-benchmark artefacts."""
    print("\n[standard benchmarks]")
    meta = json.loads((ROOT / "eval" / "model_meta.json").read_text("utf-8"))
    names = [m["name"] for m in meta["models"]]
    tasks = [t["task"] for t in meta["tasks"]]
    for n in names:
        miss = [t for t in tasks if not (RESULTS / n / f"{t}.json").exists()]
        check(not miss and (RESULTS / n / "run_meta.json").exists(),
              f"{n}: {len(tasks) - len(miss)}/{len(tasks)} task results + run_meta")
    sig = json.loads((RESULTS / "significance.json").read_text("utf-8"))
    check(sig["reference"] == "boccaccio-distill"
          and sorted(sig["comparisons"]) == sorted(n for n in names if n != "boccaccio-distill"),
          "significance: reference = distilled, all other models compared")
    have = [(n, t) for n in names for t in tasks if (RESULTS / n / f"{t}.samples.jsonl").exists()]
    if have:
        bad = []
        for n, t in have:
            metric = next(x["metric"] for x in meta["tasks"] if x["task"] == t)
            try:
                load_items(RESULTS, n, t, metric)
            except ValueError as e:
                bad.append(str(e))
        check(not bad, f"per-item samples: {len(have)}/{len(names) * len(tasks)} files, "
                       f"means match the aggregates ({len(bad)} do not)")
    else:
        print("  skip  per-item samples not restored (samples-unpack from the Zenodo archive)")
    drift: set[str] = set()
    for n, comp in sig["comparisons"].items():
        for t in tasks:
            if t not in comp:
                continue
            ref = json.loads((RESULTS / "boccaccio-distill" / f"{t}.json").read_text("utf-8"))["results"][t]
            oth = json.loads((RESULTS / n / f"{t}.json").read_text("utf-8"))["results"][t]
            metric = next(x["metric"] for x in meta["tasks"] if x["task"] == t)
            d = oth[f"{metric},none"] - ref[f"{metric},none"]
            if close(d, comp[t]["diff"], 1e-6):
                continue
            # per-item samples regenerated by collect_samples.py may differ from the original
            # aggregate by a few items on generative tasks (BF16 nondeterminism, tolerance 0.005)
            if abs(d - comp[t]["diff"]) <= 0.005:
                drift.add(t)
            else:
                check(False, f"significance {n}/{t}: diff {comp[t]['diff']:.5f} vs results {d:.5f}")
    for t in sorted(drift):
        print(f"  note  significance/{t}: per-item samples differ from the aggregate within the "
              "0.005 BF16 tolerance of collect_samples.py")
    fert = json.loads((RESULTS / "fertility_wiki-it-2025.json").read_text("utf-8"))
    stats_path = ROOT / "data" / "heldout" / "wiki_it_2025.stats.json"
    stats = json.loads(stats_path.read_text("utf-8")) if stats_path.exists() else None
    bpb = {n: json.loads((RESULTS / n / "bpb_wiki-it-2025.json").read_text("utf-8")) for n in names}
    check(len({b["docs"] for b in bpb.values()} | {fert["docs"]}) == 1,
          f"BPB and fertility on the same {fert['docs']} documents")
    check(len({b["bytes"] for b in bpb.values()}) == 1, "identical byte count for every BPB run")
    if stats:
        check(stats["documents"] == fert["docs"] and stats["bytes"] == bpb[names[0]]["bytes"],
              "held-out corpus stats = BPB/fertility corpus")
    ours = fert["entries"][0]
    check(ours["tokens"] == bpb["boccaccio-distill"]["tokens"],
          "our tokenizer: same token count in fertility and BPB")
    check(all(close(e["relative"], e["fertility"] / ours["fertility"], 1e-9) for e in fert["entries"]),
          "fertility ratios relative to our tokenizer")
    check(min(e["fertility"] for e in fert["entries"]) == ours["fertility"],
          "our tokenizer has the lowest fertility")


def cost_checks() -> None:
    """Cost-model inputs that can be verified against logs."""
    print("\n[cost model]")
    cm = json.loads((ROOT / "eval" / "cost_model.json").read_text("utf-8"))
    log = DISTILL_LOG / "answers.jsonl"
    if not log.exists():
        print("  skip  distillation log not available")
        return
    rows = [json.loads(x) for x in log.read_text("utf-8").splitlines() if x.strip()]
    filt = [json.loads(x) for x in (DISTILL_LOG / "filtered.jsonl").read_text("utf-8").splitlines()
            if x.strip()]
    api = next(s for s in cm["stages"] if s["key"] == "distill_api")
    dd = cm["distill_data"]
    check(close(sum(r.get("cost") or 0 for r in rows), api["usd"], 1e-3),
          f"teacher API cost {api['usd']} = sum of logged charges")
    check(len(rows) == dd["teacher_answers"] and len(filt) == dd["filtered"],
          f"teacher answers {len(rows)}, filtered {len(filt)}")
    check(sum(r.get("source") == "squad_it" for r in filt) == dd["squad_it_with_context"]
          and sum(r.get("source") == "refusal" for r in filt) == dd["refusal"],
          "distillation composition (SQuAD-IT with context, refusals)")


def main() -> None:
    """Run all checks; exit 1 on failure."""
    bench_checks()
    lm_checks()
    cost_checks()
    print(f"\n{len(FAILURES)} failed check(s)")
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
