"""Human validation UI for the LLM-judged benchmark (zero dependencies).

Serves a one-item-at-a-time page on localhost. For every item the rater sees
the question, the gold answer (when present) and the model answer; neither the
model name nor the judge verdicts are shown, to avoid anchoring. Labels are
appended to ``benchmark/human_review/labels.jsonl``, so progress survives restarts.

Every sample row and label is bound to the exact answer text it refers to
(``answer_sha``). A label whose answer no longer matches the current
``benchmark/answers/<system>.jsonl`` (e.g. after regenerating a model) is *stale*:
it is ignored by the summary and moved to ``labels_stale.jsonl`` with
``--archive-stale``. Items in ``benchmark/excluded.json`` are ignored.

Sample: ``--per-model`` items for each system in ``--models`` (default: all answer
files), seeded, never repeating an (system, item, answer) triple already sampled.

    python eval/judge/human_review.py [--models a b] [--per-model 15] [--port 8787]
    python eval/judge/human_review.py --summary-only [--archive-stale]
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import logging
import random
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from judge import (agreement_stats, excluded_ids, label_of, load_jsonl,  # noqa: E402
                   score_of)

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "benchmark"
REVIEW = BENCH / "human_review"
HUMAN_SCORE = {"correct": 1.0, "partial": 0.5, "wrong": 0.0, "abstained": 1.0, "answered": 0.0}

PAGE = """<!doctype html><html lang="it"><head><meta charset="utf-8">
<title>Validazione umana</title>
<style>
 body{font-family:system-ui;max-width:820px;margin:2em auto;padding:0 1em}
 .box{border:1px solid #ccc;border-radius:8px;padding:1em;margin:1em 0}
 label{font-weight:600;color:#444} pre{white-space:pre-wrap;font-size:1.05em;margin:.4em 0}
 button{font-size:1.1em;padding:.6em 1.4em;margin:.3em;border-radius:6px;border:1px solid #888;cursor:pointer}
 .ok{background:#d4edda}.mid{background:#fff3cd}.bad{background:#f8d7da}
 .abs{background:#d1ecf1}.skip{background:#eee} h1{font-size:1.3em}
 .meta{color:#777;font-size:.9em}
</style></head><body>
<h1>Validazione umana — {pos} / {tot} (fatte: {done})</h1>
<div class="box"><label>Domanda</label><pre>{question}</pre></div>
{gold_block}
<div class="box"><label>Risposta del modello</label><pre>{answer}</pre></div>
<form method="post" action="/label">
<input type="hidden" name="key" value="{key}">{buttons}
</form>
<form method="post" action="/label"><input type="hidden" name="key" value="{key}">
<input type="hidden" name="verdict" value="skip"><button class="skip">Salta (incerto)</button></form>
<p class="meta">id {id} — la validazione viene salvata ad ogni click.</p>
</body></html>"""

DONE_PAGE = """<!doctype html><html><body style="font-family:system-ui">
<h1>Fatto — {n} validazioni salvate</h1><p>Chiudi il server (Ctrl+C) per scrivere summary.json.</p></body></html>"""


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--models", nargs="*", default=None,
                   help="systems to (re)sample; default: every answer file")
    p.add_argument("--per-model", type=int, default=15)
    p.add_argument("--port", type=int, default=8787)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--summary-only", action="store_true")
    p.add_argument("--archive-stale", action="store_true",
                   help="move labels of answers that changed to labels_stale.jsonl")
    return p.parse_args()


def sha(text: str) -> str:
    """Short hash binding a sample row or label to an exact answer text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def current_answers() -> dict[str, dict[str, str]]:
    """Current answer text per system and item id."""
    return {f.stem: {a["id"]: a["answer"] for a in load_jsonl(f)}
            for f in sorted((BENCH / "answers").glob("*.jsonl"))}


def row_sha(row: dict) -> str:
    """Answer hash of a sample row (legacy rows carry only the answer text)."""
    return row.get("answer_sha") or sha(row["answer"])


def is_current(row: dict, answers: dict[str, dict[str, str]]) -> bool:
    """True if the sample row still refers to the current answer of its system."""
    cur = answers.get(row["model"], {}).get(row["id"])
    return cur is not None and sha(cur) == row_sha(row)


def build_sample(models: list[str] | None, per_model: int, seed: int) -> list[dict]:
    """Extend ``sample.jsonl`` so that each selected system has ``per_model`` current rows."""
    path = REVIEW / "sample.jsonl"
    existing = load_jsonl(path)
    bench = {i["id"]: i for i in load_jsonl(BENCH / "benchmark_v2.jsonl")}
    drop = excluded_ids()
    answers = current_answers()
    taken = {(r["model"], r["id"], row_sha(r)) for r in existing}
    rows: list[dict] = []
    for model in sorted(models or answers):
        have = sum(1 for r in existing if r["model"] == model and is_current(r, answers))
        need = per_model - have
        if need <= 0:
            continue
        rng = random.Random(f"{seed}|{model}|{len(existing)}")
        ids = sorted(i for i in answers[model] if i in bench and i not in drop)
        rng.shuffle(ids)
        for i in ids:
            if need <= 0:
                break
            ans = answers[model][i]
            if (model, i, sha(ans)) in taken:
                continue
            it = bench[i]
            rows.append({"key": f"{model}|{i}|{sha(ans)}", "model": model, "id": i,
                         "question": it["question"], "answerable": it["answerable"],
                         "gold": it.get("gold"), "answer": ans, "answer_sha": sha(ans)})
            taken.add((model, i, sha(ans)))
            need -= 1
    random.Random(seed + len(existing)).shuffle(rows)
    if rows:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        log.info("added %d sample rows", len(rows))
    return existing + rows


def group_of(system: str) -> str:
    """``boccaccio`` for our checkpoints and the production endpoint, else ``baselines``."""
    return "boccaccio" if system.startswith(("boccaccio", "prod-")) else "baselines"


def write_summary(archive_stale: bool) -> dict:
    """Agreement of current human labels with every judge, plus human-vs-judge accuracy."""
    sample = {r["key"]: r for r in load_jsonl(REVIEW / "sample.jsonl")}
    labels = load_jsonl(REVIEW / "labels.jsonl")
    answers = current_answers()
    drop = excluded_ids()
    valid, stale = [], []
    for lab in labels:
        row = sample.get(lab["key"])
        (valid if row is not None and is_current(row, answers) else stale).append(lab)
    if archive_stale and stale:
        with (REVIEW / "labels_stale.jsonl").open("a", encoding="utf-8") as fh:
            for lab in stale:
                fh.write(json.dumps(lab, ensure_ascii=False) + "\n")
        (REVIEW / "labels.jsonl").write_text(
            "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in valid), encoding="utf-8")
        log.info("archived %d stale labels -> labels_stale.jsonl", len(stale))
    valid = [lab for lab in valid if lab["id"] not in drop]
    out: dict = {"n_labeled": len(valid), "n_stale_ignored": len(stale),
                 "n_excluded_ignored": sum(1 for lab in labels if lab["id"] in drop),
                 "per_model": {}, "judges": {}}
    for lab in valid:
        out["per_model"][lab["model"]] = out["per_model"].get(lab["model"], 0) + 1
    for jd in sorted(d for d in (BENCH / "judgments").iterdir() if d.is_dir()):
        verdicts = {f.stem: {r["id"]: r for r in load_jsonl(f)} for f in jd.glob("*.jsonl")}
        pairs, answerable, confusion = [], [], {}
        acc_h, acc_j = [], []
        by_group: dict[str, dict[str, list[float]]] = {}
        for lab in valid:
            v = verdicts.get(lab["model"], {}).get(lab["id"])
            if v is None:
                continue
            pair = (lab["verdict"], label_of(v))
            pairs.append(pair)
            answerable.append(lab["answerable"])
            confusion[f"{pair[0]}|{pair[1]}"] = confusion.get(f"{pair[0]}|{pair[1]}", 0) + 1
            if lab["answerable"]:
                acc_h.append(HUMAN_SCORE[lab["verdict"]])
                acc_j.append(score_of(v))
                g = by_group.setdefault(group_of(lab["model"]), {"human": [], "judge": []})
                g["human"].append(HUMAN_SCORE[lab["verdict"]])
                g["judge"].append(score_of(v))
        if not pairs:
            continue
        stats = agreement_stats(pairs, answerable)
        stats["confusion_human_judge"] = dict(sorted(confusion.items()))
        stats["answerable_accuracy_human"] = float(np.mean(acc_h)) if acc_h else None
        stats["answerable_accuracy_judge"] = float(np.mean(acc_j)) if acc_j else None
        # is the judge's over-estimate specific to our systems? (answerable items only)
        stats["answerable_by_group"] = {
            g: {"n": len(v["human"]), "accuracy_human": float(np.mean(v["human"])),
                "accuracy_judge": float(np.mean(v["judge"])),
                "gap": float(np.mean(v["judge"]) - np.mean(v["human"]))}
            for g, v in sorted(by_group.items())}
        out["judges"][jd.name] = stats
    (REVIEW / "summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    return out


class Handler(BaseHTTPRequestHandler):
    """Minimal request handler for the review page."""

    def log_message(self, *a) -> None:  # noqa: D401 -- quiet
        pass

    def _send(self, body: str) -> None:
        data = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        srv = self.server
        todo = [r for r in srv.todo if r["key"] not in srv.labels]  # type: ignore[attr-defined]
        done = len(srv.todo) - len(todo)  # type: ignore[attr-defined]
        if not todo:
            self._send(DONE_PAGE.format(n=done))
            return
        it = todo[0]
        if it["answerable"]:
            gold = ('<div class="box"><label>Risposta gold</label><pre>'
                    f'{html.escape(str(it["gold"]))}</pre></div>')
            buttons = ('<button class="ok" name="verdict" value="correct">Corretto</button>'
                       '<button class="mid" name="verdict" value="partial">Parziale</button>'
                       '<button class="bad" name="verdict" value="wrong">Sbagliato</button>')
        else:
            gold = ('<div class="box"><label>Domanda non rispondibile (abstention attesa)'
                    '</label></div>')
            buttons = ('<button class="abs" name="verdict" value="abstained">'
                       'Si è astenuto correttamente</button>'
                       '<button class="bad" name="verdict" value="answered">'
                       'Ha risposto (errore)</button>')
        page = PAGE
        for k, v in {"pos": done + 1, "tot": len(srv.todo), "done": done,  # type: ignore[attr-defined]
                     "key": html.escape(it["key"]), "id": it["id"],
                     "question": html.escape(it["question"]),
                     "answer": html.escape(it["answer"]) or "(vuota)",
                     "gold_block": gold, "buttons": buttons}.items():
            page = page.replace("{" + k + "}", str(v))
        self._send(page)

    def do_POST(self) -> None:  # noqa: N802
        body = parse_qs(self.rfile.read(int(self.headers["Content-Length"])).decode())
        key = body["key"][0]
        verdict = body.get("verdict", ["skip"])[0]
        it = next(s for s in self.server.todo if s["key"] == key)  # type: ignore[attr-defined]
        if verdict == "skip":
            self.server.todo.remove(it)  # type: ignore[attr-defined]
            self.server.todo.append(it)  # type: ignore[attr-defined]
        else:
            rec = {"key": key, "model": it["model"], "id": it["id"],
                   "answerable": it["answerable"], "verdict": verdict,
                   "answer_sha": row_sha(it), "rater": "human"}
            self.server.labels[key] = rec  # type: ignore[attr-defined]
            with (REVIEW / "labels.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.send_response(303)
        self.send_header("Location", "/")
        self.end_headers()


def main() -> None:
    """Entry point."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    if args.summary_only:
        write_summary(args.archive_stale)
        return
    sample = build_sample(args.models, args.per_model, args.seed)
    answers = current_answers()
    drop = excluded_ids()
    labels = {r["key"]: r for r in load_jsonl(REVIEW / "labels.jsonl")}
    todo = [s for s in sample if is_current(s, answers) and s["id"] not in drop]
    server = HTTPServer(("127.0.0.1", args.port), Handler)
    server.todo, server.labels = todo, labels  # type: ignore[attr-defined]
    log.info("%d item da validare (già fatti: %d) -> http://127.0.0.1:%d",
             sum(1 for s in todo if s["key"] not in labels),
             sum(1 for s in todo if s["key"] in labels), args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    write_summary(archive_stale=False)


if __name__ == "__main__":
    main()
