"""Shared RAG retrieval for benchmark v2, identical to the production endpoint.

Replicates ``E:/boccaccioAI/modal_app.py`` (``_retrieve``, ``_clean_chunk``,
``_truecase``, ``_build_prompt`` context, abstention gate) on the production
artefacts downloaded from the Modal volume ``boccaccio-rag-data`` (``index.faiss``,
``metadata.db``): e5-small query embedding, exact top-``n`` inner-product search,
bge-reranker-v2-m3 rescoring, top-``k`` context, abstention when the top reranker
score is below the threshold. Retrieval depends only on the question, so it is run
once and every generator of ``judge.py gen`` (``*-rag`` systems) receives exactly
the same context, truecased question and abstention decision.

Output ``benchmark/rag_contexts.jsonl``: {id, question_truecased, context, sources,
top_score, abstained, abstention_message}. The equivalence with the production
retrieval is verified by ``eval/judge/checks.py``.

    python eval/judge/retrieve_contexts.py --rag-dir E:/boccaccio-rag
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from judge import load_jsonl  # noqa: E402

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "benchmark"
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
ABSTENTION_THRESHOLD = 0.85
ABSTENTION_MESSAGE = "Non ho informazioni sufficienti per rispondere a questa domanda."
MAX_CONTEXT_CHARS = 2500
TOP_K = 3
TOP_N = 20


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--rag-dir", type=Path, default=Path("E:/boccaccio-rag"))
    return p.parse_args()


def truecase(question: str, chunks: list[dict]) -> str:
    """Production truecasing: proper nouns from retrieved titles, capitalised first letter."""
    result = question
    proper: dict[str, str] = {}
    for c in chunks:
        for word in re.findall(r"[A-Z][a-zà-ùA-Z]+", c["title"]):
            proper[word.lower()] = word
    for lower, original in proper.items():
        result = re.sub(r"\b" + re.escape(lower) + r"\b", original, result,
                        flags=re.IGNORECASE)
    if result and result[0].islower():
        result = result[0].upper() + result[1:]
    return result


def clean_chunk(text: str, title: str) -> str:
    """Production chunk cleaning: drop the partial sentences at both ends."""
    prefix = f"{title}. "
    body = text[len(prefix):] if text.startswith(prefix) else text
    if body and body[0].islower():
        dot = body.find(". ")
        if dot != -1 and dot + 2 < len(body):
            body = body[dot + 2:]
    last_dot = body.rfind(". ")
    if last_dot != -1 and last_dot > len(body) * 0.5:
        body = body[:last_dot + 1]
    cleaned = f"{title}. {body}".strip()
    return cleaned if len(cleaned) > len(prefix) + 20 else text


def main() -> None:
    """Entry point."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    # Windows import order: datasets/pyarrow before torch (else pyarrow segfaults),
    # torch (via sentence_transformers) before faiss (conflicting OpenMP runtimes)
    import datasets  # noqa: E402,F401
    from sentence_transformers import CrossEncoder, SentenceTransformer  # noqa: E402
    import numpy as np  # noqa: E402
    import faiss  # noqa: E402

    # memory-mapped: the 8.7 GB flat index does not fit next to the models in RAM
    index = faiss.read_index(str(args.rag_dir / "index.faiss"), faiss.IO_FLAG_MMAP)
    log.info("index: %s, ntotal=%d, d=%d", type(index).__name__, index.ntotal, index.d)
    db = sqlite3.connect(str(args.rag_dir / "metadata.db"))
    embedder = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
    reranker = CrossEncoder(RERANKER_MODEL, device="cuda")
    items = load_jsonl(BENCH / "benchmark_v2.jsonl")
    emb = embedder.encode([f"query: {it['question']}" for it in items],
                          normalize_embeddings=True, show_progress_bar=False, batch_size=64)
    if hasattr(index, "nprobe"):
        index.nprobe = min(64, index.nlist)
    scores, indices = index.search(np.asarray(emb, dtype=np.float32), TOP_N)
    out = []
    for it, sc, ix in zip(items, scores, indices):
        cands = []
        for s, i in zip(sc, ix):
            if i < 0:
                continue
            row = db.execute("SELECT title, text FROM chunks WHERE idx = ?", (int(i),)).fetchone()
            if row is not None:
                cands.append({"title": row[0], "text": row[1], "bi_encoder_score": float(s)})
        rs = reranker.predict([(it["question"], c["text"]) for c in cands],
                              show_progress_bar=False)
        for c, r in zip(cands, rs):
            c["score"] = float(r)
        cands.sort(key=lambda c: c["score"], reverse=True)
        chunks = cands[:TOP_K]
        abstained = bool(chunks) and chunks[0]["score"] < ABSTENTION_THRESHOLD
        context = "\n\n".join(clean_chunk(c["text"], c["title"]) for c in chunks)
        out.append({"id": it["id"], "question_truecased": truecase(it["question"], chunks),
                    "context": context[:MAX_CONTEXT_CHARS],
                    "sources": [{"title": c["title"], "score": c["score"],
                                 "bi_encoder_score": c["bi_encoder_score"]} for c in chunks],
                    "top_score": chunks[0]["score"] if chunks else None,
                    "abstained": abstained, "abstention_message": ABSTENTION_MESSAGE})
    path = BENCH / "rag_contexts.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out),
                    encoding="utf-8")
    log.info("written %d contexts (%d abstained) -> %s", len(out),
             sum(r["abstained"] for r in out), path)


if __name__ == "__main__":
    main()
