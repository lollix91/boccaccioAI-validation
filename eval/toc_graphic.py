"""Draw the Table-of-Contents graphic (55 mm x 50 mm) from ``summary.json``.

Single panel: average score on the nine Italian benchmarks against declared pretraining
tokens (log scale), with BoccaccioAI highlighted. All numbers come from the same
``summary.json`` as the paper figures, so nothing is typed by hand.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

log = logging.getLogger(__name__)

PAPER = Path(r"D:\Repository\Papers E Presentazioni\Paper-BoccaccioAI")
MM = 1 / 25.4
LABELS = {"minerva-1b": "Minerva-1B", "qwen3-0.6b": "Qwen3-0.6B", "gemma-3-1b-pt": "Gemma-3-1B",
          "llama-3.2-1b": "Llama-3.2-1B"}
OFFSET = {"minerva-1b": (3, 3, "left"), "qwen3-0.6b": (-3, 3, "right"),
          "gemma-3-1b-pt": (4, -7, "left"), "llama-3.2-1b": (-3, 3, "right")}
GREY, RED = "#8c8c8c", "#c0272d"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments (``summary``, ``out_dir``).
    """
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary", type=Path, default=PAPER / "generated" / "summary.json")
    p.add_argument("--out-dir", type=Path, default=PAPER / "submission")
    return p.parse_args()


def draw(summary: dict, out_dir: Path) -> None:
    """Render the ToC graphic as vector PDF and 600 dpi PNG.

    Parameters
    ----------
    summary : dict
        Parsed ``summary.json`` written by ``eval/paper_assets.py``.
    out_dir : Path
        Output directory.
    """
    rows = [r for r in summary["rows"] if r["group"] != "ours" or r["name"] == "boccaccio-distill"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 6})
    fig, ax = plt.subplots(figsize=(55 * MM, 50 * MM))
    for r in rows:
        ours = r["group"] == "ours"
        x, y = r["pretrain_tokens"], 100 * r["avg"]
        ax.scatter(x, y, s=34 if ours else 9, color=RED if ours else GREY, zorder=3,
                   marker="*" if ours else "o", edgecolor="black", linewidth=0.3)
        if r["name"] in LABELS:
            dx, dy, ha = OFFSET[r["name"]]
            ax.annotate(LABELS[r["name"]], (x, y), textcoords="offset points", xytext=(dx, dy),
                        fontsize=5, ha=ha, color="#333333")
    ours = next(r for r in rows if r["group"] == "ours")
    ax.annotate("BoccaccioAI\n< $100", (ours["pretrain_tokens"], 100 * ours["avg"]),
                textcoords="offset points", xytext=(-6, 13), fontsize=5.5, color=RED,
                fontweight="bold")
    ax.set_xscale("log")
    ax.set_xlabel("Pretraining tokens", fontsize=6, labelpad=1)
    ax.set_ylabel("Italian benchmarks, avg. score (%)", fontsize=5.5, labelpad=1)
    ax.tick_params(labelsize=5, length=2, pad=1)
    ax.spines[["top", "right"]].set_visible(False)
    xs = [r["pretrain_tokens"] for r in rows]
    ax.set_xlim(min(xs) / 2, max(xs) * 3)
    fig.subplots_adjust(left=0.2, right=0.97, bottom=0.17, top=0.97)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "toc_graphic.pdf")
    fig.savefig(out_dir / "toc_graphic.png", dpi=600)
    plt.close(fig)
    log.info("Wrote %s", out_dir / "toc_graphic.pdf")


def main() -> None:
    """Entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    draw(json.loads(args.summary.read_text(encoding="utf-8")), args.out_dir)


if __name__ == "__main__":
    main()
