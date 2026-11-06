"""Pack and unpack the two bulky artefacts that are not stored as loose files in git.

* judge cache (14k small JSON files, written by ``eval/judge/judge.py`` and
  ``verify_questions.py``) <-> ``benchmark/judge_cache.jsonl.gz`` (tracked in git)
* per-item lm-eval samples (``eval/results/<model>/<task>.samples.jsonl``, ~2.9 GB)
  <-> ``dist/boccaccioAI-validation-samples.zip`` (published on Zenodo, git-ignored)

Both archives are deterministic (sorted entries, fixed timestamps), so a rebuilt archive of
unchanged data has the same SHA-256.

    python eval/pack_artifacts.py cache-pack
    python eval/pack_artifacts.py cache-unpack
    python eval/pack_artifacts.py samples-pack
    python eval/pack_artifacts.py samples-unpack --zip path/to/samples.zip
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "eval" / "judge" / "cache"
CACHE_ARCHIVE = ROOT / "benchmark" / "judge_cache.jsonl.gz"
RESULTS = ROOT / "eval" / "results"
SAMPLES_ZIP = ROOT / "dist" / "boccaccioAI-validation-samples.zip"
MANIFEST = "MANIFEST.sha256"
FIXED_TIME = (2026, 1, 1, 0, 0, 0)
CHUNK = 1 << 20

SAMPLES_README = """\
BoccaccioAI validation: per-item lm-evaluation-harness samples
================================================================

One file per (model, task): <model>/<task>.samples.jsonl, as written by lm-eval
(--log_samples). They are the input of eval/significance.py in
https://github.com/lollix91/boccaccioAI-validation.

Restore them into a clone of that repository:

    python eval/pack_artifacts.py samples-unpack --zip <this file>
    python eval/significance.py

MANIFEST.sha256 lists the SHA-256 of every file (sha256sum format).

License: CC BY-NC-SA 4.0. The files contain the text of the evaluated items, some of which
come from non-commercial (CC BY-NC) and share-alike (CC BY-SA) datasets. See DATA_LICENSES.md
(included) for the license of each source.
"""


def sha256_file(path: Path) -> str:
    """SHA-256 of a file, read in chunks.

    Parameters
    ----------
    path : Path
        File to hash.

    Returns
    -------
    str
        Hex digest.
    """
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(CHUNK):
            h.update(block)
    return h.hexdigest()


def load_cache(archive: Path = CACHE_ARCHIVE) -> dict[str, dict]:
    """Read the packed judge cache.

    Parameters
    ----------
    archive : Path
        ``judge_cache.jsonl.gz``.

    Returns
    -------
    dict[str, dict]
        ``{cache key: cached response}``, the key being the file stem of the loose cache.
    """
    out = {}
    with gzip.open(archive, "rt", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            out[row["key"]] = row["value"]
    return out


def cache_pack(src: Path, dst: Path) -> None:
    """Pack the loose judge cache into one gzip-compressed JSONL file."""
    files = sorted(src.glob("*.json"))
    if not files:
        sys.exit(f"no cache files in {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("wb") as raw, gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0,
                                              compresslevel=9) as gz:
        out = io.TextIOWrapper(gz, encoding="utf-8", newline="\n")
        for p in files:
            value = json.loads(p.read_text("utf-8"))
            out.write(json.dumps({"key": p.stem, "value": value}, ensure_ascii=False,
                                 separators=(",", ":")) + "\n")
        out.flush()
        out.detach()
    print(f"{len(files)} entries -> {dst} ({dst.stat().st_size / 1e6:.1f} MB)")


def cache_unpack(src: Path, dst: Path) -> None:
    """Restore the loose judge cache (existing files are kept)."""
    dst.mkdir(parents=True, exist_ok=True)
    cache = load_cache(src)
    new = 0
    for key, value in cache.items():
        p = dst / f"{key}.json"
        if not p.exists():
            p.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
            new += 1
    print(f"{len(cache)} entries in the archive, {new} written to {dst}")


def sample_files(results: Path) -> list[Path]:
    """All per-item sample files, in a stable order."""
    return sorted(results.glob("*/*.samples.jsonl"))


def samples_pack(results: Path, dst: Path) -> None:
    """Build the Zenodo archive of per-item samples with a SHA-256 manifest."""
    files = sample_files(results)
    if not files:
        sys.exit(f"no *.samples.jsonl under {results}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED, compresslevel=6,
                         allowZip64=True) as zf:
        for p in files:
            name = f"{p.parent.name}/{p.name}"
            info = zipfile.ZipInfo(name, FIXED_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            h = hashlib.sha256()
            with p.open("rb") as src, zf.open(info, "w", force_zip64=True) as out:
                while block := src.read(CHUNK):
                    h.update(block)
                    out.write(block)
            lines.append(f"{h.hexdigest()}  {name}")
            print(f"  {name}")
        zf.writestr(zipfile.ZipInfo(MANIFEST, FIXED_TIME), "\n".join(lines) + "\n")
        zf.writestr(zipfile.ZipInfo("README.txt", FIXED_TIME), SAMPLES_README)
        zf.writestr(zipfile.ZipInfo("DATA_LICENSES.md", FIXED_TIME),
                    (ROOT / "DATA_LICENSES.md").read_text("utf-8"))
    digest = sha256_file(dst)
    Path(str(dst) + ".sha256").write_text(f"{digest}  {dst.name}\n", encoding="utf-8")
    print(f"{len(files)} files -> {dst} ({dst.stat().st_size / 1e9:.2f} GB), sha256 {digest}")


def samples_unpack(zip_path: Path, results: Path) -> None:
    """Extract the samples archive into ``results`` and verify every file hash."""
    with zipfile.ZipFile(zip_path) as zf:
        manifest = zf.read(MANIFEST).decode("utf-8").splitlines()
        expected = {}
        for line in manifest:
            digest, name = line.split("  ", 1)
            expected[name] = digest
        for name in expected:
            zf.extract(name, results)
    bad = [n for n, d in expected.items() if sha256_file(results / n) != d]
    if bad:
        sys.exit(f"hash mismatch: {bad}")
    print(f"{len(expected)} sample files extracted to {results}, all hashes verified")


def main() -> None:
    """Command-line entry point."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("command", choices=["cache-pack", "cache-unpack", "samples-pack",
                                       "samples-unpack"])
    p.add_argument("--zip", type=Path, default=SAMPLES_ZIP, help="samples archive path")
    p.add_argument("--results-dir", type=Path, default=RESULTS)
    p.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    p.add_argument("--cache-archive", type=Path, default=CACHE_ARCHIVE)
    a = p.parse_args()
    if a.command == "cache-pack":
        cache_pack(a.cache_dir, a.cache_archive)
    elif a.command == "cache-unpack":
        cache_unpack(a.cache_archive, a.cache_dir)
    elif a.command == "samples-pack":
        samples_pack(a.results_dir, a.zip)
    else:
        samples_unpack(a.zip, a.results_dir)


if __name__ == "__main__":
    main()
