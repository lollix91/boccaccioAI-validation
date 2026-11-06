# boccaccioAI-validation

Evaluation code, data and results for the BoccaccioAI paper
(<https://github.com/lollix91/boccaccioAI-validation>). Every table, figure and number of
the paper is generated from this repository by `eval/paper_assets.py`, and
`eval/consistency_check.py` re-derives the reported metrics from the raw files. The model
repository (`E:\boccaccioAI` in the paths below, published on the Hugging Face Hub as
`lollix91/boccaccioAI`) is treated as read-only: nothing here modifies it.

Training chain of the deployed model: pretraining stage 1 (CulturaX) -> stage 2
(Wikipedia + mC4) -> distillation from Qwen3-235B-A22B. SFT V1 (and V2/V3, started from V1)
were trained from the same stage-2 base but are not part of the deployed model; SFT V1 is
evaluated as an alternative post-training. The lineage is verified on the weights (the
distilled model is closer to the stage-2 base than to SFT V1) and in
`scripts/07_finetune_distill.sh` of the model repository.

## Layout

```
scripts/convert_to_hf.py        export a BoccaccioAI checkpoint as a Hugging Face Llama model
scripts/verify_equivalence.py   check exported logits against the original implementation
eval/bpb.py                     bits-per-byte on held-out text (tokenizer-agnostic)
eval/tasks/                     custom lm-eval tasks: truthfulqa_it_mc1/mc2, squad_it
eval/tasks.json                 task suite (identical for every model)
eval/run_all.py                 run the suite on one model with lm-evaluation-harness
eval/run_queue.py               evaluate the next unfinished model of eval/models.json, then stop
eval/models.json                the model queue (order matters)
eval/model_meta.json            display names, declared pretraining tokens (with sources), metrics
eval/paper_assets.py            generate the paper tables, figures and LaTeX macros from results
eval/consistency_check.py       re-derive every reported metric from the raw files (exit 1 on mismatch)
eval/cost_model.json            GPU hours of every training run and the total bill (paper Table cost)
eval/collect_samples.py         regenerate missing per-item samples (verified against the results)
eval/significance.py            paired tests vs the distilled model (McNemar, permutation, Holm)
eval/build_heldout.py           held-out corpus: it.wikipedia articles created after 2025-06-01
eval/fertility.py               tokenizer fertility on the held-out corpus
eval/judge/verify_questions.py  LLM verification of the benchmark-v2 candidate questions
eval/judge/rebuild_rejected.py  rebuild benchmark_v2_rejected.jsonl from the verifier cache
eval/judge/run_production.py    benchmark v2 through the production endpoint (prod-rag/norag)
eval/judge/retrieve_contexts.py production retrieval (same passages for every *-rag system)
eval/judge/judge.py             benchmark v2: gen (GPU) / judge (OpenRouter) / report
eval/judge/human_review.py      blind human validation of the judge verdicts (local web UI)
eval/judge/checks.py            equivalence checks local pipeline vs production endpoint
run_significance.ps1            collect_samples -> significance -> paper_assets
run_bpb.ps1                     build_heldout (if needed) -> bpb on every model -> paper_assets
data/heldout/                   held-out corpus and its build cache (git-ignored)
eval/pack_artifacts.py          pack/unpack the judge cache and the per-item samples (see "Data availability")
eval/results/<model>/           per-task JSON, run metadata; per-sample JSONL only after samples-unpack (git-ignored)
benchmark/                      benchmark v2 data, answers, verdicts and reports (see below)
benchmark/judge_cache.jsonl.gz  every judge/verifier response, keyed by prompt hash (1.3 MB)
DATA_LICENSES.md                licenses of the benchmark datasets and evaluated models
models/                         exported models (git-ignored)
dist/                           build output of pack_artifacts.py (git-ignored)
```

## Setup

```bash
python -m venv --system-site-packages .venv
.venv/Scripts/python -m pip install -r requirements.txt
```

Target hardware: one 12 GB GPU (RTX 3060).

## BoccaccioAI as a Hugging Face model

The architecture (RMSNorm, RoPE with `rotate_half`, GQA 3:1, SwiGLU, tied embeddings)
is numerically identical to `LlamaForCausalLM`, and parameter names already follow the
Llama convention. The export therefore reuses the original safetensors weights.

```bash
python scripts/convert_to_hf.py \
    --weights E:/boccaccioAI/checkpoints/finetune_distill/model.safetensors \
    --tokenizer-dir E:/boccaccioAI/tokenizer/boccaccio-32k-hf \
    --output-dir models/boccaccio-distill-hf
python scripts/verify_equivalence.py --hf-dir models/boccaccio-distill-hf --source-repo E:/boccaccioAI
```

Verified (fp32, GPU): 703,544,832 parameters in both implementations, max logit
difference 2.6e-5, identical top-1 predictions.

Tokenizer note: pretraining separated documents with `</s>` and never used `<s>`. The
exported `tokenizer_config.json` therefore sets `bos_token` to `</s>` (the prefix token
lm-eval-harness prepends for empty contexts) and does not add BOS automatically.

## Running the evaluation

```bash
powershell -File run.ps1           # PowerShell: run the whole queue, one model at a time (Ctrl+C to stop, re-run to resume)
python eval/run_queue.py           # evaluate the next unfinished model, then stop
python eval/run_queue.py --status  # print DONE/TODO/SKIPPED per model
python eval/run_all.py --model models/boccaccio-distill-hf --name boccaccio-distill --limit 30  # debug
```

A model is done when `eval/results/<name>/` holds `run_meta.json` plus one JSON per
suite task, so the queue is restartable. `gated` models (Gemma, Llama, Minerva) are
skipped unless `HF_TOKEN` is set (put it in `.env`, which `run.ps1` loads), leaving a `SKIPPED` marker. `sapienzanlp/Minerva-*` and `meta-llama/Llama-3.2-1B` also need the license accepted on their HF page; `google/gemma-3-1b-pt` is already accessible with the account token.

Task suite (see `eval/tasks.json`): ARC-IT, HellaSwag-IT, Belebele ita_Latn, MMLU-IT (Okapi,
5-shot), INCLUDE-base-44 Italian, LAMBADA-IT, plus custom task in `eval/tasks/`:
TruthfulQA-IT (built-in Okapi tasks `truthfulqa_it_mc1/mc2`, dataset `alexandrainst/m_truthfulqa` config `it`) and SQuAD-IT
(`crux82/squad_it`, fixed 2000-item test subset, EM / F1 / gold-containment). Library versions, seed (1234) and GPU are
saved in `run_meta.json`.

## Paper tables and figures

```bash
python eval/paper_assets.py   # writes Paper-BoccaccioAI/generated/ (override with --out-dir)
```

Reads every score and standard error from `eval/results/*/*.json`; static metadata
(display names, declared pretraining tokens with sources, task/metric choice) is in
`eval/model_meta.json`. Parameter counts are computed from the HF configs. The benchmark-v2
numbers come from `benchmark/judge_report.json`, `benchmark/human_review/summary.json` and
`benchmark/checks/checks.json`; the compute accounting from `eval/cost_model.json`. Outputs:
`tab_benchmarks.tex`, `tab_cost_tokens.tex`, `tab_fertility.tex`, `tab_rag.tex`,
`tab_cost.tex`, `results_macros.tex` (`\bres{<model>}{<key>}`), `fig_stage_ablation.pdf`,
`fig_score_vs_tokens.pdf`, `summary.json`.

```bash
python eval/consistency_check.py   # benchmark v2, human labels, lm-eval results, BPB/fertility, cost inputs
```

The check binds every judge verdict to the current answer text through the judge cache
(read from `benchmark/judge_cache.jsonl.gz` on a fresh clone), compares the per-item
samples with the aggregate results when they have been restored (otherwise that step is
skipped, see "Data availability") and verifies the cost inputs against the distillation log
of the model repository when it is available. Known,
documented difference: the SQuAD-IT per-item samples of the distilled model were
regenerated by `collect_samples.py` and differ from the original aggregate by one item
out of 2000 (BF16 generation nondeterminism, within the 0.005 tolerance); the
significance tests use the samples, the tables the aggregate.

## Data availability

| Artefact | Where | Size |
|----------|-------|------|
| Code, results JSON, benchmark v2 (items, answers, verdicts, human labels) | this repository | small |
| Judge cache (`benchmark/judge_cache.jsonl.gz`) | this repository | 1.3 MB |
| Per-item lm-eval samples (`*.samples.jsonl`, 108 files) | Zenodo, [10.5281/zenodo.24017497](https://doi.org/10.5281/zenodo.24017497): `boccaccioAI-validation-samples.zip` | 206 MB zipped, 3.0 GB unpacked |

The samples archive is built with `python eval/pack_artifacts.py samples-pack` (output in
`dist/`, with a `.sha256` file; the published file has SHA-256
`8a42cca338e072ae4d3f716b1f20ff82c02b1e77143c51170e505ba3230ee9dd`). To rerun the
significance tests from a fresh clone:

```bash
# download boccaccioAI-validation-samples.zip from the Zenodo record above
python eval/pack_artifacts.py samples-unpack --zip boccaccioAI-validation-samples.zip
python eval/significance.py
python eval/consistency_check.py
```

`samples-unpack` verifies the SHA-256 of every extracted file against the archive manifest.
The archive contains the text of the evaluated items and is licensed CC BY-NC-SA 4.0
because some sources are non-commercial (see `DATA_LICENSES.md`).

## Significance tests

```powershell
powershell -ExecutionPolicy Bypass -File run_significance.ps1
```

This regenerates samples from the models. To reuse the published samples instead, see
"Data availability".

Re-evaluates only the tasks without per-item samples (in `eval/results/<name>__samples/`),
requires the metrics to match the existing results exactly, copies the samples, then runs
`eval/significance.py` (exact McNemar for binary metrics, paired sign-flip permutation for
TruthfulQA mc2 and the average, 95% paired bootstrap CIs, Holm over all comparisons; output
`eval/results/significance.json`) and regenerates the paper assets.

## Bits per byte

```bash
python eval/bpb.py --model models/boccaccio-distill-hf --name boccaccio-distill     --text-file data/heldout.jsonl --field text --dataset-name heldout
```

Each chunk is prefixed with the BOS (or EOS) token; BPB = NLL / (ln 2 * UTF-8 bytes).
The script also reports bytes/token and token perplexity (not comparable across tokenizers).
Held-out corpus (`eval/build_heldout.py`, run automatically by `run_bpb.ps1`): 2400
candidate it.wikipedia articles created between 2025-06-01 and 2026-09-30 (after the release
of every evaluated model), current revision as plain text, reference sections removed;
the 836 documents with more than 10% of their 13-grams in `wikimedia/wikipedia`
`20231101.it` (our pretraining dump) are discarded, leaving 1,564 documents. Statistics in
`data/heldout/wiki_it_2025.stats.json`.

```powershell
powershell -ExecutionPolicy Bypass -File run_bpb.ps1
```

## Benchmark v2 (LLM-judged factual QA)

```
benchmark/benchmark_v2_candidates.jsonl  372 candidates (43 v1 + 329 new)
benchmark/benchmark_v2.jsonl             368 accepted after LLM verification (field `verifier`)
benchmark/benchmark_v2_rejected.jsonl    4 rejected candidates
benchmark/excluded.json                  2 duplicate items dropped from every score (366 scored:
                                         271 answerable, 95 unanswerable)
benchmark/rag_contexts.jsonl             shared retrieval output for the *-rag systems
benchmark/answers/<system>.jsonl         answers of the 18 scored systems (judge.py BENCH_SYSTEMS)
benchmark/judgments/<judge>/<system>.jsonl  verdicts of the two judges
benchmark/judge_report.json              scores, CIs, paired tests, inter-judge kappa
benchmark/human_review/                  blind human labels (sample.jsonl, labels.jsonl, summary.json)
benchmark/checks/                        equivalence checks with the production endpoint
benchmark/sensitivity/                   supplementary prompt-sensitivity variant (below)
```

`benchmark/sensitivity/` keeps the results of SFT-V1 generated with the generic
prompt instead of its `### Utente:`/`### Assistente:` chat format, together with the
verdicts of both judges (`<judge-slug>__<system>.jsonl`) and the corresponding 15 human
labels (`human_labels__<system>.jsonl`). It is a prompt-sensitivity check, not a
scored benchmark row: `judge.py report` ignores it, and `checks.py` writes its metrics
into `benchmark/checks/checks.json` under `sensitivity`. It never refuses but almost never
abstains on unanswerable items; on the items the chat-format variant answers it scores
similarly to it, and on the items the chat format refuses it is only slightly less
accurate, so the chat-format refusals are weakly targeted.

Refusals on answerable items (column Ref. of the paper's Table rag) are counted with the
lexical detector `refused()` in `eval/judge/judge.py`; abstention on unanswerable items is
always the judges' label.

Systems (`BENCH_SYSTEMS` in `eval/judge/judge.py`): the production endpoint with and
without retrieval (`prod-rag`, `prod-norag`, distilled model); the base and SFT-V1
checkpoints in their training format (`boccaccio-base-p2`: plain prompt like every base
model; `boccaccio-sft-v1-chat`: `### Utente:`/`### Assistente:` chat format); the nine
parametric baselines; five baselines with the production retrieval (`*-rag`: identical
passages, truecased question and abstention decision for every generator).

```bash
python eval/judge/run_production.py                    # prod-rag / prod-norag (Modal endpoint)
python eval/judge/retrieve_contexts.py --rag-dir E:/boccaccio-rag   # production index + metadata
python eval/judge/judge.py gen                         # local systems (GPU)
python eval/judge/judge.py judge --judge google/gemini-3.8-flash
python eval/judge/judge.py judge --judge deepseek/deepseek-v4.1-flash-1017
python eval/judge/judge.py report
python eval/judge/human_review.py                      # label the pending sample items
python eval/judge/judge.py gen --models boccaccio-distill-chat --out-dir benchmark/checks
python eval/judge/checks.py --rag-dir E:/boccaccio-rag
```

Scoring: answerable items correct 1 / partial 0.5 / wrong 0, unanswerable items 1 if the
model abstains; the headline score is the per-item mean of the two judges. Inter-judge and
human-judge agreement use Cohen's kappa on the categorical labels (linearly weighted kappa
on the ordinal answerable scale), computed separately for answerable and unanswerable
items. Judge calls are cached in `eval/judge/cache/` (git-ignored working copy), so
re-running `judge` on unchanged answers is free. The tracked copy is
`benchmark/judge_cache.jsonl.gz`: after new judge calls run
`python eval/pack_artifacts.py cache-pack` and commit it; on a fresh clone
`cache-unpack` restores the working copy.

Production artefacts (`index.faiss` with 5,674,447 e5-small passages from 1,842,359
it.wikipedia pages, `metadata.db`, `model.safetensors`) come from the Modal volume
`boccaccio-rag-data`: `modal volume get boccaccio-rag-data <file> E:/boccaccio-rag`.
`checks.py` verifies that the production weights equal the exported distilled model, that
the local retrieval reproduces the production abstentions and sources, and that the
distilled model generated locally in the chat format reproduces `prod-norag`.

## License and citation

Code: Apache License 2.0 (`LICENSE`). Third-party datasets and models keep their own terms,
listed in `DATA_LICENSES.md`; the Zenodo samples archive is CC BY-NC-SA 4.0.

If you use the samples archive, please cite:

> De Lauretis, L. (2026). *BoccaccioAI validation: per-item lm-evaluation-harness samples
> (108 files, 12 models x 9 tasks)* (Version 1.0) [Dataset]. Zenodo.
> https://doi.org/10.5281/zenodo.24017497