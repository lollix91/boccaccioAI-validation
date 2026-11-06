# Third-party data and model terms

This repository contains evaluation code, results and judge/answer files. It does not
redistribute the benchmark datasets themselves, with one exception: the per-item sample
archive (`boccaccioAI-validation-samples.zip`, published on Zenodo, not in this repository)
holds the text of every evaluated item (DOI
[10.5281/zenodo.24017497](https://doi.org/10.5281/zenodo.24017497)). The code is Apache-2.0
(`LICENSE`). The licenses below were read from the Hugging Face
dataset/model metadata and cards on 2026-11-06.

## Benchmark datasets (item text appears in the sample archive)

| lm-eval task | Dataset | License | Note |
|--------------|---------|---------|------|
| `arc_it` | `alexandrainst/m_arc` | CC BY-NC 4.0 | Non-commercial. Machine translation of ARC |
| `hellaswag_it` | `alexandrainst/m_hellaswag` | CC BY-NC 4.0 | Non-commercial. Machine translation of HellaSwag |
| `m_mmlu_it` | `alexandrainst/m_mmlu` | CC BY-NC 4.0 | Non-commercial. Translated with GPT-3.5-turbo (dataset card) |
| `truthfulqa_it_mc1/mc2` | `alexandrainst/m_truthfulqa` | CC BY-NC 4.0 | Non-commercial. Machine translation of TruthfulQA |
| `belebele_ita_Latn` | `facebook/belebele` | CC BY-SA 4.0 | Share-alike |
| `include_base_44_italian` | `CohereLabs/include-base-44` | Apache-2.0 | Items come from regional exams |
| `lambada_openai_mt_it` | `EleutherAI/lambada_openai` | MIT | |
| `squad_it` | `crux82/squad_it` | HF card: "unknown"; authors' GitHub README: CC BY-SA 4.0 | Derived from SQuAD 1.1 (Wikipedia text) |

Consequence for the sample archive: it combines non-commercial (`m_*`) and share-alike
(Belebele, SQuAD-IT) material, so it is published under **CC BY-NC-SA 4.0**. Do not reuse it
commercially. The licenses of the English originals behind the `m_*` translations were not
re-checked here.

The English originals were checked on 2026-11-06 (Hugging Face metadata): ARC
(`allenai/ai2_arc`) CC BY-SA 4.0, MMLU (`cais/mmlu`) MIT, TruthfulQA
(`truthfulqa/truthful_qa`) Apache-2.0, HellaSwag (`Rowan/hellaswag`) MIT (stated in the
"Licensing Information" section of the dataset card, which links the LICENSE file of
`rowanz/hellaswag`; the GitHub file itself could not be fetched). The `m_*` translations are
CC BY-NC 4.0 as stated by their authors.

## Models whose outputs are stored (answers, samples, judge verdicts)

| Model | License |
|-------|---------|
| Qwen2.5-0.5B, Qwen3-0.6B, SmolLM2-360M, Pythia-410M/1B, Minerva-350M/1B | Apache-2.0 |
| Gemma-3-1B-pt | Gemma Terms of Use (version of 2026-04-01). Outputs are not Model Derivatives and Google claims no rights in them (sections 1.1(e), 3.3), so storing outputs is allowed. The terms apply if the weights are redistributed, which this repository does not do |
| Llama-3.2-1B | Llama 3.2 Community License (gated); no attribution is required for outputs that are not Llama Materials |
| BoccaccioAI (ours) | Apache-2.0 |
| Teacher Qwen3-235B-A22B-Instruct-2507 (`qwen/qwen3-235b-a22b-2507` on OpenRouter) | Apache-2.0. The license has no clause restricting the use of outputs to train other models, so distilling BoccaccioAI from it is allowed |

## LLM judges

The two judges (`google/gemini-3.8-flash`, `deepseek/deepseek-v4.1-flash-1017`, called
through OpenRouter) produced the verdicts in `benchmark/judgments/` and `eval/judge/cache`.
Terms read on 2026-11-06:

- **OpenRouter** (ToS updated 2026-08-31): ownership of outputs follows each model's own
  terms (section 6.1); OpenRouter adds no restriction on redistributing outputs.
- **DeepSeek V4.1 Flash**: open weights under the MIT license; commercial use, distillation
  and derivatives are allowed.
- **Gemini**: the Gemini API Additional Terms (2026-03-23) state that Google claims no
  ownership of generated content, and forbid using the service to develop competing models.
  The verdicts are used only for evaluation, not for training. No clause explicitly allows or
  forbids redistributing outputs, and the Gemini model-specific terms as served through
  OpenRouter were not located, so a small residual uncertainty remains.

This is a reading of public terms, not legal advice.

## Other content in this repository

- `benchmark/benchmark_v2*.jsonl`, `benchmark/rag_contexts.jsonl`: questions and retrieved
  passages derive from Italian Wikipedia text (CC BY-SA 4.0).
- The held-out corpus (`data/heldout/`) is git-ignored and not published.
- The Apache-2.0 license of the code does not extend to the Wikipedia-derived text above.
