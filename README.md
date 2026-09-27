# KPViz

A local, reproducible cockpit for keyphrase extraction/generation evaluation.

**Paper:** Saber Zahhar, Christophe Rodrigues, Nédra Mellouli and Nicolas Travers.
*KPViz: A Framework for Keyphrase Prediction Experiments.* JCDL '26.
https://doi.org/10.1145/3805696.3846518
**Demo video:** https://youtu.be/WS2iCDfloYM ·
**Archive (code + demo data):** https://doi.org/10.5281/zenodo.22789345

KPViz scans a folder of dataset, model and architecture **cards** plus the
inference runs produced against them, derives an analytical store (DuckDB,
byte-offset indices — your corpus is never copied), and serves a web app:
explorers for datasets, models and architectures, and a set of
research-question workbenches whose figures export straight to the paper —
PGF / PDF / PNG with auto-written editable captions, and booktabs LaTeX
tables. Plotly renders the interactive view; Matplotlib's PGF backend renders
the export, from the same figure spec.

Everything runs on your machine; the only network access is the one-time
download of tokenizer assets (cached, with a flagged offline fallback).
`python app.py --offline` guarantees no network access at all: tokenizers
that are not already cached fall back to flagged approximations at once.

## Quick start

```bash
git clone https://github.com/saberzahhar/kpviz
cd kpviz

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python tools/make_sample_data.py   # optional: a small generated tree (~1 s)
python app.py                      # serves ./data, else ./sample_data
```

The browser opens on the **Overview** page. On an empty catalog the first
scan starts automatically; afterwards, **Scan for changes** re-derives only
what is new, modified or deleted, in parallel across all CPU cores, with
per-step progress and an ETA.

**Demo data.** A full demo tree (five datasets, seven models, three
architectures, seventy-five runs) is archived on Zenodo:

> https://doi.org/10.5281/zenodo.22789345

(mirror: https://drive.google.com/drive/folders/136Gj3Gv_rMyZHwdmBWIge0kI1BIAlQpj)

Download it and place it as a `data/` folder at the repository root, then run
`python app.py`. `app.py` needs a `data/` folder (or a `sample_data/` folder,
or `--data PATH`) to start; the repository itself ships no data, but
`tools/make_sample_data.py` generates a self-contained `sample_data/` tree
(five datasets, every card type, deliberate contract edge cases) and
`tools/make_scaled_data.py` scales it up for load testing.

Optional: any TeX distribution (TeX Live, MiKTeX) enables PGF-typeset PDF
exports — the figure is then set in your paper's own fonts. Without TeX,
PDF/PNG exports still work through Matplotlib.

## The data contract

KPViz is driven entirely by the data tree — nothing about your datasets,
models, metrics variables or costs is hardcoded. Keys are contractual;
values are yours.

```
data/
  documents/document.{dataset}.json      # dataset card
  documents/document.{dataset}.jsonl     # the collection (one document per line)
  models/model.{model}.json              # model card ("inference" = parameter schema)
  architectures/architecture.{arch}.json # cost variables + linear rate models
  insights/scores.jsonl                  # document similarity pairs (leakage)
  inferences/{dataset}/{model}/{arch}/{run}/
      run_{run}.json                     # the run's parameters
      batch_%05d.json                    # batch metadata (timestamps, batch-level costs)
      batch_%05d.jsonl                   # predictions: {"_id", "inferences", "costs"?}
```

Cards are JSON; `//` and `/* */` comments are accepted (JSONC), exactly as
in the examples below. A section or annotation set may declare its languages
as `"languages": ["en"]` or `"language": "en"`. A card that cannot be parsed
is listed with its line and column on the Overview page rather than
silently ignored; the same holds for malformed JSONL lines in collections
and runs.

Folder tokens resolve to cards by file-name token first, then by declared
ids/names (`openai_api` finds `architecture.api.json` through its `arch_id`).
A token like `n.a` resolves to nothing *on purpose*: its runs stay
first-class for quality analysis and appear as performance-only wherever
cost is an axis. Run parameters resolve against the model card — missing
ones take the card's default, none is invented, and illegal values are
flagged, never dropped.

### Dataset card — `documents/document.{dataset}.json`

Declares what a document is (sections, languages), which annotation sets
exist and who produced them, and any extra metadata fields.

```jsonc
{
  "description": "SemEval-2010 scholarly documents (title + full text).",
  "domain": "Academic",
  "sub-domain": "computer-sciences",
  "metadata": {
    "split":    { "type": "split" },
    "category": { "type": "classification", "taxonomy": "ACM CCS (1998)", "depth": 3 }
  },
  "document": {
    "title+full-text": { "type": ["title", "body"], "modality": "text", "languages": ["en"] }
  },
  "annotations": {
    "author": { "type": "author", "expertise": "author",  "languages": ["en"] },
    "reader": { "type": "reader", "expertise": "student", "languages": ["en"],
                "post_annotation": { "type": "expert" } }
  }
}
```

### Model card — `models/model.{model}.json`

Identity, lineage and capabilities, plus the **`inference` schema** every
run is validated against: types, legal ranges/values, defaults, and — for
context windows — the tokenizer that measures them.

```jsonc
{
  "model_id": "https://huggingface.co/taln-ls2n/bart-base-kp20k",
  "name": "bart-base-kp20k",
  "family": [["Neural model", "Paradigms", "One2Seq"],
             ["Pre-trained models", "BART"]],
  "backend": "transformers",
  "capabilities": { "extractive": true, "abstractive": true },
  "parameters": { "total": 139420416 },
  "supervision": ["kp20k"],
  "languages": ["en"],
  "references": { "url": "…", "bibtex": "…" },
  "inference": {
    "num_beams":       { "type": "int", "min": 1 },
    "input_max_size":  { "type": "context_window",
                         "tokenizer": "transformers[bart-base]",
                         "min": 1, "max": 1024, "default": 1024 },
    "output_max_size": { "type": "context_window",
                         "tokenizer": "transformers[bart-base]",
                         "min": 1, "max": 1024, "default": 1024 }
  }
}
```

### Architecture card — `architectures/architecture.{arch}.json`

Where inference physically ran. Declares raw cost **variables** (with a
`document` or `batch` level) and linear **rate** models per cost unit;
run costs are resolved against these — never invented.

```jsonc
{
  "arch_id": "openai_api",
  "name": "OpenAI API",
  "kind": "api",
  "variables": {
    "input_tokens":  { "unit": "token", "tokenizer": "tiktoken[o200k_base]", "level": "document" },
    "output_tokens": { "unit": "token", "tokenizer": "tiktoken[o200k_base]", "level": "document" },
    "time":          { "unit": "s", "level": "document" }
  },
  "rates": {                                  // usd = 2.5e-6·input + 1e-5·output
    "usd":  { "input_tokens": 2.5e-6, "output_tokens": 1e-5 },
    "time": { "time": 1.0 }
  }
}
```

### Similarity pairs — `insights/scores.jsonl`

One pair per line; used by the data-quality workbench (e.g. train→test
leakage against a model's own supervision data).

Both key spellings are accepted (`dataset_a`/`doc_id_a` and `dataset_A`/`doc_id_A`).

```jsonc
{ "dataset_a": "kp20k", "doc_id_a": "…", "dataset_b": "kpbiomed", "doc_id_b": "…",
  "score": 0.93, "label": "near-duplicate" }
```

### Runs — `inferences/{dataset}/{model}/{arch}/{run}/`

`run_{run}.json` carries the parameters actually used (validated against the
model card). Each `batch_*.jsonl` line is one document's predictions; costs
may live at the document level or in the batch's `batch_*.json` metadata —
KPViz maps whatever is mappable at its declared level.

```jsonc
// run_04b902a7dc7b.json
{ "parameters": { "num_beams": 4, "input_max_size": 512, "output_max_size": 128 } }

// batch_00000.jsonl (one line)
{ "_id": "kp20k_testing_0",
  "inferences": ["feedback vertex set", "…"],
  "costs": { "input_tokens": 143, "output_tokens": 31, "time": 0.8 } }
```

## What you get

**Overview** — scan control, catalog summary, and integrity at a glance:
every run carries `key:value` issue tags (illegal parameter, missing
architecture/run/dataset, incomplete coverage) and every collection its
data-quality flags, each counted with its share of the collection.

**Datasets / Models / Architectures** — card explorers with per-split
statistics, PRMU distributions, run tables validated against the cards,
resolved costs, and a document browser that reads your files in place
through the byte-offset index.

**Insights** — research-question workbenches (dataset correlation, data
quality & bias, extractability & truncation, cost–performance,
hyperparameters) with statistical inference built in and one-click export.

- *Statistics* (one setting for every workbench, so a paper never mixes
  procedures): rank-based tests (Wilcoxon signed-rank, Mann–Whitney U,
  Friedman), mean-based tests (paired t, Welch t, repeated-measures ANOVA) or
  resampling (paired sign-flip permutation — the approximate-randomisation
  test of the NLP literature — and bootstrap); Holm, Bonferroni or
  Benjamini–Hochberg correction across the runs of a table; 95 % Student-t
  or bootstrap intervals as error bars and in every table; effect sizes
  (rank-biserial r, Cohen's d_z, Hedges' g, Kendall's W, partial η²);
  Pearson, Spearman and Kendall τ-b between benchmarks with Fisher-z
  intervals. Dependency-free, verified against SciPy, and every resampled
  p-value is seeded from its data, so it is reproducible.
- *Export*: pick the paper (article, *ACL, ACM, IEEE, LNCS, NeurIPS/ICLR),
  the width (column or full text width), height and legend placement, and
  preview the figure at print size. Figures are drawn at the venue's real
  `\columnwidth`/`\textwidth` in its figure font — never rescaled — and go
  into `figure` or `figure*` accordingly; copy the LaTeX, or download
  PGF / PDF / PNG / a zip bundle. Tables are booktabs, with value, interval
  and significance mark per cell, a statistics note, and a caption that
  points to their figure. Captions are auto-written from the exact
  configuration and stay editable. Every workbench's export is compiled in
  the test suite inside article, IEEEtran, llncs and acmart documents
  (pdflatex, xelatex, lualatex) with nothing allowed into the margin.

Evaluation conventions are stated in every caption: predictions lowercased,
tokenised, stemmed and deduplicated keeping rank order; P/R/F1 at k ∈
{5, 10, O, M}; documents with no gold after filtering excluded with n
reported; scores macro-averaged. PRMU follows the in-order definition on
stemmed tokens (P — all tokens appear in order; R — all appear, never in
order; M — some; U — none).

## Reproducibility

The derived store holds indices and statistics only — deleting `.kpviz/`
and rescanning rebuilds everything from your files, deterministically.
Every scan archives its exact per-step timings to `.kpviz/scan_stats/`.
The `tools/` folder contains the verification harnesses used to check the
platform itself (metric parity between the SQL and Python paths, LaTeX
compilation of exports, headless scans); each takes `--data`/`--state` and
runs against any tree.

```bash
pip install pytest psutil playwright && playwright install chromium
python -m pytest tests                  # contract, parity, statistics vs SciPy, LaTeX in venue classes, UI (~2 min)
python tools/bench/scan_profile.py --data sample_data --state /tmp/st --full
python tools/bench/probe_ui.py http://127.0.0.1:8050   # against a running app
python tools/bench/fingerprint.py --state .kpviz   # per-table content hashes
```

The test suite generates its own data tree, checks every SQL score against
an independent reference implementation, verifies that the worker count
and incremental re-scans never change a derived number, and renders and
exports every workbench. `docs/PERFORMANCE_PLAN.md` records the
performance work and its measured results. `tools/kpviz_scaling_benchmark.py` is the self-contained
scaling benchmark behind Table 2 of the paper (`pip install nltk PyStemmer`,
then `python tools/kpviz_scaling_benchmark.py`); its original output is in
`docs/benchmark_results_jcdl26.txt`.

## Status

KPViz is under active development. The data contract above is stable; the
set of insights, statistics and supported card fields will keep growing.
Issues and suggestions are welcome.

## Citation

If you use KPViz, please cite:

```bibtex
@inproceedings{zahhar2026kpviz,
  author    = {Zahhar, Saber and Rodrigues, Christophe and Mellouli, N{\'e}dra and Travers, Nicolas},
  title     = {{KPViz}: A Framework for Keyphrase Prediction Experiments},
  booktitle = {The 2026 ACM/IEEE Joint Conference on Digital Libraries (JCDL '26)},
  series    = {JCDL '26},
  year      = {2026},
  publisher = {Association for Computing Machinery},
  address   = {New York, NY, USA},
  location  = {Frisco, TX, USA},
  isbn      = {979-8-4007-2597-5},
  doi       = {10.1145/3805696.3846518}
}
```

## License

See [LICENSE](LICENSE).
