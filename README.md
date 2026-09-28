# KPViz

Evaluate keyphrase extraction and generation models on your own machine, and
put the figures straight into your paper.

KPViz reads a folder of **cards** (datasets, models, architectures) and the
**runs** produced with them. It scores every run, then serves a web app to
explore the collections, compare models, answer five research questions with
proper statistics, and export PDF/PGF/PNG figures and LaTeX tables with
ready-written captions. Your files are never copied: KPViz keeps an index and
statistics in DuckDB and reads documents in place.

**Paper:** Saber Zahhar, Christophe Rodrigues, Nédra Mellouli and Nicolas
Travers. *KPViz: A Framework for Keyphrase Prediction Experiments.* JCDL '26.
[doi:10.1145/3805696.3846518](https://doi.org/10.1145/3805696.3846518) ·
[demo video](https://youtu.be/WS2iCDfloYM) ·
[code + demo data](https://doi.org/10.5281/zenodo.22789345)

## Quick start

```bash
git clone https://github.com/saberzahhar/kpviz && cd kpviz
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python tools/make_sample_data.py   # optional: a small generated tree in sample_data/
python app.py                      # serves ./data, else ./sample_data
```

The browser opens on **Overview**. The first scan starts on its own; after
that, **Scan for changes** re-derives only the files that changed.

To try the full demo (5 datasets, 7 models, 3 architectures, 75 runs),
download the tree from [Zenodo](https://doi.org/10.5281/zenodo.22789345) and
place it as `data/` at the repository root.

Useful options (`python app.py --help` lists all):

| Option | What it does |
|---|---|
| `--data PATH` | the data tree (default `./data`, else `./sample_data`) |
| `--state PATH` | where the DuckDB store lives (default `.kpviz/` next to the data) |
| `--workers N` | scan processes (default: every core) |
| `--token-scope eval` | count model tokens on evaluation splits only (faster scans of huge training splits; default `all`) |
| `--gold-scope all` | keep per-keyphrase gold rows for training splits too (default: evaluation splits) |
| `--offline` | never use the network; uncached tokenizers become flagged approximations |

Optional extras: a TeX distribution (PGF figures set in your paper's fonts;
without it PDF/PNG still work), and `HF_TOKEN` in the environment for gated
Hugging Face tokenizers (e.g. Llama). For a very large tree, derive it once
without the UI: `python tools/scan_once.py --data PATH`.

## Your data

Everything is driven by the tree; nothing about your datasets, models or
costs is hard-coded.

```
data/
  documents/document.{dataset}.json      dataset card
  documents/document.{dataset}.jsonl     the collection, one document per line
  models/model.{model}.json              model card (with its inference-parameter schema)
  architectures/architecture.{arch}.json cost variables and rates
  insights/scores.jsonl                  optional: document similarity pairs
  inferences/{dataset}/{model}/{arch}/{run}/
      run_{run}.json                     the parameters the run used
      batch_00000.json                   optional: batch metadata (timestamps, batch costs)
      batch_00000.jsonl                  predictions, one document per line
```

Cards are JSON with `//` and `/* */` comments allowed. Examples, trimmed:

```jsonc
// documents/document.semeval-2010.json
{ "description": "SemEval-2010 scholarly documents",
  "metadata":    { "split": { "type": "split" } },
  "document":    { "title+full-text": { "type": ["title", "body"], "languages": ["en"] } },
  "annotations": { "author": { "type": "author", "languages": ["en"] },
                   "reader": { "type": "reader", "languages": ["en"] } } }

// one line of documents/document.semeval-2010.jsonl
{ "_id": "C-41", "metadata": { "split": "test" },
  "sections":    [ { "field": "title+full-text", "content": "…" } ],
  "annotations": [ { "annotator": "author", "keyphrases": ["grid computing", "…"] } ] }

// models/model.bart-base-kp20k.json
{ "name": "bart-base-kp20k", "backend": "transformers", "supervision": ["kp20k"],
  "capabilities": { "extractive": true, "abstractive": true },
  "inference": {
    "num_beams":      { "type": "int", "min": 1 },
    "input_max_size": { "type": "context_window", "tokenizer": "transformers[facebook/bart-base]",
                        "min": 1, "max": 1024, "default": 1024 } } }

// architectures/architecture.api.json
{ "arch_id": "openai_api", "kind": "api",
  "variables": { "input_tokens":  { "unit": "token", "level": "document",
                                    "tokenizer": "tiktoken[o200k_base]" },
                 "output_tokens": { "unit": "token", "level": "document",
                                    "tokenizer": "tiktoken[o200k_base]" } },
  "rates": { "usd": { "input_tokens": 2.5e-6, "output_tokens": 1e-5 } } }

// inferences/…/run_04b902a7.json, then one line of a batch_*.jsonl
{ "parameters": { "num_beams": 4, "input_max_size": 512 } }
{ "_id": "C-41", "inferences": ["grid computing", "…"],
  "costs": { "input_tokens": 1430, "output_tokens": 31 } }

// insights/scores.jsonl (one pair per line)
{ "dataset_a": "kp20k", "doc_id_a": "…", "dataset_b": "kpbiomed", "doc_id_b": "…",
  "score": 0.93, "label": "near-duplicate" }
```

What KPViz does with them:

- **Run parameters** are checked against the model card: a missing one takes
  the card's default, an illegal one is flagged, never dropped.
- **Folder names** find their card by file name, then by declared id or name.
  A run whose card is missing stays usable; what needs the card (cost,
  validation) is shown as unknown.
- **Costs** are the architecture's linear rates applied to the variables the
  runs report, per document or per batch. Nothing is invented: a cost that
  cannot be resolved is drawn as "unknown", not zero.
- **Tokenizers** are `transformers[<owner>/<repo>]`,
  `transformers[file:/path/tokenizer.json]` or `tiktoken[<encoding>]`
  (`llama3`, `llama-3.3` and similar names resolve to the Llama 3
  tokenizer). Assets are downloaded once and cached; a tokenizer that cannot
  be loaded is replaced by a flagged approximation, with the reason on the
  Overview page.
- **Problems** (unreadable cards, malformed lines, repeated document ids,
  duplicate or empty gold keyphrases, a detected language that contradicts
  the declared one) are counted and listed under *Needs attention* on the
  Overview page, never silently ignored.

## The app

- **Overview** — scan with progress and ETA, catalog size, and one table of
  everything that needs attention.
- **Datasets** — per split: document lengths (in words or in any declared
  model's tokens, against the models' input windows), PRMU classes,
  keyphrase length and part-of-speech patterns; a document browser with the
  gold, and "why did this run score this?" for any run and document.
- **Models** and **Architectures** — the cards, their runs (checked against
  the card), and costs.
- **Insights** — five workbenches, each a figure, a table and an export:
  1. Do datasets rank systems the same way? (correlation between benchmarks)
  2. Does poor input data move the scores? (language mismatch, train–test
     similarity)
  3. How much does a bounded input window cost? (present gold inside vs.
     beyond each model's window, and scores along document length)
  4. What does a point of quality cost? (Pareto frontier in USD, kWh or time)
  5. How do hyperparameters move the needle? (controlled sweeps)

A model keeps one colour everywhere, its runs are lighter shades of it, and
an architecture keeps one marker shape. Each workbench shows the few controls
most people change; the rest are under *More options*.

**Statistics** (one setting for all workbenches): rank tests (Wilcoxon,
Mann–Whitney, Friedman), mean tests (paired t, Welch t, repeated-measures
ANOVA) or resampling (permutation, bootstrap); Holm, Bonferroni or
Benjamini–Hochberg correction; 95 % intervals; effect sizes. Verified against
SciPy; resampled p-values are seeded from the data, so they are reproducible.

**Export**: PDF, PNG or the LaTeX `figure` snippet in one click; under
*Caption & export options*, the paper style (article, *ACL, ACM, IEEE, LNCS,
NeurIPS/ICLR), width, height, legend, PGF, a zip bundle, a booktabs table and
a print-size preview. Figures are drawn at the venue's real column or text
width, never rescaled. Captions are written from the exact configuration and
stay editable.

## How scores are computed

- **Text**: NFKC-normalised, lowercased, with Windows-1252 bytes mis-read as
  Latin-1 (`d\x92analyse`) repaired. Words are runs of letters, digits and
  combining marks, so accented Latin, Arabic with harakat and Indic scripts
  stay whole; each Chinese or Japanese character is a word. Words are
  stemmed with the Snowball stemmer of their declared language (28
  languages, Porter2 for English; others are matched unstemmed).
- **Predictions** are de-duplicated after stemming, keeping rank order;
  **gold** is de-duplicated per annotation set, and keyphrases with no word
  are dropped (both counted). A `+` between two words separates gold
  variants (`C++` stays whole).
- **Metrics**: P, R and F1 at 5, 10, O (the number of gold keyphrases) and M
  (all predictions); P@k = tp / min(k, #predictions), without padding.
  Scores are per document, then macro-averaged; documents without gold after
  filtering are excluded and counted.
- **PRMU** (Boudin & Gallina, 2021), on stemmed words: **P** the keyphrase
  occurs as a contiguous sequence, in order; **R** all its words occur but
  not as that sequence; **M** some occur; **U** none. A contiguous
  occurrence stays inside one section and does not cross a separating
  punctuation mark (one next to a space, like a comma or full stop);
  word-internal marks (`e-commerce`, `and/or`, `l'analyse`) do not break it.
  A PRMU filter restricts the gold only: every prediction still counts.
- **One line per document**: a repeated document id in a collection, or a
  document predicted twice by a run, keeps its first line; both are counted
  (`duplicate_doc_ids`, `duplicate_docs`).

Every caption states these conventions, so a figure carries its method.

## Scale

A scan runs in parallel on every core and re-derives only what changed. On 4
cores, 263 000 documents (430 MB) take about 60 s for the documents phase;
exact model-token counts are the main extra cost on large training splits
(`--token-scope eval` skips them outside evaluation splits). The UI reads
precomputed aggregates, so pages stay interactive on millions of documents.

## Development

```bash
pip install -r requirements-dev.txt -c constraints.txt    # the tested versions
python -m pytest tests -rs                                 # what CI runs on every push
python tools/bench/profile_scan.py --data TREE --state /tmp/p   # per-function scan profile
python tools/bench/probe_ui.py http://127.0.0.1:8050      # callbacks and latency of a running app
python tools/synth_tree.py --cards MY_CARDS --out synth    # synthetic documents and runs around your cards
```

The tests generate their own data, check every SQL score against an
independent implementation and a hand-computed oracle, verify that worker
count and incremental scans never change a number, and compile every export
inside article, IEEEtran, llncs and acmart. Every export bundle carries a
`provenance.json` (code, schema and catalog versions, input fingerprint).
`tools/kpviz_scaling_benchmark.py` is the benchmark behind Table 2 of the
paper; `docs/PERFORMANCE_PLAN.md` records the performance work.

## Citation

```bibtex
@inproceedings{zahhar2026kpviz,
  author    = {Zahhar, Saber and Rodrigues, Christophe and Mellouli, N{\'e}dra and Travers, Nicolas},
  title     = {{KPViz}: A Framework for Keyphrase Prediction Experiments},
  booktitle = {The 2026 ACM/IEEE Joint Conference on Digital Libraries (JCDL '26)},
  year      = {2026},
  publisher = {Association for Computing Machinery},
  address   = {New York, NY, USA},
  doi       = {10.1145/3805696.3846518}
}
```

## License

See [LICENSE](LICENSE).
