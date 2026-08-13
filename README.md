# KPViz — a keyphrase evaluation cockpit

KPViz scans a tree of dataset / model / architecture cards and inference
runs, derives an analytical DuckDB store (byte-offset indices — it never
copies your corpus), and serves a Dash app: entity explorers for datasets,
models and architectures, plus five research-question workbenches whose
figures export to **PGF / PDF / PNG with auto-written, editable captions**
and booktabs LaTeX tables — Plotly for the interactive layer, Matplotlib's
PGF backend for what goes in the paper.

## Install

Python 3.10+ recommended.

```bash
cd KPViz
python -m venv .venv
source .venv/bin/activate          # Windows Command Prompt: call .venv/Scripts/activate
pip install -r requirements.txt
```

That single file includes everything: Dash + DuckDB + Matplotlib +
Snowball stemmers, the exact tokenizers (`tokenizers` for
`transformers[bart-base]`, `tiktoken` for `tiktoken[o200k_base]`) and the
spaCy en/fr models used to POS-tag gold keyphrases.

Two first-use downloads happen lazily and are cached under `.kpviz/`:
the Hugging Face `bart-base` tokenizer and the `o200k_base` tiktoken
table. **Fully offline?** Drop a `tokenizer.json` into
`.kpviz/tokenizers/<name>/` — otherwise KPViz falls back to a word-ratio
heuristic and every affected number carries an `approximate` flag in the
DB and captions. Nothing breaks.

**Optional — for PGF-typeset exports:** any TeX distribution (TeX Live,
MiKTeX). KPViz probe-compiles `lualatex` → `xelatex` → `pdflatex` and uses
the first that actually works; with no TeX, PDF/PNG exports still work
through Matplotlib and the LaTeX snippet switches to `\includegraphics`.

## Run

```bash
python app.py                    # serves ./data, else ./sample_data
python app.py --data /path/to/data --port 8050
```

The browser opens on the **Overview** page. On an empty catalog the first
scan starts automatically; afterwards press **Scan for changes** — only
new / modified / deleted files (size+mtime fast path, BLAKE2 confirmation)
are re-derived, in parallel across all CPU threads, with per-step progress
and an ETA. A bundled `sample_data/` tree (real SemEval-2010 + TALN
subsets, the real bart-base-kp20k beams=4 run, synthetic minis of
kp20k/KPBiomed/KPTimes with seeded leakage, an illegal-parameter run, an
incomplete run and an `n.a` architecture) lets you demo everything
immediately.

Useful flags: `--workers N` (scan processes), `--token-scope all` /
`--gold-scope all` (default `eval` keeps tokenizer counts and gold
instance rows to testing+validation splits; distribution charts always
cover every split through aggregates), `--state DIR` (DuckDB store
location, default `.kpviz` beside the data root), `--no-autoscan`,
`--no-browser`, `--host/--port`.

Tuning flags — every one of them is auto-derived from the machine, so set
them only to override what the Overview page reports:

| flag | default | why you would change it |
|---|---|---|
| `--workers N` | usable CPUs (affinity + cgroup quota aware) | leave headroom for other jobs on a shared box |
| `--db-threads N` | `max(2, cpus − workers)`, raised to all cores between scan phases | a very wide box where DuckDB should own more threads |
| `--io-workers N` | `min(64, 2 × workers)` | network filesystems, where hashing waits on I/O rather than CPU |
| `--hash auto\|always` | `auto` — hash only when size/mtime moved | `always` after restoring a backup that reset mtimes |

`--pos-scope`, `--token-scope` and `--gold-scope` widen derivation from the
evaluation splits to every split; nothing is ever sampled or truncated at
either setting.

## The data contract

```
data/
  documents/document.{dataset}.json      # dataset card
  documents/document.{dataset}.jsonl     # the collection
  models/model.{model}.json              # model card ("inference" = param schema)
  architectures/architecture.{arch}.json # cost variables + linear rates
  insights/scores.jsonl                  # similarity pairs (leakage)
  inferences/{dataset}/{model}/{arch}/{run}/
      run_{run}.json                     # parameters
      batch_%05d.json                    # batch meta (timestamps, batch costs)
      batch_%05d.jsonl                   # {"_id", "inferences", "costs"?}
```

Keys are contractual; values are data. Folder tokens link to cards by
file-name token first, then by declared ids/names (`openai_api` finds
`architecture.api.json` through its `arch_id`; `bart-base-kp20k` finds
`model.bartbasekp20k.json` through its `name`). An architecture token like
`n.a` resolves to nothing **on purpose**: its runs stay first-class for
quality analysis and appear as dashed performance-only lines wherever cost
is an axis. Run parameters resolve against the model card — missing ones
take the card's default, none is invented otherwise, and illegal values
are *flagged, never dropped* (Overview → Issues).

Costs follow each architecture's declared linear model
(`usd = 2.5e-6·input_tokens + 1e-5·output_tokens`, …). Every raw variable
is summed at its declared level (document / batch), falling back to the
other level or to batch timestamps for wall-time — each fallback recorded
as a flag that surfaces in hovers and captions. A unit whose variables
were never observed yields `known=false`, not a crash.

## What changed in v1.5

Demo/paper pass over every page.

* **One visual language for metadata.** Dataset, model and architecture cards
  render as `key | value` chips — key in the darker half of the hue, value in
  the lighter — with one hue per *kind* of fact (`domain` blue, `lang` grey,
  `section` teal, `annotation` violet, `family` indigo, `backend` amber,
  `trained on` rose …), identical on all three pages.
* **One split language.** training `#3AA6A0` · validation `#9B7EDE` ·
  testing `#5B8DEF`, always in pipeline order, in every legend, axis, pie and
  tile on every page. Where a panel shows several annotation sets at once, the
  *pattern* separates them so the hue never has to mean two things.
* **Datasets** — tiles are now Documents · Unique keyphrases (the union of the
  annotation sets, never their sum) · Keyphrases per document (mean ± sd) ·
  Present (P) per document (mean ± sd). Document length is normalised per
  split. PRMU is titled *Keyphrase PRMU distribution*; *Keyphrase POS tags* is
  a horizontal top-4-plus-*Other* chart on one shared category set. The
  document browser filters by *intersecting* selected quality flags.
* **Models** — Parameters · Quality check · Documents (with coverage folded
  in) · costs in reading units (`$0.09 · 0.39 kWh · 1 hr 07 min`).
* **Architectures** — spend and wall time in the same units; `n.a` is no
  longer offered as an architecture (it is the declared *absence* of one).
* **RQ2** reads as a verdict: gold = what you would report, green = the clean
  subset, red = the flagged documents themselves.
* **RQ5** was rebuilt around the real question. It offers only parameters that
  vary, across every model that varies them, and tests each **(dataset, model)
  independently** — the values ran over the same documents, so they are paired
  blocks: a **Friedman** test across three or more values, Wilcoxon for two
  (both dependency-free; the chi-square tail is an incomplete-gamma
  continued fraction, checked against published critical values).
* The TeX probe now reports *why* it rejected an engine instead of silently
  downgrading every export in the session.

## What changed in v1.4

* **Annotation sets are never merged** on the Datasets page: PRMU is a grid of
  pies (row per annotator, column per split), keyphrase length and POS are
  grouped bars with hue = annotator and transparency = split, every panel on a
  bounded axis in a fixed box. The `@combined` union is not plotted; the
  similarity-pairs table is gone (it lives in Insights → data quality).
* **One significance level for all five workbenches**, chosen by a slider
  (p<0.001 · p<0.01 · p<0.05 · p<0.1); one dagger †, stated in every caption.
* **RQ2** table is *All · w/o flag · w/ flag · Δ (w/o − w/)* with inline
  `(n=…)`; **RQ3** names the context window by asset and origin
  (`512 (bart-base)`, `128k (o200k, default)`, `no window`), renames the
  conditions to *full-document* / *document truncated to model's context
  window*, and gains a second table splitting each run's documents at its own
  window with a test between the two groups. **RQ5** only offers parameters
  that actually vary across the model's runs.
* **Numbers sort as numbers** (`num_beams=1, 4, 10`) in every picker, table,
  legend and axis.
* Document quality flags are counted as `n (share of the collection)`.
* A tab left open across a server restart **reloads itself** instead of
  posting a callback the new build does not have (see Troubleshooting).
* The TeX probe now compiles a figure with the em dashes, daggers, underscores,
  Greek *and* log-axis math that real exports contain, so a half-broken engine
  can no longer certify itself; a PGF render that falls back says so on the
  console instead of silently shipping a non-TeX PDF. Wide tables are wrapped
  in `\resizebox` so a five-column export cannot run off the page.

## Correctness notice (v1.3)

**Every "F1" number produced before v1.3 was actually precision, "P" was
recall and "R" was F1.** The per-document helper returns `(p, r, f1)` while
the caller indexed it through `MEASURES = ["f1", "p", "r"]`. It surfaced when
the new SQL metric path was checked against the Python one
(`tools/test_metric_parity.py`), which now runs over every
(run, annotation set, measure, k) cell in a catalog and asserts bit-identical
agreement. Re-derive nothing — the fix is in the scoring layer, so simply
re-open any existing catalog — but **discard any figures exported earlier**.

Two export defects were fixed in the same pass, both of which only showed up
once the output was compiled rather than looked at:

* Matplotlib's PGF backend neutralises `^` and `%` and nothing else, so a
  label like `bart-base-kp20k (num_beams=4)` wrote a raw `_` into the `.pgf`
  and the figure died with *Missing $ inserted* the moment it was `\input`
  into a paper; `\mathdefault`, which Matplotlib writes into every `.pgf` but
  defines only in its own private preamble, died even earlier. Text-bearing
  spec fields are now TeX-escaped and each snippet carries
  `\providecommand{\mathdefault}[1]{#1}`.
* Non-ASCII glyphs missing from the TeX font were dropped *silently* — which
  deleted every em dash and, worse, **every significance dagger** from
  exported figures, captions and tables. They are translated to TeX
  (`†` → `$\dagger$`, `≥` → `$\geq$`, `—` → `---`, …), and `<` / `>` are put
  in math mode so `p<0.01` cannot render as `p¡0.01` under OT1.

`tools/test_latex_compile.py` compiles a deliberately hostile export with
pdflatex, xelatex and lualatex on every run, so this stays fixed.

## How it computes (the throughput story)

Tokens come from **spaCy** (blank pipelines — pure C tokenizers, no model
downloads, read out via `Doc.to_array` to avoid Python token objects);
stems from **PyStemmer** (C bindings; snowballstemmer is only a fallback —
install PyStemmer, it is the difference between minutes and hours). Every
unique phrase — gold *and* predicted — is analysed exactly once into the
global **`keyphrases` cache table** (normalised text, first-seen raw form
and language, spaCy tokens, stems, POS); documents only reference it.
POS tagging runs as its own scan phase over *globally unique untagged*
phrases, so an incremental scan tags only what is genuinely new.
Documents stream through all CPU cores in 32 MB chunks; tokenizer counts
are batch-encoded; ingestion is bulk NDJSON→DuckDB COPY. Gold *instance*
rows are stored for eval splits (`--gold-scope all` to widen); the tiny
`gold_agg` aggregates cover **every** split for the distribution charts.
**One pool, one budget.** A single `ProcessPoolExecutor` spans every phase,
created through `forkserver`/`spawn` (never plain `fork` — the scan runs in a
thread of the server process, which holds an open DuckDB connection) with an
initializer that pins `OMP`/`OPENBLAS`/`MKL`/`RAYON`/`TOKENIZERS_PARALLELISM`
to one thread each. Without that, every worker's HuggingFace-tokenizers and
BLAS pools size themselves to the whole machine: 32 workers × 32 threads is a
context-switch storm that makes a 32-core box slower than a 4-core one.
Worker count and DuckDB's memory limit come from *usable* CPU and RAM
(affinity masks and cgroup quotas, not host totals), and DuckDB gets the cores
the pool is not using during a scan, all of them afterwards.

**Nothing serial on the critical path.** Spills are drained by a dedicated
writer thread that batches one `read_json` over a *list* of files; job
planning is a generator so the pool starts on the first chunk; the corpus is
never read twice (files are content-hashed only when size/mtime moved —
`--hash always` to force); and the per-file / per-run SQL loops became
set-based. Measured on a 303 k-prediction catalog with
`tools/bench_hotspots.py` (2-core container; run it on your own hardware):

| Change | Before | After | Speedup |
|---|---|---|---|
| register 5,000 files (2 statements each → 1 bulk load) | 24,744 ms | 28 ms | **900×** |
| DELETE a run's rows (PK index → no index) | 127 ms | 4 ms | **30×** |
| cost aggregation over `preds` (Python `json.loads` → SQL) | 449 ms | 69 ms | **6.6×** |
| ingest 60,000 rows in 40 spills (per-file upsert → one INSERT) | 588 ms | 101 ms | **5.9×** |

End to end on a 2-core container: a tree of 10,217 files / 75 runs / 303 k
predictions derives in **18.5 s** from an empty database and **2.4 s** on a
no-change rescan; pool efficiency (Σ worker seconds ÷ wall × workers) is 85 %
for documents and 78 % for POS. `tools/pool_efficiency.py` prints that table
from the archived stats, and `tools/scan_once.py` derives a corpus without the
server (recommended for large trees — the scan then gets the whole machine).
Worker memory is bounded and scaled to the per-worker RAM budget, so RAM stays
flat at any corpus size. To rebuild from zero, delete `.kpviz/` and launch.

Every scan silently archives its exact timings — per step and per job —
to `.kpviz/scan_stats/scan-*.json` (wall-clock start/end, durations,
bytes, document counts, worker seconds, changes). Nothing is printed to
the console; the JSON is meant for your paper.

## What the store holds (and what it never holds)

`.kpviz/kpviz.duckdb` keeps **derived data and indices only**: per-document
stats and byte offsets, the keyphrase cache, gold instances with PRMU
classes and first-occurrence positions (chars / words / model tokens per
referenced tokenizer), per-run match primitives (which prediction hit
which gold at which rank), batch costs, run aggregates + issue tags,
similarity pairs, and a scan-versioned aggregate cache. Document and
prediction *content* is read straight from your files through
`(file, offset, length)` — the browser on the Datasets page is doing
exactly that.

Evaluation conventions (also stated in every caption): predictions are
lowercased, spaCy-tokenised, stemmed and deduplicated keeping rank order;
`+`-joined gold variants count as one keyphrase;
P@k = tp / min(k, #unique predictions), R@k = tp / #gold-after-filter,
`@O` cuts at the filtered gold count, `@M` at all predictions; documents
with no gold after filtering are excluded and n is reported; scores are
macro-averaged. **PRMU (in-order definition, on stemmed spaCy tokens):
P — all keyphrase tokens appear in the document in order; R — all appear
but never in order; M — some appear; U — none.** Occurrence positions are
the end of the earliest in-order chain on the full concatenated document,
which is also what context-window filtering uses.

## Issue tags

Every (dataset, model, run) carries `key:value` tags, shown as colored
chips on the Overview (group by dataset or by model): `illegal
parameter:<name>` (value outside the model card's schema), `missing:
architecture` (token is `n.a` or resolves to no card), `missing:run` (no
run_*.json), `missing:dataset` (inferences whose collection is absent —
runs stay first-class, metrics simply have no gold to match),
`missing:model`, `incomplete:<pct>` (document coverage), and
`unresolved_ids:<n>`. Document-level `lang_mismatch:*` /
`missing_section:*` flags use the same chip language and are counted per
dataset as `n (share of that collection)` — 157 flagged documents means
something different in a 160-document collection than in a 20k one.

## Annotation sets are never merged

The Datasets page shows one group per annotation set and never plots the
synthetic `@combined` union. PRMU becomes a grid of pies — a row per
annotator, a column per split, each titled with its own instance count — and
the keyphrase-length and POS panels become grouped bars where **hue is the
annotator and transparency is the split**, so one glance separates the two
dimensions without a second legend. Both share bounded axes (percentages of
each group's own gold) and a fixed panel box, so no panel can stretch the
page or the y axis. Keyphrase length is measured in spaCy word tokens and the
axis says so (`length (words)`); the tokenizer selector belongs to the
document-length panel above it.

## The five insight workbenches

1. **Dataset correlation** — dataset×dataset Pearson/Spearman over per-run
   scores; only (model, run) pairs evaluated on *every* selected dataset
   enter the vectors.
2. **Data quality & bias** — language-mismatch flags and similarity pairs
   (threshold + label filters); train→test leakage against each model's
   own `supervision` datasets. The table reads *All · w/o flag · w/ flag ·
   Δ (w/o − w/)*, each score carrying its own `(n=…)`, and the dagger sits on
   the delta the test actually measures — flagged documents against clean
   ones, not the reported-score shift.
3. **Extractability & truncation** — the same metric against *present* gold
   only (class P) under two conditions: **full-document** vs. **document
   truncated to model's context window**, measured with the model's own
   tokenizer. The window column names the asset, not the library, and says
   when it came from the card's default — `512 (bart-base)`,
   `128k (o200k, default)`, `no window`. Below it, doc-length bins over *all*
   gold with each run's boundary marked, plus a table splitting every run's
   documents at its own window (within vs. surpassing) and testing the two
   groups against each other.
4. **Cost–performance** — the Pareto frontier. Intersection semantics
   across datasets, dashed lines for unresolvable costs, transparency =
   document coverage, and labels that show only the hyperparameters that
   differ (`bart-base-kp20k (num_beams=4)`).
5. **Hyperparameters** — only parameters that actually *move* across the
   model's runs are offered (a constant one has no needle to move); the chart
   form adapts to the parameter's type and observed cardinality; card
   violations marked ⚠.

Selection is always models-first: pick models, then (optionally) narrow
to specific runs. One **significance level** slider at the top of Insights
(p<0.001 · p<0.01 · p<0.05 · p<0.1) governs every workbench, so a paper
cannot end up mixing thresholds: a single dagger † means "significant at the
level this caption states". The tests are a Mann–Whitney U between flagged
and clean documents (data quality), between documents inside and beyond the
window (truncation), and paired Wilcoxon signed-rank for the truncation
conditions and for each hyperparameter value against the best value on the
same dataset. Both are built in (normal approximation with tie correction,
no SciPy) and named in every caption.

Numbers read as numbers everywhere: `num_beams=1, 4, 10`, never `1, 10, 4` —
run pickers, tables, legends and bar categories all sort with embedded
integers compared as integers.

Every workbench: metric/@k/PRMU/annotation controls in one filter row, an
editable caption pre-written from the exact configuration (the box grows to
the caption, so nothing is read through a scrollbar), copy-LaTeX (figure env
+ booktabs table) buttons, and PDF / PNG / .pgf / zip-bundle downloads. The
interactive chart and the export are rendered from the same figure spec.

Direct point labels are placed, not pinned: each one is tried against nine
anchors and scored on overlap with other labels, with every marker, with the
dashed no-cost rules and with the Pareto staircase — longest label first,
frame edges weighted hardest so nothing is ever clipped. Both renderers run
the pass against their own text metrics (11 px over a web canvas vs. 7 pt
over a 7-inch figure), which is why a crowded frontier reads cleanly in the
PDF and not only on screen.

## Sample data & tooling

`tools/make_sample_data.py` regenerates `sample_data/` deterministically
(needs the original card/JSONL sources in `sample_src/`). Everything else is
a verification or measurement harness — none of it is needed at runtime, and
each takes `--data` / `--state` so it can run against your own tree:

| tool | what it proves |
|---|---|
| `scan_once.py` | headless scan; prints every step, its timing and the resulting row counts |
| `test_metric_parity.py` | the SQL metric path equals the Python reference for every (run, annotation, measure, k) cell |
| `test_latex_compile.py` | an exported figure + table typesets under pdflatex, xelatex *and* lualatex, with hostile text (underscores, em dashes, daggers, `%`, `#`, `&`, `≥`, Greek) |
| `test_exports.py` | PDF/PNG/PGF/ZIP round-trip and an RQ2 leakage delta |
| `bench_hotspots.py` | A/B of the rewritten hot spots against a live catalog |
| `pool_efficiency.py` | Σ worker-seconds ÷ (wall × workers) per phase, from the archived scan stats |
| `screenshot.py`, `test_interactions.py` | Playwright sweeps (every page, then scan → drill-down → download) |

## Troubleshooting

**Restarted the server with a tab still open?** That tab holds the callback
signatures of the build that rendered it, so it would post callbacks the new
process never registered — one `500` per poll tick, forever. KPViz stamps each
process with a build id, serves it at `/kpviz-build`, and `assets/buildcheck.js`
reloads any tab whose id no longer matches; meanwhile the server answers the
stale callback with "nothing changed" and one explanatory log line instead of a
traceback storm.

No TeX → exports say “no TeX found”, PDF uses Matplotlib's vector backend,
`.pgf` is disabled. A TeX install that *looks* usable but is not (a broken
`lualatex` whose `luaotfload` is missing, say) is rejected by the engine probe,
which compiles a figure carrying the same em dashes, daggers, underscores and
Greek that real captions carry — certifying an engine on a bare `x`/`y` figure
is how you end up with a silently non-TeX PDF. If a PGF render fails anyway,
the fallback is announced on the console, never silent. Tables wider than the
text block are wrapped in `\resizebox` so a five-column export cannot run off
the page. Offline / blocked hub → token counts become flagged
approximations (the UI says so). spaCy model missing for a language → POS
panels explain themselves and everything else works. Port busy →
`--port 8051`. Windows → fully supported (spawn-safe workers, pathlib
throughout); use `python app.py` from an activated venv. A stuck or
interrupted scan can always be re-run — derivation is idempotent and
signature-guarded. To rebuild from scratch, delete the `.kpviz/` folder.
