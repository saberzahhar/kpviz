# KPViz performance plan: memory, parallel workers, efficiency, latency

Status: **revision 4**, 27 Sep 2026, 16 days before the JCDL '26 demo (13–16 Oct). The plan was revised after six reviews of revision 1, and Waves 1 and 2 plus most of Wave 3 are implemented on branch `claude/modest-knuth-j3991q`. Section 6 has the before/after measurements.

This plan covers everything that affects how fast KPViz responds, how much memory it holds, and how well it uses the machine's cores, from the first scan to the last click of the demo. Each action names the code it touches, how to change it, the measured reason for it, and the check that closes it.

Sources:
- a full read of this repository;
- a baseline measured for this plan (Section 1), and a before/after re-measurement on the same machine (Section 6);
- the six external reviews of the code, cited as **[RA §]** for `REVIEW.md`, **[RB §]** for `KPViz-review.md` and **[R1]**–**[R5]** for the five reviewer notes;
- the six reviews of revision 1 of this plan, cited as **[P1]**–**[P6]**.

**Correctness first.** Revision 1 kept evaluation-semantics fixes out of scope. The plan reviews showed that correctness and performance meet in the same code (the incremental scan, the caches, the scoring statement), and that "parity" can otherwise mean preserving a bug ([P3], [P4], [P5]). Revision 2 therefore:
- adds a correctness block (C13–C25) as Wave 1;
- distinguishes two kinds of change:
  - an **optimisation** keeps every derived number identical under a fixed evaluation contract;
  - a **correctness fix** changes numbers on purpose, is labelled as such, and says why;
- checks parity against an **independent reference** written from the documented conventions (`tests/test_metrics.py`), not only against the previous implementation.

---

## 0. Summary

### 0.1 Where we were, where we are

Same machine, same 296 MB synthetic tree (Section 1.1). "Before" is the original code measured the same day; the UI "before" is the Section 1 baseline.

| Metric | Before | Now (this branch) | Target | Met? |
|---|---|---|---|---|
| Callbacks fired by the first page load | 96–98 | **1** | ≤ 15 | yes |
| Time until the UI settles after first load | 28.3 s | **0.22 s** | ≤ 1.5 s | yes |
| Idle requests per open tab, 10 s | 20 | **0** | 0 | yes |
| Server RSS after first load | 766 MB | **136 MB** | ≤ 300 MB | yes |
| Server RSS after the whole probe (every page, every workbench, exports) | ≥ 992 MB, unbounded | **439 MB** with a TeX engine and Matplotlib loaded (391 MB without); byte-bounded cache, now 192 MB | ≤ 400 MB | **no** (TeX case) |
| RQ3: change @k (cold, both panels, with tests and intervals) | 11.2 s | **0.64 s** (was 0.98 s in revision 2) | ≤ 0.4 s | **no** |
| RQ5: change @k (cold) | 6.4 s | **0.34 s** (was 0.93 s) | ≤ 0.4 s | yes |
| RQ2: change @k (cold) | 3.5 s | **0.26 s** (was 0.45 s) | ≤ 0.25 s | nearly |
| Any workbench, revisited or after start-up/scan | — | cache hit (background warm-up of recently used views) | instant | yes |
| RQ1 / RQ4: change @k | — | **0.09 / 0.13 s** | ≤ 0.4 s | yes |
| Export click, PNG / PDF (no TeX on this machine) | TeX probe on the request path, 6.3 s per clipboard | **0.6–0.9 s / 0.6 s**, clipboards never wait | — | — |
| Datasets page, first open | — | **1.9 s** | ≤ 0.2 s | **no** (C10) |
| Cold scan (75 k docs, 1.32 M prediction lines) | 36.8 s | **25.9 s**, including 1.2 s of content hashing the old code skipped (C13) | ≤ 25 s | nearly |
| — inferences phase | 16.9 s | **10.8 s** | — | — |
| — documents phase | 11.5 s | **8.4 s** | — | — |
| — POS phase | 4.5 s | **2.9 s** | ≤ 1.5 s | **no** |
| — finalize | 3.0 s | **2.2 s** | — | — |
| Mean CPU while matching predictions | 67 % | **84 %** | ≥ 85 % | nearly |
| No-op "Scan for changes" | 1.9 s, bumps the catalog version | **0.27 s**, no version change | ≤ 0.4 s, no bump | yes |
| Peak scan memory (parent RSS + workers PSS) | 1,853 MB | **1,194 MB** | ≤ 1.1 GB | nearly |
| Parent RSS while matching predictions | 835 MB | **382 MB** | — | — |
| DuckDB `memory_limit` while serving | 10.3 GB | **2 GB** (serve budget) | ≤ 2 GB | yes |
| Metric parity with the original code (sample tree) | — | **2,040 / 2,040 cells identical** | all | yes |
| Independent-reference parity grid (`pytest`) | vacuous-pass possible | **> 500 run × filter × k × measure cells, per document** | non-empty, exact | yes |

What is still open, in order of demo impact:
1. **RQ3 cold at 0.64 s**: three concurrent scoring statements (0.14 s each after the integer run index) plus the per-run tests. Revisits and the views warmed after start-up are instant.
2. **Datasets page**: 0.28 s server-side; about 0.8 s end to end in a fresh browser (C10 pre-aggregates would halve it).
3. **POS phase**: 2.9 s for 2,410 phrases is mostly spaCy model loading; D13b (start POS as soon as documents drain) hides it.
4. **Real-data rehearsal on the demo laptop** (A8, G1, G2, G7): the one thing this machine cannot do.

### 0.2 The ten changes that mattered most (all implemented)

1. **B1. Compute only what is on screen.** Every heavy callback takes a visibility store as its first Input and a per-client signature as State. First load went from 96 callbacks to 1.
2. **C1 + C2. One DuckDB statement per scoring request, fetched with `fetchnumpy`.** Per-document scores for all runs of a dataset in one statement, with the PRMU, position, token-limit and document filters in SQL.
3. **C3 + C4. Per-document results as float arrays over a shared document index, in a byte-bounded single-flight cache.** RSS is bounded by bytes, and concurrent identical requests compute once.
4. **B2. Poll only while a scan runs.** 0 idle requests.
5. **B3. TeX probe in a background thread, persisted.** Clipboard callbacks never wait on it.
6. **D6 + D7a. Tokenizers resolved once per scan with a negative cache, offline workers, and no-op scans that publish nothing.**
7. **D2. Coalesced prediction tasks** of 1–8 MB instead of one task per 0.08 MB batch file.
8. **D8a. POS sized to the work, gold phrases only, caches dropped before spaCy loads.**
9. **D5. Zero post-ingest UPDATEs.**
10. **C13–C25. The correctness block**: content hashes for new files, both language spellings, JSONC cards, versioned phrase cache, honest column names, source-hash code version, reported parse errors, export fixes.

### 0.3 Waves, not tiers

Revision 1 ordered the work by tier, which mixed correctness, demo path and throughput items in one list. The plan reviews asked for waves, so that small speed-ups cannot crowd out coherence work under time pressure ([P4], [P5]), and for a `blocked_by` column (Section 5).

| Wave | Content | Status |
|---|---|---|
| **1. Correctness and measurement** | A1–A6 · C12–C25 · D4 · D6 · D8d · D15 | done, except A1's large-vocabulary fixture (A1 L) and A2's in-app `mem` block |
| **2. Demo path and incremental contract** | B1–B5 · C1–C5 · C9 · C11 · D5 · D7a · D10a · D14 (cheap half) · D17 · E1 · E2 · G3 · G6 | done, except G6 (paper text), full G3, and C9's catalog snapshot |
| **3. Throughput** | D1 · D2 · D8a · D8b · D9a–e · F3 · B6 · C10 · B7 | done, except C10 and B6/B7 in part |
| **4. Rehearsal (before the 10 Oct freeze)** | A8 · G1 · G2 · G4 · G5 · G7 · F1 · B9 · RQ3/RQ5 single statement | open: needs the demo laptop and the Zenodo tree |
| **After the demo** | A7 · B8 · B10 · B11 · C6 · C7 · C8 · D3 · D7b–c · D8c · D8e · D8f · D9f · D10b–d · D11 · D12 · D13 · D14 (full) · D16 · F2 | deferred, each with the reason in its item |

Rules for Wave 4, from the plan reviews:
- G1 comes first. The synthetic tree has 2,410 unique phrases and approximate tokenizers; real data has orders of magnitude more phrases and exact tokenizers, so scan-side priorities are re-ranked after G1 ([P6]).
- No medium-risk scan rewrite after 3 Oct unless G1 shows the on-stage re-scan misses its 60 s budget ([P6]).

### 0.4 Revision 3: inference, export, reading

Added after revision 2, on the request to make the tool excellent for statistical inference, LaTeX export and figure control:

| Area | What changed | Evidence |
|---|---|---|
| Statistics | One *Statistics* bar for every workbench: rank-based / mean-based / resampling test families, Holm (default) / Bonferroni / Benjamini–Hochberg correction per table, 95 % Student-t or bootstrap intervals, effect sizes; correlation tests with Fisher-z intervals and Kendall τ-b in RQ1; intervals on every mean, difference and macro-average (Welch–Satterthwaite / per-dataset bootstrap) | `tests/test_stats.py` against SciPy |
| Speed of inference | sign-flip permutation as bit-packed matrix products (11 ms at 20 k documents), bootstrap by multinomial value counts (1 ms), Friedman vectorised (8 ms instead of a per-document loop), `t_ppf` by Newton (0.04 ms), statistics per run in parallel threads | Section 6.5 |
| Scoring | the statement returns an integer run index instead of a concatenated key per row: 267 → 140 ms at 440 k rows | parity grid unchanged |
| Warm start (B9, G4) | the views a person used are persisted and recomputed in the background after start-up and after every scan | `/kpviz-perf` → `warm` |
| LaTeX | paper presets with real column/text widths and figure fonts, `figure`/`figure*` by span, print-size preview, legend and height control, structured table cells, `table*` + adjustbox for wide tables, statistics notes, table captions that reference their figure; heatmaps as vector cells (a raster broke `.pgf`); Greek and math glyphs translated | `tests/test_latex.py`: every workbench compiles in article, ACL geometry, IEEEtran, llncs and acmart; xelatex and lualatex |
| Reading | horizontal bars when a workbench shows more than 8 runs, plots that grow with their rows, lines for numeric hyperparameters, a *How to read these results* panel (metrics, PRMU, which test and correction), n in tooltips when an interval is shown | screenshots, Section 6.3 |

### 0.5 What the plan reviews changed

| Review point | Change in revision 2 |
|---|---|
| D4 is a correctness bug, not T3 polish ([P2], [P3], [P6]) | Wave 1; implemented; a slow-flush test proves `drain()` is a barrier |
| Missing correctness items: auto-hash, `language`/`languages`, per-file purge, phrase-cache versioning, `axvspan` ([P5]) | C13–C17 added; all implemented except C15, which the contract makes moot (one collection file per dataset) |
| Lying column names, source-hash invalidation, pinned requirements, parse diagnostics ([P4]) | C18–C20, C25; implemented |
| Empty-filter collision, dataset-specific run lookup, escaped ids, optimisation vs correctness ([P3]) | C21 and the "correctness first" rule; implemented |
| Type 3 fonts, `\resizebox` on PGF, export on missing values ([P6], [P2]) | C22; implemented |
| D8d changes scores; predictions must be re-tokenised per annotation language ([P6]) | Labelled as a correctness fix; implemented, including mixed-language combined sets |
| D9f changes what is computed ([P2], [P3]) | Boundaries written down; deferred, off by default |
| F2 conflicts with E1/B1 ([P2], [P3], [P6]) | Deferred until F1 is measured; PNG only, debounced, cancellable if ever done |
| D3 adds pyarrow to every worker; do D2 first ([P2], [P6]) | Deferred to an experiment after D2 (done) |
| D12 thread and fork-safety traps ([P3], [P6]) | Moved to an experiment with named traps |
| C5 colours must persist; effort underestimated ([P2], [P4]) | Colours in `state_dir/colors.json`; effort M; done |
| D5 can be zero-UPDATE ([P2]) | Done that way |
| D14: keep it simple; do the cheap half now ([P1], [P2]) | Cheap half done; the full version is a file swap, not a tracker |
| One memory accounting model; `memory_limit` is not a process cap ([P3]) | Section 2.1 |
| Outcome-based acceptance instead of proxies (task count, CPU %, statement count) ([P3]) | D2, D3, D7, D10 criteria rewritten |
| Parity grid must cover filters, token limits, exclusion sets, per-doc; table-level hashes ([P2]) | `tests/test_metrics.py`, `tools/bench/fingerprint.py`, `tests/test_scan.py` |
| Benchmark tools must be committed; `playwright install chromium`; machine spec; 120 % budget ([P1], [P4], [P6]) | `tools/bench/`; Section 7; the machine spec is recorded in every `scan_stats` file |
| Waves and `blocked_by` ([P4], [P5]) | Section 0.3 and Section 5 |
| Coherence and a pleasing interface, not only milliseconds ([P3]) | stable colours (C5), `uirevision` (B7), loading overlays that keep the old figure (B5), axis labels that fit (C24), captions naming the gold actually used (C24) |

---

## 1. Baseline measured for this plan

This is revision 1's baseline, kept as measured. File and line references in Sections 1 and 3 point at the code before this branch; Section 6 has the same-day before/after comparison.

### 1.1 Setup

| Item | Value |
|---|---|
| Machine | 4 usable vCPU, 15.7 GB RAM, Linux, Python 3.11 |
| Libraries | Dash 4.4.1, Plotly 7.1.0, DuckDB 1.5.5, spaCy 3.8.16 (+ `en_core_web_sm`, `fr_core_news_sm`), PyStemmer 3.1.0, orjson 3.12.0 |
| Budget chosen by `config.py` | 4 workers · 2 DuckDB threads during a scan · 10.3 GB DuckDB `memory_limit` · phrase cache 949 k entries |
| Data | Review A's synthetic generators (`make_sample_src.py`, `make_scaled_data.py 20000`) over `tools/make_sample_data.py` |
| Tree | 5 datasets, 66 runs, 75,190 documents, 1.32 M prediction lines, 296 MB, 5,366 files |
| Derived rows | 320 k gold · 725 k `gold_tokpos` · 1.32 M `preds` · 1.32 M `matches` · 2,413 unique keyphrases |
| Not available | HuggingFace/tiktoken downloads (blocked, so tokenizers ran approximate) and TeX |
| Harness | a memory-sampling scan runner (parent + every child: RSS/PSS/USS per phase), a Playwright UI probe (callback round trips, settle time, idle traffic, server RSS), cProfile of one worker chunk of each kind, and kernel micro-benchmarks. Now committed as `tools/bench/scan_profile.py` and `tools/bench/probe_ui.py` (A2, A3) |

Caveats:

- Absolute times will differ on the demo laptop. The ratios are what matter.
- The synthetic vocabulary is tiny: 2,413 unique phrases, where the real corpora have millions. Phrase caches, the keyphrase merge and POS are therefore **under-stressed** here. A1 must add a large-vocabulary fixture.

### 1.2 Measurements

**Scan (cold)**

| Phase | Wall | Notes |
|---|---|---|
| discover | 0.23 s | 5,366 files, no hashing on a first scan |
| warm tokenizers | 1.96 s | network retries of unavailable assets, in the parent |
| documents | 12.8 s | 44.1 worker-s over 4 workers; mean CPU 79 %; ingest 3.1 s |
| inferences | 19.9 s | 34.7 worker-s; ingest 11.9 s; `known_doc` UPDATEs 5.6 s; gold packs 1.7 s; mean CPU 62 % |
| keyphrases (POS) | 4.9 s | 2,413 phrases, 7.2 worker-s; each worker loads the spaCy models (max 488 MB RSS per worker) |
| finalize | 3.4 s | aggregate runs 0.6 s, run_metrics 0.5 s, plus version bump and CHECKPOINT |
| **total** | **43.4 s** | |

Peak memory per phase (MB):

| Phase | Parent RSS | Workers PSS | Largest worker RSS |
|---|---|---|---|
| documents | 257 | 541 | 183 |
| inferences | 835 | 717 | 224 |
| keyphrases | 747 | 1,123 | 488 |
| finalize | 950 | 15 | 12 |

The overall peak was **1.87 GB** (parent RSS + workers PSS). The DuckDB file ended at 50.8 MB.

**Scan (no-op and repeat)**

- **No-op:** 3.17 s. 1.85 s of that is tokenizer download retries and 1.0 s is finalize. The catalog version went 1 → 2 although nothing changed.
- **First `--full`:** 36.3 s. It re-derived all 5 collections, because the stored hash went from `None` to a real value and that changes the document signature.
- **Later `--full`:** 19.5 s. It re-derived *no* collections and all 66 runs, which confirms [RA S10].
- **DuckDB file size** across these scans: 50.8 → 97.8 → 68.3 → 78.3 MB.

**UI (Playwright, headless Chromium, 1440×1000)**

| Scenario | Callbacks | Settled after | Server RSS |
|---|---|---|---|
| First load of `/` (second run: 96 in 27.7 s) | 98 | 28.3 s | 131 → 766 MB |
| – breakdown of the 96 | 56 home/sidebar polls · 18 RQ option/chain · 10 other pages' bodies and pickers · 6 export clipboards · 5 RQ compute · 1 routing | | |
| – RQ3 workbench (hidden) | 1 | in flight 22.1 s | |
| – RQ5 (hidden) | 1 | 16.2 s | |
| – RQ2 (hidden) | 1 | 8.7 s | |
| – RQ4 (hidden) | 1 | 7.0 s | |
| – export clipboards (TeX probe) | 6 | 6.3 s each | |
| Idle on Overview, 10 s | 20 | — | |
| Switch Insights tab | 3 (1 of them the tab switch itself) | 0.04 s | |
| RQ3: change @k | 27 (24 of them home/sidebar polls) | 11.2 s | 766 → 919 MB |
| RQ2: change @k | 14 | 3.5 s | → 975 MB |
| RQ5: change @k | 16 | 6.4 s | → 992 MB |

**Kernels**

| Operation (kp20k, 22 runs × 20 k docs, 440 k match rows) | Time / size |
|---|---|
| Unfiltered, precomputed `run_metrics` | 11 ms |
| Unfiltered with `per_doc=True` (Python loop) | 2,674 ms |
| Absent gold only (Python loop) | 3,044 ms |
| Present + token limit, per-doc (the RQ3 path) | 3,156 ms |
| Absent gold only, **one DuckDB statement** | **127 ms** (parity 6e-15) |
| `fetchall()` of the 440 k rows as Python lists | 1.93 s |
| Same rows as an Arrow table | 0.22 s (44.7 MB) |
| One per-doc result retained in the memo | 47.0 MB (245 MB transient peak during the call) |
| Same scores as float32 | 1.68 MB |

**Worker profile (cProfile)**

One 8 MB document chunk of 8,373 kp20k documents took 8.6 s:

- spaCy tokenisation 2.9 s cumulative;
- regex `findall` 0.82 s over 115,906 calls;
- tokenizer network retry about 1.0 s of socket reads, *inside the worker*;
- stop-word language ID 0.36 s;
- approximate `char_to_token` 0.54 s over 73,958 calls;
- `position_index` 0.41 s;
- `.tolist()` 0.13 s.

One prediction batch is only 0.1 MB (512 lines): 0.09 s to load the gold pack once per worker, then about 0.08 s of work with a cold phrase cache under cProfile.

**Spill format (1.32 M `matches` rows)**

| Format | File size | Ingest, one statement |
|---|---|---|
| NDJSON | 251.9 MB | 0.91 s |
| Parquet (zstd) | 4.4 MB | 0.31 s |
| Arrow IPC | 160.4 MB | 0.49 s |

The scan spent **11.9 s** ingesting the same rows from up to 7,900 small spill files. The repeated identity strings `(dataset, model, arch, run_id, ann_key)` in `matches` alone come to 56 MB.

### 1.3 New findings (not in the reviews, or now quantified)

- **F1.** Every worker retries the tokenizer download itself. The parent's `_warm_tokenizers` result lives in the parent's `textproc._TOKENIZERS` and never reaches the workers (`textproc.py:298-341`, `derive.py:150`). The cost is about 1 s × workers × unavailable specs, per scan.
- **F2.** `fetchall()` is 72 % of the Python scoring path (`metrics.py:224-228`).
- **F3.** The matching phase is dominated by per-task and per-file overhead, not by matching itself. There are 2,640 tasks (one per batch file, 0.08 MB on average), each writing up to three spill files.
- **F4.** POS memory scales as workers × model size (`derive.pos_chunk`, `textproc._spacy_tagger`). On a 16-core machine that is about 8 GB just to hold tagging models.
- **F5.** The DuckDB file grows by up to 2× on re-derivation. Purge-then-insert fragments it.
- **F6.** The `known_doc` / `batches.n_docs` UPDATEs take 5.6 s, serially, on the scan thread (`scanner.py:929-945`).
- **F7.** The Overview `dcc.Interval` (`home.py:48`) is pre-mounted, so it polls on every page. Each tick re-renders the whole progress panel and competes for the GIL with real work.
- **F8.** DuckDB keeps a 10.3 GB `memory_limit` while serving (`db.py:208`, `config.py:76-86`), and parent RSS stays at about 1 GB after a scan.
- **F9.** Regex work re-scans text several times per document: section words, stop-word language ID, approximate counts, and a prefix re-tokenisation *per gold phrase* for approximate positions, which is quadratic (`textproc.py:354-361, 394-408`).
- **F10.** A live "delete `.kpviz/` and re-scan" at real scale does not fit on stage. The documents phase derived 71.6 MB in 44.1 worker-seconds: about 1.6 MB/s per worker, or 5.6 MB/s on 4 workers. Extrapolating linearly to the 14.2 GB `documents.tar` gives roughly **40–45 minutes on 4 cores for the documents phase alone**, dominated by training splits. Real full-text documents may cost more or less per byte, so A8 must measure it.
- **F11.** One page load runs the heavy scoring callbacks concurrently in Flask threads. They contend for the GIL, so RQ3 took 22 s at first load but 11 s alone.

---

## 2. Rules of engagement

1. **Measure, change, re-measure.** No action is "done" without its acceptance number from Workstream A.
2. **Optimisations never move numbers.** Every one must pass the parity and determinism gates (A5): the independent reference grid, worker-count determinism, incremental = clean rebuild.
3. **Correctness fixes move numbers on purpose.** They are labelled, explained, and ship under one code-version change with the other fixes of the same wave. Parity is then checked against the corrected reference, never against the old bug.
4. **Push arithmetic into DuckDB.** It runs vectorised, uses every core and releases the GIL. The Python in callbacks should only shape results.
5. **Nothing expensive or blocking on the request path.** No network, TeX, model loading or DuckDB writes inside a UI callback.
6. **Bound every cache by bytes**, never by entry count.
7. **Work is proportional to what is visible and to what changed.**
8. **Keep the architecture.** Keep one process, one DuckDB writer and one scan pool. Do not add Celery/Redis, a multi-process WSGI server, the `fork` start method or a frontend rewrite (Section H).
9. **Budgets are enforced at 120 %.** A scripted step that exceeds 120 % of its budget fails (G3) ([P4]).

### 2.1 Memory accounting model

One model for every memory number in this plan ([P3]):

| Quantity | What it counts | Measured by | Budget |
|---|---|---|---|
| **Server steady state** | RSS of the server process in serve mode: Python and library baseline (about 130 MB) + DuckDB buffer manager + result cache + export cache + in-flight requests | `probe_ui.py` (RSS after each action) | ≤ 400 MB after the full probe (E1) |
| — result cache | sum of `nbytes` of the cached per-document arrays and aggregates | `metrics.cache_stats()` | 192 MB (`--ui-cache-mb`) |
| — export cache | bytes of cached renders | `export._cache_size` | 64 MB |
| — DuckDB | buffer manager only. `memory_limit` caps DuckDB's buffers, **not** the process | `duckdb_memory()` | serve limit min(2 GB, 20 % RAM) |
| **Scan peak** | parent RSS + Σ worker PSS (shared pages counted once), sampled every 250 ms | `tools/bench/scan_profile.py` | ≤ 1.1 GB on fixture M |
| — scan DuckDB | `memory_limit` during a scan = max(512 MB, min(50 % RAM, RAM − workers × worker peak − 1 GB)) | `config.describe()` | — |
| **Export worker** (F1, when it lands) | its own RSS, counted in the server steady state | `probe_ui.py` | inside E1 |

`tracemalloc` complements these for Python allocations; it does not replace process-level numbers.

---

## 3. Workstreams and actions

Format per action: **ID. Title** · tier or wave · effort (S < ½ day, M 1–3 days, L > 3 days) · risk. Then *Where*, *Do*, *Why* (measured where possible) and *Done when*. Tiers (T0–T3) are revision 1's labels; Section 0.3 maps them to waves and Section 5 gives the current status of every item.

### C0. Correctness prerequisites (Wave 1)

Every later number depends on these. Each is small; together they are what the paper's reproducibility claim rests on ([P4], [P5]). All ship under one `SCHEMA_VERSION` bump (5) and one code-version change.

**C13. Content-hash every new file** · done
- *Where:* `scanner._discover`.
- *Was:* in `auto` mode new files were never hashed, so the "content hash" guarantee was false after the first scan, and the first `--full` re-derived everything ([P5], Section 1.2).
- *Now:* new files and files with no stored hash are hashed at discovery (1.2 s for 5,366 files, 296 MB). A legacy `None` hash with an unchanged stat counts as unchanged, so upgrading does not re-derive.

**C14. Both language spellings** · done
- *Where:* `util.declared_langs`, `cards`, `derive`.
- *Now:* a section or annotation set may declare `"languages": [...]` or `"language": "..."`. The sample tree uses both, and `tests/test_contract.py` checks that a `languages`-only annotation set is analysed in its language.

**C15. Purge by (dataset, file)** · not needed
- The contract names exactly one collection file per dataset (`documents/document.{dataset}.jsonl`), so purging by dataset purges by file. Revisit only if the contract ever allows split collections.

**C16. Version the phrase cache on what produces it** · done
- *Where:* `scanner._sync_phrase_cache`.
- *Now:* `keyphrases` records the code version and the installed spaCy model versions. New normalisation code drops the rows (the documents re-derive under the same code-version change and restage every gold phrase). A new tagger model, or a store that predates the record, keeps the keys and re-tags. `tests/test_scan.py::test_phrase_cache_retags_on_tagger_change`.

**C17. Shading after the axis limits** · done
- *Where:* `figures.to_mpl`. `axvspan` now runs after `set_xlim`, so the truncation shading in RQ3 is correct under every `xrange`.

**C18. Honest column names, one schema bump** · done
- `gold.first_char`/`first_word` stored *end* offsets; they are now `end_char`/`end_word` ([P4]). The same bump removes `agg_cache`, `color_assign` (C5), `preds.known_doc` and `batches.n_docs` (D5).

**C19. Code version from the source** · done
- `CODE_VERSION = "<manual rev>-<blake2b(derive.py + textproc.py)>"`. A forgotten manual bump can no longer serve stale numbers ([P4]).

**C20. Parse errors are output, not silence** · done
- Cards are read as JSONC (`//`, `/* */`, BOM), exactly as the README shows them.
- An unreadable card is listed on the Overview with file, line and column (`kv card_errors`).
- Malformed JSONL lines are counted per collection and per run, with the byte offset of the first one, and lines without `_id` and duplicate document ids are reported (`kv collection_issues`, `run_issues`, run tags).

**C21. Identity and filter defects** · done
- An empty document selection means "no documents", even after the unfiltered result for the same runs was cached ([RB E06]).
- RQ3 looks up each run's context window per dataset ([RB E09]), and invalid parameter values fall back instead of crashing.
- Cost-variable names reach SQL as bound JSON-path parameters, so a name with quotes cannot break the scan.
- Timestamps: ISO 8601 with fractions and offsets, converted to naive UTC; slash dates still accepted.

**C22. Exports that compile and embed correctly** · done
- Matplotlib saves inside the rc context, so `pdf.fonttype = 42` applies: no Type 3 fonts.
- Two-column figures use `figure*` at the natural width instead of `\resizebox` on a PGF ([P6]).
- Missing values (`None`) render as gaps instead of crashing the export.
- The render cache key excludes the caption and the table (E2).

**C23. Contract details the synthetic tree now exercises** · done
- Coverage is computed within the run's majority split and capped at 1.0; runs spanning several splits get a `multi_split` tag.
- A run without its run card gets `missing:run`; `n.a` architectures stay performance-only.

**C24. What the reader sees is what was computed** · done
- Captions name the gold actually used per dataset ("author gold (kp20k)", "author+reader combined gold (semeval2010)").
- RQ4's Pareto frontier sorts by (cost, −performance) and drops non-finite points.
- A mixed-language `@combined` gold set analyses each gold phrase in its own language.
- RQ3's axis label lists window sizes only; which run has which window moves to the caption, so the label no longer overflows the figure.

**C25. Pinned requirements** · done
- `dash<5`, `plotly<8`, `duckdb>=1.1,<2`, NumPy explicit; the spaCy models stay pinned to 3.8.0 wheels ([P4]). CI wiring stays open (A7).

### A. Measurement, regression and determinism harness (do first)

**A1. Fixture generator in the repo** · T0 · M · low
- *Where:* new `tools/bench/make_fixture.py`, seeded from Review A's `make_sample_src.py` + `make_scaled_data.py`.
- *Do:*
  1. Make it self-contained, so there is no unshipped `sample_src/`.
  2. Offer sizes **S** (about 1 MB, for CI), **M** (the demo scale above) and **L** (100 k docs per dataset).
  3. **L** must also have a long-document collection (SemEval-like, 8 k+ words), two languages, and a realistic vocabulary of at least 1 M unique keyphrases with a Zipfian mix. Otherwise the phrase caches, the merge and POS are never stressed.
  4. Add a `--dup-lines` option to plant duplicate prediction lines.
- *Done when:* `python tools/bench/make_fixture.py --size M` rebuilds the Section 1 tree byte-for-byte, with a fixed seed.

**A2. Scan profiler inside the app** · T0 · S · low
- *Where:* `scanner.py` (`ScanState`, `_write_scan_stats`), new `kpviz/perf.py`.
- *Do:*
  1. Sample parent RSS and children RSS/PSS/USS every 250 ms, by phase, using `/proc/<pid>/smaps_rollup` (no new dependency) or `psutil` when present.
  2. Record CPU%, task count, spill bytes, ingest statements, rows per statement, and pool idle.
  3. Write a `mem` block and a `pool` block into `scan_stats/*.json`.
  4. Extend `tools/pool_efficiency.py` to print them.
- *Done when:* every scan archives per-phase peak memory and efficiency.

**A3. UI probe** · T0 · S · low
- *Where:* new `tools/bench/probe_ui.py` (Playwright). Accept `--chromium PATH` so it can use a preinstalled browser.
- *Do:*
  - Measure first-load callbacks, time to settle and bytes.
  - Measure idle requests per 10 s on each page.
  - Measure per-RQ interaction latency (change @k, dataset, a filter), repeated 5× with p50/p95.
  - Report server RSS before load, after load and after N interactions, plus console errors.
- *Done when:* it prints the Section 1.2 UI table in about 90 s.

**A4. Kernel micro-benchmarks** · T0 · S · low
- *Where:* `tools/bench/kernels.py` (extends `tools/bench_hotspots.py`).
- *Do:* time the scoring paths, fetch paths, figure build (`to_plotly`), export renders (PNG, PDF), and the document/prediction chunk kernels with cProfile top-N.
- *Done when:* each kernel has a median over 5 runs, and its result is written to JSON.

**A5. Parity and determinism gates** · T0 · M · low
- *Where:* `tools/test_metric_parity.py` → `tests/test_parity.py` (pytest), plus new `tests/test_determinism.py`.
- *Do:*
  1. Assert a non-empty comparison set. The current script can pass vacuously on an empty set [RB M03].
  2. Compare the SQL path to the reference over k × {none, P, RMU, token limit, position, doc filter} × measure × run.
  3. Check that cached and uncached results are identical, including empty filters ([RB E06]).
  4. Fingerprint each derived table: an ordered hash of `SELECT * ORDER BY all`. Fingerprints must be identical for `--workers 1, 2, N`, and between an incremental re-scan and a clean rebuild.
- *Why:* parallelism must never change numbers. The language-blind phrase cache already makes results depend on scheduling ([RA E9], [RB E02]). This gate will catch it.
- *Done when:* the gates run in under 3 min on fixture S, and a deliberately injected change fails them.

**A6. Per-callback server timing** · T1 · S · low
- *Where:* `appfactory.py` (Flask `before_request` / `after_request`).
- *Do:*
  1. For `_dash-update-component`, record the output id, duration, response bytes and thread count in a ring buffer.
  2. Expose p50/p95 per callback at `/kpviz-perf` (JSON).
  3. Log callbacks slower than 500 ms.
- *Done when:* a slow callback can be named without a browser.

**A7. Performance budget and CI gate** · T2 · S · low
- *Where:* `tools/bench/budget.json`, `.github/workflows/perf.yml`.
- *Do:*
  1. Store the targets from Section 0.1 per fixture.
  2. Fail CI on a regression of more than 15 % on fixture S (every PR) or fixture M (nightly).
  3. Run A5 on every PR.
- *Done when:* a PR that re-enables hidden-panel computation fails CI.

**A8. Baseline on the demo laptop with the real tree** · T1 · S · low
- *Do:* run A2 + A3 on the Zenodo data on the demo machine. Record the numbers next to Section 1.
- *Why:* every decision in G depends on real numbers (F10).

### B. UI latency and request traffic

**B1. Compute only the visible page and workbench** · T1 · M · low · biggest single win
- *Where:* `appfactory.py:62-94` (pre-mounted pages), `pages/insights.py:42-63` (panels + tab switch), every `rq*.update`, `datasets.py:402`, `models.py:217`, `architectures.py` body callbacks.
- *Do:*
  1. Add `dcc.Store(id="page-active")`, written by the routing callback, and `dcc.Store(id="ins-active")`, written by the tab switch. Default: the landing page and `rq4`.
  2. Make the active store the **first Input** of each heavy callback. If this callback's page or RQ is not active, `raise PreventUpdate`.
  3. Add a per-client `dcc.Store({"type": "fig-sig", "rq": RQ})`. Compute `sig = stable_hash([inputs, catalog_generation])`. If `sig == last_sig`, `raise PreventUpdate` (re-shown, unchanged). Otherwise return the outputs plus `sig`.
  4. Keep the signature per client, never module-global. A server-side dict would leave a second tab's graphs empty [RA S1].
  5. Apply the same pattern to `ds-body`, `md-body`, the architectures body, and the Overview inventory and issues.
  6. Make `app.layout` a callable (Dash evaluates it on every page load). Fill each dropdown's initial options and value there, and mark the dataset-refresh, model→run chain and annotation-option callbacks `prevent_initial_call=True`. They then fire only on real changes; today they account for 18 + 10 first-load requests.
  7. Keep pre-mounting: it preserves control state across navigation for free.
- *Why:* about 40 of the 96 first-load callbacks belong to hidden pages and workbenches. Hidden RQ3/RQ5/RQ2/RQ4 were in flight for 22/16/9/7 s, slowing everything through GIL contention (F11). The 1 Hz poll then fires for as long as the page is busy (B2 removes it).
- *Done when:* together with B2 and B3, first load fires ≤ 15 callbacks and settles in ≤ 1.5 s (fixture M). Switching to an unchanged tab fires one callback. Two browser tabs both render.

**B2. Poll only while a scan runs** · T1 · S · low
- *Where:* `home.py:48, 245-287`; `appfactory.py:97-109`; `assets/buildcheck.js:29`; `assets/autosize.js:36-37`.
- *Do:*
  1. Move the `dcc.Interval` into the app frame as `scan-poll`, built with `disabled=not STATE.running` inside the callable layout from B1 step 6, so each page load sees the current state. Today `app.layout` is built once at start-up.
  2. The scan and re-scan buttons set `disabled=False`. The poll callback returns `disabled=True` on the tick after `running` turns false, after emitting the final panel and `catalog-version`.
  3. Fold `sidebar_scan` into the same callback: one request per tick instead of two.
  4. Return `no_update` for the progress panel when a hash of the snapshot *minus* `elapsed_s`/`eta_s` is unchanged. Render elapsed and ETA in a clientside callback from `started_at`.
  5. In `buildcheck.js`, check on `visibilitychange`, on `focus` and every 60 s, instead of every 4 s.
  6. In `autosize.js`, observe `.caption-box` containers, not `document.documentElement` with `subtree: true`.
- *Why:* 20 requests per 10 s per idle tab (F7). The poll also sent 56 of the 96 first-load requests and 24 of the 27 requests in one RQ3 interaction.
- *Done when:* A3 shows 0 requests in 10 s of idle on every page. A scan still streams progress at 1 Hz and the sidebar still updates.

**B3. TeX probe off the request path** · T1 · S · low
- *Where:* `export.py:69-105` (`tex_engine`, `lru_cache` without a lock), `appfactory.py:176`.
- *Do:*
  1. Start the probe in a daemon thread at `build_app()`, under a `threading.Lock`.
  2. Persist the result to `.kpviz/tex_probe.json`, keyed by (engine path, `engine --version` first line, Matplotlib version).
  3. Order engines pdflatex → xelatex → lualatex, or honour a "my paper compiles with …" setting.
  4. `clipboards` reads the status without blocking and shows "checking TeX…" until it is known.
  5. The probe must set rcParams only under `_render_lock`, as [R1 A.10] notes.
- *Why:* 6 callbacks at first load waited 6.3 s each, and up to 23 s with a TeX install ([RA S5]).
- *Done when:* clipboard callbacks answer in ≤ 50 ms at first load, with or without TeX installed.

**B4. Clientside callbacks for pure-UI toggles** · T1 · S · low
- *Where:* `appfactory.py:82-94` (`route`), `insights.py:52-63` (`switch`).
- *Do:*
  1. Rewrite both as `clientside_callback`. They only set `style`/`className`, and now also write the B1 `page-active` / `ins-active` stores.
  2. Leave caption → LaTeX strings in Python, so one escaping implementation stays authoritative. Trigger them on `n_blur` and on spec change instead of on every keystroke.
- *Done when:* navigation and tab switches make zero server round trips beyond what B1 triggers.

**B5. Loading feedback that never blocks** · T1 · S · low
- *Where:* `insights_common.figure_block`, `ui.graph`, the dataset/model bodies; `appfactory.py:60` (`update_title=None`).
- *Do:*
  1. Wrap the graph and table in `dcc.Loading(delay_show=300, overlay_style={"visibility": "visible", "opacity": .5})`. The previous figure stays readable under the overlay.
  2. Set a short `update_title`.
- *Why:* once C1 lands, most callbacks finish before 300 ms and never show a spinner. Only real work does.

**B6. Split monolithic callbacks** · T2 · M · low
- *Where:* `rq3.py:96-349` (two panels in one callback), `datasets.py:130-374` (`_body`: header, KPIs, five charts and the browser), `models.py:149-205`.
- *Do:*
  - **RQ3:** give panel (a) and panel (b) separate callbacks. They share the per-doc result through C4 single-flight.
  - **Datasets:**
    - one callback each for header+KPIs, length chart, PRMU, keyphrase length + POS, and the browser;
    - the tokenizer drives only the length chart;
    - move the browser out of `ds-body`, which also fixes the search box resetting ([RA S7]).
  - **Models:** load the quality glance in its own callback.
- *Why:* independent requests run concurrently, because DuckDB releases the GIL. The first panel paints early, and changing one control recomputes one panel.
- *Done when:* changing the Datasets tokenizer re-runs one callback, and the browser keeps its search text.

**B7. Cheaper figures and payloads** · T3 · S · low
- *Where:* `figures.py:293-464` (`to_plotly` builds validated `go.Figure` objects).
- *Do:*
  1. Measure `to_plotly` with A4 first.
  2. If it costs more than 20 ms, build plain dict figures or use `go.Figure(..., skip_invalid=True)`.
  3. Round floats in hover/customdata to 4 significant digits.
  4. Set `layout.uirevision` so zoom survives updates.
  5. Use `dash.Patch()` for label and opacity toggles in RQ4 instead of rebuilding the figure.
- *Done when:* the RQ3 payload (157 KB today) is ≤ 60 KB, and a label toggle sends a patch.

**B8. Compress responses** · T3 · S · low
- *Where:* `appfactory.build_app`.
- *Do:* add `flask-compress` (gzip or brotli) for callback JSON and the Plotly bundle.
- *Why:* it cuts first paint when the browser is not on the same machine (projector setups, remote access). Measure with A3.

**B9. Prefetch the default views after publish** · T2 · S · low
- *Where:* new `kpviz/prefetch.py`, hooked after `bump_scan_version()` and at server start.
- *Do:*
  1. In a low-priority thread, compute each workbench's default view into the C4 cache.
  2. Use a DuckDB cursor limited to 1–2 threads so it never competes with a user request.
  3. A user request for the same key joins the in-flight computation (C4 single-flight).
- *Done when:* the first click on every workbench after start is a cache hit.

**B10. Bookmarkable state without recomputation** · T3 · M · low
- *Do:* encode page, RQ and controls in `dcc.Location.search` ([RA U12]). Restoring a URL reuses the C4 cache and the B1 signatures.

**B11. WSGI server** · T3 · S · low
- *Do:* serve through `waitress` (threads = 8) instead of the Flask development server. Keep a single process.
- *Why:* robustness and a bounded thread count. It does not remove the GIL. C1 and C2 are the actual fix.

### C. Query engine and metrics

**C1. One DuckDB statement for filtered and per-document scores** · T1 · M · medium (parity-gated)
- *Where:* `metrics.py:113-148` (`gold_masks`), `:171-272` (`run_scores`). Callers: `rq2.py:157`, `rq3.py:135-152, 240`, `rq5.py:168`, `models.py:131`.
- *Do:*
  1. Port Review A's `sql_scores.py` statement: gold filter CTE → `nok` → one matches row per `rid` → `unnest` hits → tp → per-doc P/R/F1 → aggregate.
  2. Compute **all three measures in one pass**, and return them for every run at once.
  3. Pass `doc_ids` / `exclude_doc_ids` as a registered Arrow/temp table joined in SQL, not as a Python `if` per row.
  4. Keep `run_metrics` as the unfiltered fast path.
  5. Prepare the statement once and bind parameters.
- *Why:* measured 3.0 s → 0.127 s at parity 6e-15. DuckDB parallelises across cores and releases the GIL.
- *Done when:* A5 parity holds over the full grid, and RQ3 / RQ5 / RQ2 interactions are ≤ 0.4 / 0.4 / 0.25 s on fixture M.

**C2. Arrow/NumPy fetches for any large result** · T1 · S · low
- *Where:*
  - `metrics.run_scores`;
  - `datasets.py:218-223` (per-document lengths pulled into Python for a histogram);
  - `rq3.py:252-255` (20 k-row `dict(db.q(...))` per tokenizer);
  - `scanner._build_gold_pack` (`scanner.py:824-836`).
- *Do:*
  1. Use `.fetchnumpy()` / `.to_arrow_table()`, never `fetchall()` beyond about 10 k rows.
  2. Compute histograms in SQL with `width_bucket`/`histogram`, and fetch the bins only.
  3. Build gold packs with `COPY (SELECT ...) TO ... (FORMAT json)` straight from DuckDB.
- *Why:* 1.93 s → 0.22 s for 440 k list rows (F2).

**C3. Per-document results as dense arrays** · T2 · M · medium
- *Where:* schema (`db.py`), `scanner` documents ingest, `metrics.py`, `stats.py`, `rq2/rq3/rq5`.
- *Do:*
  1. Add `doc_ord INTEGER`, dense per dataset. Right after the documents phase, one statement builds `doc_index(dataset, doc_id, doc_ord)` with `row_number() OVER (PARTITION BY dataset ORDER BY doc_id)` and adds `doc_ord` to `gold`, which is small. Ship `doc_ord` inside the gold pack so prediction workers write it straight into `preds`/`matches`. There is no post-hoc UPDATE over the large tables (compare D5).
  2. Change the per-doc result per run to `(doc_ord: int32[] sorted, score: float32[])`.
  3. Pair documents with `np.intersect1d(assume_unique=True)` or a presence bitmap.
  4. Vectorise `stats.py` rank computations with NumPy argsort. The asymptotic formulas stay as they are, and parity against the current implementation is tested.
- *Why:* 47 MB → 1.7 MB per cached result, and paired tests stop iterating over Python dicts.
- *Done when:* one per-doc result measures ≤ 2 MB and the statistics are identical to 1e-12.

**C4. Byte-bounded, single-flight result cache** · T1 · S · low
- *Where:* `metrics.py:278-299` (`_MEMO`, `_MEMO_MAX = 4096`), `db.cache_get/put`.
- *Do:*
  1. Replace the memo with an LRU sized in bytes: 256 MB default, `--ui-cache-mb` to change it. Entry size is the sum of array `nbytes`, or an estimate for dicts before C3.
  2. Add single flight: a per-key `Future`, so concurrent identical requests wait on one computation.
  3. Cap concurrent heavy computations with a semaphore of 2. That stops first-load bursts from thrashing the GIL and stacking 245 MB transient peaks.
  4. Key the cache on the catalog generation (D14), or on `scan_version` until D14 lands.
- *Why:* RSS grew 766 → 992 MB over three interactions, and the only bound is 4,096 entries of up to 47 MB each.
- *Done when:* after 50 random interactions, RSS stays within the cache budget + 150 MB.

**C5. No DuckDB writes from UI callbacks** · T2 · S · low
- *Where:* `db.cache_put` (`db.py:392-395`), `db.color_seq` (`:406-414`, called by `naming.encode_runs` on every RQ render), `db.kv_set`.
- *Do:*
  1. Keep scalar results in C4 only, and drop the `agg_cache` table writes.
  2. Keep colour slots stable without DuckDB writes: the assignment lives in `state_dir/colors.json`, new entities are appended in sorted order and the file is replaced atomically. Colours follow the entity across sessions and rebuilds ([RA C8], [P2]), and no UI path takes `_wlock`.
- *Effort, revised:* M, not S ([P4]). It removes `agg_cache` and `color_assign`, so it ships with the schema bump of C18.
- *Why:* UI writes take `_wlock`, which the ingest thread holds for whole `read_json` batches, so a click during a scan can queue behind the ingest.
- *Done when:* grep finds no `_wlock` / `execute` on any UI callback path, and interactions during a scan stay fast.

**C6. Integer surrogate keys** · T3 · M · medium
- *Do:*
  1. Add `run_ord SMALLINT` for (dataset, model, arch, run_id), `ann_ord UTINYINT`, and `doc_ord` (C3).
  2. Store the ordinals in `preds` and `matches` instead of five VARCHARs. Select runs by an integer list.
  3. Bump the schema version.
- *Why:* 56 MB of repeated identity strings in `matches` at demo scale is decompressed into every scan of that table. Integer joins and filters are faster, and so are hash tables.

**C7. Physical clustering for zone maps** · T3 · S · low
- *Do:* insert `matches` and `gold` ordered by (dataset, ann_key, run_ord, doc_ord), either at ingest or in a finalize rewrite. Verify with `EXPLAIN ANALYZE` that single-dataset queries skip row groups.

**C8. Precompute the common filtered views** · Wave 4 · S · low · blocked by the evaluation-convention fixes ([P6]); precompute only a handful of views, never every filter combination ([P3])
- *Where:* `metrics.rebuild_run_metrics`.
- *Do:* extend `run_metrics` with a `view` column: `all`, `P` with position, and `RMU` (absent). These are the default views of RQ3 panel (a) and of Fig. 1 (absent R@M), so they become 10 ms lookups.
- *Cost:* about 0.5 s per view at finalize, on fixture M.

**C9. Remove N+1 queries and repeated derivations from callbacks** · T2 · M · low
- *Where:*
  - `rq3._run_limit` (`rq3.py:63-82`): one query per run, called three times per run per interaction; it also ignores the dataset, [RB E09];
  - `rq5._varying` / `_models_with_variation` (`rq5.py:28-46`): one query per model, called from three callbacks;
  - `rq5.py:154-155`: one query per (dataset, model);
  - `insights_common.effective_runs` → `run_options` → `run_labels`: recomputed in the update *and* in the chain callback;
  - `ann_for` / `ann_options` / `_annotators`: queried per call.
- *Do:*
  1. Build an immutable per-generation **`Catalog` snapshot** at publish. It holds runs with parsed params, varying params per model, the context limit per (run, dataset), annotation keys per dataset, labels and encodings.
  2. Callbacks read from it in O(1).
- *Done when:* A6 shows ≤ 3 SQL statements per RQ interaction, excluding the scoring statement.

**C10. Datasets page from pre-aggregates** · T3 · S · low
- *Where:* `datasets.py:144-197, 274-327`.
- *Do:*
  1. Materialise `doc_gold_stats(dataset, doc_ord, split, ann_key, n_kp, n_p)` and per-dataset unique-stem counts at finalize.
  2. Compute histograms in SQL.
  3. Join POS on `kp` instead of `display`.
  4. Page the browser in SQL with natural order.

**C11. DuckDB serve-mode memory** · T2 · S · low
- *Where:* `db.set_scan_mode` (`db.py:254-264`), `config.duckdb_memory_bytes`.
- *Do:* on leaving scan mode, run:
  - `SET memory_limit` to a serve budget, by default min(2 GB, 20 % of RAM);
  - `SET allocator_flush_threshold='64MB'`;
  - `SET allocator_background_threads=true`.

  All three settings exist in the installed DuckDB 1.5.5. Restore the scan budget on entering scan mode.
- *Why:* parent RSS stayed at about 1 GB after the scan, with a 10.3 GB limit (F8).
- *Done when:* server RSS drops within 5 s after a heavy query, and the steady state meets E1.

**C12. `db.q` retries only what is retryable** · T2 · S · low
- *Where:* `db.py:276-282`.
- *Do:* retry only on connection or cursor invalidation: `duckdb.ConnectionException`, and an `InvalidInputException` / `InternalException` whose message says the connection or cursor is closed ([P5]). Parser, binder, catalog, conversion and constraint errors raise at once, with the original traceback ([RA G2], [R2 #2]).

### D. Scan engine: parallel workers and throughput

**D1. CPU budget from measurement, not assumption** · T2 · S · low
- *Where:* `config.py:45-56`.
- *Do:*
  1. Default `workers = cpus − 1` on ≤ 8 cores and `cpus − 2` above, because the parent's ingest thread, pool driver and DuckDB need a core.
  2. Give DuckDB `max(1, cpus − workers)` threads during a scan.
  3. Add `--workers auto|N` and a sweep in A2 (N ∈ {1, 2, …, cpus}) that reports throughput, peak PSS and CPU %.
  4. Choose the default from the sweep, on 4, 8 and 16 cores. The metric is scan wall time within the memory budget, not CPU utilisation ([P3], [P5]); sweep workers ∈ {cpus/2, cpus, 2·cpus}.
- *Why:* today the default is all cores, plus DuckDB threads on top ([RA S16]). Yet measured CPU was only 62–79 %, which points at serial phases and per-task overhead (D2, D5), not oversubscription. The sweep separates the two.

**D2. Coalesce small prediction files into real tasks** · T2 · M · medium
- *Where:* `scanner._derive_all_runs` (`jobs_iter`, `scanner.py:896-912`), `derive.derive_preds_chunk`.
- *Do:*
  1. Make a task carry a **list** of `(path, start, end, batch_idx, file_id)` from one run, filled up to a target of `clamp(total_bytes / (workers × 4), 2 MB, 16 MB)`.
  2. Split big files by byte range as today.
  3. Keep dataset-major order, so each worker loads each gold pack once.
  4. Track `remaining` per run key across tasks.
- *Why:* 2,640 tasks of 0.08 MB produced up to 7,900 spill files and 11.9 s of ingest. One `read_json` of the same rows takes 0.9 s (F3).
- *Done when:* the inferences phase is faster end to end within the memory budget, with determinism (A5) intact ([P3]). Task count and CPU % are diagnostics, not the target.
- *Order:* bound the task size in bytes (the target above) so a coalesced task cannot grow past the per-worker memory budget; D8f refines this for long documents ([P1]).

**D3. Columnar spills (Parquet)** · after the demo · M · medium · an experiment, decided on data after D2 ([P2], [P6])
- *Where:* `derive._write_ndjson`, `ingest.py`, `db.ingest_ndjson`.
- *Do:*
  1. Workers accumulate **column lists**, not row dicts, and write Parquet (zstd) with `pyarrow`. That adds one dependency, already present in most environments.
  2. The ingestor issues `INSERT INTO t SELECT * FROM read_parquet([...])`.
  3. Keep NDJSON as a fallback when `pyarrow` is missing.
- *Why:* for `matches`, 252 MB of NDJSON becomes 4.4 MB of Parquet, and the one-statement ingest drops from 0.91 s to 0.31 s. Workers also stop holding a list of row dicts *and* a `BytesIO` copy of the whole chunk (`derive.py:99-109`).
- *Done when:* spill size **plus** serialisation time, ingest time, worker peak memory and total scan time all improve, and the post-ingest tables are byte-identical to the NDJSON path (A5 fingerprints), including list columns, NULL vs missing and large integers, on multi-file ingests ([P3], [P4]).
- *Caveat:* `pyarrow` in every worker costs import time and RSS, which works against D8 and E1 ([P2]). After D2 the NDJSON ingest is 4.9 s of worker-side time spread over the phase; measure whether Parquet still pays.

**D4. Ingestor: acknowledged barrier and byte-aware backpressure** · Wave 1 · S · low · a correctness fix, done before D2 ([P2], [P3], [P6])
- *Where:* `ingest.py:55-147`.
- *Do:*
  1. Have `drain()` wait on a `Future` that completes *after* the write. Today `n_pending` drops when a path is dequeued, before it is written ([RB D02]).
  2. Keep a persistent error state that wakes waiters, and route every flush path through one exception boundary.
  3. Apply backpressure on the sum of queued spill bytes, not on the file count.
- *Done when:* a deliberately slow flush keeps `drain()` waiting, and an injected write error surfaces on every later call (`tests/test_units.py`).

**D5. Remove the serial post-ingest UPDATEs** · T2 · S · low
- *Where:* `scanner.py:929-945`.
- *Do:*
  1. Drop `preds.known_doc` and derive unresolved ids with an anti-join inside `_aggregate_runs_sql` ([P2]).
  2. Drop `batches.n_docs`; `_aggregate_runs_sql` counts documents itself.
  3. The result is zero post-ingest UPDATEs, not fewer.
- *Why:* 5.6 s serial on the scan thread (F6).
- *Done when:* the inferences phase has no statement touching all of `preds`.

**D6. Resolve tokenizers once per scan and share the result with workers** · T1 · S · low
- *Where:* `scanner._warm_tokenizers` (`:624-639`), `textproc.ModelTokenizer._try_hf` (`:311-341`), `derive.init_worker`, `derive_doc_chunk:150`.
- *Do:*
  1. The parent resolves each spec to either a local `tokenizer.json` path / tiktoken cache or `approx`.
  2. Persist a negative-cache marker `tokenizers/<name>/.unavailable` with a TTL (default 24 h).
  3. Pass the resolution map in the task args. `init_worker` sets `HF_HUB_OFFLINE=1`, so workers load only from local paths.
  4. Honour `HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE` / `TIKTOKEN_CACHE_DIR` in the parent too.
  5. Add a "retry tokenizer downloads" button on Overview that clears the marker and `textproc._TOKENIZERS`.
  6. Put the exact/approximate status in the doc signature ([RA S10]), so a newly available exact tokenizer triggers re-derivation.
- *Why:* 1.85 s per no-op scan in the parent, plus about 1 s per worker per spec inside every scan (F1).
- *Done when:* with the network off, a no-op scan spends ≤ 10 ms on tokenizers and workers open no sockets.

**D7. No-op and small-edit scans do only what changed** · T1 (a) / T3 (b, c) · S / M · low
- *Where:* `scanner._do_scan` (`:354-441`), `_aggregate_runs_sql`, `metrics.rebuild_run_metrics`.
- *Do:*
  - **a.** If discover finds no new, modified or deleted file, no card changed and no signature changed: skip the pool, POS, scores, finalize, the version bump and the cache clear, and return.
  - **b.** Partition finalize: re-aggregate and re-score only the affected runs (`DELETE … WHERE run IN changed`, then `INSERT … SELECT … WHERE run IN changed`). Recompute costs for every run only when an architecture card changed.
  - **c.** Store `checked_at` separately from the catalog generation.
- *Why:* 3.2 s today, and the version bump re-fires every `catalog-version`-driven callback. Before B1 that meant every hidden workbench. Step 1 of the paper's walkthrough is exactly this button.
- *Done when:*
  - a no-op scan takes ≤ 0.4 s with no generation change and no cache invalidation;
  - an edit that cannot change a derived row (a card description) re-derives nothing;
  - editing one run touches only that run's rows (b);
  - after any edit, the incremental store equals a clean rebuild, table by table (A5 fingerprints). This is the contract behind the paper's "recomputes only what changed" ([P5]).

**D8. Worker memory**
- **a. Size the POS phase to the work** · T2 · S · low
  - *Where:* `scanner._pos_phase` (`:1182-1253`), `_pos_chunk_size` (`:1176`).
  - *Do:*
    - Use `pos_workers = min(workers, ceil(n / 5000), floor(pos_mem_budget / model_rss))`.
    - Below about 5 k phrases, tag in a single task.
    - Tag only phrases referenced by gold rows. The UI reads POS for gold only (`datasets.py:320-327, 474-477`); prediction-only phrases were 37 % of the sample ([RA S13]) and dominate at real scale.
    - Store `''` for "tagged, no tags" so those phrases are not re-tagged every scan.
  - *Why:* 4.9 s and 1.1 GB of worker PSS for 2,413 phrases. The largest worker was 488 MB with the models loaded (F4).
  - *Done when:* fixture M's POS phase is ≤ 1.5 s at ≤ 400 MB, and fixture L's POS memory does not scale with the worker count.
- **b. Stop the gold pack converging to a fully decoded copy per worker** · T3 · S · low
  - *Where:* `derive._pack_entry` (`:384-398`) writes decoded dicts back into the long-lived pack.
  - *Do:* decode into a chunk-local dict and drop it at the end of the chunk, or keep a bounded LRU of decoded entries. Keep the raw bytes as the only long-lived form.
- **c. Share the pack across workers by memory mapping** · T3 · M · medium
  - *Do:* the parent writes one pack file plus a sorted `(doc_id → offset, len)` index per dataset, and workers `mmap` both. Page-cache pages are shared, so PSS counts the pack once rather than once per worker.
  - Do this only if the fixture L profile (A2) shows RSS scaling as workers × pack. Moved to the post-demo appendix ([P5]).
  - *Done when:* total process-group PSS grows less with the worker count than before. Sharing reduces the replicated pack; it cannot remove each worker's private state ([P3]).
- **d. Caches without cliffs, keyed for determinism** · T2 · S · medium
  - *Where:* `textproc.PhraseCache` (`:146-189`), `derive._MATCH_STEMS` (`:79-96`).
  - *Do:*
    - Replace clear-all resets with a two-generation scheme: on overflow the current dict becomes "old", lookups promote from old to new, and the old generation is dropped at the next overflow ([R2 #6–7], [R4 3.1]).
    - Bound `_persisted` without triggering re-persist storms.
    - Key by `(lang2, norm)`. This is the determinism fix of [RA E9], and A5 checks it across worker counts.
    - Analyse predictions **per annotation language**, not once in the dataset's main language and re-stemmed ([P6]). A mixed-language combined set analyses each gold in its own language.
    - This changes numbers where languages differ, so it is labelled as a correctness fix and ships under the same `CODE_VERSION` change as the other convention fixes ([P6]).
    - Test the cliff scenario explicitly: vocabulary larger than the cache cap, skewed phrase frequencies, then a second scan ([P4]).
  - Record hit rates in the task results.
- **e. Columnar accumulation and streamed writes** · T3 · S · low
  - Comes with D3. No list of row dicts, and no whole-chunk `BytesIO`.
- **f. Memory-aware chunk size** · T3 · S · low
  - *Where:* `scanner._eff_chunk` (`:304-309`).
  - *Do:* bound the chunk by the per-worker memory budget and the file's mean line length. Long-document collections get smaller chunks.

**D9. Documents phase CPU**
- **a. Normalise once per document** · T2 · S · low
  - *Where:* `derive.py:199-228`, `textproc.norm_text/tokenize/spacy_doc_tokens`.
  - *Do:* compute `norm_text(full_text)` once, and pass normalised section slices to the regex word counter, the language ID and spaCy.
- **b. Language ID in one pass** · T2 · S · low
  - *Where:* `textproc.detect_language` (`:253-271`).
  - *Do:* one pass over the tokens with a merged `token → languages` dict, instead of one generator per candidate language. Same scores, so parity-testable. Measured 0.36 s per 8 MB chunk.
- **c. Approximate token positions in O(log n)** · T2 · S · low
  - *Where:* `textproc.count_batch` (`:361`), `char_to_token` (`:407-408`).
  - *Do:* compute word-end offsets once per document. The count is their length, and a position is a `bisect`. This removes the per-phrase prefix `findall`, which is quadratic (F9).
  - *Done when:* results are identical to the current approximate values.
- **d. One exact encode per document** · T3 · S · low
  - *Where:* `derive.flush_tokens` (`:160-184`), `textproc.encode_cached`.
  - *Do:*
    - Use one `encode_batch` with offsets per tokenizer for both counts and positions.
    - For tiktoken, compute ends as `accumulate(map(len, decode_tokens_bytes(ids)))`.
    - Build a char→byte prefix table once per document.
  - Fix special-token offsets first ([RB E04]) so that the optimisation preserves correct coordinates.
- **e. Stay in NumPy for spaCy attributes** · T3 · S · low
  - *Where:* `textproc.spacy_doc_tokens` (`:125-132`).
  - *Do:* select with boolean masks on the `to_array` result instead of `.tolist()` over every token.
- **f. Seeded sampling of training splits for distribution charts** · after the demo · M · **medium** — it changes what the tool computes ([P2], [P3])
  - *Where:* `derive_doc_chunk`, new `--dist-sample N` (default 50 k per split).
  - *Do:*
    - Sample only the expensive NLP for training documents: spaCy, stemming, PRMU, language ID and tokenizer counts run when a seeded uniform sample selects the document.
    - **Every** training document still gets its cheap `documents` row (id, split, byte offsets), because RQ2's leakage criterion joins counterpart training documents by id (`rq2._leak_docs`).
    - Captions state "n = 50,000 of 530,809, uniform sample" ([RA S14]). The Overview says that language flags on training splits come from the sample.
  - *Boundaries* ([P2], [P3]):
    - never sampled: document identities, splits and byte offsets; coverage ratios; RQ2 leakage relationships; the document browser; every test-split number;
    - sampled only: descriptive distributions of training splits, named per chart;
    - a stable selection rule (hash of `doc_id` with a stored seed), so the sample is identical across scans and machines, and the sample identity is stored with the figure spec;
    - an explicit `--dist-sample` mode, off by default. The default stays exhaustive.
  - *Why:* training splits dominate the real 14.2 GB tree (F10). This is the single largest lever on a real cold scan.
- **g. Hoist per-document closures** · T3 · S · low
  - *Do:* define `stream` and `agg_add` once per chunk (`derive.py:220, 259`).

**D10. Finer incrementality** · T3 · M–L · medium
- *Do:*
  - **a.** Build the document signature from the card fields that affect derivation (sections, languages, annotations), not `card.raw` [RA S10].
  - **b.** Detect appends to a JSONL collection (prefix hash unchanged, file grew) and derive only the tail.
  - **c.** Give runs per-batch signatures: purge and re-derive only the changed batch files.
  - **d.** Have a run's signature reference a **hash of its dataset's gold pack**, not the document file's signature. Editing document text that does not change gold then leaves the runs untouched.
- *Done when:* only affected dependencies recompute, and the incremental output equals a clean rebuild (A5). Appending documents can legitimately change coverage or resolve previously unresolved predictions, so "zero runs re-derived" is the target only when the appended ids are not referenced by any run ([P3], [P5]).
- *Blocked by:* D14, so that incremental re-derivation never exposes torn reads ([P5]).

**D11. Hash while reading** · T3 · S · low
- *Where:* `scanner._discover` (`:482-527`), worker chunk readers.
- *Do:* workers compute a blake2b per byte range as they read, and the parent combines them in order. Remove the separate hashing pass, and the spurious re-derivation after the first scan (Section 1.2 "First `--full`", [RA S11], [R5 1.1]).

**D12. Warm workers through forkserver preload** · experiment, after the demo · S · **medium** ([P3], [P6])
- *Where:* `scanner._mp_context` (`:74-96`).
- *Do:* `ctx.set_forkserver_preload(["kpviz.derive", "kpviz.textproc", "spacy", "Stemmer", "orjson", "numpy"])`. The API is present on the installed Python.
- *Why:* workers fork from an image that has already imported the heavy modules. Start-up is faster and the import pages are shared copy-on-write, which lowers PSS.
- *Traps:*
  - Preloading spaCy imports NumPy, and OpenBLAS sizes its thread pool at import, before `init_worker` pins threads. Set `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and `RAYON_NUM_THREADS` in the parent before the forkserver starts ([P6]).
  - Imported libraries can start threads inside the forkserver, which breaks its fork-safety assumptions ([P3]). Check the thread count of the forkserver after preload.
  - Linux and macOS only; keep `spawn` as the fallback ([P2]).
  - Warm `_blank`, the stemmers and the tagger models explicitly in the preload, or nothing is shared ([P2]).
- *Done when:* A2 shows lower per-worker USS and a faster first task, and a per-worker thread count equal to the pinned value.

**D13. Scheduling across phases** · T3 · S · low
- *Do:*
  - **a.** Submit the largest collections' chunks first (longest-processing-time first) to shorten the tail.
  - **b.** Submit gold-phrase POS tasks as soon as the documents phase drains, interleaved with prediction tasks, so the POS phase stops being a separate serial tail.
  - **c.** Batch `STATE` progress updates, instead of taking the lock twice per result.

**D14. Atomic catalog generations** · T3 · L · medium
- *Where:* `scanner._do_scan`, the purge functions, `db.py`.
- *Do:*
  1. Derive into staging tables or a new schema. Validate, then publish in one transaction: rows, card snapshot, signatures and generation id.
  2. The UI reads only the published generation, and every cache is keyed on it.
  3. On cancel or failure, keep the previous generation and delete unclaimed spills ([RB D01], [R1 A.8], [R4 2.7]).
- *Why (latency):* no torn reads during a scan, and a single cache invalidation at publish.
- *Keep it simple* ([P1]): build the new generation in a fresh `.duckdb` file and swap it in with an atomic rename, or rely on one DuckDB transaction; no application-level generation tracker. That also delivers D16.
- *Cheap half, done now* ([P2]): a failed or cancelled scan records `last_scan_ok = false`, bumps the version so every cache is invalidated, and deletes unclaimed spills; the next scan then never takes the no-op shortcut.

**D15. Scan start under a lock, cancellation everywhere** · T2 · S · low
- *Where:* `scanner.start_scan` (`:238-246`), the POS, hashing and ingest loops.
- *Do:* do the check-and-set under a module lock ([RA G1]), and add cancel checkpoints to every long loop, including the POS phase and tokenizer batches ([P4]).
- *Done when:* eight concurrent starts run one scan; a cancel mid-phase leaves the previous catalog readable and marks the store for a full next pass ([P3]).

**D16. Keep the store compact** · T3 · S · low
- *Do:* with D14, write each new generation into a fresh file and swap it in. That compacts by construction. Until then, add a `kpviz compact` command that rebuilds with `CREATE TABLE … AS` into a new file (F5).

**D17. A scan-time DuckDB budget that leaves room for workers** · T2 · S · low
- *Where:* `config.duckdb_memory_bytes` (`:76-86`).
- *Do:*
  1. Budget = RAM − workers × (measured peak worker PSS from A2) − 1 GB reserved for UI and OS.
  2. Cap it at 50 % of RAM, and spill to `temp_directory` beyond that.
  3. Drop to the serve budget afterwards (C11).

### E. Steady-state server memory

**E1. A written memory budget** · T1 · S · low
- *Do:* set the ceiling at demo scale to ≤ 400 MB server RSS after an hour of use: DuckDB serve budget + 256 MB result cache + ≤ 64 MB export cache + Python baseline. A3 checks it after 50 scripted interactions.

**E2. Export cache bounded by bytes, keyed without the caption** · T2 · S · low
- *Where:* `export.py:24-41`.
- *Do:*
  1. Bound the cache at 64 MB.
  2. Key figure bytes on the spec *minus* `caption`. The caption is not drawn in the figure, so editing it must not evict renders ([RB V04]).

**E3. No large transient Python structures in callbacks** · T2 · S · low
- *Do:* after C1–C3 there is nothing to trim (the 245 MB transient peak disappears). Check it with `tracemalloc` in A4. Do not rely on `gc.collect()` or `malloc_trim` hacks.

### F. Export latency

**F1. Render exports in a dedicated process** · T2 · S · low
- *Where:* `export.py` (`fig_png`, `fig_pdf`, `fig_pgf` under `_render_lock`).
- *Do:* use a `ProcessPoolExecutor(max_workers=1)` created at start, with Matplotlib imported and rcParams isolated. The server sends a spec and receives bytes.
- *Why:* rendering stops holding the server's GIL, global rcParams races disappear, and a TeX crash cannot take the server down.
- The worker is pre-warmed at start (Matplotlib imported, fonts cached), so no click pays process start-up ([P1]); it counts against E1.

**F2. Speculative pre-render** · deferred until F1 is measured · S · medium
- *Do (if still needed):* after 1–2 s of idleness on a stable `fig-spec`, pre-render **PNG only** in the export process, at low priority, deduplicated, and cancelled when the spec changes ([P2], [P3], [P6]).
- *Why deferred:* every control change produces a new spec, so eager rendering would spend about 1 s of CPU per slider tick while the user explores, against E1 and B1. With the caption-free cache key (E2) and the persisted TeX probe (B3), the measured export click is 0.6–0.9 s today.
- *Done when:* the download click is ≤ 100 ms **and** interactive latency does not regress, with the speculative work included in the memory account.

**F3. Cache failures, demote engines, say why** · T2 · S · low
- *Where:* `export.py:170-186` (`fig_pgf` returns `None` and re-runs TeX on every click).
- *Do:* cache failures per (engine, spec), demote a failing engine and try the next ([RA X3, X4]), and show the reason in the export hint.

### G. Demo readiness

**G1. Real-data rehearsal on the demo laptop** · T1 · M · —
- *Do:*
  1. Run A2 and A3 with the Zenodo tree (A8).
  2. Decide what the on-stage "delete `.kpviz/` and re-scan" (paper §6 step 5) runs on. Pick a demo subtree (test splits + sampled training via D9f) sized to finish in ≤ 60 s on that laptop.
  3. State the subset honestly on screen.
- *Why:* a full real-scale cold scan is estimated at 40–45 minutes on 4 cores for the documents phase alone (F10).

**G2. Offline kit** · T1 · S · —
- *Do:*
  1. Pre-download every tokenizer asset named by the demo cards, so token positions are exact on stage:
     - bart-base into `.kpviz/tokenizers/`;
     - `o200k_base` into `TIKTOKEN_CACHE_DIR`;
     - Llama through `HF_TOKEN`, or from a local `tokenizer.json` path ([RA D4]).
  2. Set `HF_HUB_OFFLINE=1`.
  3. Persist the TeX probe (B3).
  4. Use a pinned environment lock ([RA U2], [RB M02]).

**G3. Scripted walkthrough test** · Wave 2 · S · — ([P6])
- *Do:* a Playwright script plays §6 end to end:
  1. Scan for changes.
  2. Open RQ3 and select the kptimes models.
  3. Check the figure and caption.
  4. Copy LaTeX and download PGF.
  5. Delete the store and re-scan.
  6. Assert the figure spec hash is identical.

  Each step fails if it exceeds 120 % of its budget ([P4]). Run it daily until the demo, and on the demo laptop at the freeze.

**G4. Warm start** · T2 · S · —
- *Do:* start the app a few minutes before the session. B9 warms every default view, so the first click of the demo is a cache hit.

**G5. Performance strip on the Overview (optional, high demo value)** · T2 · S · low
- *Do:* a compact live strip:
  - during a scan: documents/s, predictions/s, worker utilisation (Σ worker-s / (wall × workers)) and peak memory, all from A2;
  - afterwards: the last interaction latency (A6).
- *Why:* it makes the paper's efficiency claims visible, so the audience watches the engine rather than a spinner.

**G6. Align the paper's performance claims with what the code runs** · Wave 2 · S · — ([P5], [P6])
- *Do:* quote end-to-end numbers from A on named hardware, for example "N predictions matched in X s on a 4-core laptop; interactions ≤ Y ms". Re-time Table 2 on the shipped code path: spaCy tokenisation, same-definition PRMU ([RA C1], [R1 B.2]).
- Keep a claim table: each paper claim → the code path → the measurement or test that supports it ([P5]). Q&A is likely to land on Table 2's tokenizer row, the in-order PRMU definition and Porter2 ([P6]). The byte-offset story of Fig. 2 must survive any schema change (C6 derived keys only) ([P4]).

**G7. Fallback kit** · T1 · S · —
- *Do:* keep a pre-scanned copy of `.kpviz/` (restorable in seconds), pre-rendered exports and screenshots of every step, and the demo video. If anything misbehaves live, restore and continue.

### H. Guardrails: what not to do

- **No multi-process WSGI** (gunicorn workers). `ScanState` is in-process, and DuckDB allows one writer per file.
- **No Celery/Redis background callbacks.** They add processes, memory and moving parts. C1–C4 make callbacks fast enough, and F1 isolates the only CPU-heavy UI work (rendering).
- **No `fork` start method** in the server. The existing reasoning in `scanner._mp_context` stands.
- **No free-threaded Python** experiment before the demo. This dependency stack (spaCy, tokenizers, DuckDB) has not been validated on it here.
- **No regex tokeniser swap** to "win" Table 2. That changes token boundaries, so it is a semantic change and belongs to [RA C1].
- **No new cache bounded by entry count.**
- **No optimisation without its A-series benchmark**, and no merge without A5.

---
## 4. Dependencies between actions

The `Blocked by` column in Section 5 is authoritative. The main chains:

```
A1..A6, C13..C25 ──► every performance action (baseline, gates, correct inputs)
D4 ─► D2 ─► D3 (experiment)          D8f ─► D2 at real scale (task bytes bounded)
D6 ─► D7a                            D14 (cheap half) ─► D7a ─► D7b ─► D10b..d
D14 (full) ─► D10, D16               C18 ─► C5, D5 (one schema bump)
C1 ─► C3 ─► C6 ─► C7                 evaluation fixes (C21..C24, D8d) ─► C8
C4 ─► B9, G4                         B1 ─► B6, B9, B10
F1 ─► F2 (only if still needed)      A8 ─► G1 ─► G3 (real budgets), G7, D9f decision
```

---

## 5. Tracker

Status: **done** = implemented and verified on this branch; **partial** = the part named is done; **open** = not started; **deferred** = after the demo, with its reason in the item; **n/a** = not needed.

| ID | Action | Wave | Blocked by | Status | Acceptance and evidence |
|---|---|---|---|---|---|
| A1 | Fixture generators | 1 | — | partial | `tools/make_sample_src.py`, `make_sample_data.py`, `make_scaled_data.py` committed, seeded; the large-vocabulary L fixture is open |
| A2 | Scan memory/pool profiler | 1 | — | partial | `tools/bench/scan_profile.py` (parent + children RSS/PSS/USS per phase); machine spec in every `scan_stats`; the in-app `mem` block is open |
| A3 | UI probe | 1 | — | done | `tools/bench/probe_ui.py`, Section 6.2 |
| A4 | Kernel benchmarks | 3 | A1 | open | — |
| A5 | Parity + determinism gates | 1 | — | done | `tests/test_metrics.py` (independent reference, > 500 cells), `tests/test_scan.py` (workers 1 vs 4, incremental = clean), `tools/bench/fingerprint.py`, `tools/bench/metrics_snapshot.py` (2,040/2,040 vs original code) |
| A6 | Per-callback timing | 1 | — | done | `/kpviz-perf`; callbacks > 0.5 s logged |
| A7 | Budget + CI | after | A5 | open | no CI in the repository yet |
| A8 | Real-data baseline | 4 | demo laptop | open | — |
| B1 | Visible-only computation | 2 | — | done | first load 1 callback, settle 0.22 s |
| B2 | Poll only while scanning | 2 | B1 | done | 0 idle requests |
| B3 | TeX probe off request path | 2 | — | done | background thread, persisted in `tex_probe.json` |
| B4 | Clientside toggles | 2 | B1 | done | `assets/route.js` |
| B5 | Loading overlay | 2 | — | done | `ui.loading` (300 ms delay, previous figure stays visible) |
| B6 | Split monolithic callbacks | 3 | B1 | partial | Datasets browser split out (keeps its search); RQ3 panels still share one callback |
| B7 | Lighter figures + Patch | 3 | A4 | partial | `uirevision`; payload trimming open |
| B8 | Compression | after | — | open | — |
| B9 | Prefetch default views | 4 | C4 | done | recently used views (persisted) warmed after start-up and after each scan |
| B10 | URL state | after | B1 | open | — |
| B11 | waitress | after | — | open | Flask threaded server kept |
| C1 | One-statement scoring | 2 | A5 | done | parity grid; RQ1/RQ4 ≤ 0.13 s, RQ2 0.45 s |
| C2 | NumPy fetch | 2 | — | done | `db.qnp`; gold packs written by DuckDB itself (`COPY … FORMAT JSON`, 1.25 → 0.58 s) |
| C3 | Per-doc arrays over a doc index | 2 | C1 | done | `metrics.DocIndex`, `PerDoc`; stats vectorised, identical ranks and ties |
| C4 | Byte LRU + single-flight | 2 | — | done | `ByteLRU`, 192 MB default |
| C5 | No UI writes to DuckDB | 2 | C18 | done | colours in `colors.json`; `agg_cache` removed |
| C6 | Integer keys | after | C18 | open | — |
| C7 | Clustering / zone maps | after | C6 | open | — |
| C8 | Precomputed filtered views | after | C21–C24 | open | — |
| C9 | Catalog snapshot, no N+1 | 2 | — | partial | helpers memoised per catalog version (RQ3 run limits, RQ5 variations, options); a single snapshot object is open |
| C10 | Datasets pre-aggregates | 3 | D4 | partial | browser paged in SQL; aggregates open (page 1.9 s) |
| C11 | Serve-mode DuckDB memory | 2 | — | done | 2 GB serve limit, allocator flush |
| C12 | Retry only retryable | 1 | — | done | `db._retryable` |
| C13 | Content-hash new files | 1 | — | done | `tests/test_scan.py` no-op and incremental tests |
| C14 | `language` and `languages` | 1 | — | done | `tests/test_contract.py` |
| C15 | Purge by file | 1 | — | n/a | one collection file per dataset |
| C16 | Versioned phrase cache | 1 | C19 | done | `test_phrase_cache_retags_on_tagger_change` |
| C17 | Shading after axis limits | 1 | — | done | `figures.to_mpl` |
| C18 | Honest names, one schema bump | 1 | — | done | `SCHEMA_VERSION = 5` |
| C19 | Source-hash code version | 1 | — | done | `scanner.CODE_VERSION` |
| C20 | JSONC cards, reported parse errors | 1 | — | done | `test_broken_inputs_are_reported_not_dropped` |
| C21 | Identity and filter defects | 1 | — | done | `test_empty_document_filter_is_not_no_filter` |
| C22 | Export correctness | 1 | — | done | `test_matplotlib_pdf_has_no_type3_fonts`, `test_figure_environment_follows_size`, `test_missing_values_export` |
| C23 | Coverage, run tags | 1 | — | done | `test_issue_tags` |
| C24 | Captions and encodings | 1 | — | done | `test_caption_names_the_gold_actually_used`, `test_pareto_ties_and_non_finite` |
| C25 | Pinned requirements | 1 | — | done | `requirements.txt` |
| D1 | Measured CPU budget | 3 | A2 | partial | 4-vCPU sweep kept the all-cores default; 8/16-core sweeps open (A8) |
| D2 | Coalesced prediction tasks | 3 | D4 | done | inferences 16.9 → 10.8 s, CPU 67 → 84 % |
| D3 | Parquet spills | after | D2 | deferred | experiment |
| D4 | Acknowledged ingest barrier | 1 | — | done | `test_drain_waits_for_the_write`, `test_ingest_error_is_sticky` |
| D5 | Zero post-ingest UPDATEs | 2 | C18 | done | no statement touches all of `preds` |
| D6 | Tokenizers once, offline workers | 1 | — | done | negative cache, `HF_HUB_OFFLINE` in workers, `--offline`; `test_offline_tokenizer_never_downloads` |
| D7a | No-op scans publish nothing | 2 | D6, D14 cheap | done | 0.27 s, version unchanged (`test_noop_scan_publishes_nothing`) |
| D7b–c | Partial finalize | after | D7a | open | — |
| D8a | POS sized to work, gold only | 3 | — | done | 4.5 → 2.9 s |
| D8b | Bounded decoded pack | 3 | — | done | 200 k decoded entries per worker |
| D8c | mmap-shared pack | after | A1 L | deferred | only if the L profile shows need |
| D8d | Language-keyed caches, per-language analysis | 1 | — | done | `test_language_keyed_phrase_cache`; determinism test |
| D8e | Columnar accumulation | after | D3 | deferred | — |
| D8f | Memory-aware chunks | after | A1 L | open | — |
| D9a–c | Normalise once, one-pass LID, O(log n) positions | 3 | A5 | done | `test_detect_language_matches_reference`, `test_approx_positions_equal_prefix_rescan`, `test_norm_offsets_map_back_to_source` |
| D9d–e | One exact encode, NumPy attributes | 3 | C22 | done | determinism and parity gates |
| D9f | Training-split sampling | after | G1 | deferred | boundaries written; off by default |
| D9g | Hoisted closures | after | — | open | — |
| D10a | Signature from derivation-relevant card fields | 2 | — | done | `test_description_edit_rederives_nothing` |
| D10b–d | Append / per-batch / gold-hash incrementality | after | D14 | open | — |
| D11 | Hash while reading | after | C13 | open | C13 covers correctness; the double read remains |
| D12 | forkserver preload | 3 | — | done | `kpviz/_preload.py`: thread pools pinned before NumPy; tokenizers of the declared languages built once; always in the server, in one-shot scans only for non-English trees (measured both ways, Section 6.6) |
| D13 | Longest-first, POS overlap | 3 / after | — | partial | (a) document chunks of every collection submitted largest first; POS overlap open |
| D14 | Atomic generations | 2 (cheap) / after (full) | — | partial | failed or cancelled scans invalidate caches and force a full next pass |
| D15 | Scan lock + cancel points | 1 | — | done | `test_concurrent_start_starts_one_scan` |
| D16 | Compaction | after | D14 | open | — |
| D17 | Scan DuckDB budget | 2 | A2 | done | `config.duckdb_memory_bytes` |
| E1 | Memory budget | 2 | C4 | done | 391 MB after the full probe (Section 2.1 model) |
| E2 | Export cache in bytes, caption-free key | 2 | — | done | `test_caption_edit_does_not_rerender` |
| E3 | No large transients | 2 | C3 | partial | by construction; a `tracemalloc` check is open |
| F1 | Export process | 4 | — | open | — |
| F2 | Speculative pre-render | after | F1 | deferred | — |
| F3 | Failure caching and demotion | 3 | B3 | done | engines pdflatex → xelatex → lualatex, failures cached, reason shown |
| G1 | Real-data rehearsal | 4 | A8 | open | re-scan ≤ 60 s on the chosen subtree |
| G2 | Offline kit | 4 | D6 | partial | `--offline`, negative cache, `file:` tokenizer specs; the kit itself is built on the demo laptop |
| G3 | Scripted walkthrough | 2 | B1 | partial | `tests/test_ui.py` renders and exports every workbench; the §6 end-to-end script is open |
| G4 | Warm start | 4 | B9 | done | automatic (B9) |
| G5 | Performance strip | 4 | A2 | open | — |
| G6 | Paper claim alignment | 2 | A8 | open | — |
| G7 | Fallback kit | 4 | G1 | open | restore ≤ 30 s |

---

## 6. Measured results

### 6.1 Machine and method

| Item | Value |
|---|---|
| Machine | 4 usable vCPU, 15.7 GB RAM, Linux 6.18, Python 3.11 |
| Libraries | Dash 4.4.1, Plotly 7.1.0, DuckDB 1.5.5, spaCy 3.8.16 (+ `en_core_web_sm`, `fr_core_news_sm` 3.8.0), PyStemmer 3.1.0, orjson 3.12.0, NumPy 2.4.6, Matplotlib 3.11.2 |
| Tree | `tools/make_scaled_data.py . sample_src scaled_data 20000`: 5 datasets, 66 runs, 75,190 documents, 1.32 M prediction lines, 296 MB, 5,366 files |
| Network | offline (`KPVIZ_OFFLINE=1`), so tokenizers are approximate; no TeX |
| Before | the original code, run the same day on the same tree, page cache warm for both |

The before cold scan measured 36.8 s here against 43.4 s in Section 1; Section 1's run had a cold page cache. Both columns of 6.2 use the same-day runs.

### 6.2 Scan

| Phase | Before | Now |
|---|---|---|
| discover | 0.22 s (0 files hashed) | 1.39 s (5,366 files hashed, C13) |
| documents | 11.50 s | 8.38 s |
| inferences | 16.87 s | 10.80 s |
| keyphrases (POS) | 4.45 s | 2.85 s |
| finalize | 3.04 s | 2.21 s |
| **total, cold** | **36.8 s** | **25.9 s** |
| no-op re-scan | 1.90 s, version 1 → 2 | 0.27 s, version unchanged |

| Peak memory (MB) | Before: parent RSS / workers PSS | Now: parent RSS / workers PSS |
|---|---|---|
| documents | 276 / 528 | 229 / 487 |
| inferences | 835 / 699 | 382 / 812 |
| keyphrases | 762 / 1,091 | 326 / 908 |
| finalize | 933 / 15 | 679 / 15 |
| **overall peak (parent + workers)** | **1,853** | **1,194** |

Derived row counts are identical, except `keyphrases` (2,413 → 2,410): the phrase cache is now keyed by language and holds gold phrases only (D8a, D8d), a labelled correctness change.

### 6.3 UI (`probe_ui.py`, headless Chromium, 1440 × 1000)

| Action | Callbacks | Slowest callback | Server RSS |
|---|---|---|---|
| server start | — | — | 126 MB |
| first load `/` | 1 | 0.37 s | 136 MB |
| idle 10 s on Overview | 0 requests | — | — |
| open Insights (RQ4 default) | 7 | 0.27 s | 154 MB |
| first open of RQ1 / RQ2 / RQ3 / RQ5 | 11 / 11 / 12 / 9 | 0.14 / 0.48 / 1.15 / 0.90 s | 154 → 281 MB |
| change @k on RQ1 / RQ2 / RQ3 / RQ4 / RQ5 | 2 / 2 / 3 / 2 / 2 | 0.09 / 0.45 / 0.98 / 0.13 / 0.93 s | → 380 MB |
| tab away and back (unchanged inputs) | 15–20, all answered by `PreventUpdate` or the signature | ≤ 0.30 s | flat |
| export PNG / PDF | — | 0.91 / 0.64 s | — |
| open Datasets / Models / Architectures | 11 / 2 / 2 | 1.89 / 0.09 / 0.12 s | 394 MB |
| end of probe | — | — | 391 MB, 0 console errors |

### 6.4 Correctness gates

`python -m pytest tests`: 59 tests, about 2 min, all passing (37 in revision 2; added: statistics against SciPy, LaTeX compilation in venue classes, every workbench under three statistics settings). They cover the data contract as the README writes it (JSONC cards, both language spellings, both similarity-key spellings, ISO timestamps, missing run cards, illegal parameters, broken inputs reported), the independent scoring reference over every dataset × annotation set × filter × k × measure, worker-count determinism, incremental = clean rebuild, no-op scans, concurrent scan starts, the ingest barrier, and render + export of every workbench.

### 6.5 Workbench latency after revision 3

Callbacks called directly on the scaled store, result cache cleared (cold) and repeated (warm); default statistics (rank-based, Holm, t-intervals):

| Workbench | Cold | Warm |
|---|---|---|
| RQ1 correlation (3 datasets) | 0.23 s | 0.06 s |
| RQ2 data quality (22 runs) | 0.26 s | 0.10 s |
| RQ3 extractability, both panels (22 runs) | 0.64 s | 0.12 s |
| RQ4 cost–performance with intervals (3 datasets) | 0.50 s | 0.04 s |
| RQ5 hyperparameters (3 datasets, Friedman) | 0.34 s | 0.10 s |

In the browser (probe): first load 0.29 s; export PDF typeset through pdflatex 1.7 s, PNG 1.0 s.

### 6.6 Revision 4: a real card set, and a function-level pass

Verified on the authors' own cards (7 models, 3 architectures, 51,825 similarity pairs) with documents and runs synthesised around them by `tools/synth_tree.py` (62.9 k documents, 137 runs, 83 MB). Everything resolves, costs and renders; the planted problems — and only those — are flagged. The pass below was profiled with `tools/bench/profile_scan.py` (every worker task's own cProfile, merged).

| Finding (function) | Fix | Effect |
|---|---|---|
| Every worker compiled spaCy's French tokenizer-exception regex (3.8 s each) | forkserver preload of the declared languages (D12) | inferences phase 7.4 → 1.4 s on this tree |
| DuckDB's client tries `import pandas` for every bound parameter; failed imports are not cached, so each walked `sys.path` | mark genuinely absent optional modules as absent once (`db._mark_missing_optional_modules`) | RQ1 0.17 → 0.02 s warm; every query cheaper |
| RQ2 resolved the same flagged document set to ordinals once per run, and re-ran the similarity join on every call | `PerDoc.select_ords`, one resolution per distinct set; leak sets memoised per catalog version | RQ2 1.26 → 0.38 s |
| Gold packs fetched every document's lists into Python and re-serialised them | `COPY … FORMAT JSON` straight from DuckDB | 1.25 → 0.58 s |
| `kv_set_many` was an `executemany` (1.5 ms a row) | one `unnest` insert | 0.40 → 0.08 s |
| Document chunks submitted in dataset order, 4 MB minimum | largest first, 1 MB minimum | worker tail removed |
| `detect_language` counted per token in Python | C-level filter + `Counter` | −40 % per call, identical output |
| `python - <<EOF` scans crashed every forkserver child (`__file__ == "<stdin>"`) | require a real file | bug fixed |
| Size-named training splits (`train_large`) counted as evaluation data | one training-split rule in Python and SQL | correctness fix |
| `tiktoken[llama3]` said "not cached locally" — tiktoken has no such encoding | say so, and what would work | clarity |

Measured on this machine (cold scan, fresh process): the authors' card tree 11.3 s (no-op re-scan 0.14 s), peak 783 MB; workbenches cold (result cache cleared) RQ1 0.27 s · RQ2 0.48 s (its 51 k-pair similarity join included, then memoised) · RQ3 0.14–0.21 s · RQ4 0.12 s · RQ5 0.10 s; warm ≤ 0.19 s. The 296 MB synthetic tree: 24.2 s cold (25.9 s in revision 3), no-op 0.21 s. Derived tables are byte-identical before and after every change in this pass (`fingerprint.py`).

---

## 7. How to reproduce

Every script below is in the repository.

```bash
pip install -r requirements.txt
pip install pytest psutil playwright
playwright install chromium        # or pass --chromium PATH to probe_ui.py

python -m pytest tests             # contract, parity, determinism, UI

python tools/make_sample_src.py sample_src
python tools/make_sample_data.py --src sample_src --out sample_data
python tools/make_scaled_data.py . sample_src scaled_data 20000          # 296 MB, 66 runs

python tools/bench/scan_profile.py --data scaled_data --state /tmp/st --json scan.json   # cold
python tools/bench/scan_profile.py --data scaled_data --state /tmp/st                    # no-op
python tools/bench/fingerprint.py --state /tmp/st                                        # table hashes

python app.py --data scaled_data --state /tmp/st --no-browser --port 8091 &
python tools/bench/probe_ui.py http://127.0.0.1:8091 --json ui.json --screenshots shots

# metric parity between two code versions or stores
python tools/bench/metrics_snapshot.py --data sample_data --state /tmp/st_a --out a.json
python tools/bench/metrics_snapshot.py --compare a.json b.json
```

Record the machine spec with every result (`scan_stats/*.json` already carries it). Keep the before/after results of each wave in `docs/benchmark_results_jcdl26.txt` with the date ([P4]).
