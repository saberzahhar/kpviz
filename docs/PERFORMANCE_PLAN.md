# KPViz performance plan: memory, parallel workers, efficiency, latency

Status: plan only (no code changed). Written 27 Sep 2026, 16 days before the JCDL '26 demo (13–16 Oct).

This plan covers everything that affects how fast KPViz responds, how much memory it holds, and how well it uses the machine's cores, from the first scan to the last click of the demo. Each action names the code it touches, how to change it, the measured reason for it, and the check that closes it.

Sources: a full read of this repository, a new baseline measured for this plan (Section 1), and the six external reviews supplied with the request. The reviews are cited as **[RA §]** for `REVIEW.md`, **[RB §]** for `KPViz-review.md` and **[R1]**–**[R5]** for the five reviewer notes.

Evaluation-semantics fixes (PRMU definition, padding, gold de-duplication, missing-document policy, and so on) are out of scope here, except where they interact with parallelism or caching. They change published numbers, so they belong in separate, labelled PRs with a `CODE_VERSION` bump ([RA §2], [RB §3]).

---

## 0. Summary

### 0.1 Where we are and where we are going

| Metric (demo-scale synthetic tree, 4 vCPU) | Measured today | Target | Main actions |
|---|---|---|---|
| Callbacks fired by the first page load | 96–98 (56 of them the 1 Hz poll) | ≤ 15 | B1, B2, B3 |
| Time until the UI settles after first load | 28.3 s | ≤ 1.5 s | B1, B3, C1 |
| Server RSS: before load → after first load | 131 → 766 MB | ≤ 300 MB | B1, C3, C4, C11 |
| Server RSS growth per heavy interaction | +17 to +153 MB; only a 4,096-entry cap bounds it | ≤ 5 MB steady; hard byte cap | C3, C4 |
| RQ3 interaction (change @k) | 11.2 s | ≤ 0.4 s | C1–C3, C9 |
| RQ5 interaction | 6.4 s | ≤ 0.4 s | C1, C9 |
| RQ2 interaction | 3.5 s | ≤ 0.25 s | C1 |
| Idle requests per open browser tab | 20 per 10 s, forever, on every page | 0 | B2 |
| Filtered/per-document scoring kernel (22 runs × 20 k docs) | 2.7–3.2 s | ≤ 0.15 s | C1, C2 |
| One memoised per-document result | 47 MB | ≤ 2 MB | C3 |
| Cold scan (75 k docs, 1.32 M prediction lines) | 43.4 s | ≤ 25 s (to verify) | D2, D5, D6, D8, D9, D12 |
| Mean CPU use while matching predictions | 62 % | ≥ 90 % | D2, D3, D5 |
| No-op "Scan for changes" | 3.2 s, and it bumps the catalog version | ≤ 0.4 s, no bump | D6, D7 |
| POS phase for 2,413 new phrases | 4.9 s and 1.1 GB of worker memory | ≤ 1.5 s, ≤ 400 MB | D8a |
| Peak scan memory (parent RSS + workers PSS) | 1.87 GB | ≤ 1.1 GB | D8, D12, D17 |
| NDJSON spill written for `matches` alone | 252 MB | ≤ 10 MB | D2, D3 |
| DuckDB file after one re-derivation | 50.8 → 97.8 MB | ≈ live size | D14, D16 |

The targets are engineering goals. Workstream A validates each of them; none is a measurement yet.

### 0.2 The ten changes that matter most

1. **B1. Compute only what is on screen.** Hidden pages and workbenches fire about 40 of the first-load callbacks, including the four that take 7–22 s, so they account for nearly all of the 28 s. The 1 Hz poll fires most of the rest (56 of 96) for as long as the page stays busy.
2. **C1 + C2. One DuckDB statement per scoring request, fetched as Arrow/NumPy.** This takes 3 s down to 0.13 s. Most of the old cost was `fetchall()` building Python lists: 1.93 of the 2.7 s.
3. **C3 + C4. Per-document results as float32 arrays, in a byte-bounded single-flight cache.** An entry shrinks from 47 MB to 1.7 MB, and concurrent identical requests are computed once.
4. **B2. Stop polling when nothing is scanning.** The Overview poll runs on every page. It sent 56 of the 96 first-load requests and 24 of the 27 requests during a single RQ3 interaction.
5. **B3. Take the TeX probe off the request path.** Six callbacks at first load wait on it.
6. **D6 + D7. Resolve tokenizers once per scan and make no-op scans free.** Today every worker retries the download (about 1 s each), and every scan bumps the catalog version, which re-fires every workbench.
7. **D2. Coalesce tiny prediction files into real tasks.** 2,640 tasks of 0.08 MB each became up to 7,900 spill files and 11.9 s of ingest.
8. **D8a. Size the POS phase by the amount of work.** It loads spaCy models into every worker (488 MB each) to tag 2,413 phrases.
9. **D5. Remove the serial post-ingest `UPDATE`s.** They cost 5.6 s on the scan thread.
10. **G1–G3. Rehearse the demo on the demo laptop with the real data, from a script.** At real scale a live "delete `.kpviz/` and re-scan" is estimated at 40–45 minutes on 4 cores for the documents phase alone (Section 1.3), so the on-stage step needs a planned subset.

### 0.3 Schedule to the demo

| Window | Tier | Content |
|---|---|---|
| 28–30 Sep | Tier 0 | Workstream A essentials: A1–A5 (about 2–3 days, reusing Review A's scripts) |
| 30 Sep – 7 Oct | Tier 1 (demo-critical) | B1–B5, C1, C2, C4, D6, D7a, A6, A8, E1, G2, G7 (about 8 days) |
| 8–9 Oct | Tier 2, in this order, as time allows | G3, B9 + G4, F1 + F2, C9, C11, D8a, D5, C3, C5, D2, D12, B6, G5; then A7, C12, D1, D8d, D9a–c, D9f, D15, D17, E2, E3, F3, G6 |
| 10 Oct | Freeze | Bug fixes only. Real-data rehearsal on the demo laptop (G1) |
| after the demo | Tier 3 | B7, B8, B10, B11, C6, C7, C8, C10, D3, D4, D7b–c, D8b, D8c, D8e, D8f, D9d, D9e, D9g, D10, D11, D13, D14, D16 |

With one developer, Tier 0 and Tier 1 fill most of the time to the freeze, and Tier 2 is picked in the order listed. If only Tier 1 lands, the demo still gets the large perceived wins: fast first load, near-instant "Scan for changes", and sub-second workbenches.

---

## 1. Baseline measured for this plan

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
| Harness | a memory-sampling scan runner (parent + every child: RSS/PSS/USS per phase), a Playwright UI probe (callback round trips, settle time, idle traffic, server RSS), cProfile of one worker chunk of each kind, and kernel micro-benchmarks. It lives in the session scratchpad today; A1–A4 turn it into committed tooling |

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
2. **Numbers never move in a performance PR.** Every one of them must pass the parity and determinism gates (A5). Semantic changes go in separate PRs, with a `CODE_VERSION` bump and release notes.
3. **Push arithmetic into DuckDB.** It runs vectorised, uses every core and releases the GIL. The Python in callbacks should only shape results.
4. **Nothing expensive or blocking on the request path.** No network, TeX, model loading or DuckDB writes inside a UI callback.
5. **Bound every cache by bytes**, never by entry count.
6. **Work is proportional to what is visible and to what changed.**
7. **Keep the architecture.** Keep one process, one DuckDB writer and one scan pool. Do not add Celery/Redis, a multi-process WSGI server, the `fork` start method or a frontend rewrite (Section H).

---

## 3. Workstreams and actions

Format per action: **ID. Title** · tier · effort (S < ½ day, M 1–3 days, L > 3 days) · risk. Then *Where*, *Do*, *Why* (measured where possible) and *Done when*.

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
  2. Make colour slots deterministic without writes: sorted entity order per scope, computed at finalize and published with the generation. This also fixes colours changing after a rebuild [RA C8].
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

**C8. Precompute the common filtered views** · T3 · S · low
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
- *Do:* retry only on connection or cursor invalidation. Raise parser, binder and catalog errors at once, with the original traceback ([RA G2], [R2 #2]).

### D. Scan engine: parallel workers and throughput

**D1. CPU budget from measurement, not assumption** · T2 · S · low
- *Where:* `config.py:45-56`.
- *Do:*
  1. Default `workers = cpus − 1` on ≤ 8 cores and `cpus − 2` above, because the parent's ingest thread, pool driver and DuckDB need a core.
  2. Give DuckDB `max(1, cpus − workers)` threads during a scan.
  3. Add `--workers auto|N` and a sweep in A2 (N ∈ {1, 2, …, cpus}) that reports throughput, peak PSS and CPU %.
  4. Choose the default from the sweep, on 4, 8 and 16 cores.
- *Why:* today the default is all cores, plus DuckDB threads on top ([RA S16]). Yet measured CPU was only 62–79 %, which points at serial phases and per-task overhead (D2, D5), not oversubscription. The sweep separates the two.

**D2. Coalesce small prediction files into real tasks** · T2 · M · medium
- *Where:* `scanner._derive_all_runs` (`jobs_iter`, `scanner.py:896-912`), `derive.derive_preds_chunk`.
- *Do:*
  1. Make a task carry a **list** of `(path, start, end, batch_idx, file_id)` from one run, filled up to a target of `clamp(total_bytes / (workers × 4), 2 MB, 16 MB)`.
  2. Split big files by byte range as today.
  3. Keep dataset-major order, so each worker loads each gold pack once.
  4. Track `remaining` per run key across tasks.
- *Why:* 2,640 tasks of 0.08 MB produced up to 7,900 spill files and 11.9 s of ingest. One `read_json` of the same rows takes 0.9 s (F3).
- *Done when:* fixture M has ≤ 100 prediction tasks, inference ingest is ≤ 3 s, and CPU during matching is ≥ 85 %.

**D3. Columnar spills (Parquet)** · T3 · M · medium
- *Where:* `derive._write_ndjson`, `ingest.py`, `db.ingest_ndjson`.
- *Do:*
  1. Workers accumulate **column lists**, not row dicts, and write Parquet (zstd) with `pyarrow`. That adds one dependency, already present in most environments.
  2. The ingestor issues `INSERT INTO t SELECT * FROM read_parquet([...])`.
  3. Keep NDJSON as a fallback when `pyarrow` is missing.
- *Why:* for `matches`, 252 MB of NDJSON becomes 4.4 MB of Parquet, and the one-statement ingest drops from 0.91 s to 0.31 s. Workers also stop holding a list of row dicts *and* a `BytesIO` copy of the whole chunk (`derive.py:99-109`).
- *Done when:* tmp spill bytes are ≤ 5 % of today's, and parity (A5) holds.

**D4. Ingestor: acknowledged barrier and byte-aware backpressure** · T3 · S · low
- *Where:* `ingest.py:55-147`.
- *Do:*
  1. Have `drain()` wait on a `Future` that completes *after* the write. Today `n_pending` drops when a path is dequeued, before it is written ([RB D02]).
  2. Keep a persistent error state that wakes waiters, and route every flush path through one exception boundary.
  3. Apply backpressure on the sum of queued spill bytes, not on the file count.
- *Done when:* a deliberately delayed write keeps `drain()` waiting, and an injected write error surfaces.

**D5. Remove the serial post-ingest UPDATEs** · T2 · S · low
- *Where:* `scanner.py:929-945`.
- *Do:*
  1. Compute `known_doc` in the worker, from a compact per-dataset id set covering **all** splits: sorted ids plus offsets, shipped next to the gold pack.
  2. Alternatively, drop the column and derive `unresolved_ids` with an anti-join inside `_aggregate_runs_sql`.
  3. Compute `batches.n_docs` inside `_aggregate_runs_sql` instead of `UPDATE`.
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
- *Done when:* a no-op scan takes ≤ 0.4 s with no generation change, and editing one run touches only that run's rows.

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
  - Do this only if the fixture L profile (A2) shows RSS scaling as workers × pack.
- **d. Caches without cliffs, keyed for determinism** · T2 · S · medium
  - *Where:* `textproc.PhraseCache` (`:146-189`), `derive._MATCH_STEMS` (`:79-96`).
  - *Do:*
    - Replace clear-all resets with a two-generation scheme: on overflow the current dict becomes "old", lookups promote from old to new, and the old generation is dropped at the next overflow ([R2 #6–7], [R4 3.1]).
    - Bound `_persisted` without triggering re-persist storms.
    - Key by `(lang2, norm)`. This is the determinism fix of [RA E9], and A5 checks it across worker counts.
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
- **f. Seeded sampling of training splits for distribution charts** · T2 · M · low
  - *Where:* `derive_doc_chunk`, new `--dist-sample N` (default 50 k per split).
  - *Do:*
    - Sample only the expensive NLP for training documents: spaCy, stemming, PRMU, language ID and tokenizer counts run when a seeded uniform sample selects the document.
    - **Every** training document still gets its cheap `documents` row (id, split, byte offsets), because RQ2's leakage criterion joins counterpart training documents by id (`rq2._leak_docs`).
    - Captions state "n = 50,000 of 530,809, uniform sample" ([RA S14]). The Overview says that language flags on training splits come from the sample.
  - *Why:* training splits dominate the real 14.2 GB tree (F10). This is the single largest lever on a real cold scan.
- **g. Hoist per-document closures** · T3 · S · low
  - *Do:* define `stream` and `agg_add` once per chunk (`derive.py:220, 259`).

**D10. Finer incrementality** · T3 · M–L · medium
- *Do:*
  - **a.** Build the document signature from the card fields that affect derivation (sections, languages, annotations), not `card.raw` [RA S10].
  - **b.** Detect appends to a JSONL collection (prefix hash unchanged, file grew) and derive only the tail.
  - **c.** Give runs per-batch signatures: purge and re-derive only the changed batch files.
  - **d.** Have a run's signature reference a **hash of its dataset's gold pack**, not the document file's signature. Editing document text that does not change gold then leaves the runs untouched.
- *Done when:* appending one document re-derives one chunk and zero runs (today it is 1 collection + 9 runs [R1 C.3]).

**D11. Hash while reading** · T3 · S · low
- *Where:* `scanner._discover` (`:482-527`), worker chunk readers.
- *Do:* workers compute a blake2b per byte range as they read, and the parent combines them in order. Remove the separate hashing pass, and the spurious re-derivation after the first scan (Section 1.2 "First `--full`", [RA S11], [R5 1.1]).

**D12. Warm workers through forkserver preload** · T2 · S · low
- *Where:* `scanner._mp_context` (`:74-96`).
- *Do:* `ctx.set_forkserver_preload(["kpviz.derive", "kpviz.textproc", "spacy", "Stemmer", "orjson", "numpy"])`. The API is present on the installed Python.
- *Why:* workers fork from an image that has already imported the heavy modules. Start-up is faster and the import pages are shared copy-on-write, which lowers PSS.
- *Done when:* A2 shows lower per-worker USS and a faster first task.

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

**D15. Scan start under a lock, cancellation everywhere** · T2 · S · low
- *Where:* `scanner.start_scan` (`:238-246`), the POS, hashing and ingest loops.
- *Do:* do the check-and-set under a module lock ([RA G1]), and add cancel checkpoints to every long loop.

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

**F2. Speculative pre-render** · T2 · S · low
- *Do:* when a `fig-spec` has been stable for 1 s, render PDF, PNG and PGF in the export process in the background. A newer spec replaces the pending job. Clicking "PDF" or ".pgf" then returns cached bytes.
- *Why:* step 4 of the demo is "Copy LaTeX and PGF", so it must be instant.

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

**G3. Scripted walkthrough test** · T2 · S · —
- *Do:* a Playwright script plays §6 end to end:
  1. Scan for changes.
  2. Open RQ3 and select the kptimes models.
  3. Check the figure and caption.
  4. Copy LaTeX and download PGF.
  5. Delete the store and re-scan.
  6. Assert the figure spec hash is identical.

  Each step asserts its latency budget. Run it daily until the demo, and on the demo laptop at the freeze.

**G4. Warm start** · T2 · S · —
- *Do:* start the app a few minutes before the session. B9 warms every default view, so the first click of the demo is a cache hit.

**G5. Performance strip on the Overview (optional, high demo value)** · T2 · S · low
- *Do:* a compact live strip:
  - during a scan: documents/s, predictions/s, worker utilisation (Σ worker-s / (wall × workers)) and peak memory, all from A2;
  - afterwards: the last interaction latency (A6).
- *Why:* it makes the paper's efficiency claims visible, so the audience watches the engine rather than a spinner.

**G6. Align the paper's performance claims with what the code runs** · T2 · S · —
- *Do:* quote end-to-end numbers from A on named hardware, for example "N predictions matched in X s on a 4-core laptop; interactions ≤ Y ms". Re-time Table 2 on the shipped code path: spaCy tokenisation, same-definition PRMU ([RA C1], [R1 B.2]).

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

```
A1..A5 ──► every other action (baseline + gates)
B1 ─┬─► B6, B9, B10        C1 ─► C3 ─► C6 ─► C7
B3 ─┘                      C4 ─► B9, C5
D6 ─► D7a                  C9 ─► C5 (catalog snapshot carries colours)
D2 ─► D3 ─► D8e            D14 ─► D16, C4 (generation keys)
D8d ─► A5 determinism passes    D9f ─► G1 (real-scale demo subtree)
F1 ─► F2                   A8 ─► G1 ─► G3, G7
```

---

## 5. Tracker

| ID | Action | Tier | Effort | Risk | Expected effect | Acceptance (fixture M unless noted) |
|---|---|---|---|---|---|---|
| A1 | Fixture generator S/M/L | T0 | M | low | reproducible baseline | fixed seed, Section 1 tree |
| A2 | Scan memory/pool profiler | T0 | S | low | per-phase peaks archived | `mem`/`pool` blocks in scan_stats |
| A3 | UI probe | T0 | S | low | callback, idle and RSS numbers | Section 1.2 table in ≤ 90 s |
| A4 | Kernel benchmarks | T0 | S | low | per-kernel medians | JSON output |
| A5 | Parity + determinism gates | T0 | M | low | numbers can't move silently | injected change fails |
| A6 | Per-callback timing | T1 | S | low | p50/p95 per callback | `/kpviz-perf` |
| A7 | Budget + CI | T2 | S | low | regressions blocked | CI red on regression |
| A8 | Real-data baseline | T1 | S | low | demo numbers | recorded |
| B1 | Visible-only computation | T1 | M | low | removes ~40 hidden callbacks and the 7–22 s ones | with B2+B3: ≤ 15 callbacks, settle ≤ 1.5 s |
| B2 | Poll only while scanning | T1 | S | low | 20/10 s → 0 idle | 0 idle requests |
| B3 | TeX probe off request path | T1 | S | low | −6 slow callbacks | clipboards ≤ 50 ms |
| B4 | Clientside toggles | T1 | S | low | 0 round trips to switch | — |
| B5 | Loading overlay | T1 | S | low | visible progress | — |
| B6 | Split monolithic callbacks | T2 | M | low | concurrent panels | 1 callback per control |
| B7 | Lighter figures + Patch | T3 | S | low | smaller payloads | RQ3 ≤ 60 KB |
| B8 | Compression | T3 | S | low | faster first paint remotely | — |
| B9 | Prefetch default views | T2 | S | low | first click cached | cache hit |
| B10 | URL state | T3 | M | low | shareable views | — |
| B11 | waitress | T3 | S | low | robustness | — |
| C1 | One-statement scoring | T1 | M | med | 3.0 s → 0.13 s | parity; RQ3 ≤ 0.4 s |
| C2 | Arrow/NumPy fetch | T1 | S | low | 1.93 s → 0.22 s fetch | no large `fetchall` |
| C3 | Per-doc arrays (`doc_ord`) | T2 | M | med | 47 → 1.7 MB per entry | stats parity |
| C4 | Byte LRU + single-flight | T1 | S | low | bounded RSS | ≤ budget + 150 MB |
| C5 | No UI writes to DuckDB | T2 | S | low | no lock waits in scans | no `_wlock` in callbacks |
| C6 | Integer keys | T3 | M | med | faster joins, less memory | parity |
| C7 | Clustering / zone maps | T3 | S | low | row groups skipped | `EXPLAIN ANALYZE` |
| C8 | Precomputed filtered views | T3 | S | low | 10 ms defaults | parity |
| C9 | Catalog snapshot, no N+1 | T2 | M | low | ≤ 3 SQL per interaction | A6 |
| C10 | Datasets pre-aggregates | T3 | S | low | page ≤ 200 ms | — |
| C11 | Serve-mode DuckDB memory | T2 | S | low | RSS returns to OS | E1 |
| C12 | Retry only retryable | T2 | S | low | honest errors | — |
| D1 | Measured CPU budget | T2 | S | low | right worker count | sweep chart |
| D2 | Coalesced prediction tasks | T2 | M | med | ingest 11.9 → ≤ 3 s | ≤ 100 tasks, CPU ≥ 85 % |
| D3 | Parquet spills | T3 | M | med | 252 MB → ≤ 13 MB | parity |
| D4 | Acknowledged ingest barrier | T3 | S | low | correct drain | fault test |
| D5 | Drop post-ingest UPDATEs | T2 | S | low | −5.6 s serial | no full-table UPDATE |
| D6 | Tokenizers resolved once, offline workers | T1 | S | low | −1.85 s per scan, −1 s per worker | 0 sockets in workers |
| D7 | No-op / partial finalize | T1/T3 | S/M | low | 3.2 → ≤ 0.4 s | no generation change |
| D8a | POS sized to work, gold only | T2 | S | low | 4.9 s / 1.1 GB → ≤ 1.5 s / 400 MB | fixture L flat in workers |
| D8b | Bounded decoded pack | T3 | S | low | worker RSS bounded | — |
| D8c | mmap-shared pack | T3 | M | med | PSS ÷ workers | only if A2 shows need |
| D8d | Generational caches keyed by language | T2 | S | med | no cliffs, deterministic | A5 across workers |
| D8e | Columnar accumulation | T3 | S | low | less worker RAM | — |
| D8f | Memory-aware chunks | T3 | S | low | long docs bounded | — |
| D9a–c | Normalise once, one-pass LID, O(log n) approx positions | T2 | S | low | up to −1.2 s per 8 MB chunk | identical outputs |
| D9d–e,g | Single exact encode, NumPy attributes, hoisted closures | T3 | S | low | fewer passes | identical outputs |
| D9f | Training-split sampling | T2 | M | low | ≈ train/test× faster real cold scan | caption states n of N |
| D10 | Finer incrementality | T3 | M–L | med | append = one chunk | 0 runs re-derived |
| D11 | Hash while reading | T3 | S | low | no double read | no spurious re-derive |
| D12 | forkserver preload | T2 | S | low | lower PSS, faster start | A2 |
| D13 | Longest-first, POS overlap | T3 | S | low | shorter tail | — |
| D14 | Atomic generations | T3 | L | med | no torn reads | fault injection |
| D15 | Scan lock + cancel points | T2 | S | low | one scan at a time | double-click test |
| D16 | Compaction | T3 | S | low | file ≈ live size | — |
| D17 | Scan DuckDB budget | T2 | S | low | no OOM on laptops | A2 sweep |
| E1 | Memory budget | T1 | S | low | ≤ 400 MB steady | A3 after 50 interactions |
| E2 | Export cache in bytes, caption-free key | T2 | S | low | bounded, fewer re-renders | — |
| E3 | No large transients | T2 | S | low | flat RSS | `tracemalloc` |
| F1 | Export process | T2 | S | low | no GIL contention | — |
| F2 | Speculative pre-render | T2 | S | low | instant download | click ≤ 100 ms |
| F3 | Failure caching and demotion | T2 | S | low | no repeated TeX runs | — |
| G1 | Real-data rehearsal | T1 | M | — | known on-stage timings | re-scan ≤ 60 s subtree |
| G2 | Offline kit | T1 | S | — | exact tokens, no network | airplane-mode run |
| G3 | Scripted walkthrough | T2 | S | — | daily regression | all steps in budget |
| G4 | Warm start | T2 | S | — | instant first click | — |
| G5 | Performance strip | T2 | S | low | efficiency visible | — |
| G6 | Paper claim alignment | T2 | S | — | defensible numbers | — |
| G7 | Fallback kit | T1 | S | — | recoverable demo | restore ≤ 30 s |

---

## 6. How the baseline was produced

The Section 1 numbers came from the sequence below. `make_sample_src.py`, `make_scaled_data.py` and `bench_scores.py` are in Review A's tools zip. `scan_profile.py` and `probe.py` are the plan author's session scripts and are **not in the repository**: A2 and A3 specify their committed replacements (per-phase parent/children RSS·PSS·USS sampling, and Playwright callback, idle and RSS probing).

```bash
pip install -r requirements.txt psutil playwright pyarrow
python make_sample_src.py sample_src
python tools/make_sample_data.py --src sample_src --out sample_data
python make_scaled_data.py . sample_src ../scaled_data 20000      # 296 MB, 66 runs
python scan_profile.py . ../scaled_data ../scaled_state           # cold scan + memory per phase
python scan_profile.py . ../scaled_data ../scaled_state           # no-op scan
python app.py --data ../scaled_data --state ../scaled_state --no-browser --port 8085 &
python probe.py http://127.0.0.1:8085 <server-pid>                # UI: callbacks, idle, RSS
# stop the server first (DuckDB single-writer lock):
python bench_scores.py ../scaled_data ../scaled_state kp20k       # Python loop vs one SQL statement
```
