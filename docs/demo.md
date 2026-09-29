# Demo guide

A five-minute walk through KPViz, and the checks to run before showing it
to a room. The script uses the demo tree from
[Zenodo](https://doi.org/10.5281/zenodo.22789345); the generated sample data
(`python tools/make_sample_data.py`) works too, with smaller numbers.

## Before the session

- [ ] Scan the tree in advance: `python tools/scan_once.py --data data`.
      The Overview then shows "up to date" instead of a first scan.
- [ ] Warm the tokenizer cache on the demo machine (one online scan), then
      check *Overview → Engine*: every tokenizer **exact**. Offline without a
      cache, windows become flagged approximations.
- [ ] Start with `./run.sh --no-browser` and open
      `http://127.0.0.1:8050/insights?present#rq4`. Presentation mode (the
      **P** key) enlarges the page and the figures; it is remembered.
- [ ] Pre-open one browser tab per workbench as a safety net:
      `/insights#rq1` … `/insights#rq5`.
- [ ] Keep a run folder aside (for example one extra `num_beams` value) to
      drop into `inferences/` during step 6.
- [ ] Check the export once: *View → Paper* shows the figure as printed
      (typeset by TeX when it is installed). Without TeX, PDF and PNG still
      work; PGF needs `pdflatex`.
- [ ] Rehearse at the projector's resolution, in presentation mode.

## The five minutes

| Time | Screen | Say | Do |
|---|---|---|---|
| 0:00 | Insights › RQ4 Quality vs. cost | "Each mark is a run; the grey staircase is what no other run beats on both quality and cost." | Switch *Cost* to wall-clock time; the frontier changes. |
| 0:50 | RQ3 Context windows | "A bounded input window puts gold out of reach: the gap is what truncation costs, before any model runs." | Point at the † in the table: the difference is significant after correction. |
| 1:40 | RQ2 Data quality | "Documents whose language contradicts their card change the score." | Toggle one flag criterion. |
| 2:20 | Datasets › a flagged document | "The marks in the text are the present gold, found exactly as the scorer finds it." | Open a document, pick a run under *Why this run scored…*: ✓/✗ per prediction, four cut-offs. |
| 3:20 | Back to RQ4 › export | "Every figure is ready for the paper." | *View → Both*: the interactive figure beside the printed one; *Export → Paper: ACL*, then *PDF*. |
| 4:10 | Overview | "And it keeps itself honest." | Drop the spare run folder in, *Scan for changes*: one new run, everything else unchanged; *Needs attention* lists what is off. |

## If something goes wrong

| Symptom | Cause | Fix |
|---|---|---|
| "The catalog is being updated" banner | a scan is running | wait: figures refresh once, when it finishes |
| "The last scan did not complete" | a scan was interrupted | *Overview → Scan for changes* |
| Windows marked approximate | tokenizer assets not cached | scan once online, or accept the flagged approximation |
| An empty workbench | too few datasets or runs for its question | the message in place of the figure says what to change |
| A figure looks stale after a scan in another tab | the page predates the scan | reload: routing lives in the URL, nothing is lost |
