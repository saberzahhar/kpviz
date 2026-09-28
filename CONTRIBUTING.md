# Contributing to KPViz

Thank you for helping. Bug reports, data that breaks the scanner, and ideas
for the research-question workbenches are all welcome.

## Report a problem

Open an issue with the *Bug report* form. The most useful reports include:

- the command you ran and the full console output;
- the Overview page's *Needs attention* table and *Engine* line;
- a minimal data tree that reproduces it. `tools/synth_tree.py` builds one
  around your own cards, without your documents.

## Set up

```bash
git clone https://github.com/saberzahhar/kpviz && cd kpviz
python -m venv .venv && source .venv/bin/activate
pip install -e . -r requirements-dev.txt -c constraints.txt
```

A TeX distribution (TeX Live or MiKTeX, with `pdflatex`) is needed for the
export tests; CI installs it and allows no skips.

## Before you open a pull request

```bash
python -m pyflakes kpviz tests tools app.py
python -m pytest tests -q -rs
```

Both must pass, as they do in CI. Then:

- **Keep numbers honest.** A change that can move a score needs a test
  against an independent computation (see `tests/test_evaluation.py`). Worker
  count and incremental scans must never change a number
  (`tests/test_scan.py`).
- **Keep the interface contract.** Read `docs/design.md`. New colours go
  through the tokens in `kpviz/assets/kpviz.css`, and a new categorical palette
  is validated before it ships.
- **Look at it.** For a UI change, run the app on the sample data
  (`python tools/make_sample_data.py && python app.py`) and attach a
  screenshot. `python tools/screenshot.py` captures every page.
- **Measure speed.** For a change on a hot path, include before/after
  numbers from `tools/bench/` (see `docs/PERFORMANCE_PLAN.md`).

## Style

Match the surrounding code: small functions, comments that say *why*, and
names a reader of the paper would recognise. Docstrings and UI text use plain
words and sentence case.

## Code of conduct

This project follows the [Code of Conduct](CODE_OF_CONDUCT.md).
