## What and why

<!-- One or two sentences: the problem, and how this change solves it. -->

## How it was checked

- [ ] `python -m pyflakes kpviz tests tools app.py`
- [ ] `python -m pytest tests -q -rs`
- [ ] For a UI change: a screenshot on the sample data
- [ ] For a change that can move a score: a test against an independent computation
- [ ] For a hot path: before/after numbers from `tools/bench/`
