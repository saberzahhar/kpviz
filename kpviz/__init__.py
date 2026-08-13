"""KPViz — a keyphrase evaluation cockpit.

Scans a KPViz data tree (dataset / model / architecture cards, document
collections, inference runs), derives an analytical DuckDB store (never
copying the raw corpus — byte offsets + derived indices only) and serves a
Dash app with entity explorers and five research-question workbenches with
publication-grade LaTeX / PGF / PDF / PNG export.
"""

__version__ = "1.5.0"
