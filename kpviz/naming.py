"""Display names, distinguishing-hyperparameter labels and visual encoding.

One visual identity on every page (see the dataviz method):
  * hue = the model, from a slot persisted per catalog (colour follows the
    entity, never its rank or the page: a model is the same colour on the
    Models page, in every workbench and in every export);
  * shape = the architecture it ran on (persisted slot as well);
  * a model's runs are lightness steps of its hue (and dash patterns for
    lines), in the natural order of their labels;
  * labels show the model name plus only the hyperparameters that differ
    between its displayed runs; legends list models and architectures, not
    runs (a run is named on hover and, selectively, next to its mark).
"""
from __future__ import annotations

import json
import re

from . import db
from .cards import CardIndex
from .util import fmt_num

_NUM_IN_TEXT = re.compile(r"(\d+(?:\.\d+)?)")


def natural_key(s) -> tuple:
    """Sort key where embedded numbers compare as numbers.

    Plain string order puts num_beams=10 between 1 and 4; every ordering the
    user reads — run pickers, tables, bar categories — goes through this."""
    parts = _NUM_IN_TEXT.split(str(s))
    return tuple((0, float(t)) if i % 2 else (1, t.lower())
                 for i, t in enumerate(parts))


def value_key(v) -> tuple:
    """Order parameter *values* of mixed type: numbers first, then naturally."""
    if isinstance(v, bool):
        return (1, 0.0, str(v))
    if isinstance(v, (int, float)):
        return (0, float(v), "")
    return (2, 0.0, str(natural_key(param_value_str(v))))

# categorical slots (light mode) — fixed order, never cycled
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
           "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
OTHER_GRAY = "#898781"

PLOTLY_SHAPES = ["circle", "square", "diamond", "triangle-up",
                 "cross", "x", "star", "triangle-down"]
MPL_SHAPES = ["o", "s", "D", "^", "P", "X", "*", "v"]


def slot_color(i: int) -> str:
    return PALETTE[i] if 0 <= i < len(PALETTE) else OTHER_GRAY


DASHES = ["solid", "dash", "dot", "dashdot", "longdash", "longdashdot"]


def shade(hex_color: str, t: float) -> str:
    """`hex_color` mixed with white by t ∈ [0, 1) (0 = unchanged): one hue,
    several lightness steps."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    mix = lambda c: round(c + (255 - c) * max(0.0, min(0.9, t)))  # noqa: E731
    return f"#{mix(r):02x}{mix(g):02x}{mix(b):02x}"


def assign_all_slots(idx) -> None:
    """Colour slots for every model and shape slots for every architecture
    of the catalog, in sorted order, once per scan: a figure's encoding then
    never depends on which page was opened first (pages only read slots)."""
    rows = db.q("SELECT DISTINCT model, arch FROM runs ORDER BY 1, 2")
    db.color_seq("model", sorted({r[0] for r in rows}))
    db.color_seq("arch", sorted({r[1] for r in rows}))


def model_color(model: str) -> str:
    """The model's hue — the same on every page and in every export."""
    return slot_color(db.color_seq("model", [model]).get(model, 99))


def arch_shape(arch: str) -> tuple[str, str]:
    """(Plotly symbol, Matplotlib marker) of an architecture."""
    i = db.color_seq("arch", [arch]).get(arch, 0) % len(PLOTLY_SHAPES)
    return PLOTLY_SHAPES[i], MPL_SHAPES[i]


# ---- PRMU: green → yellow → orange → red, by how much of the phrase the
# document holds (P verbatim … U none of it). Validated as a set: lightness
# band, chroma floor, colour-vision separation of neighbours (worst ΔE 13.5)
# and normal-vision separation. Every chip also prints its letter, so the
# class never rests on the colour alone (assets/kpviz.css mirrors these
# values; a test keeps the two in step).
PRMU_COLORS = {"P": "#2e9158", "R": "#d7ad1d", "M": "#d6601a", "U": "#a82424"}
# ---- conditions compared inside one run (all documents vs. a subset, all
# gold vs. the gold inside the window): neutral greys, darker = the subset
# under test, so hue keeps meaning "which model" on every figure
CONDITION = {"context": "#c2c1ba", "kept": "#8a8983", "focus": "#4d4c47"}

PRMU_NAMES = {"P": "Present", "R": "Reordered", "M": "Mixed", "U": "Unseen"}


# ---- splits: one colour and one order, everywhere ------------------------
# A split means the same thing on every page, so it gets a fixed hue and a
# fixed reading order (the pipeline order: you train, you tune, you test).
# Validated as a categorical set on the light surface: lightness band,
# chroma floor, colour-vision separation (worst pair ΔE 13.8, protan) and
# 3:1 contrast all pass.
_TRAIN, _VALID, _TEST = "#1a9e8f", "#c2761c", "#5b50c8"
SPLIT_COLORS = {
    "training": _TRAIN, "train": _TRAIN,
    "validation": _VALID, "valid": _VALID, "val": _VALID,
    "dev": _VALID, "development": _VALID,
    "testing": _TEST, "test": _TEST, "eval": _TEST,
}
_SPLIT_RANK = {
    "training": 0, "train": 0,
    "validation": 1, "valid": 1, "val": 1, "dev": 1, "development": 1,
    "testing": 2, "test": 2, "eval": 2,
}


def split_color(split) -> str:
    return SPLIT_COLORS.get(str(split or "").strip().lower(), OTHER_GRAY)


def split_rank(split) -> tuple:
    s = str(split or "").strip().lower()
    return (_SPLIT_RANK.get(s, 9), natural_key(s))


def order_splits(splits) -> list:
    return sorted({s for s in splits if s is not None}, key=split_rank)


def short_run(run_id: str, n: int = 8) -> str:
    return run_id if len(run_id) <= n else run_id[:n]


def tokenizer_label(tokz: str | None) -> str:
    """"transformers[bart-base]" -> "bart-base"; "tiktoken[o200k_base]" -> "o200k".

    The library name ("transformers") is not the answer to "measured with
    what?" — the asset is."""
    if not tokz:
        return "—"
    asset = tokz.split("[", 1)[1].rstrip("]") if "[" in tokz else tokz
    asset = asset.rsplit("/", 1)[-1]                     # org/name -> name
    if tokz.startswith("tiktoken") and asset.endswith("_base"):
        asset = asset[:-5]                               # o200k_base -> o200k
    return asset


def limit_str(v: int | float | None) -> str:
    """128000 -> "128k", 1024 -> "1024" (round thousands only, never lossy)."""
    if v is None:
        return "—"
    v = int(v)
    if v >= 1000 and v % 1000 == 0:
        return f"{v // 1000}k"
    return str(v)


def window_str(tokz: str | None, limit, is_default: bool = False,
               reserved: int = 0) -> str:
    """"512 (bart-base)" · "128k (o200k)" · "1024 (bart-base, default)" ·
    "1022 (bart-base, 2 reserved)" — `limit` is what the document can use."""
    if limit is None:
        return "—"
    inner = (tokenizer_label(tokz) + (", default" if is_default else "")
             + (f", {reserved:,} reserved" if reserved else ""))
    return f"{limit_str(limit)} ({inner})"


def param_value_str(v) -> str:
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, float):
        return fmt_num(v)
    if isinstance(v, (list, tuple)):
        return "[" + ",".join(str(x) for x in v) + "]"
    if isinstance(v, str) and len(v) > 18:
        return v[:15] + "…"
    return str(v)


def distinct_params(resolved_list: list[dict]) -> list[str]:
    """Names of parameters whose values differ across the given runs
    (resolved dicts as stored: {param: {value, source}})."""
    values: dict[str, set] = {}
    for res in resolved_list:
        for p, info in (res or {}).items():
            v = info.get("value") if isinstance(info, dict) else info
            values.setdefault(p, set()).add(json.dumps(v, sort_keys=True, default=str))
    return sorted([p for p, vs in values.items() if len(vs) > 1],
                  key=lambda p: (-len(values[p]), p))


_ROWS_MEMO: dict = {}
_ROWS_VERSION: int | None = None


def run_rows(datasets: list[str] | None = None) -> list[dict]:
    """All runs with parsed JSON columns (memoised per scan version)."""
    global _ROWS_MEMO, _ROWS_VERSION
    v = db.scan_version()
    if v != _ROWS_VERSION:
        _ROWS_MEMO, _ROWS_VERSION = {}, v
    memo_key = tuple(sorted(datasets)) if datasets else None
    hit = _ROWS_MEMO.get(memo_key)
    if hit is not None:
        return hit
    sql = """SELECT dataset, model, arch, run_id, resolved, violations,
                    coverage, n_docs, expected_docs, costs, arch_known, params
             FROM runs"""
    args = []
    if datasets:
        sql += f" WHERE dataset IN ({','.join('?' * len(datasets))})"
        args = list(datasets)
    rows = []
    for r in db.q(sql + " ORDER BY model, run_id, dataset", *args):
        rows.append({
            "dataset": r[0], "model": r[1], "arch": r[2], "run_id": r[3],
            "resolved": json.loads(r[4] or "{}"),
            "violations": json.loads(r[5] or "[]"),
            "coverage": r[6], "n_docs": r[7], "expected_docs": r[8],
            "costs": json.loads(r[9] or "{}"), "arch_known": r[10],
            "params": json.loads(r[11] or "{}"),
        })
    _ROWS_MEMO[memo_key] = rows
    return rows


def group_key(model: str, arch: str, run_id: str) -> str:
    return f"{model}||{arch}||{run_id}"


def parse_group_key(k: str) -> tuple[str, str, str]:
    a = k.split("||")
    return (a[0], a[1], a[2]) if len(a) == 3 else (k, "", "")


def run_labels(idx: CardIndex, runs: list[dict]) -> dict[str, str]:
    """{group_key: 'model-name (only, differing, params)'} — unique labels.

    Only parameters that *differ between the displayed runs of the same
    model* are written; the short run id is appended only when still
    ambiguous."""
    by_model: dict[str, list[dict]] = {}
    seen: dict[str, dict] = {}
    for r in runs:
        k = group_key(r["model"], r["arch"], r["run_id"])
        if k not in seen:
            seen[k] = r
            by_model.setdefault(r["model"], []).append(r)
    labels: dict[str, str] = {}
    for model, rs in by_model.items():
        name = idx.model(model).name
        diffs = distinct_params([r["resolved"] for r in rs]) if len(rs) > 1 else []
        used: dict[str, int] = {}
        for r in rs:
            parts = []
            for p in diffs:
                info = r["resolved"].get(p) or {}
                v = info.get("value")
                if v is not None:
                    parts.append(f"{p}={param_value_str(v)}")
            lab = name + (f" ({', '.join(parts)})" if parts else "")
            if lab in used:  # still ambiguous -> disambiguate with run id
                used[lab] += 1
                lab = f"{lab} · {short_run(r['run_id'])}"
            else:
                used[lab] = 1
            labels[group_key(r["model"], r["arch"], r["run_id"])] = lab
    return labels


def encode_runs(idx: CardIndex, runs: list[dict]) -> dict[str, dict]:
    """{group_key: {color, dash, shape, mpl_marker, size, label, model,
    model_name, arch}} — hue = model, shape = architecture, runs of one model
    = lightness steps of its hue (and dash patterns for lines)."""
    uniq: dict[str, dict] = {}
    for r in runs:
        uniq.setdefault(group_key(r["model"], r["arch"], r["run_id"]), r)
    labels = run_labels(idx, list(uniq.values()))
    mslot = db.color_seq("model", sorted({r["model"] for r in uniq.values()}))
    aslot = db.color_seq("arch", sorted({r["arch"] for r in uniq.values()}))
    by_model: dict[str, list[str]] = {}
    for k, r in uniq.items():
        by_model.setdefault(r["model"], []).append(k)
    rank: dict[str, tuple[int, int]] = {}
    for ks in by_model.values():
        ks.sort(key=lambda k: natural_key(labels[k]))
        for i, k in enumerate(ks):
            rank[k] = (i, len(ks))
    out: dict[str, dict] = {}
    for k, r in uniq.items():
        i, n = rank[k]
        base = slot_color(mslot.get(r["model"], 99))
        si = aslot.get(r["arch"], 0) % len(PLOTLY_SHAPES)
        out[k] = {
            # darkest first; the lightest step stays readable on white
            "color": shade(base, 0.55 * i / max(1, n - 1)) if n > 1 else base,
            "base": base,
            "dash": DASHES[i % len(DASHES)] if n > 1 else "solid",
            "shape": PLOTLY_SHAPES[si], "mpl_marker": MPL_SHAPES[si],
            "size": 10.0, "label": labels[k], "model": r["model"],
            "model_name": idx.model(r["model"]).name, "arch": r["arch"],
        }
    return out


def arch_label(idx, token: str) -> str:
    """How an architecture is named to a reader: its card's name ("OpenAI
    API"), never the folder token; a run declared without one ("n.a") is
    "no architecture"."""
    from .cards import is_unknown_token
    if is_unknown_token(token):
        return "no architecture"
    card = idx.arch(token) if idx is not None else None
    return card.name if card is not None and card.known else token


def legend_items(enc: dict[str, dict], keys=None, lines: bool = False,
                 arch_names=None) -> list[dict]:
    """Legend entries for an encoding: one per model (its hue) and, when
    the marks come from several architectures, one per architecture (its
    shape, in grey), each block under its own title so the legend teaches
    the encoding. `keys` restricts to the runs actually drawn; `arch_names`
    (a card index) gives architectures their card names."""
    vals = [enc[k] for k in (keys if keys is not None else enc) if k in enc]
    models: dict[str, dict] = {}
    for e in sorted(vals, key=lambda e: natural_key(e["model_name"])):
        models.setdefault(e["model"], {
            "name": e["model_name"], "color": e["base"], "group": e["model"],
            "shape": "circle", "mpl_marker": "o", "line": lines,
            "block": "model (colour)"})
    items = list(models.values())
    archs = {e["arch"]: (e["shape"], e["mpl_marker"]) for e in vals}
    if len(archs) > 1 and not lines:
        for a in sorted(archs, key=natural_key):
            items.append({"name": arch_label(arch_names, a), "color": OTHER_GRAY,
                          "group": f"arch:{a}", "shape": archs[a][0],
                          "mpl_marker": archs[a][1], "line": False,
                          "block": "architecture (shape)"})
    return items
