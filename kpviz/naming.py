"""Display names, distinguishing-hyperparameter labels and visual encoding.

The rules (see the dataviz method):
  * color follows the entity, never its rank — assignments are persisted;
  * one model on screen -> hue/shape/size carry its *differing* hyper-
    parameters (up to three dimensions, ordered by cardinality);
  * several models -> hue carries the domain/family group, shape the model,
    size the parameter count; labels show the model name plus only the
    hyperparameters that differ between its displayed runs.
"""
from __future__ import annotations

import json
import math
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


# ---- splits: one colour and one order, everywhere ------------------------
# A split means the same thing on every page, so it gets a fixed hue and a
# fixed reading order (the pipeline order: you train, you tune, you test).
SPLIT_COLORS = {
    "training": "#3AA6A0", "train": "#3AA6A0",
    "validation": "#9B7EDE", "valid": "#9B7EDE", "val": "#9B7EDE",
    "dev": "#9B7EDE", "development": "#9B7EDE",
    "testing": "#5B8DEF", "test": "#5B8DEF", "eval": "#5B8DEF",
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


def window_str(tokz: str | None, limit, is_default: bool = False) -> str:
    """"512 (bart-base)" · "128k (o200k)" · "1024 (bart-base, default)"."""
    if limit is None:
        return "—"
    inner = tokenizer_label(tokz) + (", default" if is_default else "")
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


def _size_from_params(n_params: int | None) -> float:
    """Marker size (px area-ish) from parameter count, log-scaled."""
    if not n_params:
        return 10.0
    return 8.0 + 2.6 * max(0.0, math.log10(n_params) - 7)  # 10M->8, 100B->18.4


def encode_runs(idx: CardIndex, runs: list[dict]) -> dict[str, dict]:
    """Visual encoding {group_key: {color, mpl_marker, shape, size, label,
    color_dim, shape_dim, size_dim}} following the single-/multi-model rules."""
    uniq: dict[str, dict] = {}
    for r in runs:
        uniq.setdefault(group_key(r["model"], r["arch"], r["run_id"]), r)
    labels = run_labels(idx, list(uniq.values()))
    models = sorted({r["model"] for r in uniq.values()})
    out: dict[str, dict] = {}

    if len(models) == 1 and len(uniq) > 1:
        # one model: encode its differing hyperparameters
        rs = list(uniq.values())
        diffs = distinct_params([r["resolved"] for r in rs])
        dims = diffs[:3]

        def pval(r, p):
            info = r["resolved"].get(p) or {}
            return json.dumps(info.get("value"), sort_keys=True, default=str)

        val_order = {p: sorted({pval(r, p) for r in rs}) for p in dims}
        seq = db.color_seq("run", sorted(uniq.keys()))
        for k, r in uniq.items():
            ci = (val_order[dims[0]].index(pval(r, dims[0]))
                  if dims else seq[k] % len(PALETTE))
            si = (val_order[dims[1]].index(pval(r, dims[1]))
                  if len(dims) > 1 else 0)
            zi = (val_order[dims[2]].index(pval(r, dims[2]))
                  if len(dims) > 2 else 0)
            nz = len(val_order[dims[2]]) if len(dims) > 2 else 1
            out[k] = {
                "color": slot_color(ci % len(PALETTE)),
                "shape": PLOTLY_SHAPES[si % len(PLOTLY_SHAPES)],
                "mpl_marker": MPL_SHAPES[si % len(MPL_SHAPES)],
                "size": 10.0 + (6.0 * zi / max(1, nz - 1) if nz > 1 else 0.0),
                "label": labels[k],
                "color_dim": dims[0] if dims else "run",
                "shape_dim": dims[1] if len(dims) > 1 else None,
                "size_dim": dims[2] if len(dims) > 2 else None,
            }
        return out

    # several models: hue = domain/family group, shape = model, size = #params
    def group_of(model: str) -> str:
        card = idx.model(model)
        if card.domains:
            return card.domains[0].get("domain") or "Other"
        return card.family_top or "Other"

    groups = sorted({group_of(m) for m in models})
    gseq = db.color_seq("group", groups)
    mseq = db.color_seq("model", models)
    for k, r in uniq.items():
        g = group_of(r["model"])
        card = idx.model(r["model"])
        out[k] = {
            "color": slot_color(gseq[g] % len(PALETTE)),
            "shape": PLOTLY_SHAPES[mseq[r["model"]] % len(PLOTLY_SHAPES)],
            "mpl_marker": MPL_SHAPES[mseq[r["model"]] % len(MPL_SHAPES)],
            "size": _size_from_params(card.n_parameters),
            "label": labels[k],
            "color_dim": "domain" if any(idx.model(m).domains for m in models) else "family",
            "shape_dim": "model",
            "size_dim": "#parameters",
        }
    return out
