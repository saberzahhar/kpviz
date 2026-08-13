"""Cost resolution.

An architecture card declares raw *variables* (each with a level:
document or batch) and *rates* — linear models {cost_unit: {var: coeff}}.
Runs carry observed variable values at the batch level (batch_XXXXX.json
"costs") and/or at the document level (per-line "costs" in the jsonl).

This module turns those observations into per-run cost totals for every
unit the architecture prices, without ever inventing data:

  * a variable is summed at its *declared* level when fully observed there,
    falling back to the other level (flagged) when not;
  * wall-clock time may be derived from batch timestamps as a last resort
    (flagged `derived_from_timestamps`);
  * a unit whose needed variables are simply not observed anywhere yields
    known=False — the UI then draws the run as a dashed performance-only
    line rather than breaking.
"""
from __future__ import annotations

from .cards import ArchCard

FLAG_FALLBACK_LEVEL = "level_fallback"
FLAG_PARTIAL = "partial_coverage"
FLAG_DERIVED_TIME = "derived_from_timestamps"
FLAG_UNDECLARED = "undeclared_variable"


def resolve_var_totals(arch: ArchCard,
                       doc_sums: dict[str, tuple[float, int]],
                       batch_sums: dict[str, tuple[float, int]],
                       n_docs: int, n_batches: int,
                       wall_from_timestamps: float | None) -> dict[str, dict]:
    """Combine observations into one total per raw variable.

    doc_sums / batch_sums: {var: (sum, observation_count)}.
    Returns {var: {total, level, doc_cov, batch_cov, flags[]}}.
    """
    out: dict[str, dict] = {}
    vars_seen = set(doc_sums) | set(batch_sums) | set(arch.variables)
    for var in sorted(vars_seen):
        d_sum, d_n = doc_sums.get(var, (0.0, 0))
        b_sum, b_n = batch_sums.get(var, (0.0, 0))
        d_cov = (d_n / n_docs) if n_docs else 0.0
        b_cov = (b_n / n_batches) if n_batches else 0.0
        declared = arch.var_level(var)          # 'document' | 'batch' | None
        flags: list[str] = []
        if declared is None and arch.known:
            flags.append(FLAG_UNDECLARED)

        total, level = None, None
        order = (["document", "batch"] if declared == "document"
                 else ["batch", "document"] if declared == "batch"
                 else (["document", "batch"] if d_n else ["batch", "document"]))
        for lv in order:
            cov = d_cov if lv == "document" else b_cov
            s = d_sum if lv == "document" else b_sum
            n = d_n if lv == "document" else b_n
            if n > 0:
                total, level = s, lv
                if declared and lv != declared:
                    flags.append(FLAG_FALLBACK_LEVEL)
                if cov < 0.999:
                    flags.append(FLAG_PARTIAL)
                break
        # last resort for wall time: derive it from batch timestamps
        if total is None and var == "time" and wall_from_timestamps:
            total, level = wall_from_timestamps, "batch"
            flags.append(FLAG_DERIVED_TIME)
        if total is None:
            continue
        out[var] = {"total": total, "level": level,
                    "doc_cov": round(d_cov, 4), "batch_cov": round(b_cov, 4),
                    "flags": flags}
    return out


def resolve_costs(arch: ArchCard, var_totals: dict[str, dict],
                  n_docs: int) -> dict[str, dict]:
    """Apply the linear rate models.

    Returns {unit: {total, per_doc, known, complete, flags[], terms}} for
    every unit the architecture prices. Unknown architecture -> {}.
    """
    out: dict[str, dict] = {}
    if not arch.known:
        return out
    for unit, coeffs in arch.rates.items():
        needed = [v for v, c in coeffs.items() if c]
        missing = [v for v in needed if v not in var_totals]
        flags: list[str] = []
        if missing:
            out[unit] = {"total": None, "per_doc": None, "known": False,
                         "complete": False, "flags": [f"missing:{v}" for v in missing],
                         "terms": {}}
            continue
        total = 0.0
        terms: dict[str, float] = {}
        complete = True
        for v in needed:
            info = var_totals[v]
            contrib = coeffs[v] * info["total"]
            terms[v] = contrib
            total += contrib
            if FLAG_PARTIAL in info["flags"]:
                complete = False
                flags.append(f"{FLAG_PARTIAL}:{v}")
            if FLAG_DERIVED_TIME in info["flags"]:
                flags.append(FLAG_DERIVED_TIME)
            if FLAG_FALLBACK_LEVEL in info["flags"]:
                flags.append(f"{FLAG_FALLBACK_LEVEL}:{v}")
        out[unit] = {"total": total,
                     "per_doc": (total / n_docs) if n_docs else None,
                     "known": True, "complete": complete,
                     "flags": sorted(set(flags)), "terms": terms}
    return out


def cost_formula(arch: ArchCard, unit: str) -> str:
    """Human-readable linear model, e.g. 'usd = 2.5e-06·input_tokens + 1e-05·output_tokens'."""
    coeffs = arch.rates.get(unit) or {}
    terms = [f"{c:g}·{v}" for v, c in coeffs.items() if c]
    return f"{unit} = " + (" + ".join(terms) if terms else "0")
