"""Dependency-free statistical tests for the insight workbenches.

Both tests use the normal approximation with tie correction (and, for
Wilcoxon, zero-difference exclusion). They are two-sided. Below the
minimum sample size they return None rather than a misleading p-value.

Marking convention: a single dagger † at the significance level the reader
picks (Insights → significance level), written into every caption.
"""
from __future__ import annotations

import math

MIN_N = 6


def _rankdata(values: list[float]) -> tuple[list[float], float]:
    """Average ranks + tie-correction term Σ(t³−t)."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    tie_term = 0.0
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        r = (i + j) / 2 + 1
        t = j - i + 1
        if t > 1:
            tie_term += t ** 3 - t
        for k in range(i, j + 1):
            ranks[order[k]] = r
        i = j + 1
    return ranks, tie_term


def _norm_sf(z: float) -> float:
    """P(Z > z) for the standard normal."""
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def mann_whitney_u(x: list[float], y: list[float]) -> tuple[float | None, float | None]:
    """Two-sided Mann–Whitney U (independent samples). Returns (U, p)."""
    n1, n2 = len(x), len(y)
    if n1 < MIN_N or n2 < MIN_N:
        return None, None
    ranks, tie_term = _rankdata(list(x) + list(y))
    r1 = sum(ranks[:n1])
    u1 = r1 - n1 * (n1 + 1) / 2
    u = min(u1, n1 * n2 - u1)
    mu = n1 * n2 / 2
    n = n1 + n2
    sigma2 = n1 * n2 / 12 * ((n + 1) - tie_term / (n * (n - 1)))
    if sigma2 <= 0:
        return u, 1.0
    z = (u - mu + 0.5) / math.sqrt(sigma2)      # continuity correction
    p = 2 * _norm_sf(abs(z))
    return u, min(1.0, p)


def wilcoxon_signed_rank(x: list[float], y: list[float]) -> tuple[float | None, float | None]:
    """Two-sided paired Wilcoxon signed-rank on aligned samples. (W, p)."""
    diffs = [a - b for a, b in zip(x, y) if a != b]
    n = len(diffs)
    if n < MIN_N:
        return None, None
    ranks, tie_term = _rankdata([abs(d) for d in diffs])
    w_pos = sum(r for r, d in zip(ranks, diffs) if d > 0)
    w_neg = sum(r for r, d in zip(ranks, diffs) if d < 0)
    w = min(w_pos, w_neg)
    mu = n * (n + 1) / 4
    sigma2 = n * (n + 1) * (2 * n + 1) / 24 - tie_term / 48
    if sigma2 <= 0:
        return w, 1.0
    z = (w - mu + 0.5) / math.sqrt(sigma2)
    p = 2 * _norm_sf(abs(z))
    return w, min(1.0, p)


def _gammap_q(a: float, x: float) -> float:
    """Regularised upper incomplete gamma Q(a, x) — series + continued
    fraction (Numerical Recipes 6.2); enough for a chi-square tail."""
    if a <= 0 or x <= 0:
        return 1.0
    lg = math.lgamma(a)
    if x < a + 1.0:                       # series for P(a, x)
        ap, ssum, d = a, 1.0 / a, 1.0 / a
        for _ in range(500):
            ap += 1.0
            d *= x / ap
            ssum += d
            if abs(d) < abs(ssum) * 1e-14:
                break
        return 1.0 - ssum * math.exp(-x + a * math.log(x) - lg)
    tiny = 1e-300                          # continued fraction for Q(a, x)
    b, c, d = x + 1.0 - a, 1.0 / tiny, 1.0 / (x + 1.0 - a)
    h = d
    for i in range(1, 500):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-14:
            break
    return math.exp(-x + a * math.log(x) - lg) * h


def chi2_sf(x: float, df: int) -> float:
    """P(X > x) for a chi-square with df degrees of freedom."""
    if df <= 0 or x <= 0:
        return 1.0
    return min(1.0, max(0.0, _gammap_q(df / 2.0, x / 2.0)))


def friedman(groups: list[list[float]]) -> tuple[float | None, float | None]:
    """Friedman test over k >= 3 *paired* groups (same subjects, k conditions).

    The right question for a hyperparameter is "does the value matter at all?",
    across all of its values at once — not a pile of pairwise tests. Every value
    was run on the same documents, so the blocks are the documents: rank the k
    values within each document and ask whether the rank sums differ.
    Returns (chi2, p), or (None, None) below MIN_N complete blocks."""
    k = len(groups)
    if k < 3 or any(len(g) != len(groups[0]) for g in groups):
        return None, None
    n = len(groups[0])
    if n < MIN_N:
        return None, None
    rank_sums = [0.0] * k
    ties = 0.0
    for i in range(n):
        ranks, tie_term = _rankdata([g[i] for g in groups])
        ties += tie_term
        for j in range(k):
            rank_sums[j] += ranks[j]
    stat = (12.0 * sum((r - n * (k + 1) / 2.0) ** 2 for r in rank_sums)
            / (n * k * (k + 1)))
    denom = 1.0 - ties / (n * k * (k * k - 1)) if ties else 1.0
    if denom <= 0:
        return None, None
    stat /= denom
    return stat, chi2_sf(stat, k - 1)


ALPHAS = (0.001, 0.01, 0.05, 0.10)   # the significance levels offered in the UI
ALPHA_DEFAULT = 0.05
DAGGER = "†"


def sig_mark(p: float | None, alpha: float = ALPHA_DEFAULT) -> str:
    """'†' when the result is significant at `alpha`, '' otherwise.

    One mark, one threshold: the level is chosen by the reader (Insights →
    significance level) and written into every caption, so a dagger never
    means two different things in the same paper."""
    if p is None:
        return ""
    return DAGGER if p < alpha else ""


def alpha_str(alpha: float) -> str:
    return f"{alpha:g}"


def sig_caption(alpha: float = ALPHA_DEFAULT) -> str:
    return f"{DAGGER} p<{alpha_str(alpha)} (two-sided)"


def p_str(p: float | None, n_note: str = "n<6") -> str:
    if p is None:
        return n_note
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def paired_common(a: dict[str, float], b: dict[str, float]) -> tuple[list, list]:
    """Align two per-document score maps on their common documents."""
    common = sorted(set(a) & set(b))
    return [a[d] for d in common], [b[d] for d in common]
