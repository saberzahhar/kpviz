"""Dependency-free statistical tests for the insight workbenches.

Rank tests are exact for small samples without ties (Wilcoxon signed-rank
n ≤ 50, Mann–Whitney n1 + n2 ≤ 50, Spearman and Kendall n ≤ 9 systems) and
use the normal approximation with tie correction and a continuity
correction that never moves the statistic past the null centre otherwise.
All are two-sided. Below the minimum sample size, or on unaligned or
non-finite input, they return None rather than a misleading p-value.

Marking convention: a single dagger † at the significance level the reader
picks (Insights → significance level), written into every caption.
"""
from __future__ import annotations

import functools
import math

try:
    import numpy as _np
except Exception:  # pragma: no cover - numpy ships with the stack
    _np = None

MIN_N = 6


def _rankdata(values) -> tuple[list[float], float]:
    """Average ranks + tie-correction term Σ(t³−t).

    Vectorised for large samples (20 k paired documents × 22 runs is the
    common case): the same average ranks — exact halves, so identical — as
    the pure-Python loop kept below for small inputs."""
    n = len(values)
    if _np is not None and n > 256:
        a = _np.asarray(values, dtype=float)
        sorter = _np.argsort(a, kind="mergesort")
        inv = _np.empty(n, dtype=_np.intp)
        inv[sorter] = _np.arange(n)
        sa = a[sorter]
        obs = _np.r_[True, sa[1:] != sa[:-1]]
        dense = obs.cumsum()[inv]
        count = _np.r_[_np.nonzero(obs)[0], n]
        ranks = 0.5 * (count[dense] + count[dense - 1] + 1)
        t = _np.diff(count).astype(float)
        return ranks, float(((t ** 3) - t).sum())
    values = list(values)
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


EXACT_MAX_N = 50        # exact rank distributions up to this many items


class UnalignedError(ValueError):
    """Paired samples of different lengths: pairing must be explicit."""


def _pair(x, y):
    """Aligned paired arrays; unequal lengths are a caller error, never a
    silent truncation."""
    a, b = _arr(x), _arr(y)
    if len(a) != len(b):
        raise UnalignedError(f"paired samples differ in length ({len(a)} vs {len(b)})")
    return a, b


def _finite(*arrays) -> bool:
    return all(bool(_np.isfinite(a).all()) for a in arrays)


@functools.lru_cache(maxsize=256)
def _signed_rank_cdf(n: int) -> tuple:
    """Exact null distribution of the Wilcoxon W+ for n untied, non-zero
    differences, as cumulative probabilities (counts of subsets of 1..n by
    sum, a knapsack over the ranks)."""
    top = n * (n + 1) // 2
    c = [0] * (top + 1)
    c[0] = 1
    for r in range(1, n + 1):
        for s_ in range(top, r - 1, -1):
            c[s_] += c[s_ - r]
    tot = float(2 ** n)
    out, acc = [], 0
    for v in c:
        acc += v
        out.append(acc / tot)
    return tuple(out)


@functools.lru_cache(maxsize=512)
def _mw_cdf(n1: int, n2: int) -> tuple:
    """Exact null distribution of U for untied samples (Gaussian binomial
    coefficients by the recurrence c(m, n, u) = c(m-1, n, u-n) + c(m, n-1, u))."""
    prev = [[1]] + [[1] for _ in range(n2)]          # m = 0: U = 0 only
    for m in range(1, n1 + 1):
        cur = [[1]]                                   # n = 0: U = 0 only
        for n in range(1, n2 + 1):
            a, b = prev[n], cur[n - 1]
            size = m * n + 1
            row = [0] * size
            for u, v in enumerate(b):
                row[u] += v
            for u, v in enumerate(a):
                if u + n < size:
                    row[u + n] += v
            cur.append(row)
        prev = cur
    counts = prev[n2]
    tot = float(sum(counts))
    out, acc = [], 0
    for v in counts:
        acc += v
        out.append(acc / tot)
    return tuple(out)


def mann_whitney_u(x: list[float], y: list[float]) -> tuple[float | None, float | None]:
    """Two-sided Mann–Whitney U (independent samples). Returns (U, p);
    exact without ties up to n1 + n2 = EXACT_MAX_N."""
    n1, n2 = len(x), len(y)
    if n1 < MIN_N or n2 < MIN_N:
        return None, None
    if _np is not None and not _finite(_arr(x), _arr(y)):
        return None, None
    if _np is not None:
        both = _np.concatenate((_np.asarray(x, dtype=float),
                                _np.asarray(y, dtype=float)))
    else:
        both = list(x) + list(y)
    ranks, tie_term = _rankdata(both)
    r1 = float(sum(ranks[:n1]))
    u1 = r1 - n1 * (n1 + 1) / 2
    u = min(u1, n1 * n2 - u1)
    mu = n1 * n2 / 2
    n = n1 + n2
    if tie_term == 0 and n <= EXACT_MAX_N:
        cdf = _mw_cdf(min(n1, n2), max(n1, n2))
        return u, min(1.0, 2 * cdf[int(round(u))])
    sigma2 = n1 * n2 / 12 * ((n + 1) - tie_term / (n * (n - 1)))
    if sigma2 <= 0:
        return u, 1.0
    # continuity correction towards the centre, never past it
    z = max(0.0, abs(u - mu) - 0.5) / math.sqrt(sigma2)
    p = 2 * _norm_sf(z)
    return u, min(1.0, p)


def wilcoxon_signed_rank(x: list[float], y: list[float]) -> tuple[float | None, float | None]:
    """Two-sided paired Wilcoxon signed-rank on aligned samples. (W, p).
    Zero differences are dropped (Wilcoxon's convention); exact without
    ties up to EXACT_MAX_N differences."""
    if len(x) != len(y):
        raise UnalignedError(f"paired samples differ in length ({len(x)} vs {len(y)})")
    if _np is not None and not _finite(_arr(x), _arr(y)):
        return None, None
    if _np is not None and len(x) > 256:
        d = _np.asarray(x, dtype=float) - _np.asarray(y, dtype=float)
        d = d[d != 0]
        n = len(d)
        if n < MIN_N:
            return None, None
        ranks, tie_term = _rankdata(_np.abs(d))
        ranks = _np.asarray(ranks, dtype=float)
        w_pos = float(ranks[d > 0].sum())
        w_neg = float(ranks[d < 0].sum())
    else:
        diffs = [a - b for a, b in zip(x, y) if a != b]
        n = len(diffs)
        if n < MIN_N:
            return None, None
        ranks, tie_term = _rankdata([abs(d) for d in diffs])
        w_pos = sum(r for r, d in zip(ranks, diffs) if d > 0)
        w_neg = sum(r for r, d in zip(ranks, diffs) if d < 0)
    w = min(w_pos, w_neg)
    mu = n * (n + 1) / 4
    if tie_term == 0 and n <= EXACT_MAX_N:
        return w, min(1.0, 2 * _signed_rank_cdf(n)[int(round(w))])
    sigma2 = n * (n + 1) * (2 * n + 1) / 24 - tie_term / 48
    if sigma2 <= 0:
        return w, 1.0
    # continuity correction towards the centre, never past it: balanced
    # differences give z = 0 and p = 1 (as SciPy's corrected statistic)
    z = max(0.0, abs(w - mu) - 0.5) / math.sqrt(sigma2)
    p = 2 * _norm_sf(z)
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
    if _np is not None and n > 64 and k <= 16:
        # every block at once: within-block average ranks from an n×k×k
        # comparison, and Σ(t³−t) per block as Σ_j (c_j² − 1), c_j being
        # how many values in the block equal the j-th (same numbers as the
        # per-block loop below, 100× faster at 20 k documents)
        X = _np.column_stack([_np.asarray(g, dtype=float) for g in groups])
        less = (X[:, None, :] < X[:, :, None]).sum(axis=2)
        eq = (X[:, None, :] == X[:, :, None]).sum(axis=2)
        R = less + (eq + 1) / 2.0
        rank_sums = R.sum(axis=0).tolist()
        ties = float((eq.astype(float) ** 2 - 1).sum())
    else:
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


# ===========================================================================
# Inference toolkit: test families, effect sizes, confidence intervals,
# multiple-comparison control. Dependency-free (NumPy only); every
# resampling procedure is seeded from its own data, so a p-value or an
# interval is reproducible across runs and machines.
# ===========================================================================

FAMILIES = {
    # paired / independent / k >= 3 paired
    "rank": ("Wilcoxon signed-rank", "Mann–Whitney U", "Friedman"),
    "mean": ("paired t-test", "Welch t-test", "repeated-measures ANOVA"),
    "resample": ("paired permutation (sign-flip)", "bootstrap (two-sample)",
                 "Friedman"),
}
EFFECTS = {
    "rank": ("rank-biserial r", "rank-biserial r", "Kendall's W"),
    "mean": ("Cohen's d_z", "Hedges' g", "partial η²"),
    "resample": ("rank-biserial r", "rank-biserial r", "Kendall's W"),
}
ADJUST = {"none": "no correction", "holm": "Holm–Bonferroni",
          "bonferroni": "Bonferroni", "bh": "Benjamini–Hochberg (FDR)"}
RESAMPLES = 1000
CI_LEVEL = 0.95


def _arr(v):
    return _np.asarray(v, dtype=float)


def data_seed(*parts) -> int:
    """A stable 63-bit seed from the data itself (arrays, strings, numbers).
    Same inputs → same resamples → same p-value, on every machine."""
    import hashlib
    h = hashlib.blake2b(digest_size=8)
    for p in parts:
        if _np is not None and isinstance(p, _np.ndarray):
            h.update(_np.ascontiguousarray(p, dtype=float).tobytes())
        elif isinstance(p, (list, tuple)):
            h.update(_arr(p).tobytes() if p and isinstance(p[0], (int, float))
                     else repr(p).encode())
        else:
            h.update(repr(p).encode())
        h.update(b"|")
    return int.from_bytes(h.digest(), "little") >> 1


# ---- distributions ---------------------------------------------------------

def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta (Numerical Recipes 6.4)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 400):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-14:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta I_x(a, b)."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbt = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
           + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(lbt) * _betacf(a, b, x) / a
    return 1.0 - math.exp(lbt) * _betacf(b, a, 1.0 - x) / b


def t_sf(t: float, df: float) -> float:
    """P(T > t) for Student's t with df degrees of freedom."""
    if df <= 0 or t != t:
        return float("nan")
    if math.isinf(t):
        return 0.0 if t > 0 else 1.0
    tail = 0.5 * betainc(df / 2.0, 0.5, df / (df + t * t))
    return tail if t > 0 else 1.0 - tail


def _t_pdf(t: float, df: float) -> float:
    return math.exp(math.lgamma((df + 1) / 2) - math.lgamma(df / 2)
                    - 0.5 * math.log(df * math.pi)
                    - (df + 1) / 2 * math.log1p(t * t / df))


@functools.lru_cache(maxsize=4096)
def _t_ppf(q: float, df: float) -> float:
    z = norm_ppf(q)
    # Cornish–Fisher start (Abramowitz & Stegun 26.7.5), then Newton on the cdf
    g1 = (z ** 3 + z) / 4
    g2 = (5 * z ** 5 + 16 * z ** 3 + 3 * z) / 96
    g3 = (3 * z ** 7 + 19 * z ** 5 + 17 * z ** 3 - 15 * z) / 384
    x = z + g1 / df + g2 / df ** 2 + g3 / df ** 3
    for _ in range(50):
        f = (1.0 - t_sf(x, df)) - q
        step = f / max(_t_pdf(x, df), 1e-300)
        x -= step
        if abs(step) < 1e-12 * max(1.0, abs(x)):
            break
    return x


def t_ppf(q: float, df: float) -> float:
    """Quantile of Student's t (q in (0, 1)); cached, a few Newton steps."""
    if q == 0.5:
        return 0.0
    if q < 0.5:
        return -_t_ppf(1 - q, round(float(df), 9))
    return _t_ppf(q, round(float(df), 9))


def f_sf(f: float, d1: float, d2: float) -> float:
    """P(F > f) for the F distribution."""
    if f <= 0:
        return 1.0
    return betainc(d2 / 2.0, d1 / 2.0, d2 / (d2 + d1 * f))


def norm_ppf(q: float) -> float:
    """Standard normal quantile (Acklam's rational approximation, |ε|<1e-9
    after one Newton step)."""
    if not 0.0 < q < 1.0:
        return float("nan")
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)
    pl = 0.02425
    if q < pl:
        r = math.sqrt(-2 * math.log(q))
        x = (((((c[0]*r+c[1])*r+c[2])*r+c[3])*r+c[4])*r+c[5]) / \
            ((((d[0]*r+d[1])*r+d[2])*r+d[3])*r+1)
    elif q > 1 - pl:
        r = math.sqrt(-2 * math.log(1 - q))
        x = -(((((c[0]*r+c[1])*r+c[2])*r+c[3])*r+c[4])*r+c[5]) / \
            ((((d[0]*r+d[1])*r+d[2])*r+d[3])*r+1)
    else:
        r = q - 0.5
        s = r * r
        x = (((((a[0]*s+a[1])*s+a[2])*s+a[3])*s+a[4])*s+a[5])*r / \
            (((((b[0]*s+b[1])*s+b[2])*s+b[3])*s+b[4])*s+1)
    e = 0.5 * math.erfc(-x / math.sqrt(2)) - q
    u = e * math.sqrt(2 * math.pi) * math.exp(x * x / 2)
    return x - u / (1 + x * u / 2)


# ---- parametric tests ------------------------------------------------------

def paired_t(x, y):
    """Two-sided paired t-test on aligned samples. (t, p, Cohen's d_z)."""
    a, b = _pair(x, y)
    d = a - b
    n = len(d)
    if n < MIN_N:
        return None, None, None
    sd = float(d.std(ddof=1))
    m = float(d.mean())
    if sd == 0:
        return (0.0, 1.0, 0.0) if m == 0 else (math.inf, 0.0, math.copysign(math.inf, m))
    t = m / (sd / math.sqrt(n))
    return t, min(1.0, 2 * t_sf(abs(t), n - 1)), m / sd


def welch_t(x, y):
    """Two-sided Welch t-test. (t, p, Hedges' g)."""
    a, b = _arr(x), _arr(y)
    n1, n2 = len(a), len(b)
    if n1 < MIN_N or n2 < MIN_N:
        return None, None, None
    v1, v2 = float(a.var(ddof=1)), float(b.var(ddof=1))
    se2 = v1 / n1 + v2 / n2
    diff = float(a.mean() - b.mean())
    if se2 == 0:
        return (0.0, 1.0, 0.0) if diff == 0 else (math.inf, 0.0, None)
    t = diff / math.sqrt(se2)
    df = se2 ** 2 / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1))
    sp = math.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    g = (diff / sp) * (1 - 3 / (4 * (n1 + n2) - 9)) if sp > 0 else 0.0
    return t, min(1.0, 2 * t_sf(abs(t), df)), g


def rm_anova(groups):
    """One-way repeated-measures ANOVA over k paired groups (rows = blocks).
    (F, p, partial η²). The p-value is Greenhouse–Geisser corrected: the
    degrees of freedom are scaled by ε̂, estimated from the double-centred
    covariance of the conditions, so a violation of sphericity (unequal
    variances of the pairwise differences) does not inflate significance.
    ε̂ = 1 under sphericity, and for k = 2 always."""
    k = len(groups)
    if k < 2 or any(len(g) != len(groups[0]) for g in groups):
        return None, None, None
    X = _np.column_stack([_arr(g) for g in groups])
    n = X.shape[0]
    if n < MIN_N:
        return None, None, None
    gm = X.mean()
    ss_cond = n * float(((X.mean(axis=0) - gm) ** 2).sum())
    ss_subj = k * float(((X.mean(axis=1) - gm) ** 2).sum())
    ss_tot = float(((X - gm) ** 2).sum())
    ss_err = ss_tot - ss_cond - ss_subj
    df1, df2 = k - 1, (k - 1) * (n - 1)
    if ss_err <= 1e-15:
        return (math.inf if ss_cond > 0 else 0.0,
                0.0 if ss_cond > 0 else 1.0, 1.0 if ss_cond > 0 else 0.0)
    F = (ss_cond / df1) / (ss_err / df2)
    eps = 1.0
    if k > 2:
        S = _np.cov(X, rowvar=False)
        S = S - S.mean(axis=0) - S.mean(axis=1)[:, None] + S.mean()
        tr2 = float(_np.trace(S @ S))
        if tr2 > 0:
            eps = min(1.0, max(1.0 / (k - 1), float(_np.trace(S)) ** 2 / ((k - 1) * tr2)))
    return F, f_sf(F, df1 * eps, df2 * eps), ss_cond / (ss_cond + ss_err)


# ---- resampling ------------------------------------------------------------

ASYMPTOTIC_N = 200     # resampling tests switch to their normal limit here


def paired_permutation(x, y, resamples: int = RESAMPLES, seed: int | None = None):
    """Two-sided paired permutation test (random sign flips of the per-item
    differences) on the mean difference — the approximate randomisation test
    of the NLP literature, at the document level. (mean diff, p).

    Below ASYMPTOTIC_N documents: Monte Carlo, p = (1 + #{|T*| ≥ |T|}) /
    (R + 1), so it is never 0 and its resolution is 1/(R+1). From
    ASYMPTOTIC_N on: the sign-flip statistic Σ±dᵢ is a sum of independent
    terms with variance Σdᵢ², so p = 2Φ(−|Σdᵢ|/√Σdᵢ²) — what the Monte Carlo
    estimate converges to, without its 1/(R+1) floor (which made Holm-
    adjusted p unreachable below α = 0.01 with a dozen tests)."""
    a, b = _pair(x, y)
    d = a - b
    n = len(d)
    if n < MIN_N or not _finite(d):
        return None, None
    obs = float(d.sum())
    if not d.any():
        return 0.0, 1.0
    if n >= ASYMPTOTIC_N:
        ss = float((d * d).sum())
        return obs / n, min(1.0, 2 * _norm_sf(abs(obs) / math.sqrt(ss)))
    rng = _np.random.default_rng(data_seed(d) if seed is None else seed)
    d32 = d.astype(_np.float32)
    tot = float(d32.sum())
    tol = 1e-6 * float(_np.abs(d).sum())
    chunk = max(1, min(resamples, 4_000_000 // max(n, 1)))
    nbytes = (n + 7) // 8
    hits, done = 0, 0
    while done < resamples:
        c = min(chunk, resamples - done)
        bits = _np.unpackbits(rng.integers(0, 256, size=(c, nbytes),
                                           dtype=_np.uint8), axis=1)[:, :n]
        t = 2.0 * (bits.astype(_np.float32) @ d32) - tot
        hits += int((_np.abs(t) >= abs(obs) - tol).sum())
        done += c
    return obs / n, (hits + 1) / (resamples + 1)


def _boot_means(v, resamples: int, rng):
    """Means of `resamples` bootstrap resamples of v.

    A mean depends only on how many times each *value* is drawn, and per-
    document P/R/F1 take few distinct values (0, 1/3, 1/2, …): drawing the
    value counts from a multinomial is then the same distribution as drawing
    items, at a cost independent of n (0.6 ms instead of 160 ms at 20 k
    documents). Many distinct values (differences, lengths) fall back to
    drawing item indices in chunks."""
    n = len(v)
    u, c = _np.unique(v, return_counts=True)
    if len(u) <= 512:
        cnt = rng.multinomial(n, c / n, size=resamples)
        return (cnt @ u) / n
    out = _np.empty(resamples)
    chunk = max(1, min(resamples, 4_000_000 // max(n, 1)))
    done = 0
    while done < resamples:
        c = min(chunk, resamples - done)
        idx = rng.integers(0, n, size=(c, n),
                           dtype=_np.uint16 if n <= 65535 else _np.int64)
        out[done:done + c] = v[idx].mean(axis=1)
        done += c
    return out


def bootstrap_ci(x, level: float = CI_LEVEL, resamples: int = RESAMPLES,
                 seed: int | None = None):
    """Percentile bootstrap interval for the mean. (lo, hi) or (None, None)."""
    v = _arr(x)
    if len(v) < 2:
        return None, None
    rng = _np.random.default_rng(data_seed(v) if seed is None else seed)
    m = _boot_means(v, resamples, rng)
    a = (1 - level) / 2
    return float(_np.quantile(m, a)), float(_np.quantile(m, 1 - a))


def t_ci(x, level: float = CI_LEVEL):
    """Student-t interval for the mean. (lo, hi) or (None, None)."""
    v = _arr(x)
    n = len(v)
    if n < 2:
        return None, None
    m = float(v.mean())
    half = t_ppf(1 - (1 - level) / 2, n - 1) * float(v.std(ddof=1)) / math.sqrt(n)
    return m - half, m + half


def paired_bootstrap(x, y, level: float = CI_LEVEL, resamples: int = RESAMPLES,
                     seed: int | None = None):
    """Paired bootstrap of the mean difference: documents are resampled
    together, so both systems see the same resample. Two-sided p from the
    centred bootstrap distribution. (diff, p, lo, hi)."""
    a, b = _pair(x, y)
    d = a - b
    n = len(d)
    if n < MIN_N:
        return None, None, None, None
    rng = _np.random.default_rng(data_seed(d) if seed is None else seed)
    m = _boot_means(d, resamples, rng)
    obs = float(d.mean())
    hits = int((_np.abs(m - obs) >= abs(obs) - 1e-12).sum())
    a = (1 - level) / 2
    return (obs, (hits + 1) / (resamples + 1),
            float(_np.quantile(m, a)), float(_np.quantile(m, 1 - a)))


def two_sample_bootstrap(x, y, level: float = CI_LEVEL,
                         resamples: int = RESAMPLES, seed: int | None = None):
    """Independent groups: each resampled on its own. (diff, p, lo, hi).
    From ASYMPTOTIC_N documents per group the p-value is the bootstrap's
    normal limit, 2Φ(−|Δ|/SE) with SE² = s₁²/n₁ + s₂²/n₂ (no 1/(R+1) floor);
    the interval stays the percentile bootstrap."""
    a, b = _arr(x), _arr(y)
    if len(a) < MIN_N or len(b) < MIN_N or not _finite(a, b):
        return None, None, None, None
    rng = _np.random.default_rng(data_seed(a, b) if seed is None else seed)
    m = _boot_means(a, resamples, rng) - _boot_means(b, resamples, rng)
    obs = float(a.mean() - b.mean())
    if min(len(a), len(b)) >= ASYMPTOTIC_N:
        se = math.sqrt(float(a.var(ddof=1)) / len(a) + float(b.var(ddof=1)) / len(b))
        p = 1.0 if se == 0 and obs == 0 else (
            0.0 if se == 0 else min(1.0, 2 * _norm_sf(abs(obs) / se)))
    else:
        hits = int((_np.abs(m - obs) >= abs(obs) - 1e-12).sum())
        p = (hits + 1) / (resamples + 1)
    al = (1 - level) / 2
    return (obs, p, float(_np.quantile(m, al)), float(_np.quantile(m, 1 - al)))


# ---- effect sizes ----------------------------------------------------------

def rank_biserial_paired(x, y) -> float | None:
    """Matched-pairs rank-biserial correlation: (W+ − W−)/(W+ + W−), in
    [−1, 1]; positive when x tends to exceed y."""
    a, b = _pair(x, y)
    d = a - b
    d = d[d != 0]
    if not len(d):
        return 0.0 if len(x) else None
    ranks, _t = _rankdata(_np.abs(d))
    ranks = _np.asarray(ranks, dtype=float)
    wp, wn = float(ranks[d > 0].sum()), float(ranks[d < 0].sum())
    return (wp - wn) / (wp + wn)


def rank_biserial_indep(x, y) -> float | None:
    """Rank-biserial for two independent groups (= Cliff's δ): P(X>Y) − P(X<Y)."""
    a, b = _arr(x), _arr(y)
    n1, n2 = len(a), len(b)
    if not n1 or not n2:
        return None
    ranks, _t = _rankdata(_np.concatenate((a, b)))
    r1 = float(_np.asarray(ranks[:n1], dtype=float).sum())
    u1 = r1 - n1 * (n1 + 1) / 2
    return 2 * u1 / (n1 * n2) - 1


def kendalls_w(chi2: float | None, n: int, k: int) -> float | None:
    """Kendall's coefficient of concordance from Friedman's statistic."""
    if chi2 is None or n <= 0 or k <= 1:
        return None
    return chi2 / (n * (k - 1))


# ---- multiple comparisons --------------------------------------------------

def adjust_p(pvals: list, method: str = "holm") -> list:
    """Adjusted p-values for one family of tests; None entries (untested)
    are kept and do not count towards the family size."""
    idx = [i for i, p in enumerate(pvals) if p is not None]
    m = len(idx)
    out = list(pvals)
    if method in (None, "none") or m <= 1:
        return out
    ps = [pvals[i] for i in idx]
    order = sorted(range(m), key=lambda j: ps[j])
    adj = [0.0] * m
    if method == "bonferroni":
        adj = [min(1.0, p * m) for p in ps]
    elif method == "holm":
        run = 0.0
        for rank, j in enumerate(order):
            run = max(run, min(1.0, (m - rank) * ps[j]))
            adj[j] = run
    elif method == "bh":
        run = 1.0
        for rank in range(m - 1, -1, -1):
            j = order[rank]
            run = min(run, ps[j] * m / (rank + 1))
            adj[j] = min(1.0, run)
    else:
        raise ValueError(method)
    for j, i in enumerate(idx):
        out[i] = adj[j]
    return out


# ---- correlation -----------------------------------------------------------

def pearson(a, b):
    """(r, p, lo, hi): two-sided t-test p (exact under bivariate normality)
    and a Fisher-z 95 % interval; no interval when |r| = 1."""
    x, y = _arr(a), _arr(b)
    n = len(x)
    if n < 3 or len(y) != n or not _finite(x, y) or x.std() == 0 or y.std() == 0:
        return None, None, None, None
    r = float(_np.corrcoef(x, y)[0, 1])
    r = max(-1.0, min(1.0, r))
    if abs(r) >= 1.0:
        return r, 0.0, None, None
    t = r * math.sqrt((n - 2) / (1 - r * r))
    p = min(1.0, 2 * t_sf(abs(t), n - 2))
    lo = hi = None
    if n > 3:
        z, se = math.atanh(r), 1 / math.sqrt(n - 3)
        q = norm_ppf(1 - (1 - CI_LEVEL) / 2)
        lo, hi = math.tanh(z - q * se), math.tanh(z + q * se)
    return r, p, lo, hi


EXACT_PERM_N = 9      # correlations over ≤ 9 systems: all n! orderings


@functools.lru_cache(maxsize=16)
def _perms(n: int):
    import itertools
    return _np.array(list(itertools.permutations(range(n))), dtype=_np.int8)


def _exact_perm_p(x, y, stat) -> float:
    """Two-sided exact permutation p of a correlation statistic: the share
    of the n! pairings of y with x whose |statistic| reaches the observed
    one (ties handled by the statistic itself)."""
    obs = abs(stat(x, y))
    P = _perms(len(x))
    hits = 0
    for perm in P:
        v = stat(x, y[perm])
        if abs(v) >= obs - 1e-12:
            hits += 1
    return hits / len(P)


def _spearman_r(x, y) -> float:
    rx = _np.asarray(_rankdata(x)[0], dtype=float)
    ry = _np.asarray(_rankdata(y)[0], dtype=float)
    sx, sy = rx.std(), ry.std()
    if sx == 0 or sy == 0:
        return 0.0
    return float(((rx - rx.mean()) * (ry - ry.mean())).mean() / (sx * sy))


def spearman(a, b):
    """(ρ, p, lo, hi): Pearson on average ranks. Up to EXACT_PERM_N systems
    the p-value is exact (all n! pairings — with 3 systems a perfect ρ has
    p = 1/3, not 0); above it, the t-approximation. The interval is Fisher-z
    with the Bonett–Wright variance (1.06/(n−3)), and absent when |ρ| = 1
    or n ≤ 3 (it would be a point)."""
    x, y = _arr(a), _arr(b)
    n = len(x)
    if n < 3 or len(y) != n or not _finite(x, y) or x.std() == 0 or y.std() == 0:
        return None, None, None, None
    rx = _np.asarray(_rankdata(x)[0], dtype=float)
    ry = _np.asarray(_rankdata(y)[0], dtype=float)
    r, p, _lo, _hi = pearson(rx, ry)
    if r is None:
        return None, None, None, None
    if n <= EXACT_PERM_N:
        p = _exact_perm_p(x, y, _spearman_r)
    if n <= 3 or abs(r) >= 1:
        return r, p, None, None
    z, se = math.atanh(r), math.sqrt(1.06 / (n - 3))
    q = norm_ppf(1 - (1 - CI_LEVEL) / 2)
    return r, p, math.tanh(z - q * se), math.tanh(z + q * se)


def kendall_tau_b(a, b):
    """(τ_b, p): Kendall's τ-b with ties. Up to EXACT_PERM_N systems the
    p-value is exact (all n! pairings); above, the normal approximation with
    the tie-corrected variance. O(n²) — n is a number of systems here."""
    x, y = list(map(float, a)), list(map(float, b))
    n = len(x)
    if n < 3 or len(y) != n or not all(map(math.isfinite, x + y)):
        return None, None
    if n <= EXACT_PERM_N:
        tau = _tau_b(x, y)
        if tau is None:
            return None, None
        xa, ya = _arr(x), _arr(y)
        return tau, _exact_perm_p(xa, ya, lambda u, v: _tau_b(list(u), list(v)) or 0.0)
    conc = disc = tx = ty = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx, dy = x[i] - x[j], y[i] - y[j]
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                tx += 1
            elif dy == 0:
                ty += 1
            elif (dx > 0) == (dy > 0):
                conc += 1
            else:
                disc += 1
    n0 = n * (n - 1) / 2
    denom = math.sqrt((conc + disc + tx) * (conc + disc + ty))
    if denom == 0:
        return None, None
    tau = (conc - disc) / denom

    def ties(v):
        from collections import Counter
        return [t for t in Counter(v).values() if t > 1]
    t1, t2 = ties(x), ties(y)
    v0 = n * (n - 1) * (2 * n + 5)
    vt = sum(t * (t - 1) * (2 * t + 5) for t in t1)
    vu = sum(t * (t - 1) * (2 * t + 5) for t in t2)
    v1 = sum(t * (t - 1) for t in t1) * sum(u * (u - 1) for u in t2)
    v2 = (sum(t * (t - 1) * (t - 2) for t in t1)
          * sum(u * (u - 1) * (u - 2) for u in t2))
    var = ((v0 - vt - vu) / 18 + v1 / (2 * n * (n - 1))
           + v2 / (9 * n * (n - 1) * (n - 2)))
    if var <= 0:
        return tau, None
    z = (conc - disc) / math.sqrt(var)
    _ = n0
    return tau, min(1.0, 2 * _norm_sf(abs(z)))


def _tau_b(x, y) -> float | None:
    n = len(x)
    conc = disc = tx = ty = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx, dy = x[i] - x[j], y[i] - y[j]
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                tx += 1
            elif dy == 0:
                ty += 1
            elif (dx > 0) == (dy > 0):
                conc += 1
            else:
                disc += 1
    denom = math.sqrt((conc + disc + tx) * (conc + disc + ty))
    return (conc - disc) / denom if denom else None


def kendall_ci(tau, n):
    """Fisher-z interval for Kendall's τ with the Fieller–Hartley–Pearson
    variance 0.437/(n−4)."""
    if tau is None or n <= 4 or abs(tau) >= 1:
        return None, None
    z, se = math.atanh(tau), math.sqrt(0.437 / (n - 4))
    q = norm_ppf(1 - (1 - CI_LEVEL) / 2)
    return math.tanh(z - q * se), math.tanh(z + q * se)


# ---- one entry point per design -------------------------------------------

class StatsCfg:
    """The reader's inference settings (Insights → Statistics), shared by
    every workbench, so a paper never mixes procedures."""

    def __init__(self, alpha=ALPHA_DEFAULT, family="rank", adjust="holm",
                 ci="t", resamples=RESAMPLES):
        self.alpha = alpha
        self.family = family if family in FAMILIES else "rank"
        self.adjust = adjust if adjust in ADJUST else "holm"
        self.ci = ci if ci in ("none", "t", "bootstrap") else "t"
        try:
            self.resamples = max(100, min(100_000, int(resamples)))
        except (TypeError, ValueError):
            self.resamples = RESAMPLES

    def key(self):
        return (self.alpha, self.family, self.adjust, self.ci, self.resamples)

    def test_name(self, design: str, n: int | None = None) -> str:
        """The test a design runs under this family; with the sample size,
        the variant actually used (resampling tests switch to their normal
        limit from ASYMPTOTIC_N documents)."""
        name = FAMILIES[self.family][("paired", "indep", "multi").index(design)]
        if (self.family == "resample" and design in ("paired", "indep")
                and n is not None and n >= ASYMPTOTIC_N):
            name += " (normal limit)"
        if self.family == "mean" and design == "multi":
            name += ", Greenhouse–Geisser corrected"
        return name

    def effect_name(self, design: str) -> str:
        return EFFECTS[self.family][("paired", "indep", "multi").index(design)]

    # P, R and F1 live in [0, 1]: that parameter space — known, not taken
    # from the sample — is the only thing an interval is intersected with
    BOUNDS = (0.0, 1.0)

    def _bound(self, lo, hi):
        if lo is None or hi is None:
            return lo, hi
        b0, b1 = self.BOUNDS
        return max(lo, b0), min(hi, b1)

    def mean_ci(self, x):
        """Interval for a mean score: the Student-t or percentile-bootstrap
        interval as computed, intersected with the metric's range [0, 1]
        (a t-interval for a handful of documents can reach below 0).
        Never clipped to the observed sample's extremes."""
        if self.ci == "none" or x is None or len(x) < 2:
            return None, None
        v = _arr(x)
        if not _finite(v):
            return None, None
        if self.ci == "bootstrap":
            return self._bound(*bootstrap_ci(v, resamples=self.resamples))
        return self._bound(*t_ci(v))

    def macro_ci(self, arrays):
        """Interval for the macro-average of per-dataset means — the same
        estimand as the plotted point: every dataset given, equal weights.
        Datasets are treated as fixed strata, documents as the sampled
        units. If any dataset cannot support a variance estimate (fewer
        than two documents) the interval is unavailable (None, None) rather
        than an interval for a different average.

        t: Welch–Satterthwaite degrees of freedom. Bootstrap: every dataset
        resampled on its own from one random stream — datasets are put in a
        canonical order first, so the result does not depend on the order
        they were passed in, and two datasets with identical scores still
        get independent resamples."""
        arrays = [None if a is None else _arr(a) for a in arrays]
        if self.ci == "none" or not arrays:
            return None, None
        if any(a is None or len(a) < 2 or not _finite(a) for a in arrays):
            return None, None
        if len(arrays) == 1:
            return self.mean_ci(arrays[0])
        D = len(arrays)
        m = sum(float(a.mean()) for a in arrays) / D
        if self.ci == "bootstrap":
            keyed = sorted(arrays, key=lambda a: (data_seed(a), len(a)))
            rng = _np.random.default_rng(data_seed(*keyed, D))
            mm = sum(_boot_means(a, self.resamples, rng) for a in keyed) / D
            al = (1 - CI_LEVEL) / 2
            return self._bound(float(_np.quantile(mm, al)),
                               float(_np.quantile(mm, 1 - al)))
        parts = [float(a.var(ddof=1)) / len(a) / D ** 2 for a in arrays]
        se2 = sum(parts)
        if se2 <= 0:
            return m, m
        df = se2 ** 2 / sum(p * p / (len(a) - 1) for p, a in zip(parts, arrays))
        half = t_ppf(1 - (1 - CI_LEVEL) / 2, df) * math.sqrt(se2)
        return self._bound(m - half, m + half)

    def ci_text(self) -> str:
        if self.ci == "none":
            return ""
        if self.ci == "bootstrap":
            return (f"{int(CI_LEVEL * 100)} % percentile-bootstrap intervals "
                    f"({self.resamples:,} resamples of documents), "
                    "intersected with [0, 1]")
        return (f"{int(CI_LEVEL * 100)} % Student-t intervals, intersected "
                "with [0, 1]")

    def paired(self, x, y) -> dict:
        """x vs y on the same documents: {'p', 'effect', 'diff', 'lo', 'hi',
        'n', 'test', 'effect_name'} — the test and effect actually used."""
        x, y = _arr(x), _arr(y)
        out = {"p": None, "effect": None, "diff": None, "lo": None, "hi": None,
               "n": int(min(len(x), len(y))),
               "test": self.test_name("paired", len(x)),
               "effect_name": self.effect_name("paired")}
        if len(x) != len(y):
            out["why"] = "unaligned samples"
            return out
        if out["n"] < MIN_N or not _finite(x, y):
            return out
        d = x - y
        out["diff"] = float(d.mean())
        if not d.any():
            # identical on every document: no evidence of a difference (the
            # rank test would drop every pair and report "too few")
            out.update(p=1.0, effect=0.0, lo=0.0 if self.ci != "none" else None,
                       hi=0.0 if self.ci != "none" else None)
            return out
        if self.family == "rank":
            _w, out["p"] = wilcoxon_signed_rank(x, y)
            out["effect"] = rank_biserial_paired(x, y)
        elif self.family == "mean":
            _t, out["p"], out["effect"] = paired_t(x, y)
        else:
            _d, out["p"] = paired_permutation(x, y, self.resamples)
            out["effect"] = rank_biserial_paired(x, y)
        if self.ci == "bootstrap":
            _d, _p, out["lo"], out["hi"] = paired_bootstrap(
                x, y, resamples=self.resamples)
        elif self.ci == "t":
            out["lo"], out["hi"] = t_ci(d)
        return out

    def indep(self, x, y) -> dict:
        x, y = _arr(x), _arr(y)
        out = {"p": None, "effect": None, "diff": None, "lo": None, "hi": None,
               "n": (len(x), len(y)),
               "test": self.test_name("indep", min(len(x), len(y))),
               "effect_name": self.effect_name("indep")}
        if not len(x) or not len(y) or not _finite(x, y):
            return out
        out["diff"] = float(x.mean() - y.mean())
        if self.family == "rank":
            _u, out["p"] = mann_whitney_u(x, y)
            out["effect"] = rank_biserial_indep(x, y)
        elif self.family == "mean":
            _t, out["p"], out["effect"] = welch_t(x, y)
        else:
            _d, out["p"], lo, hi = two_sample_bootstrap(
                x, y, resamples=self.resamples)
            out["effect"] = rank_biserial_indep(x, y)
            if self.ci == "bootstrap":
                out["lo"], out["hi"] = lo, hi
        if self.ci == "t" and len(x) > 1 and len(y) > 1:
            v1, v2 = float(x.var(ddof=1)), float(y.var(ddof=1))
            n1, n2 = len(x), len(y)
            se2 = v1 / n1 + v2 / n2
            if se2 > 0:
                df = se2 ** 2 / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1))
                half = t_ppf(1 - (1 - CI_LEVEL) / 2, df) * math.sqrt(se2)
                out["lo"], out["hi"] = out["diff"] - half, out["diff"] + half
        elif self.ci == "bootstrap" and out["lo"] is None and \
                len(x) >= MIN_N and len(y) >= MIN_N:
            _d, _p, out["lo"], out["hi"] = two_sample_bootstrap(
                x, y, resamples=self.resamples)
        return out

    def multi(self, groups) -> dict:
        """k paired conditions. Two conditions are a paired comparison: the
        result then names the paired test and effect it used."""
        out = {"p": None, "effect": None, "test": self.test_name("multi"),
               "effect_name": self.effect_name("multi")}
        if len(groups) < 2 or not len(groups[0]):
            return out
        if len(groups) == 2:
            r = self.paired(groups[0], groups[1])
            return {"p": r["p"], "effect": r["effect"], "two": True,
                    "test": r["test"], "effect_name": r["effect_name"]}
        if self.family == "mean":
            _f, out["p"], out["effect"] = rm_anova(groups)
        else:
            chi, out["p"] = friedman([list(map(float, g)) for g in groups])
            out["effect"] = kendalls_w(chi, len(groups[0]), len(groups))
        return out

    def adjust_all(self, pvals: list) -> list:
        return adjust_p(pvals, self.adjust)

    def method_text(self, design: str, n_tests: int | None = None) -> str:
        """One sentence a methods section can quote."""
        test = self.test_name(design)
        effect = self.effect_name(design)
        if design == "multi":
            test += (f" ({self.test_name('paired')} with "
                     f"{self.effect_name('paired')} for two values)")
        bits = [f"two-sided {test}"]
        if self.family == "rank":
            bits.append(f"exact null distribution without ties up to "
                        f"{EXACT_MAX_N} documents, normal approximation otherwise")
        if self.family == "resample":
            bits.append(f"{self.resamples:,} resamples seeded from the data "
                        f"(smallest attainable p = 1/{self.resamples + 1:,}) "
                        f"below {ASYMPTOTIC_N} documents, the tests' normal "
                        "limit from there on")
        adj = ""
        if n_tests and n_tests > 1 and self.adjust != "none":
            adj = f"; p-values {ADJUST[self.adjust]}-adjusted over {n_tests} tests"
        return (", ".join(bits) + adj
                + f"; effect size: {effect}"
                + f"; {DAGGER} marks p<{alpha_str(self.alpha)}"
                + ("" if self.adjust == "none" or not n_tests or n_tests < 2
                   else " after adjustment"))


def fmt_effect(e: float | None) -> str:
    if e is None or e != e:
        return "—"
    if math.isinf(e):
        return "∞" if e > 0 else "−∞"
    return f"{e:+.2f}"


def fmt_ci(lo, hi, digits: int = 3) -> str:
    if lo is None or hi is None:
        return ""
    return f"[{lo:.{digits}f}, {hi:.{digits}f}]"
