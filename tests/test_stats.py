"""The inference toolkit against SciPy (a test-only reference; the app itself
depends on NumPy alone)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from kpviz import stats

sp = pytest.importorskip("scipy.stats")


def _data(n, seed=0, shift=0.03):
    r = np.random.default_rng(seed)
    x = r.choice([0, .2, .25, 1 / 3, .4, .5, 2 / 3, 1], size=n)
    y = np.clip(x - shift + r.normal(0, .15, n), 0, 1).round(3)
    return x, y


@pytest.mark.parametrize("n", [12, 400, 5000])
def test_parametric_tests_match_scipy(n):
    x, y = _data(n)
    t, p, dz = stats.paired_t(x, y)
    ref = sp.ttest_rel(x, y)
    assert math.isclose(t, ref.statistic, rel_tol=1e-9)
    assert math.isclose(p, ref.pvalue, rel_tol=1e-6, abs_tol=1e-12)
    t, p, _g = stats.welch_t(x, y)
    ref = sp.ttest_ind(x, y, equal_var=False)
    assert math.isclose(t, ref.statistic, rel_tol=1e-9)
    assert math.isclose(p, ref.pvalue, rel_tol=1e-6, abs_tol=1e-12)


def test_distributions():
    for df in (1, 3, 10, 57.3, 400):
        for t in (-3.2, -0.4, 0.0, 1.7, 6.0):
            assert math.isclose(stats.t_sf(t, df), sp.t.sf(t, df), rel_tol=1e-8, abs_tol=1e-14)
        for q in (0.025, 0.5, 0.975):
            assert math.isclose(stats.t_ppf(q, df), sp.t.ppf(q, df), rel_tol=1e-6, abs_tol=1e-9)
    for f in (0.2, 1.0, 4.5):
        assert math.isclose(stats.f_sf(f, 3, 120), sp.f.sf(f, 3, 120), rel_tol=1e-8)
    for q in (1e-6, 0.02, 0.5, 0.9, 0.999):
        assert math.isclose(stats.norm_ppf(q), sp.norm.ppf(q), rel_tol=1e-8)


def test_rank_tests_match_scipy():
    x, y = _data(800)
    _w, p = stats.wilcoxon_signed_rank(x, y)
    ref = sp.wilcoxon(x, y, zero_method="wilcox", correction=True, method="approx")
    assert math.isclose(p, ref.pvalue, rel_tol=1e-6)
    _u, p = stats.mann_whitney_u(x[:300], y[300:])
    ref = sp.mannwhitneyu(x[:300], y[300:], alternative="two-sided",
                          use_continuity=True, method="asymptotic")
    assert math.isclose(p, ref.pvalue, rel_tol=1e-6)
    g = [x[:200], y[:200], (x[:200] + y[:200]) / 2]
    chi, p = stats.friedman([list(v) for v in g])
    ref = sp.friedmanchisquare(*g)
    assert math.isclose(chi, ref.statistic, rel_tol=1e-9)
    assert math.isclose(p, ref.pvalue, rel_tol=1e-6)


def test_rm_anova_matches_textbook():
    # classic example: 5 subjects × 3 conditions
    X = [[45, 50, 55], [42, 42, 45], [36, 41, 43], [39, 35, 40], [51, 55, 59],
         [44, 49, 56]]
    F, p, eta = stats.rm_anova([[r[j] for r in X] for j in range(3)])
    # hand computation
    A = np.array(X, float)
    gm = A.mean()
    ssc = 6 * ((A.mean(0) - gm) ** 2).sum()
    sss = 3 * ((A.mean(1) - gm) ** 2).sum()
    sse = ((A - gm) ** 2).sum() - ssc - sss
    F_ref = (ssc / 2) / (sse / 10)
    assert math.isclose(F, F_ref, rel_tol=1e-12)
    assert math.isclose(p, sp.f.sf(F_ref, 2, 10), rel_tol=1e-8)
    assert math.isclose(eta, ssc / (ssc + sse), rel_tol=1e-12)


def test_correlations_match_scipy():
    r = np.random.default_rng(3)
    a = r.normal(size=22)
    b = a * .6 + r.normal(size=22)
    b[3] = b[4]                              # a tie
    rr, p, lo, hi = stats.pearson(a, b)
    ref = sp.pearsonr(a, b)
    assert math.isclose(rr, ref.statistic, rel_tol=1e-9)
    assert math.isclose(p, ref.pvalue, rel_tol=1e-6)
    ci = ref.confidence_interval(0.95)
    assert math.isclose(lo, ci.low, rel_tol=1e-6) and math.isclose(hi, ci.high, rel_tol=1e-6)
    rho, p, _l, _h = stats.spearman(a, b)
    ref = sp.spearmanr(a, b)
    assert math.isclose(rho, ref.statistic, rel_tol=1e-9)
    assert math.isclose(p, ref.pvalue, rel_tol=1e-6)
    a2 = np.round(a, 0)
    tau, p = stats.kendall_tau_b(a2, b)
    ref = sp.kendalltau(a2, b, method="asymptotic")
    assert math.isclose(tau, ref.statistic, rel_tol=1e-9)
    assert math.isclose(p, ref.pvalue, rel_tol=1e-6)


def test_adjustments_match_reference():
    p = [0.01, 0.04, None, 0.03, 0.2, 0.001]
    real = [v for v in p if v is not None]
    holm = stats.adjust_p(p, "holm")
    bh = stats.adjust_p(p, "bh")
    bon = stats.adjust_p(p, "bonferroni")
    assert holm[2] is None and bh[2] is None
    m = len(real)
    # Holm by hand
    order = np.argsort(real)
    exp = np.empty(m)
    run = 0
    for rank, j in enumerate(order):
        run = max(run, min(1, (m - rank) * real[j]))
        exp[j] = run
    assert [v for v in holm if v is not None] == pytest.approx(list(exp))
    assert [v for v in bon if v is not None] == pytest.approx([min(1, v * m) for v in real])
    ref = sp.false_discovery_control(real, method="bh")
    assert [v for v in bh if v is not None] == pytest.approx(list(ref))


def test_resampling_is_reproducible_and_calibrated():
    x, y = _data(3000, shift=0.0)
    d1, p1 = stats.paired_permutation(x, y, 2000)
    d2, p2 = stats.paired_permutation(x, y, 2000)
    assert (d1, p1) == (d2, p2)                       # seeded from the data
    # permutation p agrees with the paired t-test at large n (same statistic)
    _t, pt, _dz = stats.paired_t(x, y)
    assert abs(p1 - pt) < 0.05
    xs, ys = _data(3000, shift=0.05)
    _d, p = stats.paired_permutation(xs, ys, 1000)
    assert p == pytest.approx(1 / 1001)               # floor at resolution
    diff, p, lo, hi = stats.paired_bootstrap(xs, ys, resamples=2000)
    assert lo < diff < hi and p < 0.01
    lo, hi = stats.bootstrap_ci(x, resamples=4000)
    tl, th = stats.t_ci(x)
    assert abs(lo - tl) < 0.01 and abs(hi - th) < 0.01


def test_effect_sizes():
    x = np.array([1., 2, 3, 4, 5, 6, 7, 8])
    y = x - 1
    assert stats.rank_biserial_paired(x, y) == 1.0
    assert stats.rank_biserial_paired(y, x) == -1.0
    a, b = np.arange(10.), np.arange(10.) + 100
    assert stats.rank_biserial_indep(a, b) == -1.0
    # Cliff's delta by definition
    r = np.random.default_rng(1)
    a, b = r.normal(size=40), r.normal(.4, size=50)
    cliff = np.mean(np.sign(a[:, None] - b[None, :]))
    assert stats.rank_biserial_indep(a, b) == pytest.approx(cliff)


def test_cfg_routes_by_family():
    x, y = _data(500)
    for fam in ("rank", "mean", "resample"):
        cfg = stats.StatsCfg(family=fam, ci="bootstrap", resamples=500)
        r = cfg.paired(x, y)
        assert r["p"] is not None and r["lo"] < r["diff"] < r["hi"]
        r = cfg.indep(x[:200], y[200:])
        assert r["p"] is not None
        m = cfg.multi([x[:100], y[:100], x[100:200]])
        assert m["p"] is not None and m["effect"] is not None
        assert "two-sided" in cfg.method_text("paired", 22)
