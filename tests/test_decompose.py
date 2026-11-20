"""Pointing steps, segments and the Paper V decomposition (systematix.steps / .decompose)."""
import numpy as np
import pytest

import systematix as sx
from systematix import decompose as dc
from systematix.steps import consensus_steps, pointing_steps, segment_labels

CAD = 0.0204   # d, long cadence


def _quarter(t0=100.0, days=30.0, gap=(115.0, 116.0)):
    t = np.arange(t0, t0 + days, CAD)
    return t[(t < gap[0]) | (t > gap[1])]


def test_step_between_adjacent_cadences():
    t = _quarter()
    pos = np.zeros((len(t), 2))
    pos[t > 108.0, 0] = 0.05
    assert pointing_steps(t, pos[:, 0], pos[:, 1]) == pytest.approx([108.0], abs=CAD)


def test_step_across_gap_uses_medians():
    t = _quarter()
    pos = np.zeros((len(t), 2))
    pos[t > 116.0, 1] = 0.03                      # the pointing moved while no data were taken
    steps = pointing_steps(t, pos[:, 0], pos[:, 1])
    assert len(steps) == 1 and 115.0 < steps[0] < 116.0


def test_small_settling_flagged_and_nan_ignored():
    t = _quarter()
    pos = np.zeros((len(t), 2))
    pos[t > 100.5, 0] = 0.05                      # within the first day: settling, ignored
    pos[t > 110.0, 0] += 0.01                     # below threshold
    q = np.zeros(len(t), int)
    bad = (t > 120.0) & (t < 120.1)
    pos[bad, 0] = 5.0                              # a flagged excursion
    q[bad] = 1                                     # a DEFAULT_BITMASK bit (attitude tweak)
    pos[np.argmin(np.abs(t - 125.0)), 1] = np.nan
    assert pointing_steps(t, pos[:, 0], pos[:, 1], q) == []


def test_steps_merge_within_half_day():
    t = _quarter()
    pos = np.zeros((len(t), 2))
    pos[t > 108.0, 0] = 0.05
    pos[t > 108.2, 0] = 0.10
    assert len(pointing_steps(t, pos[:, 0], pos[:, 1])) == 1


def test_consensus_keeps_steps_in_half_the_stars():
    lists = [[10.0, 50.0], [10.2], [10.1, 70.0], []]
    assert consensus_steps(lists) == pytest.approx([10.1])
    assert consensus_steps([[], []]) == []


def test_segment_labels_restart_each_quarter():
    t = np.r_[np.linspace(0, 10, 50), np.linspace(20, 30, 50)]
    q = np.r_[np.full(50, 2), np.full(50, 3)]
    lab = segment_labels(t, q, [5.0, 25.0])
    assert set(lab[:50]) == {200, 201} and set(lab[50:]) == {300, 301}
    # the same quarter alone gives the same labels
    np.testing.assert_array_equal(segment_labels(t[50:], q[50:], [5.0, 25.0]), lab[50:])


def _obs_with_truth(rng, steps=(108.0,)):
    t = _quarter()
    n = len(t)
    cad = np.arange(n) + 1000
    vec = rng.standard_normal((n, 16)) / np.sqrt(n)
    cbv = {"cadenceno": cad, "vectors": vec, "gap": np.zeros(n, bool)}
    obs = sx.Observation(time=t, flux=np.ones(n), flux_err=np.full(n, 1e-4), quarter=5, cadenceno=cad,
                         quarter_meta={5: {"MODULE": 2, "OUTPUT": 1}})
    D = dc.design_blocks(obs, list(steps), cbv_loader=lambda q, m, o: cbv)
    M = np.hstack([D[b] for b in dc.BLOCKS])
    w = rng.normal(0, 1e-3, M.shape[1])
    w[-dc.N_CBV:] = rng.normal(0, 0.03, dc.N_CBV)
    return obs, D, M, w


def test_fit_recovers_weights_and_jitter():
    rng = np.random.default_rng(1)
    obs, D, M, w = _obs_with_truth(rng)
    err = np.full(len(obs), 1e-4)
    rel = M @ w + rng.normal(0, np.hypot(1e-4, 2e-4), len(obs))
    fit, res, jit, cond = dc.fit_flat(rel, err, D)
    wf, Cf = fit["all"]
    pull = (wf - w) / np.sqrt(np.diag(Cf))
    assert np.all(np.isfinite(wf)) and np.abs(pull).max() < 5
    assert jit == pytest.approx(2e-4, rel=0.1)
    assert D["seg_ids"].tolist() == [500, 501]


def test_keyed_weights_relative_offsets():
    rng = np.random.default_rng(2)
    obs, D, M, w = _obs_with_truth(rng)
    rel = M @ w
    fit, *_ = dc.fit_flat(rel, np.full(len(obs), 1e-5), D)
    t0c = dc.canonical_t0([[t0 for (_, t0, _) in D["ramp_meta"]]])
    keys, kw, C = dc.keyed_weights(fit, D, 5, t0c)
    assert keys[0] == ("offset", 1)
    assert kw[0] == pytest.approx(w[1] - w[0], abs=1e-6)          # segment 1 relative to segment 0
    assert sum(k[0] == "ramp" for k in keys) == 2 * len(D["ramp_meta"]) // 2
    assert C.shape == (len(keys), len(keys))
    for k in keys:
        assert dc.parse_key(dc.key_str(k)) == k


@pytest.mark.parametrize("order", [0, 1, 4])
def test_poly_order(order):
    rng = np.random.default_rng(4)
    obs, D, M, w = _obs_with_truth(rng)
    t = _quarter()
    cbv = {"cadenceno": obs.cadenceno, "vectors": rng.standard_normal((len(t), 16)), "gap": np.zeros(len(t), bool)}
    Dk = dc.design_blocks(obs, [108.0], cbv_loader=lambda q, m, o: cbv, poly_order=order)
    nseg = len(Dk["seg_ids"])
    assert Dk["polynomial"].shape == (len(obs), order * nseg)
    assert len(Dk["poly_meta"]) == order * nseg
    assert [p for (_, s, p) in Dk["poly_meta"] if s == Dk["seg_ids"][0]] == list(range(1, order + 1))
    np.testing.assert_array_equal(Dk["offsets"], D["offsets"])     # offsets do not depend on the order
    if order:
        fit, *_ = dc.fit_flat(np.hstack([Dk[b] for b in dc.BLOCKS]) @ rng.normal(0, 1e-3, sum(
            Dk[b].shape[1] for b in dc.BLOCKS)), np.full(len(obs), 1e-4), Dk)
        assert np.all(np.isfinite(fit["polynomial"][0]))
    with pytest.raises(ValueError):
        dc.design_blocks(obs, [108.0], cbv_loader=lambda q, m, o: cbv, poly_order=-1)


def test_short_segment_has_no_polynomial():
    rng = np.random.default_rng(3)
    t = _quarter()
    obs, D, M, w = _obs_with_truth(rng, steps=(t[-20],))           # last segment: 19 cadences
    assert np.all(D["polynomial"][:, -dc.POLY_ORDER:] == 0)
    fit, *_ = dc.fit_flat(M @ w, np.full(len(obs), 1e-4), D)
    assert np.all(np.isnan(fit["polynomial"][0][-dc.POLY_ORDER:]))
    assert np.all(np.isfinite(fit["offsets"][0]))
