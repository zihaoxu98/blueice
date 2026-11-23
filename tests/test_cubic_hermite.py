"""Tests for the C1 cubic Hermite interpolation and source-wise interpolation of binned likelihoods."""
import numpy as np
import pytest
from scipy.interpolate import PchipInterpolator

from blueice.likelihood import BinnedLogLikelihood
from blueice.pdf_morphers import CubicHermiteGridInterpolator
from blueice.test_helpers import conf_for_test


def random_grids(rng, n_dims):
    """Random ascending grids, the last dimension with a single anchor if n_dims > 1."""
    grids = [np.sort(rng.choice(np.linspace(-3, 3, 13), rng.integers(3, 6), replace=False))
             for _ in range(n_dims)]
    if n_dims > 1:
        grids[-1] = np.array([0.5])
    return grids


def random_point(rng, grids):
    return np.array([rng.uniform(g[0], g[-1]) for g in grids])


def test_1d_equals_scipy_pchip():
    rng = np.random.default_rng(0)
    grid = np.array([-2., -1., 0., 0.5, 2.])
    scores = rng.uniform(0, 1, size=(5, 7))
    itp = CubicHermiteGridInterpolator([grid], scores)
    reference = PchipInterpolator(grid, scores, axis=0)
    for z in np.concatenate([rng.uniform(-2, 2, 50), grid]):
        np.testing.assert_allclose(itp([z]), reference(z), rtol=0, atol=1e-13)


@pytest.mark.parametrize("n_dims", [1, 2, 3])
def test_cubic_hermite_interpolation(n_dims):
    rng = np.random.default_rng(n_dims)
    grids = random_grids(rng, n_dims)
    scores = rng.uniform(0.1, 1, size=tuple(len(g) for g in grids) + (4,))
    itp = CubicHermiteGridInterpolator(grids, scores)

    # passes through the anchor values
    for index in np.ndindex(*scores.shape[:-1]):
        z = np.array([g[i] for g, i in zip(grids, index)])
        np.testing.assert_allclose(itp(z), scores[index], rtol=0, atol=1e-12)

    # analytic gradient matches central finite differences
    h = 1e-6
    for _ in range(20):
        z = random_point(rng, grids)
        value, gradient = itp.value_and_gradient(z)
        np.testing.assert_allclose(value, itp(z), rtol=0, atol=1e-14)
        for k, g in enumerate(grids):
            if len(g) == 1:
                assert np.all(gradient[k] == 0)
                continue
            zp, zm = z.copy(), z.copy()
            zp[k] = min(z[k] + h, g[-1])
            zm[k] = max(z[k] - h, g[0])
            numeric = (itp(zp) - itp(zm)) / (zp[k] - zm[k])
            np.testing.assert_allclose(gradient[k], numeric, rtol=1e-5, atol=1e-6)

    # C1: no kink at interior anchors (multilinear interpolation would have one)
    for k, g in enumerate(grids):
        for anchor in g[1:-1]:
            z = random_point(rng, grids)
            z[k] = anchor
            left = z.copy()
            left[k] -= 1e-9
            np.testing.assert_allclose(itp.value_and_gradient(left)[1],
                                       itp.value_and_gradient(z)[1], rtol=0, atol=1e-6)


def test_cubic_hermite_monotone_in_1d():
    grid = np.arange(5.)
    scores = np.array([0., 0., 1., 5., 5.])[:, None]  # PCHIP does not overshoot
    itp = CubicHermiteGridInterpolator([grid], scores)
    values = np.array([itp([z])[0] for z in np.linspace(0, 4, 401)])
    assert np.all(values >= 0) and np.all(values <= 5 + 1e-12)
    assert np.all(np.diff(values) >= -1e-12)


def make_likelihood(source_wise, morpher='GridInterpolator'):
    # s0 depends on both shape parameters, s1 only on some_multiplier
    conf = conf_for_test(n_sources=2, mc=True, n_events_for_pdf=int(2e5),
                         analysis_space=[['x', np.linspace(-10, 10, 41)]])
    conf['sources'][1].update(sigma=3, mu=0., extra_dont_hash_settings=['mu'])
    # (re)use the same cached MC pmfs in likelihoods with and without source-wise interpolation
    for source in conf['sources']:
        source.setdefault('extra_dont_hash_settings', []).append('source_wise_interpolation')
    conf['livetime_days'] = 2.
    conf['source_wise_interpolation'] = source_wise
    lf = BinnedLogLikelihood(conf, likelihood_config={'morpher': morpher})
    lf.add_rate_parameter('s0')
    lf.add_rate_parameter('s1')
    lf.add_shape_parameter('mu', (-1., 0., 1., 2.))
    lf.add_shape_parameter('some_multiplier', (0.5, 1., 2.))
    lf.prepare()
    data = np.zeros(3000, dtype=[('x', float), ('source', int)])
    data['x'] = np.random.RandomState(0).normal(0.2, 2, len(data))
    lf.set_data(data)
    return lf


POINTS = [
    dict(s0_rate_multiplier=1.3, s1_rate_multiplier=0.7, mu=0.3, some_multiplier=1.4),
    dict(s0_rate_multiplier=0.9, s1_rate_multiplier=1.6, mu=-0.8, some_multiplier=0.6),
    dict(s0_rate_multiplier=1.1, s1_rate_multiplier=0.5, mu=1., some_multiplier=2.),  # anchors
]


def check_gradient(lf, kwargs, one_sided_shape=False):
    value, gradient = lf.loglikelihood_and_gradient(**kwargs)
    assert value == pytest.approx(lf(**kwargs), abs=1e-9)
    h = 1e-6
    for name in gradient:
        up, down = dict(kwargs), dict(kwargs)
        up[name] += h
        down[name] -= h
        if name in lf.shape_parameters and up[name] > lf.get_bounds(name)[1]:
            numeric = (lf(**kwargs) - lf(**down)) / h  # at the upper edge: left derivative
        elif one_sided_shape and name in lf.shape_parameters:
            numeric = (lf(**up) - lf(**kwargs)) / h  # multilinear: right derivative at anchors
        else:
            numeric = (lf(**up) - lf(**down)) / (2 * h)
        assert gradient[name] == pytest.approx(numeric, rel=1e-4, abs=1e-4), name


def test_source_wise_linear_matches_joint_grid(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # shared pdf cache: identical anchor pmfs in both likelihoods
    joint = make_likelihood(source_wise=False)
    source_wise = make_likelihood(source_wise=True)
    assert len(joint.anchor_models) == 12
    assert set(source_wise.source_interpolators) == {'s0', 's1'}
    assert source_wise.source_interpolators['s1'].shape_indices == [1]
    assert source_wise.supports_gradient()
    for point in POINTS:
        assert source_wise(**point) == pytest.approx(joint(**point), abs=1e-9)
        _, g_joint = joint.loglikelihood_and_gradient(**point)
        _, g_source_wise = source_wise.loglikelihood_and_gradient(**point)
        for name in g_joint:
            assert g_source_wise[name] == pytest.approx(g_joint[name], rel=1e-9, abs=1e-9)
        check_gradient(source_wise, point, one_sided_shape=True)


def test_source_wise_cubic_hermite(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    lf = make_likelihood(source_wise=True, morpher='CubicHermiteInterpolator')
    assert lf.supports_gradient()
    s0 = lf.source_interpolators['s0']
    assert s0.renormalize

    # the pmf is rescaled to the (smoothly) interpolated normalization of the anchor pmfs
    for point in POINTS:
        zs = [point['mu'], point['some_multiplier']]
        expected_sum = s0.interpolator(np.array(zs))[-1]  # interpolated anchor normalization
        assert s0.density(zs).sum() == pytest.approx(expected_sum, rel=1e-12)
        check_gradient(lf, point)

    # exact at the anchors
    for (mu, some_multiplier), source in lf.anchor_sources['s0'].items():
        zs = [mu, some_multiplier]
        np.testing.assert_allclose(s0.density(zs), source.get_pmf_grid()[0], rtol=1e-12, atol=1e-15)
        assert s0.rate(zs) == pytest.approx(source.expected_events, rel=1e-12)

    # no kink at the anchor mu = 0
    h = 1e-7
    point = dict(POINTS[0], mu=0.)
    left = (lf(**point) - lf(**dict(point, mu=-h))) / h
    right = (lf(**dict(point, mu=h)) - lf(**point)) / h
    assert left == pytest.approx(right, rel=1e-3, abs=1e-3)


def test_cubic_hermite_requires_source_wise_interpolation():
    conf = conf_for_test(n_sources=1, mc=True)
    with pytest.raises(ValueError, match='source_wise_interpolation'):
        BinnedLogLikelihood(conf, likelihood_config={'morpher': 'CubicHermiteInterpolator'})


def test_source_wise_without_beeston_barlow():
    conf = conf_for_test(n_sources=1, mc=True, analysis_space=[['x', np.linspace(-5, 5, 6)]],
                         source_wise_interpolation=True)
    lf = BinnedLogLikelihood(conf, likelihood_config={
        'model_statistical_uncertainty_handling': 'bb_single', 'bb_single_source': 0})
    lf.add_shape_parameter('mu', (-1., 0., 1.))
    with pytest.raises(NotImplementedError):
        lf.prepare()
