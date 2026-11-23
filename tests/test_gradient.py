"""Tests for MultilinearGridInterpolator and the analytic gradient of BinnedLogLikelihood."""
import numpy as np
import pytest
from scipy import stats
from scipy.interpolate import RegularGridInterpolator

from blueice.likelihood import BinnedLogLikelihood, LogLikelihoodSum
from blueice.pdf_morphers import MultilinearGridInterpolator
from blueice.test_helpers import conf_for_test


def random_grids(rng, n_dims, allow_single=False):
    lo = 1 if allow_single else 2
    return [np.sort(rng.choice(np.linspace(-3, 3, 13), rng.integers(lo, 6), replace=False))
            for _ in range(n_dims)]


@pytest.mark.parametrize("n_dims", [1, 2, 3, 4])
@pytest.mark.parametrize("extra", [(), (7,), (3, 4)])
def test_multilinear_matches_regular_grid_interpolator(n_dims, extra):
    rng = np.random.default_rng(n_dims * 10 + len(extra))
    grids = random_grids(rng, n_dims)
    scores = rng.normal(size=tuple(len(g) for g in grids) + extra)
    itp = MultilinearGridInterpolator(grids, scores)
    reference = RegularGridInterpolator(grids, scores)
    points = [np.array([rng.uniform(g[0], g[-1]) for g in grids]) for _ in range(30)]
    points += [np.array([rng.choice(g) for g in grids]) for _ in range(10)]  # on anchors
    points += [np.array([g[0] for g in grids]), np.array([g[-1] for g in grids])]
    for z in points:
        np.testing.assert_allclose(itp(z), reference(z)[0], rtol=0, atol=1e-13)
    # the result is a fresh array: modifying it must not affect later calls
    result = itp(points[0])
    result *= 0
    np.testing.assert_allclose(itp(points[0]), reference(points[0])[0], rtol=0, atol=1e-13)
    with pytest.raises(ValueError):
        itp(np.array([g[-1] + 0.1 for g in grids]))


@pytest.mark.parametrize("n_dims", [1, 2, 3])
def test_multilinear_corner_weight_gradient(n_dims):
    rng = np.random.default_rng(n_dims)
    grids = random_grids(rng, n_dims, allow_single=True)
    scores = rng.normal(size=tuple(len(g) for g in grids) + (5,))
    itp = MultilinearGridInterpolator(grids, scores)
    points = [np.array([rng.uniform(g[0], g[-1]) for g in grids]) for _ in range(20)]
    points += [np.array([rng.choice(g) for g in grids]) for _ in range(5)]
    h = 1e-6
    for z in points:
        rows, weights, weight_gradients = itp.corner_weights(z)
        np.testing.assert_allclose(weights @ itp.values[rows], itp(z).ravel(), atol=1e-13)
        for k, g in enumerate(grids):
            if len(g) == 1:
                assert np.all(weight_gradients[:, k] == 0)
                continue
            analytic = weight_gradients[:, k] @ itp.values[rows]
            # one-sided like the analytic derivative: to the right, except at the upper edge
            if z[k] + h <= g[-1]:
                zp = z.copy()
                zp[k] += h
                numeric = (itp(zp) - itp(z)).ravel() / h
            else:
                zm = z.copy()
                zm[k] -= h
                numeric = (itp(z) - itp(zm)).ravel() / h
            np.testing.assert_allclose(analytic, numeric, rtol=1e-5, atol=1e-6)


def make_binned_likelihood(shape=True):
    # MC (histogram) sources: binned likelihoods need pmf grids
    conf = conf_for_test(n_sources=2, mc=True, n_events_for_pdf=int(2e5),
                         analysis_space=[['x', np.linspace(-10, 10, 41)]])
    conf['sources'][1]['sigma'] = 3  # make the two sources differ
    conf['livetime_days'] = 2.
    lf = BinnedLogLikelihood(conf)
    lf.add_rate_parameter('s0')
    lf.add_rate_parameter('s1')
    if shape:
        lf.add_shape_parameter('mu', (-1., 0., 1.))
        lf.add_shape_parameter('some_multiplier', (0.5, 1., 2.))
    lf.prepare()
    data = np.zeros(3000, dtype=[('x', float), ('source', int)])
    data['x'] = np.random.RandomState(0).normal(0.2, 2, len(data))
    lf.set_data(data)
    return lf


def numeric_gradient(lf, name, kwargs, one_sided=False, h=1e-6):
    up = dict(kwargs)
    up[name] += h
    if one_sided:
        return (lf(**up) - lf(**kwargs)) / h
    down = dict(kwargs)
    down[name] -= h
    return (lf(**up) - lf(**down)) / (2 * h)


@pytest.mark.parametrize("point", [
    dict(s0_rate_multiplier=1.3, s1_rate_multiplier=0.7, mu=0.3, some_multiplier=1.4),
    dict(s0_rate_multiplier=0.9, s1_rate_multiplier=1.6, mu=-0.8, some_multiplier=0.6),
    dict(s0_rate_multiplier=1.1, s1_rate_multiplier=0.5, mu=0., some_multiplier=1.),  # anchors
])
@pytest.mark.parametrize("livetime_days", [None, 3.])
def test_binned_gradient(point, livetime_days):
    lf = make_binned_likelihood()
    assert lf.supports_gradient()
    kwargs = dict(point, livetime_days=livetime_days)
    value, gradient = lf.loglikelihood_and_gradient(**kwargs)
    assert value == pytest.approx(lf(**kwargs), abs=1e-9)
    assert set(gradient) == {'s0_rate_multiplier', 's1_rate_multiplier', 'mu', 'some_multiplier'}
    for name in gradient:
        shape = name in ('mu', 'some_multiplier')
        numeric = numeric_gradient(lf, name, kwargs, one_sided=shape)
        assert gradient[name] == pytest.approx(numeric, rel=1e-4, abs=1e-4), name


def test_binned_gradient_without_shape_parameters():
    lf = make_binned_likelihood(shape=False)
    kwargs = dict(s0_rate_multiplier=1.3, s1_rate_multiplier=0.7)
    value, gradient = lf.loglikelihood_and_gradient(**kwargs)
    assert value == pytest.approx(lf(**kwargs), abs=1e-9)
    for name in kwargs:
        assert gradient[name] == pytest.approx(numeric_gradient(lf, name, kwargs), rel=1e-5)


def test_binned_gradient_out_of_bounds():
    lf = make_binned_likelihood()
    value, gradient = lf.loglikelihood_and_gradient(mu=1.5)
    assert value == -float('inf')
    assert all(g == 0 for g in gradient.values())


def test_binned_gradient_infinite_likelihood():
    # without s1 (the wide source) there are bins with data but no expected events
    lf = make_binned_likelihood()
    value, gradient = lf.loglikelihood_and_gradient(s1_rate_multiplier=0.)
    assert value == lf(s1_rate_multiplier=0.) == -float('inf')
    assert all(g == 0 for g in gradient.values())


def test_likelihood_sum_gradient():
    lf0 = make_binned_likelihood()
    lf1 = make_binned_likelihood(shape=False)
    lsum = LogLikelihoodSum([lf0, lf1], likelihood_weights=[1., 0.5])
    assert lsum.supports_gradient()
    kwargs = dict(s0_rate_multiplier=1.3, s1_rate_multiplier=0.7, mu=0.3, some_multiplier=1.4)
    value, gradient = lsum.loglikelihood_and_gradient(**kwargs)
    assert value == pytest.approx(lsum(**kwargs), abs=1e-9)
    _, g0 = lf0.loglikelihood_and_gradient(**kwargs)
    _, g1 = lf1.loglikelihood_and_gradient(s0_rate_multiplier=1.3, s1_rate_multiplier=0.7)
    assert gradient['mu'] == pytest.approx(g0['mu'])
    assert gradient['s0_rate_multiplier'] == pytest.approx(
        g0['s0_rate_multiplier'] + 0.5 * g1['s0_rate_multiplier'])


def test_no_gradient_for_beeston_barlow():
    conf = conf_for_test(n_sources=1, mc=True, analysis_space=[['x', np.linspace(-5, 5, 6)]])
    lf = BinnedLogLikelihood(conf, likelihood_config={
        'model_statistical_uncertainty_handling': 'bb_single', 'bb_single_source': 0})
    lf.add_rate_parameter('s0')
    lf.prepare()
    lf.set_data(np.zeros(5, dtype=[('x', float), ('source', int)]))
    assert not lf.supports_gradient()
    with pytest.raises(NotImplementedError):
        lf.loglikelihood_and_gradient()


def test_binned_gradient_with_efficiency():
    conf = conf_for_test(n_sources=2, mc=True, n_events_for_pdf=int(2e5),
                         analysis_space=[['x', np.linspace(-10, 10, 41)]])
    conf['sources'][1]['sigma'] = 3
    conf['sources'][0]['apply_efficiency'] = True
    conf['sources'][0]['efficiency_name'] = 'eff'
    conf['eff'] = 1.
    lf = BinnedLogLikelihood(conf)
    lf.add_rate_parameter('s0')
    lf.add_rate_parameter('s1')
    lf.add_shape_parameter('eff', (0.2, 1.5))
    lf.prepare()
    data = np.zeros(3000, dtype=[('x', float), ('source', int)])
    data['x'] = np.random.RandomState(0).normal(0.2, 2, len(data))
    lf.set_data(data)
    assert lf.supports_gradient()
    kwargs = dict(s0_rate_multiplier=1.3, s1_rate_multiplier=0.7, eff=0.8)
    value, gradient = lf.loglikelihood_and_gradient(**kwargs)
    assert value == pytest.approx(lf(**kwargs), abs=1e-9)
    for name in kwargs:
        assert gradient[name] == pytest.approx(numeric_gradient(lf, name, kwargs), rel=1e-5), name


def test_full_output_does_not_expose_internal_arrays():
    lf = make_binned_likelihood(shape=False)
    reference = lf()
    _, mus, ps = lf(full_output=True)
    mus *= 0
    ps *= 0
    assert lf() == reference


def test_data_modified_in_place():
    lf = make_binned_likelihood(shape=False)
    lf()
    lf.data_events_per_bin.histogram[20] += 5
    expected = (np.einsum('s,s...->...', *lf(full_output=True)[1:]))
    reference = np.sum(stats.poisson(expected).logpmf(lf.data_events_per_bin.histogram))
    assert lf() == pytest.approx(reference, abs=1e-8)


def test_no_gradient_for_subclasses_changing_the_likelihood():
    class ModifiedLikelihood(BinnedLogLikelihood):
        def _compute_likelihood(self, mus, pmfs):
            return 2 * super()._compute_likelihood(mus, pmfs)

    conf = conf_for_test(n_sources=1, mc=True, analysis_space=[['x', np.linspace(-5, 5, 6)]])
    lf = ModifiedLikelihood(conf)
    lf.add_rate_parameter('s0')
    lf.prepare()
    lf.set_data(np.zeros(5, dtype=[('x', float), ('source', int)]))
    assert not lf.supports_gradient()
