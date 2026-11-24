"""Tests for the analytic gradient of UnbinnedLogLikelihood and its source-wise interpolation."""
import numpy as np
import pytest

from blueice.likelihood import UnbinnedLogLikelihood
from blueice.test_helpers import conf_for_test


def make_data(seed=0):
    data = np.zeros(500, dtype=[('x', float), ('source', int)])
    data['x'] = np.random.RandomState(seed).normal(0.2, 2, len(data))
    data['x'][:3] = 9.9  # far in the tail: no MC events there, outlier events
    return data


def make_likelihood(source_wise, morpher='GridInterpolator', shape=True, mc=True, seed=0):
    """s0 depends on both shape parameters, s1 only on some_multiplier.

    mc=True: histogram (MC) sources, needed for the C1 interpolation, which renormalizes with
    the sources' pmf grids. mc=False: analytic Gaussian pdfs, identical in every likelihood.
    """
    conf = conf_for_test(n_sources=2, mc=mc, n_events_for_pdf=int(2e5),
                         analysis_space=[['x', np.linspace(-10, 10, 41)]])
    conf['sources'][1].update(sigma=3, mu=0., extra_dont_hash_settings=['mu'])
    conf['livetime_days'] = 2.
    conf['source_wise_interpolation'] = source_wise
    lf = UnbinnedLogLikelihood(conf, likelihood_config={'morpher': morpher})
    lf.add_rate_parameter('s0')
    lf.add_rate_parameter('s1')
    if shape:
        lf.add_shape_parameter('mu', (-1., 0., 1., 2.))
        lf.add_shape_parameter('some_multiplier', (0.5, 1., 2.))
    lf.prepare()
    lf.set_data(make_data(seed))
    return lf


POINTS = [
    dict(s0_rate_multiplier=1.3, s1_rate_multiplier=0.7, mu=0.3, some_multiplier=1.4),
    dict(s0_rate_multiplier=0.9, s1_rate_multiplier=1.6, mu=-0.8, some_multiplier=0.6),
    dict(s0_rate_multiplier=1.1, s1_rate_multiplier=0.5, mu=1., some_multiplier=2.),  # anchors
]


def check_gradient(lf, kwargs, one_sided_shape=False):
    value, gradient = lf.loglikelihood_and_gradient(**kwargs)
    assert value == pytest.approx(lf(**kwargs), abs=1e-9)
    assert set(gradient) == set(kwargs) - {'livetime_days'}
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


@pytest.mark.parametrize("source_wise", [False, True])
def test_unbinned_gradient_linear(source_wise, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    lf = make_likelihood(source_wise)
    assert lf.supports_gradient()
    for point in POINTS:
        check_gradient(lf, point, one_sided_shape=True)
        check_gradient(lf, dict(point, livetime_days=3.), one_sided_shape=True)


def test_unbinned_gradient_without_shape_parameters(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    lf = make_likelihood(source_wise=False, shape=False)
    check_gradient(lf, dict(s0_rate_multiplier=1.3, s1_rate_multiplier=0.7))


def test_unbinned_source_wise_matches_joint_grid(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    joint = make_likelihood(source_wise=False, mc=False)
    source_wise = make_likelihood(source_wise=True, mc=False)
    for point in POINTS:
        assert source_wise(**point) == pytest.approx(joint(**point), abs=1e-9)
        _, g_joint = joint.loglikelihood_and_gradient(**point)
        _, g_source_wise = source_wise.loglikelihood_and_gradient(**point)
        for name in g_joint:
            assert g_source_wise[name] == pytest.approx(g_joint[name], rel=1e-9, abs=1e-9)


def test_unbinned_cubic_hermite(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    lf = make_likelihood(source_wise=True, morpher='CubicHermiteInterpolator')
    assert lf.supports_gradient()
    s0 = lf.source_interpolators['s0']
    assert s0.renormalize
    for point in POINTS:
        check_gradient(lf, point)

    # exact at the anchors
    events = lf.base_model.to_analysis_dimensions(lf._data)
    for (mu, some_multiplier), source in lf.anchor_sources['s0'].items():
        np.testing.assert_allclose(s0.density([mu, some_multiplier]), source.pdf(*events),
                                   rtol=1e-12, atol=1e-15)

    # between anchors, the pdf values are rescaled such that the interpolated pmf has the
    # interpolated normalization of the anchor pmfs
    zs = np.array([0.3, 1.4])
    rate, density, total, normalization = s0._split(s0.interpolator(zs))
    np.testing.assert_allclose(s0.density(zs), density * normalization / total, rtol=1e-12)
    assert total != pytest.approx(normalization, rel=1e-9, abs=0)  # i.e. it matters

    # no kink at the anchor mu = 0
    h = 1e-7
    point = dict(POINTS[0], mu=0.)
    left = (lf(**point) - lf(**dict(point, mu=-h))) / h
    right = (lf(**dict(point, mu=h)) - lf(**point)) / h
    assert left == pytest.approx(right, rel=1e-3, abs=1e-3)


def test_cubic_hermite_requires_source_wise_interpolation():
    conf = conf_for_test(n_sources=1, mc=True)
    with pytest.raises(ValueError, match='source_wise_interpolation'):
        UnbinnedLogLikelihood(conf, likelihood_config={'morpher': 'CubicHermiteInterpolator'})


def test_no_gradient_for_subclasses_changing_the_likelihood(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    class ModifiedLikelihood(UnbinnedLogLikelihood):
        def _compute_likelihood(self, mus, pdf_values_at_events):
            return 2 * super()._compute_likelihood(mus, pdf_values_at_events)

    conf = conf_for_test(n_sources=1, mc=True)
    lf = ModifiedLikelihood(conf)
    lf.add_rate_parameter('s0')
    lf.prepare()
    lf.set_data(np.zeros(5, dtype=[('x', float), ('source', int)]))
    assert not lf.supports_gradient()


def test_unbinned_cubic_hermite_keeps_data_independent_part_between_data_sets(tmp_path, monkeypatch):
    """The sum of the pmf does not depend on the data: it is computed once, and a second data set
    gives the same likelihood and gradient as a likelihood that was never given other data."""
    monkeypatch.chdir(tmp_path)
    # The sources' MC histograms are random, and shared between anchors through the pdf cache. Fill the
    # cache first, so that both likelihoods below are built from the same models.
    make_likelihood(source_wise=True, morpher='CubicHermiteInterpolator')
    reused = make_likelihood(source_wise=True, morpher='CubicHermiteInterpolator', seed=0)
    rows = reused._pmf_sum_caches['s0']['pmf_sum_rows'][1]
    reused.set_data(make_data(1))
    assert reused._pmf_sum_caches['s0']['pmf_sum_rows'][1] is rows  # not recomputed
    fresh = make_likelihood(source_wise=True, morpher='CubicHermiteInterpolator', seed=1)
    for name in reused.anchor_sources:
        for a, b in zip(reused.anchor_sources[name].values(), fresh.anchor_sources[name].values()):
            np.testing.assert_array_equal(a.get_pmf_grid()[0], b.get_pmf_grid()[0])  # same models
    for kwargs in POINTS:
        value, gradient = reused.loglikelihood_and_gradient(**kwargs)
        fresh_value, fresh_gradient = fresh.loglikelihood_and_gradient(**kwargs)
        assert value == fresh_value
        assert gradient == fresh_gradient
        assert reused(**kwargs) == fresh(**kwargs)


def test_unbinned_cubic_hermite_cached_pmf_sum_equals_uncached(tmp_path, monkeypatch):
    from blueice.likelihood import SourceShapeInterpolator
    monkeypatch.chdir(tmp_path)
    lf = make_likelihood(source_wise=True, morpher='CubicHermiteInterpolator')
    events = lf.base_model.to_analysis_dimensions(make_data(2))
    arguments = dict(morpher=lf.source_morphers['s0'], anchor_sources=lf.anchor_sources['s0'],
                     source_index=0, shape_indices=lf.source_shape_indices['s0'], events=events)
    uncached = SourceShapeInterpolator(**arguments)
    cached = SourceShapeInterpolator(**arguments, pmf_sum_cache={})
    np.testing.assert_array_equal(cached.interpolator.values, uncached.interpolator.values)
    assert cached._slices == uncached._slices
