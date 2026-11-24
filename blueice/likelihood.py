"""Log likelihood constructors: the heart of blueice.


"""

from collections import OrderedDict
from copy import deepcopy
from functools import wraps

import numpy as np
from multihist import Histdd
from scipy import special, stats
from tqdm import tqdm

from .exceptions import NotPreparedException, InvalidParameterSpecification, InvalidParameter
from .model import Model
from .parallel import create_models_ipyparallel, compute_many
from .utils import combine_dicts, inherit_docstring_from
from . import inference

from .pdf_morphers import (GridInterpolator, CubicHermiteInterpolator, MultilinearGridInterpolator,
                           RadialInterpolator)
from .pdf_morphers_cubic import CubicSplineInterpolator
MORPHERS = {x.__name__: x for x in [GridInterpolator, RadialInterpolator, CubicSplineInterpolator,
                                    CubicHermiteInterpolator]}

__all__ = ['LogLikelihoodBase', 'BinnedLogLikelihood', 'UnbinnedLogLikelihood', 'LogLikelihoodSum',
           'LogLikelihoodReParam', 'SourceShapeInterpolator']


##
# Decorators for methods which have to be run after prepare or set_data
##

def _needs_preparation(f):
    @wraps(f)
    def wrapper(self, *args, **kwargs):
        if not self.is_prepared:
            if not len(self.shape_parameters):
                # preparation is going to be trivial, just do it
                self.prepare()
            else:
                raise NotPreparedException("%s requires you to first prepare the likelihood function using prepare()" %
                                           f.__name__)
        return f(self, *args, **kwargs)
    return wrapper


def _needs_data(f):
    @wraps(f)
    def wrapper(self, *args, **kwargs):
        if not self.is_data_set:
            raise NotPreparedException("%s requires you to first set the data using set_data()" % f.__name__)
        return f(self, *args, **kwargs)
    return wrapper


class LogLikelihoodBase:
    """Log likelihood function with several rate and/or shape parameters

    likelihood_config options:
        unphysical_behaviour
        outlier_likelihood
        parallelize_models: True (default) or False
        block_during_paralellization: True or False (default)
    """

    def __init__(self, pdf_base_config, likelihood_config=None, **kwargs):
        """
        :param pdf_base_config: dictionary with configuration passed to the Model
        :param likelihood_config: dictionary with options for LogLikelihood itself
        :param kwargs: Overrides for pdf_base_config, not likelihood config!
        :return:
        """
        self.pdf_base_config = combine_dicts(pdf_base_config, kwargs, deep_copy=True)

        if likelihood_config is None:
            likelihood_config = {}
        self.config = likelihood_config
        self.config.setdefault('morpher', 'GridInterpolator')
        self.source_wise_interpolation = self.pdf_base_config.get('source_wise_interpolation', False)
        if not MORPHERS[self.config['morpher']].preserves_sums and not self.source_wise_interpolation:
            # e.g. CubicHermiteInterpolator: only source-wise interpolation renormalizes the
            # interpolated densities (see SourceShapeInterpolator)
            raise ValueError("%s requires source_wise_interpolation" % self.config['morpher'])
        self.source_interpolators = OrderedDict()  # source name -> SourceShapeInterpolator

        # Base model: no variations of any settings
        self.base_model = Model(self.pdf_base_config)
        self.source_name_list = [s.name for s in self.base_model.sources]

        self.source_allowed_negative = [s.config.get("allow_negative",False)
                                        for s in self.base_model.sources]
        self.source_apply_efficiency = np.array([s.config.get("apply_efficiency", False)
                                                 for s in self.base_model.sources])
        self.source_efficiency_names = np.array([s.config.get("efficiency_name", "efficiency")
                                                 for s in self.base_model.sources])

        self.rate_parameters = OrderedDict()     # sourcename_rate -> logprior
        self.shape_parameters = OrderedDict()    # settingname -> (anchors, logprior, base_z).
        # where anchors is dict: representative number -> actual setting
        # From here on representative number will be called 'z-score'.
        # base_value is the default z-score that will be used.
        # We'll take care of sorting the keys in self.prepare()

        self.is_prepared = False
        self.is_data_set = False
        self._has_non_numeric = False

        # If there are NO shape parameters:
        self.ps = None                # ps of the data

        # If there are shape parameters:
        self.anchor_models = OrderedDict()  # dictionary mapping model zs -> actual model
        self.anchor_sources = OrderedDict()  # dictionary mapping source_name -> zs -> source
        # Interpolators created by morphers. These map zs to...
        self.mus_interpolator = None                          # rates for each source
        self.ps_interpolator = None                          # (source, event) p-values (unbinned), or pmf grid (binned)
        # number of events per bin observed in Monte Carlo / calibration data that gave rise to the model.
        self.n_model_events_interpolator = lambda x: None
        self.n_model_events = None

    @property
    def source_shape_parameters(self):
        """Dict of sources with shape parameters. source name -> dict of shape parameters."""
        source_shape_parameters = OrderedDict()
        for sn, source, apply_eff, eff_name in zip(
                self.source_name_list,
                self.base_model.sources,
                self.source_apply_efficiency,
                self.source_efficiency_names
                ):
            ignore_parameters = set(source.config['dont_hash_settings'])
            # The efficiecny parameter doesn't need to be hashed but it needs to be passed to the morpher
            if apply_eff:
                ignore_parameters.discard(eff_name)
            shape_parameters = OrderedDict({k: v for k, v in self.shape_parameters.items() if k not in ignore_parameters})
            if shape_parameters:
                source_shape_parameters[sn] = shape_parameters
        return source_shape_parameters

    def _get_shape_indices(self, source_name):
        """Return the indices of the shape parameters used by the source."""
        shape_keys = self.source_shape_parameters[source_name].keys()
        return [i for i, k in enumerate(self.shape_parameters.keys()) if k in shape_keys]

    def _get_model_anchor(self, anchor, source_name):
        """Return the shape anchors of the full model, given the shape anchors of a signle source.
        All values of shape parameters not used by this source will be set to None.
        """
        shape_indices = self._get_shape_indices(source_name)
        model_anchor = [None] * len(self.shape_parameters)
        for i, idx in enumerate(shape_indices):
            model_anchor[idx] = anchor[i]
        return tuple(model_anchor)

    def prepare(self, n_cores=1, ipp_client=None):
        """Prepares a likelihood function with shape parameters for use.
        This will compute the models for each shape parameter anchor value combination.
        """
        if len(self.shape_parameters):
            if self.source_wise_interpolation:
                # Create morphers for each source individually
                self.source_morphers = OrderedDict()

                def create_source_morpher(shape_parameters):
                    return MORPHERS[self.config['morpher']](
                        self.config.get('morpher_config', {}),
                        shape_parameters)

                for sn, shape_parameters in self.source_shape_parameters.items():
                    self.source_morphers[sn] = create_source_morpher(shape_parameters)
                zs_list = set()
                for source_name, morpher in self.source_morphers.items():
                    anchor_points = morpher.get_anchor_points(bounds=None)
                    for anchor in anchor_points:
                        zs = self._get_model_anchor(anchor, source_name)
                        zs_list.add(zs)
                zs_list = list(zs_list)

            else:
                self.morpher = MORPHERS[self.config['morpher']](self.config.get('morpher_config', {}),
                                                                self.shape_parameters)
                zs_list = self.morpher.get_anchor_points(bounds=self.get_bounds())

            # Create the configs for each new model
            configs = []
            for zs in zs_list:
                config = deepcopy(self.pdf_base_config)
                for i, (setting_name, (anchors, _, _)) in enumerate(self.shape_parameters.items()):
                    # Translate from zs to settings using the anchors dict. Maybe not all settings are numerical.
                    if zs[i] is not None:
                        config[setting_name] = anchors[zs[i]]
                if ipp_client is None and n_cores != 1:
                    # We have to compute in parallel: must have delayed computation on
                    config['delay_pdf_computation'] = True
                configs.append(config)

            # Create the new models
            if n_cores == 1:
                models = [Model(c) for c in tqdm(configs, desc="Computing/loading models on one core")]

            elif ipp_client is not None:
                models = create_models_ipyparallel(configs, ipp_client,
                                                   block=self.config.get('block_during_paralellization', False))

            else:
                models = [Model(c) for c in tqdm(configs, desc="Preparing model computation tasks")]

                hashes = set()
                for m in models:
                    for s in m.sources:
                        hashes.add(s.hash)

                compute_many(hashes, n_cores)

                # Reload models so computation takes effect
                models = [Model(c) for c in tqdm(configs, desc="Loading computed models")]

            if self.source_wise_interpolation:
                for source_name, morpher in self.source_morphers.items():
                    anchors = morpher.get_anchor_points(bounds=None)
                    self.anchor_sources[source_name] = OrderedDict()
                    for anchor in anchors:
                        model_anchor = self._get_model_anchor(anchor, source_name)
                        model_index = zs_list.index(model_anchor)
                        source_index = self.source_name_list.index(source_name)
                        self.anchor_sources[source_name][anchor] = models[model_index].sources[source_index]
                # Which shape parameters each source depends on. Computed once: deriving it from
                # source_shape_parameters at every likelihood evaluation is slow.
                self.source_shape_indices = OrderedDict(
                    (sn, self._get_shape_indices(sn)) for sn in self.source_morphers)
                mus_interpolators = OrderedDict()
                for sn, base_source in zip(self.source_name_list, self.base_model.sources):
                    if sn in self.source_morphers:
                        mus_interpolators[sn] = self.source_morphers[sn].make_interpolator(
                            f=lambda s: s.expected_events,
                            extra_dims=[1],
                            anchor_models=self.anchor_sources[sn])
                    else:
                        mus_interpolators[sn] = base_source.expected_events

                def mus_interpolator(*args):
                    # take zs, convert to values for each source's interpolator call the respective interpolator
                    mus = []
                    for sn in self.source_name_list:
                        if sn in self.source_shape_indices:
                            these_args = [args[0][i] for i in self.source_shape_indices[sn]]
                            mus.append(mus_interpolators[sn](np.asarray(these_args))[0])
                        else:
                            mus.append(mus_interpolators[sn])
                    return np.array(mus)
                self.mus_interpolator = mus_interpolator

            else:
                # Add the new models to the anchor_models dict
                for zs, model in zip(zs_list, models):
                    self.anchor_models[tuple(zs)] = model

                # Build the interpolator for the rates of each source.
                self.mus_interpolator = self.morpher.make_interpolator(
                    f=lambda m: m.expected_events(),
                    extra_dims=[len(self.source_name_list)],
                    anchor_models=self.anchor_models)

        self.is_data_set = False
        self.is_prepared = True

    @_needs_preparation
    def set_data(self, d):
        """Prepare the dataset d for likelihood function evaluation
        :param d: Dataset, must be an indexable object that provides the measurement dimensions
        For example, if your models are on 's1' and 's2', d must be something for which d['s1'] and d['s2'] give
        the s1 and s2 values of your events as numpy arrays.
        """
        self._data = d
        self.is_data_set = True

    def add_rate_parameter(self, source_name, log_prior=None):
        """Add a rate parameter names source_name + "_rate_multiplier" to the likelihood function..
        The values of this parameter will MULTIPLY the expected rate of events for the source.
        The rates of sources can also vary due to shape parameters.
        :param source_name: Name of the source for which you want to vary the rate
        :param log_prior: prior logpdf function on rate multiplier (not on rate itself!)
        """
        self.rate_parameters[source_name] = log_prior

    def add_shape_parameter(self, setting_name, anchors, log_prior=None, base_value=None):
        """Add a shape parameter to the likelihood function
        :param setting_name: Name of the setting to vary
        :param anchors: a list/tuple/array of setting values (if they are numeric)
                        OR a dictionary with some numerical value -> setting values (for non-numeric settings).
        :param base_value: for non-numeric settings, the number which represents the base model value of the setting.
        For example, if you have LCE maps with varying reflectivities, use
            add_shape_variation('s1_relative_ly_map', {0.98: 'lce_98%.pklz', 0.99: 'lce_99%.pklz, ...})
        then the argument s1_relative_ly_map of the likelihood function takes values between 0.98 and 0.99.
        """
        is_numeric = isinstance(self.pdf_base_config.get(setting_name), (float, int))
        if not isinstance(anchors, dict):
            # Convert anchors list to a dictionary
            if not is_numeric:
                raise InvalidParameterSpecification("When specifying anchors only by setting values, "
                                                    "base setting must have a numerical default.")
            anchors = {z: z for z in anchors}

        if not is_numeric:
            self._has_non_numeric = True
        if not is_numeric and base_value is None:
            raise InvalidParameterSpecification("For non-numeric settings, you must specify what number will represent "
                                                "the default value (the base model setting)")
        if is_numeric and base_value is not None:
            raise InvalidParameterSpecification("For numeric settings, base_value is an unnecessary argument.")

        self.shape_parameters[setting_name] = (anchors, log_prior, base_value)

    def get_bounds(self, parameter_name=None):
        """Return bounds on the parameter parameter_name"""
        if parameter_name is None:
            return [self.get_bounds(p) for p in self.shape_parameters.keys()]
        if parameter_name in self.shape_parameters:
            anchor_settings = list(self.shape_parameters[parameter_name][0].keys())
            return min(anchor_settings), max(anchor_settings)
        elif parameter_name.endswith('_rate_multiplier'):
            for source_name, allow_negative in zip(self.source_name_list,self.source_allowed_negative):
                if parameter_name.startswith(source_name) and allow_negative==True:
                    return float('-inf'), float('inf')
            return 0, float('inf')
        else:
            raise InvalidParameter("Non-existing parameter %s" % parameter_name)

    def _source_wise_rates(self, zs):
        """Expected events of all sources at shape parameters zs (source-wise interpolation)."""
        mus = np.array(self.base_model.expected_events(), dtype=float)
        for interpolator in self.source_interpolators.values():
            mus[interpolator.source_index] = interpolator.rate(zs)
        return mus

    def _source_wise_densities(self, zs):
        """Pmfs (binned) or pdf values at the events (unbinned) of all sources at shape
        parameters zs (source-wise interpolation); self.ps holds those of the base model."""
        ps = self.ps.copy()
        for interpolator in self.source_interpolators.values():
            ps[interpolator.source_index] = interpolator.density(zs)
        return ps

    # Methods the analytic gradients reproduce; subclasses overriding them do not get gradients
    _GRADIENT_REQUIRES_METHODS = ('__call__', '_kwargs_to_settings', 'adjust_expectations',
                                  '_compute_likelihood', 'get_bounds')

    def _gradient_possible(self, reference_class):
        """Configuration checks shared by the analytic gradients of binned and unbinned
        likelihoods (reference_class: the class whose methods the gradient reproduces)."""
        for name in self._GRADIENT_REQUIRES_METHODS:
            if getattr(type(self), name) is not getattr(reference_class, name):
                return False
        if self._has_non_numeric:
            return False
        if any(lp is not None for lp in self.rate_parameters.values()):
            return False
        if any(lp is not None for _, lp, _ in self.shape_parameters.values()):
            return False
        if any(self.source_allowed_negative):
            return False
        if not len(self.shape_parameters):
            return True
        if self.source_wise_interpolation:
            return all(i.supports_gradient() for i in self.source_interpolators.values())
        return (isinstance(self.ps_interpolator, MultilinearGridInterpolator)
                and isinstance(self.mus_interpolator, MultilinearGridInterpolator))

    def _shape_point(self, shape_parameter_settings):
        """Array of the shape parameter values, or None if any is outside its anchors."""
        zs = []
        for setting_name in self.shape_parameters:
            z = shape_parameter_settings[setting_name]
            minbound, maxbound = self.get_bounds(setting_name)
            if not minbound <= z <= maxbound:
                return None
            zs.append(z)
        return np.asarray(zs)

    def _rate_factors(self, rate_multipliers, livetime_days, shape_parameter_settings):
        """Return livetime_scale, factor and efficiencies (arrays per source) such that the
        expected events of source s are mus_s * factor_s * efficiencies_s, as in __call__, where
        factor = rate multiplier x livetime_scale."""
        livetime_scale = 1.
        if livetime_days is not None:
            if 'livetime_days' not in self.pdf_base_config:
                raise ValueError("Cannot scale live-time, base value absent")
            if self.pdf_base_config['livetime_days'] == 0:
                raise NotImplementedError("Gradient not available for zero base livetime")
            livetime_scale = livetime_days / self.pdf_base_config['livetime_days']
        factor = np.asarray(rate_multipliers, dtype=float) * livetime_scale
        efficiencies = np.ones(len(self.source_name_list))
        for source_i, (sae, sen) in enumerate(zip(self.source_apply_efficiency,
                                                  self.source_efficiency_names)):
            if sae:
                efficiencies[source_i] = shape_parameter_settings.get(sen, 1)
        return livetime_scale, factor, efficiencies

    def _rate_gradients(self, dll_dexpected, mus, livetime_scale, efficiencies):
        """Gradient entries of the rate multipliers, given dll_dexpected[s] = d ll / d M_s for
        the expected events M_s = mus_s * rate multiplier_s * livetime_scale * efficiencies_s."""
        return {source_name + '_rate_multiplier':
                dll_dexpected[source_i] * mus[source_i] * livetime_scale * efficiencies[source_i]
                for source_i, source_name in enumerate(self.source_name_list)}

    def _add_efficiency_gradients(self, gradient, dll_dexpected, mus, factor):
        """Efficiency parameters also scale the expected events M_s of the sources using them:
        add d ll / d M_s * d M_s / d efficiency = dll_dexpected[s] * mus_s * factor_s."""
        for source_i, (sae, sen) in enumerate(zip(self.source_apply_efficiency,
                                                  self.source_efficiency_names)):
            if sae and sen in gradient:
                gradient[sen] += dll_dexpected[source_i] * mus[source_i] * factor[source_i]

    @_needs_data
    def __call__(self, livetime_days=None, compute_pdf=False, full_output=False, **kwargs):
        """Evaluate the likelihood function. Pass any values for parameters as keyword arguments.
        For values not passed, their base values will be assumed.
        For rate uncertainties, pass sourcename_rate_multiplier.
        :param lifetime_days: lifetime in days to use, will affect rates of all sources.
        :param full_output: instead of returning just the loglikelihood, return also the adjusted mus and ps as well.
        :param compute_pdf: compute new PDFs instead of interpolating the PDF at the requested parameters.
        """
        result = 0
        rate_multipliers, shape_parameter_settings = self._kwargs_to_settings(**kwargs)

        if len(self.shape_parameters):
            if compute_pdf:
                if self._has_non_numeric:
                    raise NotImplementedError("compute_pdf only works for numerical values")

                mus, ps, n_model_events = self._compute_single_pdf(**kwargs)

            else:
                # We can use the interpolators. They require the settings to come in order:
                zs = []
                for setting_name, (_, log_prior, _) in self.shape_parameters.items():
                    z = shape_parameter_settings[setting_name]
                    zs.append(z)

                    # Test if the z value is out of range; if so, return -inf (since we can't extrapolate)
                    minbound, maxbound = self.get_bounds(setting_name)
                    if not minbound <= z <= maxbound:
                        return -float('inf')

                    if log_prior is not None:
                        result += log_prior(z)

                # The RegularGridInterpolators want numpy arrays: give it to them...
                zs = np.asarray(zs)

                mus = self.mus_interpolator(zs)
                ps = self.ps_interpolator(zs)
                n_model_events = self.n_model_events_interpolator(zs)

        else:
            # No shape parameters
            mus = self.base_model.expected_events()
            ps = self.ps
            n_model_events = self.n_model_events

        # Apply the rate multipliers
        for source_i, source_name in enumerate(self.source_name_list):
            mult = rate_multipliers[source_i]
            mus[source_i] *= mult
            log_prior = self.rate_parameters.get(source_name, None)
            if log_prior is not None:
                result += log_prior(mult)

        # Scale the live-time
        if livetime_days is not None:
            if 'livetime_days' not in self.pdf_base_config:
                raise ValueError("Cannot scale live-time, base value absent")
            if self.pdf_base_config['livetime_days'] == 0:
                if livetime_days != 0:
                    raise ValueError("Cannot scale from 0 to non-0 livetime")
                assert np.all(mus == 0), "Got non-0 mus with 0 livetime?!"
            else:
                mus *= livetime_days / self.pdf_base_config['livetime_days']

        # Apply efficiency to those sources that use it:
        if True in self.source_apply_efficiency:
            effs = []
            for sae, sen in zip(self.source_apply_efficiency,
                                self.source_efficiency_names):
                if sae:
                    # if that particular efficiency is not in the shape parameters, apply 1
                    effs.append(shape_parameter_settings.get(sen, 1))

            mus[self.source_apply_efficiency] *= np.array(effs)

        # Check for negative rates. Depending on the config, either error or return -float('inf') as loglikelihood
        # If any source is allowed to be negative, check the sources one by one
        if not any(self.source_allowed_negative):
            if not np.all((mus >= 0) & (mus < float('inf'))):
                if self.config.get('unphysical_behaviour') == 'error':
                    raise ValueError("Unphysical rates: %s" % str(mus))
                else:
                    return -float('inf')
        else:
            if (not any(mus < float('inf'))) or (np.sum(mus) < 0):
                if self.config.get('unphysical_behaviour') == 'error':
                    raise ValueError("Unphysical rates: %s" % str(mus))
                else:
                    return -float('inf')

            for mu,allowed_negative in zip(mus,self.source_allowed_negative):
                if not (0 <= mu) and (not allowed_negative):
                    if self.config.get('unphysical_behaviour') == 'error':
                        raise ValueError("Unphysical rates: %s" % str(mus))
                    else:
                        return -float('inf')

        # Perform fits to background calibration data if needed:
        # Currently only performed (analytically) for Binned likelihood via the Beeston-Barlow method
        mus, ps = self.adjust_expectations(mus, ps, n_model_events)

        # Get the loglikelihood. At last!
        result += self._compute_likelihood(mus, ps)

        if full_output:
            return result, mus, ps
        else:
            return result

    def adjust_expectations(self, mus, ps, n_model_events):
        """Adjust uncertain (mus, pmfs) based on the observed data.

        If the density is derived from a finite-statistics sample (n_model_events array of events per bin),
        we can take into account this uncertainty by modifying the likelihood function.

        For a binned likelihood, this means adding the expected number of events for each bin for each source as
        nuisance parameters constrained by Poisson terms around the number of events observed in the model.
        While these nuisance parameters could be optimized numerically along with the main parameters,
        for a given value of the main parameters these per-bin nuisance parameters can often be estimated analytically,
        as shown by Beeston & Barlow (1993).
        """
        return mus, ps

    def _kwargs_to_settings(self, **kwargs):
        """Return shape parameters, rate_multipliers from kwargs.
          shape_parmeters is a dict mapping setting name -> value | representative number
          rate_multipliers is a list of rate multipliers for each source in self.source_name_list
        """
        # Validate the kwargs: must be either shape parameters, or <known_source>_rate_multiplier
        for k in kwargs.keys():
            if k in self.shape_parameters:
                continue
            if k.endswith('_rate_multiplier'):
                s_name = k[:-16]
                if s_name in self.source_name_list:
                    continue
            raise InvalidParameter("%s is not a known shape or rate parameter!" % k)

        shape_parameter_settings = dict()
        for setting_name, (_, _, base_value) in self.shape_parameters.items():
            z = kwargs.get(setting_name)

            if z is None:
                # Parameter was not given: get the default value of (the number representing) this shape parameter
                base_setting = self.pdf_base_config.get(setting_name)
                is_numeric = isinstance(base_setting, (float, int))
                if is_numeric:
                    assert base_value is None
                    z = base_setting
                else:
                    z = base_value

            if not isinstance(z, (int, float)):
                raise ValueError("Arguments to likelihood function must be numeric, not %s" % type(z))

            shape_parameter_settings[setting_name] = z

        rate_multipliers = []
        for source_i, source_name in enumerate(self.source_name_list):
            rate_multipliers.append(kwargs.get(source_name + '_rate_multiplier', 1))

        return rate_multipliers, shape_parameter_settings

    ##
    # Convenience functions for uncertainties.
    # Adding more general priors is the user's responsibility
    # (either provide prior argument to add_x_parameter, or wrap the loglikelihood function)
    ##
    def add_rate_uncertainty(self, source_name, fractional_uncertainty):
        """Adds a rate parameter to the likelihood function with Gaussian prior"""
        self.add_rate_parameter(source_name, log_prior=stats.norm(1, fractional_uncertainty).logpdf)

    def add_shape_uncertainty(self, setting_name, fractional_uncertainty, anchor_zs=(-2, -1, 0, 1, 2), base_value=None):
        """Adds a shape parameter to the likelihood function, with Gaussian prior around the default value.
        :param fractional_uncertainty: Relative uncertainty on the default value.
        Other parameters as in add_shape_parameter.
        """
        # Call add_shape_parameter without a prior first, then inject the prior later.
        # It's a bit of a hack, but there is some validation / default-setting code for base_value we don't want to
        # replicate.
        self.add_shape_parameter(setting_name, anchor_zs, base_value=base_value)
        anchors, log_prior, base_value = self.shape_parameters[setting_name]
        self.shape_parameters[setting_name] = (anchors,
                                               stats.norm(base_value, base_value * fractional_uncertainty).logpdf,
                                               base_value)

    def _compute_single_model(self, **kwargs):
        """Return a model formed using the base config, using kwargs as overrides"""
        _, shape_parameter_settings = self._kwargs_to_settings(**kwargs)
        config = combine_dicts(self.pdf_base_config, shape_parameter_settings, deep_copy=True)

        config['never_save_to_cache'] = True
        return Model(config, **shape_parameter_settings)

    ##
    # Methods to override
    ##
    def _compute_likelihood(self, *args, **kwargs):
        raise NotImplementedError

    def _compute_single_pdf(self, **kwargs):
        """Return likelihood arguments for a single newly computed model,
        formed using the base config, using kwargs as overrides.
        Returns mus, ps, n_model_events
        """
        raise NotImplementedError


class UnbinnedLogLikelihood(LogLikelihoodBase):

    @inherit_docstring_from(LogLikelihoodBase)
    def set_data(self, d):
        LogLikelihoodBase.set_data(self, d)
        if len(self.shape_parameters):
            if self.source_wise_interpolation:
                # Interpolate the pdf values at the events (and the rate) of each source with
                # shape parameters separately; self.ps holds those of the other sources
                events = self.base_model.to_analysis_dimensions(d)
                self.ps = self.base_model.score_events(d)
                self.source_interpolators = OrderedDict()
                # what does not depend on the data is kept between data sets, per source
                pmf_sum_caches = self.__dict__.setdefault('_pmf_sum_caches', {})
                for source_name, morpher in self.source_morphers.items():
                    self.source_interpolators[source_name] = SourceShapeInterpolator(
                        morpher,
                        anchor_sources=self.anchor_sources[source_name],
                        source_index=self.source_name_list.index(source_name),
                        shape_indices=self.source_shape_indices[source_name],
                        events=events,
                        pmf_sum_cache=pmf_sum_caches.setdefault(source_name, {}))
                self.mus_interpolator = self._source_wise_rates
                self.ps_interpolator = self._source_wise_densities
            else:
                self.ps_interpolator = self.morpher.make_interpolator(
                    f=lambda m: m.score_events(d),
                    extra_dims=[len(self.source_name_list), len(d)],
                    anchor_models=self.anchor_models)
        else:
            self.ps = self.base_model.score_events(d)

    @inherit_docstring_from(LogLikelihoodBase)
    def _compute_single_pdf(self, **kwargs):
        model = self._compute_single_model(**kwargs)
        mus = model.expected_events()
        ps = model.score_events(self._data)
        return mus, ps, None

    def _compute_likelihood(self, mus, pdf_values_at_events):
        return extended_loglikelihood(mus, pdf_values_at_events,
                                      outlier_likelihood=self.config.get('outlier_likelihood', 1e-12))

    def supports_gradient(self):
        """Whether loglikelihood_and_gradient can be used for this likelihood."""
        return self._gradient_possible(UnbinnedLogLikelihood)

    def _interpolate_with_derivatives(self, zs):
        """Rates (n_sources) and pdf values at the events (n_sources, n_events) at shape parameters
        zs, and their derivatives: a list of (source index, indices of the shape parameters,
        d rate / d z (k,), d pdf values / d z (k, n_events)) for the sources with shape parameters."""
        derivatives = []
        if self.source_wise_interpolation:
            mus = np.array(self.base_model.expected_events(), dtype=float)
            ps = self.ps.copy()
            for interpolator in self.source_interpolators.values():
                rate, pdf, rate_gradient, pdf_gradient = interpolator.values_and_gradients(zs)
                mus[interpolator.source_index] = rate
                ps[interpolator.source_index] = pdf
                derivatives.append((interpolator.source_index, interpolator.shape_indices,
                                    rate_gradient, pdf_gradient))
        else:
            mus, mus_gradient = self.mus_interpolator.value_and_gradient(zs)
            ps, ps_gradient = self.ps_interpolator.value_and_gradient(zs)
            all_shape_indices = list(range(len(zs)))
            for source_i in range(len(self.source_name_list)):
                derivatives.append((source_i, all_shape_indices,
                                    mus_gradient[:, source_i], ps_gradient[:, source_i]))
        return mus, ps, derivatives

    @_needs_data
    def loglikelihood_and_gradient(self, livetime_days=None, **kwargs):
        """Return the log likelihood (same value as __call__) and its gradient.

        The gradient is a dict with an entry for every rate multiplier
        (<source>_rate_multiplier) and every shape parameter of this likelihood. For shape
        parameters it is the derivative of the interpolation (for multilinear interpolation
        one-sided at the anchors). Only for the configurations accepted by supports_gradient().

        The extended log likelihood is ll = -sum_s M_s + sum_i log P_i, with P_i = sum_s M_s p_si,
        where M_s is the rate of source s (interpolated rate x rate multiplier x live-time scale x
        efficiency) and p_si its pdf value at event i. Hence
            d ll / d M_s = B_s - 1, with B_s = sum_i p_si / P_i
            d ll / d z = sum_s [ dM_s/dz (B_s - 1) + M_s sum_i (dp_si/dz) / P_i ]
        Events whose likelihood is replaced by outlier_likelihood (P_i <= 0) do not contribute.
        """
        if not self.supports_gradient():
            raise NotImplementedError("Gradient not available for this likelihood configuration")
        rate_multipliers, shape_parameter_settings = self._kwargs_to_settings(**kwargs)
        shape_names = list(self.shape_parameters.keys())
        zero_gradient = {sn + '_rate_multiplier': 0. for sn in self.source_name_list}
        zero_gradient.update({sn: 0. for sn in shape_names})

        if len(shape_names):
            zs = self._shape_point(shape_parameter_settings)
            if zs is None:
                return -float('inf'), zero_gradient
            mus, ps, derivatives = self._interpolate_with_derivatives(zs)
        else:
            mus = np.array(self.base_model.expected_events(), dtype=float)
            ps, derivatives = self.ps, []

        livetime_scale, factor, efficiencies = self._rate_factors(
            rate_multipliers, livetime_days, shape_parameter_settings)
        expected = mus * factor * efficiencies  # M_s, as mus in __call__

        if not np.all((expected >= 0) & (expected < float('inf'))):
            if self.config.get('unphysical_behaviour') == 'error':
                raise ValueError("Unphysical rates: %s" % str(expected))
            return -float('inf'), zero_gradient

        ll = self._compute_likelihood(expected, ps)
        if not np.isfinite(ll):
            return ll, zero_gradient

        # 1 / P_i for the events that are not outliers; nan pdf values are ignored (as nansum)
        ps = np.nan_to_num(ps)
        p_events = np.einsum('s,si->i', expected, ps)
        with np.errstate(divide='ignore'):
            inverse_p_events = np.where(p_events > 0, 1. / p_events, 0.)
        dll_dexpected = np.einsum('si,i->s', ps, inverse_p_events) - 1.  # B_s - 1

        gradient = self._rate_gradients(dll_dexpected, mus, livetime_scale, efficiencies)
        shape_gradient = np.zeros(len(shape_names))
        for source_i, shape_indices, rate_gradient, pdf_gradient in derivatives:
            pdf_term = np.einsum('ki,i->k', np.nan_to_num(pdf_gradient), inverse_p_events)
            shape_gradient[shape_indices] += factor[source_i] * efficiencies[source_i] * (
                rate_gradient * dll_dexpected[source_i] + mus[source_i] * pdf_term)
        for k, setting_name in enumerate(shape_names):
            gradient[setting_name] = shape_gradient[k]

        self._add_efficiency_gradients(gradient, dll_dexpected, mus, factor)
        return ll, gradient


class BinnedLogLikelihood(LogLikelihoodBase):

    def __init__(self, pdf_base_config, likelihood_config=None, **kwargs):
        LogLikelihoodBase.__init__(self, pdf_base_config, likelihood_config, **kwargs)
        pdf_base_config['pdf_interpolation_method'] = 'piecewise'

        self.model_statistical_uncertainty_handling = self.config.get('model_statistical_uncertainty_handling')

    @inherit_docstring_from(LogLikelihoodBase)
    def prepare(self, *args):
        LogLikelihoodBase.prepare(self, *args)
        self.ps, self.n_model_events = self.base_model.pmf_grids()

        if len(self.shape_parameters):
            if self.source_wise_interpolation:
                self._prepare_source_wise_interpolation()
                return

            self.ps_interpolator = self.morpher.make_interpolator(
                f=lambda m: m.pmf_grids()[0],
                extra_dims=list(self.ps.shape),
                anchor_models=self.anchor_models)

            if self.model_statistical_uncertainty_handling is not None:
                self.n_model_events_interpolator = self.morpher.make_interpolator(f=lambda m: m.pmf_grids()[1],
                                                                                extra_dims=list(self.ps.shape),
                                                                                anchor_models=self.anchor_models)

    def _prepare_source_wise_interpolation(self):
        """Interpolate the rate and pmf of each source separately, only in the shape parameters
        the source depends on (anchor models from LogLikelihoodBase.prepare)."""
        if self.model_statistical_uncertainty_handling is not None:
            raise NotImplementedError("Source-wise interpolation does not support "
                                      "model_statistical_uncertainty_handling")
        for source_name, morpher in self.source_morphers.items():
            self.source_interpolators[source_name] = SourceShapeInterpolator(
                morpher,
                anchor_sources=self.anchor_sources[source_name],
                source_index=self.source_name_list.index(source_name),
                shape_indices=self.source_shape_indices[source_name])
        self.mus_interpolator = self._source_wise_rates
        self.ps_interpolator = self._source_wise_densities

    @inherit_docstring_from(LogLikelihoodBase)
    def set_data(self, d):
        LogLikelihoodBase.set_data(self, d)
        # Bin the data in the analysis space
        dimnames, bins = zip(*self.base_model.config['analysis_space'])
        self.data_events_per_bin = Histdd(bins=bins, axis_names=dimnames)
        self.data_events_per_bin.add(*self.base_model.to_analysis_dimensions(d))

    @inherit_docstring_from(LogLikelihoodBase)
    def _compute_single_pdf(self, **kwargs):
        model = self._compute_single_model(**kwargs)
        mus = model.expected_events()
        ps, n_model_events = model.pmf_grids()
        return mus, ps, n_model_events

    @_needs_data
    @inherit_docstring_from(LogLikelihoodBase)
    def adjust_expectations(self, mus, pmfs, n_model_events):
        if self.model_statistical_uncertainty_handling is None:
            # Nothing to adjust. The returned arrays may be handed out (full_output), so never
            # return internal arrays; interpolated pmfs are already fresh arrays.
            return mus.copy(), (pmfs.copy() if pmfs is self.ps else pmfs)

        # Avoid mutating the arrays we're also going to return
        mus = mus.copy()
        pmfs = pmfs.copy()

        if self.model_statistical_uncertainty_handling == 'bb_single':

            source_i = self.config.get('bb_single_source')
            if source_i is None:
                raise ValueError("You need to specify bb_single_source to use bb_single_source expectation adjustment")
            source_i = self.base_model.get_source_i(source_i)

            assert pmfs.shape == n_model_events.shape

            # Get the number of events expected for the sources we will NOT adjust
            counts_per_bin = pmfs.copy()
            for i, (mu, _x) in enumerate(zip(mus, counts_per_bin)):
                if i != source_i:
                    _x *= mu
                else:
                    _x *= 0.
            u_bins = np.sum(counts_per_bin, axis=0)

            a_bins = n_model_events[source_i]

            p_calibration = mus[source_i] / n_model_events[source_i].sum()
            w_calibration = pmfs[source_i] / a_bins * n_model_events[source_i].sum()

            A_bins_1, A_bins_2 = beeston_barlow_roots(a_bins, w_calibration * p_calibration, u_bins, self.data_events_per_bin.histogram)
            assert np.all(A_bins_1 <= 0)  # it seems(?) the 1st root is always negative

            # For U=0, the solution above is singular; we need to use a special case instead
            A_bins_special = (self.data_events_per_bin.histogram + a_bins) / (1. + p_calibration)
            A_bins = np.choose(u_bins == 0, [A_bins_2, A_bins_special])

            assert np.all(0 <= A_bins)
            pmfs[source_i] = A_bins * w_calibration
            pmfs[source_i] /= pmfs[source_i].sum()
            mus[source_i] = (A_bins * w_calibration).sum() * p_calibration

        return mus, pmfs

    def _compute_likelihood(self, mus, pmfs):
        """Return binned Poisson log likelihood
        :param mus: numpy array with expected rates for each source
        :param pmfs: array (sources, *analysis_space) of PMFs for each source in each bin
        """
        # sum_s mu_s * pmf_s, without modifying the inputs. einsum (no BLAS) rather than tensordot:
        # multi-threaded BLAS on arrays this small wastes all cores and can be much slower.
        expected_total = np.einsum('s,s...->...', mus, pmfs)
        observed_counts = self.data_events_per_bin.histogram

        if np.any(expected_total < 0):
            # scipy.stats.poisson.logpmf gives nan for negative expectations
            return np.nan

        # Poisson log pmf, k log(mu) - mu - log(k!), same conventions as scipy.stats.poisson.logpmf
        # (xlogy gives 0 for k = mu = 0, -inf for k > 0 = mu). sum log(k!) only depends on the
        # data; it is cached together with a copy of the counts, so that replacing or modifying
        # the data histogram in place is detected.
        cached = getattr(self, '_log_factorial_cache', None)
        if cached is None or not np.array_equal(cached[0], observed_counts):
            cached = (np.array(observed_counts, copy=True),
                      np.sum(special.gammaln(observed_counts + 1)))
            self._log_factorial_cache = cached
        return (np.sum(special.xlogy(observed_counts, expected_total))
                - np.sum(expected_total)
                - cached[1])


    def supports_gradient(self):
        """Whether loglikelihood_and_gradient can be used for this likelihood."""
        if self.model_statistical_uncertainty_handling is not None:
            return False
        return self._gradient_possible(BinnedLogLikelihood)

    @_needs_data
    def loglikelihood_and_gradient(self, livetime_days=None, **kwargs):
        """Return the log likelihood (same value as __call__) and its gradient.

        The gradient is a dict with an entry for every rate multiplier
        (<source>_rate_multiplier) and every shape parameter of this likelihood. For shape
        parameters it is the derivative of the interpolation (for multilinear interpolation
        one-sided at the anchors). Only for the configurations accepted by supports_gradient().

        With expected events lambda_b = sum_s M_s p_sb in bin b, where M_s is the rate of source s
        (interpolated rate x rate multiplier x live-time scale x efficiency) and p_sb its pmf:
            d ll / d lambda_b = n_b / lambda_b - 1
            d ll / d z = sum_b (d ll / d lambda_b) sum_s (dM_s/dz p_sb + M_s dp_sb/dz)
        """
        if not self.supports_gradient():
            raise NotImplementedError("Gradient not available for this likelihood configuration")
        rate_multipliers, shape_parameter_settings = self._kwargs_to_settings(**kwargs)
        n_sources = len(self.source_name_list)
        shape_names = list(self.shape_parameters.keys())
        zero_gradient = {sn + '_rate_multiplier': 0. for sn in self.source_name_list}
        zero_gradient.update({sn: 0. for sn in shape_names})

        # Interpolate rates and pmfs (as __call__), keeping what is needed for their derivatives
        if len(shape_names):
            zs = self._shape_point(shape_parameter_settings)
            if zs is None:
                return -float('inf'), zero_gradient
            if self.source_wise_interpolation:
                mus = np.array(self.base_model.expected_events(), dtype=float)
                ps = self.ps.copy()
                source_derivatives = []
                for interpolator in self.source_interpolators.values():
                    rate, pmf, rate_gradient, pmf_gradient = interpolator.values_and_gradients(zs)
                    mus[interpolator.source_index] = rate
                    ps[interpolator.source_index] = pmf
                    source_derivatives.append((interpolator, rate_gradient, pmf_gradient))
            else:
                rows, weights, weight_gradients = self.ps_interpolator.corner_weights(zs)
                ps = self.ps_interpolator.combine(rows, weights).reshape(self.ps.shape)
                mus_anchor = self.mus_interpolator.values[rows]  # (n_corners, n_sources)
                mus = weights @ mus_anchor
        else:
            ps = self.ps
            mus = np.asarray(self.base_model.expected_events(), dtype=float).copy()

        livetime_scale, factor, efficiencies = self._rate_factors(
            rate_multipliers, livetime_days, shape_parameter_settings)
        expected = mus * factor * efficiencies  # M_s, as mus in __call__

        if not np.all((expected >= 0) & (expected < float('inf'))):
            if self.config.get('unphysical_behaviour') == 'error':
                raise ValueError("Unphysical rates: %s" % str(expected))
            return -float('inf'), zero_gradient

        ll = self._compute_likelihood(expected, ps)
        if not np.isfinite(ll):
            # e.g. data in a bin without expected events; no meaningful gradient
            return ll, zero_gradient

        # d ll / d lambda_b = n_b / lambda_b - 1 (and -1 for empty bins with lambda_b = 0)
        observed_counts = self.data_events_per_bin.histogram
        expected_total = np.einsum('s,s...->...', expected, ps)
        with np.errstate(divide='ignore', invalid='ignore'):
            dll_dlambda = np.where(observed_counts > 0, observed_counts / expected_total, 0.) - 1.
        dll_dlambda_flat = dll_dlambda.ravel()
        # d ll / d M_s = sum_b dll/dlambda_b * p_sb
        a_s = np.einsum('sb,b->s', ps.reshape(n_sources, -1), dll_dlambda_flat)

        gradient = self._rate_gradients(a_s, mus, livetime_scale, efficiencies)

        if len(shape_names):
            if self.source_wise_interpolation:
                shape_gradient = np.zeros(len(shape_names))
                for interpolator, rate_gradient, pmf_gradient in source_derivatives:
                    # d ll / d z_k from this source, for the shape parameters it depends on
                    source_i = interpolator.source_index
                    scale = factor[source_i] * efficiencies[source_i]
                    pmf_term = np.einsum('kb,b->k', pmf_gradient.reshape(len(rate_gradient), -1),
                                         dll_dlambda_flat)
                    shape_gradient[interpolator.shape_indices] += scale * (
                        rate_gradient * a_s[source_i] + mus[source_i] * pmf_term)
            else:
                # d lambda_b / d z_k = sum_c dw_ck [ sum_s mus_cs f_s e_s p_sb + sum_s M_s P_csb ]
                pmf_size = ps[0].size
                per_corner = np.empty(len(rows))
                for c, row_i in enumerate(rows):
                    pmf_c = self.ps_interpolator.values[row_i].reshape(n_sources, pmf_size)
                    per_corner[c] = (np.dot(mus_anchor[c] * factor * efficiencies, a_s)
                                     + np.dot(expected, np.einsum('sb,b->s', pmf_c, dll_dlambda_flat)))
                shape_gradient = per_corner @ weight_gradients
            for k, setting_name in enumerate(shape_names):
                gradient[setting_name] = shape_gradient[k]

        self._add_efficiency_gradients(gradient, a_s, mus, factor)
        return ll, gradient

class SourceShapeInterpolator:
    """Rate and density of one source, interpolated in the shape parameters this source depends on.

    The density is the pmf over the analysis-space bins (binned likelihoods) or the pdf values at
    the events of the dataset (unbinned likelihoods, if events are given). Methods take the values
    of all shape parameters of the likelihood (zs) and pick out those of this source
    (shape_indices); gradients are with respect to this source's shape parameters, in the order of
    shape_indices.

    Everything is interpolated together, as one vector
        [pdf values at the events (unbinned only)..., pmf bins..., expected events, normalization]
    so that the grid cell and interpolation weights are found only once per evaluation.

    If the morpher's interpolation does not preserve sums (preserves_sums False, e.g.
    CubicHermiteInterpolator), the interpolated density is rescaled by N / S, where S is the sum of
    the interpolated pmf and N the interpolated normalization of the anchor pmfs. Pmfs normalized
    at all anchors thus stay normalized; for unbinned likelihoods the pdf values at the events are
    rescaled the same way (so the source must provide its pmf grid). With interpolations that
    preserve sums, unbinned likelihoods do not need the pmf grid at all.
    """

    def __init__(self, morpher, anchor_sources, source_index, shape_indices, events=None,
                 pmf_sum_cache=None):
        """
        :param morpher: morpher of this source (over its shape parameters only)
        :param anchor_sources: dict {anchor z-scores of this source: Source}
        :param source_index: index of the source in the likelihood's source list
        :param shape_indices: indices of this source's shape parameters among all shape parameters
        :param events: for unbinned likelihoods, the events in analysis-space coordinates
                       (a tuple of arrays, as passed to Source.pdf); None for binned likelihoods
        :param pmf_sum_cache: optional dict, kept by the caller between data sets of the same
                              anchor models. Unbinned likelihoods that renormalize only need the sum
                              of the interpolated pmf, which does not depend on the events: its
                              stored interpolation rows are computed once and kept here, so that a
                              new data set only interpolates the quantities that depend on the events
        """
        self.source_index = source_index
        self.shape_indices = list(shape_indices)
        self.events = events
        self.renormalize = not morpher.preserves_sums
        uses_pmf = events is None or self.renormalize
        example_source = next(iter(anchor_sources.values()))

        # Position of each quantity in the interpolated vector
        sizes = OrderedDict()
        if events is not None:
            sizes['pdf'] = len(events[0])
        if uses_pmf:
            self.pmf_shape = example_source.get_pmf_grid()[0].shape
            sizes['pmf'] = int(np.prod(self.pmf_shape))
        sizes['rate'] = 1
        if self.renormalize:
            sizes['normalization'] = 1
        pmf_sum_rows = None
        if events is not None and self.renormalize and pmf_sum_cache is not None:
            pmf_sum_rows = self._pmf_sum_rows(morpher, anchor_sources, sizes['pmf'], pmf_sum_cache)
        if pmf_sum_rows is not None:
            # Interpolate everything but the pmf, then add the stored rows of its sum
            self._slices, n_outputs = self._layout(
                OrderedDict((name, size) for name, size in sizes.items() if name != 'pmf'))
            self.interpolator = morpher.make_interpolator(
                f=self._anchor_vector, extra_dims=[n_outputs], anchor_models=anchor_sources)
            self.interpolator.insert_outputs(self._slices['pdf'].stop, pmf_sum_rows)
            sizes['pmf'] = 1
            self._slices, _ = self._layout(sizes)
        else:
            self._slices, n_outputs = self._layout(sizes)
            self.interpolator = morpher.make_interpolator(
                f=self._anchor_vector, extra_dims=[n_outputs], anchor_models=anchor_sources)
            if events is not None and self.renormalize and hasattr(self.interpolator, 'sum_outputs'):
                # Unbinned: only the sum of the interpolated pmf is needed (to renormalize the pdf
                # at the events), not the pmf itself. Interpolate just that sum, which is exactly
                # the sum of the interpolated pmf but costs O(1) instead of O(n_bins) per call.
                self.interpolator.sum_outputs(self._slices['pmf'])
                sizes['pmf'] = 1
                self._slices, _ = self._layout(sizes)
        self._last_evaluation = (None, None)  # (zs of this source, result of _evaluate)

    @staticmethod
    def _pmf_sum_rows(morpher, anchor_sources, n_bins, cache):
        """Stored interpolation rows of the sum of the pmf, or None if the morpher's interpolator
        cannot sum and insert outputs. Computed once per set of anchor models and kept in cache."""
        entry = cache.get('pmf_sum_rows')
        if entry is None or entry[0] is not anchor_sources:
            interpolator = morpher.make_interpolator(
                f=lambda source: source.get_pmf_grid()[0].ravel(), extra_dims=[n_bins],
                anchor_models=anchor_sources)
            if not (hasattr(interpolator, 'sum_outputs') and hasattr(interpolator, 'insert_outputs')):
                return None
            interpolator.sum_outputs(slice(0, n_bins))
            entry = cache['pmf_sum_rows'] = (anchor_sources, interpolator.values)
        return entry[1]

    @staticmethod
    def _layout(sizes):
        """Slices of the quantities in the interpolated vector, and its length."""
        slices, start = OrderedDict(), 0
        for name, size in sizes.items():
            slices[name] = slice(start, start + size)
            start += size
        return slices, start

    def _anchor_vector(self, source):
        parts, pmf = [], None
        if 'pdf' in self._slices:
            parts.append(source.pdf(*self.events))
        if 'pmf' in self._slices:
            pmf = source.get_pmf_grid()[0]
            parts.append(pmf.ravel())
        parts.append([source.expected_events])
        if 'normalization' in self._slices:
            if pmf is None:
                pmf = source.get_pmf_grid()[0]
            parts.append([pmf.sum()])
        return np.concatenate(parts)

    def _own_zs(self, zs):
        return np.asarray([zs[i] for i in self.shape_indices], dtype=float)

    def supports_gradient(self):
        return hasattr(self.interpolator, 'value_and_gradient')

    def _split(self, vector):
        """Return rate, density and, if renormalizing, pmf sum and normalization from an
        interpolated vector, or the same parts of its gradient (shape (k, n))."""
        def part(name):
            return vector[..., self._slices[name]]
        density = part('pdf') if self.events is not None else part('pmf')
        rate = part('rate')[..., 0]
        if not self.renormalize:
            return rate, density, None, None
        return rate, density, part('pmf').sum(axis=-1), part('normalization')[..., 0]

    def _shape_density(self, density):
        """The binned pmf has the shape of the analysis space."""
        if self.events is None:
            return density.reshape(density.shape[:-1] + self.pmf_shape)
        return density

    def _evaluate(self, zs):
        """(rate, density) at zs. The last result is kept, since the likelihood asks for the rates
        and the densities of all sources separately."""
        z = self._own_zs(zs)
        if self._last_evaluation[0] != tuple(z):
            rate, density, total, normalization = self._split(self.interpolator(z))
            if self.renormalize and total > 0:
                density = density * (normalization / total)
            self._last_evaluation = (tuple(z), (rate, self._shape_density(density)))
        return self._last_evaluation[1]

    def rate(self, zs):
        """Expected events of the source at zs."""
        return self._evaluate(zs)[0]

    def density(self, zs):
        """Pmf (binned) or pdf values at the events (unbinned) at zs, a new array."""
        return self._evaluate(zs)[1].copy()

    def values_and_gradients(self, zs):
        """Return rate, density and their gradients, shapes (k,) and (k, *density shape), at zs."""
        z = self._own_zs(zs)
        vector, vector_gradient = self.interpolator.value_and_gradient(z)
        rate, density, total, normalization = self._split(vector)
        rate_gradient, density_gradient, total_gradient, normalization_gradient = \
            self._split(vector_gradient)
        if self.renormalize and total > 0:
            # density * scale with scale = normalization / total: product and quotient rule
            scale = normalization / total
            scale_gradient = (normalization_gradient - scale * total_gradient) / total
            density_gradient = density_gradient * scale + scale_gradient[:, None] * density[None, :]
            density = density * scale
        return (rate, self._shape_density(density), rate_gradient,
                self._shape_density(density_gradient))

def extended_loglikelihood(mu, ps, outlier_likelihood=0.0):
    """Evaluate an extended unbinned likelihood function
    :param mu: array of n_sources: expected number of events
    :param ps: array of (n_sources, n_events): pdf value for each source and event
    :param outlier_likelihood: if an event has p=0, give it this likelihood (instead of 0, which makes the whole
    loglikelihood infinite)
    :return: ln(likelihood)
    """
    p_events = np.nansum(mu[:, np.newaxis] * ps, axis=0)
    if outlier_likelihood != 0:
        # Replace all likelihoods which are not positive numbers (i.e. 0, negative, or nan) with outlier_likelihood
        p_events[True ^ (p_events > 0)] = outlier_likelihood
    return -mu.sum() + np.sum(np.log(p_events))


def beeston_barlow_root1(a, p, U, d):
    """Solution to the Beeston-Barlow equations for a single finite-statics source and several infinite-statistics
    sources. This is the WRONG root, as far as we can tell -- DO NOT USE IT!!
    We retained it only to keep checking that it is the wrong root. It will be removed soon, when we are more confident.
    """
    return ((-U*p - U + a*p + d*p -
             np.sqrt(U**2*p**2 + 2*U**2*p + U**2 + 2*U*a*p**2 + 2*U*a*p -
                     2*U*d*p**2 - 2*U*d*p + a**2*p**2 + 2*a*d*p**2 + d**2*p**2))/(2*p*(p + 1)))


def beeston_barlow_root2(a, p, U, d):
    """Solution to the Beeston-Barlow equations for a single finite-statics source and several infinite-statistics
    sources. This is the 'right' root, as far as we can tell anyway."""
    return ((-U*p - U + a*p + d*p +
             np.sqrt(U**2*p**2 + 2*U**2*p + U**2 + 2*U*a*p**2 + 2*U*a*p -
                     2*U*d*p**2 - 2*U*d*p + a**2*p**2 + 2*a*d*p**2 + d**2*p**2))/(2*p*(p + 1)))


def beeston_barlow_roots(a, p, U, d):
    return beeston_barlow_root1(a, p, U, d), beeston_barlow_root2(a, p, U, d)


class LogLikelihoodReParam(object):
    """Class wrapper that allows reparameterization of likelihood.
    The conversion from old parameters to new parameters is required to provide in the conv_config."""

    def __init__(self, likelihood, conv_config):
        self.__likelihood = likelihood
        self.conv_config = conv_config
        self.check_conv_config()
        self.pdf_base_config = self.__likelihood.pdf_base_config

    def __call__(self, compute_pdf=False, livetime_days=None, **kwargs):
        kwargs = deepcopy(self._parameter_converter(**kwargs))
        ret = self.__likelihood(compute_pdf=compute_pdf,
                               livetime_days=livetime_days,
                               **kwargs)
        return ret

    def check_conv_config(self):
        """Make sure the new parameters are consistent"""
        conv_config = self.conv_config
        config = self.base_model.config

        new_params_1 = [k for k in conv_config.keys() if not k.endswith("_rate_multiplier")]
        new_params_2 = []

        for k, v in conv_config.items():
            if isinstance(v, dict):
                for p in v["params"]:
                    if p not in new_params_2:
                        new_params_2.append(p)

        # order doesn't matter
        assert set(new_params_1) == set(new_params_2), "New parameters are not consistent, double check conv_config..."

        # also check if new params are in config or not
        missing_params = ""
        for new_p in new_params_1:
            if not config.get(new_p, False):
                missing_params += f"{new_p}, "

        # remove the period if there are missing parameters..
        if missing_params != "":
            missing_params = missing_params[:-2]

        assert missing_params == "", f"{missing_params} are missing in the config"

    @property
    def rate_parameters(self):
        """remove rate multipliers in the original likelihood that are converted from shape parameters"""
        rate_parameters = deepcopy(self.__likelihood.rate_parameters)
        for k in self.__likelihood.rate_parameters.keys():
            if k+"_rate_multiplier" in self.conv_config.keys():
                rate_parameters.pop(k)

        return rate_parameters

    @property
    def shape_parameters(self):
        """add extra shape parameters from the conv_config"""
        shape_parameters = deepcopy(self.__likelihood.shape_parameters)
        for k, v in self.conv_config.items():
            if not k.endswith("_rate_multiplier"):
                anchors = {z: z for z in v[0]}
                shape_parameters[k] = (anchors, v[1], v[2])

        return shape_parameters

    @property
    def base_model(self):
        model = deepcopy(self.__likelihood.base_model)
        model.simulate = self._simulate
        return model

    def set_data(self, df):
        self.__likelihood.set_data(df)

    def get_bounds(self, parameter_name=None):
        """Return bounds on the parameter parameter_name"""
        if parameter_name is None:
            return [self.get_bounds(p) for p in self.shape_parameters.keys()]
        elif parameter_name in list(self.__likelihood.rate_parameters.keys()) + list(self.__likelihood.shape_parameters.keys()):
            return self.__likelihood.get_bounds(parameter_name)
        # in the newly added parameters
        else:
            anchor_settings = list(self.shape_parameters[parameter_name][0].keys())
            return min(anchor_settings), max(anchor_settings)

    def _simulate(self, kwargs=None, livetime_days=None):
        """simulate function including coupling parameters and rate parameters"""
        if kwargs is None:
            kwargs = dict()

        kwargs = deepcopy(self._parameter_converter(with_suffix=False, **kwargs))

        rate_multipliers = dict()
        for k, v in kwargs.items():
            if k in self.__likelihood.rate_parameters.keys():
                rate_multipliers[k] = v

        return self.__likelihood.base_model.simulate(rate_multipliers=rate_multipliers, livetime_days=livetime_days)

    def _parameter_converter(self, with_suffix=True, **kwargs):
        """
        convert new parameters to old parameters so that the old likelihood can use.
        :param conv_config: dict about how we do the conversion, in the form of:
        dict(op1=dict(params=[np0, np1, ...], func)
        :param kwargs: kwargs to be converted
        :return: converted kwargs
        """
        removed_params = []

        if not with_suffix:
            kwargs_copy = dict()
            for k, v in kwargs.items():
                if k in self.__likelihood.rate_parameters.keys():
                    kwargs_copy[k + "_rate_multiplier"] = v
                else:
                    kwargs_copy[k] = v

            kwargs = deepcopy(kwargs_copy)

        pass_kwargs = OrderedDict()

        for k, v in self.conv_config.items():
            # shape param -> rate param
            if k.endswith("_rate_multiplier"):
                base_value_s = [self.pdf_base_config.get(p) for p in v["params"]]
                params = [kwargs.get(p, base_value) for p, base_value in zip(v["params"], base_value_s)]
                pass_kwargs[k] = v["func"](*params) / v["func"](*base_value_s)

                # params converted into other params won't enter the original likelihood
                for p in v["params"]:
                    if p not in removed_params:
                        removed_params.append(p)

        # retain the rest
        for k, v in kwargs.items():
            if k not in removed_params:
                pass_kwargs[k] = v

        # remove suffix if without suffix
        if not with_suffix:
            pass_kwargs_copy = OrderedDict()
            for k, v in pass_kwargs.items():
                _name = k.split("_rate_multiplier")[0]
                pass_kwargs_copy[_name] = v

            pass_kwargs = deepcopy(pass_kwargs_copy)

        return pass_kwargs


class LogLikelihoodSum(object):
    """Class that takes a list of likelihoods to be minimized together, and
    provides an interface to the inference methods and evaluation similar to likelihoods.

    Note that the pfd_base_config is a bit of a fudge; only storing guesses from the last likelihood.
    As different guesses for different likelihoods should be a cause for concern, the safest method is to pass
    manual guesses to the minimization.

    The likelihood_weights allows you to weight terms in the likelihood-- useful for Asimov estimation (I think) as
    well as if you have multiple subvolumes, in which case each subvolume constraint term gets a (1/N) weight to
    avoid overconstraining the total llh
    """
    def __init__(self, likelihood_list, likelihood_weights=None):
        self.likelihood_list = []
        self.rate_parameters = dict()
        self.shape_parameters = dict()
        self.source_list = []       # DOES NOT EXIST IN LF!

        # in order to pass to confidence interval:
        self.pdf_base_config = {}   # might also have to be fudged

        # These weights are useful  if you need to split the constraints
        # among multiple sub-volumes in analysis space.
        self.likelihood_weights = likelihood_weights
        if likelihood_weights is None:
            self.likelihood_weights = [1 for _ in likelihood_list]
        self.likelihood_parameters = []

        for ll in likelihood_list:
            self.likelihood_list.append(ll)
            self.rate_parameters.update(ll.rate_parameters)
            self.shape_parameters.update(ll.shape_parameters)
            parameter_names = []

            for rate_parameter_name in ll.rate_parameters.keys():
                parameter_names.append(rate_parameter_name + '_rate_multiplier')
                base_value = ll.pdf_base_config.get(rate_parameter_name)
                if base_value is not None:
                    self.pdf_base_config[rate_parameter_name] = base_value
            for shape_parameter_name in ll.shape_parameters.keys():
                parameter_names.append(shape_parameter_name)
                base_value = ll.pdf_base_config.get(shape_parameter_name)
                if base_value is not None:
                    self.pdf_base_config[shape_parameter_name] = base_value
            self.likelihood_parameters.append(parameter_names)

    def __call__(self, compute_pdf=False, livetime_days=None, **kwargs):
        ret = 0.
        for i, (ll, parameter_names, ll_weight) in enumerate(
                zip(self.likelihood_list,
                    self.likelihood_parameters,
                    self.likelihood_weights)):
            pass_kwargs = {k: v
                           for k, v in kwargs.items()
                           if k in parameter_names}
            livetime = livetime_days
            if isinstance(livetime_days, list):
                livetime = livetime_days[i]
            ret += ll_weight * ll(compute_pdf=compute_pdf,
                                  livetime_days=livetime,
                                  **pass_kwargs)
        return ret

    def supports_gradient(self):
        """Whether every term provides loglikelihood_and_gradient."""
        return all(hasattr(ll, 'loglikelihood_and_gradient')
                   and getattr(ll, 'supports_gradient', lambda: True)()
                   for ll in self.likelihood_list)

    def loglikelihood_and_gradient(self, livetime_days=None, **kwargs):
        """Return the weighted sum of the terms' log likelihoods and of their gradients (dict)."""
        ret = 0.
        gradient = {}
        for i, (ll, parameter_names, ll_weight) in enumerate(
                zip(self.likelihood_list,
                    self.likelihood_parameters,
                    self.likelihood_weights)):
            pass_kwargs = {k: v
                           for k, v in kwargs.items()
                           if k in parameter_names}
            livetime = livetime_days
            if isinstance(livetime_days, list):
                livetime = livetime_days[i]
            value, term_gradient = ll.loglikelihood_and_gradient(livetime_days=livetime,
                                                                 **pass_kwargs)
            ret += ll_weight * value
            for k, g in term_gradient.items():
                gradient[k] = gradient.get(k, 0.) + ll_weight * g
        return ret, gradient

    def split_results(self, result_dict):
        ret = []
        for i,parameter_names in enumerate(self.likelihood_parameters):
            ret.append({k: v for k, v in result_dict.items() if k in parameter_names})
        return ret

    def get_bounds(self, parameter_name=None):
        """Return bounds on the parameter parameter_name,"""
        if parameter_name is None:
            return [self.get_bounds(p)
                    for p in self.shape_parameters]
        if parameter_name in self.shape_parameters.keys():
            bounds = []
            for ll in self.likelihood_list:
                if parameter_name in ll.shape_parameters.keys():
                    bounds.append(ll.get_bounds(parameter_name))
            bounds = np.array(bounds)
            ret= np.max(bounds[:,0]), np.min(bounds[:,1])
            if ret[1] <= ret[0]:
                raise InvalidParameterSpecification("lower bound %s higher than upper bound!" % parameter_name)
            return ret

        elif parameter_name.endswith('_rate_multiplier'):
            return 0, float('inf')
        else:
            raise InvalidParameter("Non-existing parameter %s" % parameter_name)


class LogAncillaryLikelihood(object):
    """Ancillary (constraint) analytical likelihoods"""

    def __init__(self, func, parameter_list, config=None, func_kwargs=None):
        """
        :param func: python function taking an _OrderedDict_ of (named) input values, plus func_kwargs extra arguments.
           i.e. func({parameters:config[parameter]}, **func_kwargs)
        :param parameter_list: list of names of parameters for which a dict is pulled from the config.
        :param config: df config containing default values for parameters
        :param func_kwargs: other parameters to pass to function
        """
        if config is None:
            config = dict()
        if func_kwargs is None:
            func_kwargs = dict()

        self.rate_parameters = dict()
        self.shape_parameters = dict()
        self.source_list = []    # DOES NOT EXIST IN LF!
        # in order to pass to confidence interval
        self.pdf_base_config = config    # might also have to be fudged

        self.func = func
        self.func_kwargs = func_kwargs
        for parameter_name in parameter_list:
            self.shape_parameters.update(OrderedDict([(parameter_name,(None,None,None))]))

    def get_bounds(self, parameter_name=None):
        if parameter_name is None:
            return [self.get_bounds(p) for p in self.shape_parameters]
        if parameter_name in self.shape_parameters.keys():
            # other likelihoods can be more constrictive.
            return -np.inf, np.inf
        else:
            raise InvalidParameter("Non-existing parameter %s" % parameter_name)

    def __call__(self, **kwargs):
        # Use an ordered dict here, so function can rely on order of arguments
        pass_kwargs = OrderedDict()
        for parameter_name in self.shape_parameters:
            pass_kwargs[parameter_name] = self.pdf_base_config[parameter_name]
        pass_kwargs.update(kwargs)

        return self.func(pass_kwargs, **self.func_kwargs)


# Add the inference methods from .inference
for methodname in inference.__all__:
    for q in (LogLikelihoodBase, LogLikelihoodSum, LogAncillaryLikelihood, LogLikelihoodReParam):
        setattr(q, methodname, getattr(inference, methodname))
