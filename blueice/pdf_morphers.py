"""Morphers: interpolate multidimensional functions of models
"""
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.interpolate import RegularGridInterpolator  # noqa: F401 (kept for importers)
from scipy.spatial import KDTree
from tqdm import tqdm

from .exceptions import NoShapeParameters
from .utils import arrays_to_grid, inherit_docstring_from, combine_dicts

__all__ = ['Morpher', 'GridInterpolator', 'CubicHermiteInterpolator', 'RadialInterpolator', 'latin']


class Morpher(object):

    # Whether the interpolation preserves sums over its outputs, e.g. the normalization of
    # interpolated pdfs. If not, source-wise interpolation renormalizes (SourceShapeInterpolator).
    preserves_sums = True

    def __init__(self, config, shape_parameters):
        """Initialize the morpher, telling it which shape_parameters we're going to use
        See model for format of shape_parameters
        """
        self.config = config
        self.shape_parameters = shape_parameters
        if not len(self.shape_parameters):
            raise NoShapeParameters("Attempt to initialize a morpher without shape parameters")

    def get_anchor_points(self, bounds, n_models=None):
        """Returns list of anchor z-coordinates at which we should sample n_models between bounds.
        The morpher may choose to ignore your bounds and n_models argument if it doesn't support them.
        """
        raise NotImplementedError

    def make_interpolator(self, f, extra_dims, anchor_models):
        """Return a function which interpolates the extra_dims-valued function f(model)
        between the anchor points.
        :param f: Function which takes a Model as argument, and produces an extra_dims shaped array.
        :param extra_dims: tuple of integers, shape of return value of f.
        :param anchor_models: dictionary {z-score: Model} of anchor models at which to evaluate f.
        """
        raise NotImplementedError


class MultilinearGridInterpolator(object):
    """Multilinear interpolation of array-valued scores on a regular grid of anchor points.

    Equivalent to ``RegularGridInterpolator(grid_arrays, anchor_scores)(z)[0]`` for a single point z
    (method 'linear', bounds_error=True), but much faster for the typical blueice use: few shape
    dimensions and a large output array (e.g. sources x bins). The anchor scores are stored as a
    contiguous (n_anchor_points, n_outputs) array; a call finds the grid cell, computes the 2**n_dims
    corner weights and accumulates the weighted corner rows, skipping corners with zero weight.

    The weights at any point sum to one and are the same for every output, so sums over the
    outputs (e.g. the normalization of a pmf) are interpolated exactly: preserves_sums is True.
    """

    preserves_sums = True

    def __init__(self, grid_arrays, anchor_scores):
        self.grid_arrays = [np.asarray(g, dtype=float) for g in grid_arrays]
        self.n_dims = len(self.grid_arrays)
        anchor_scores = np.asarray(anchor_scores, dtype=float)
        self.grid_shape = anchor_scores.shape[:self.n_dims]
        if tuple(len(g) for g in self.grid_arrays) != self.grid_shape:
            raise ValueError("Shape of anchor_scores %s does not match the grid %s"
                             % (anchor_scores.shape, [len(g) for g in self.grid_arrays]))
        for g in self.grid_arrays:
            if len(g) > 1 and not np.all(np.diff(g) > 0):
                raise ValueError("Grid points must be strictly ascending")
        self.output_shape = anchor_scores.shape[self.n_dims:]
        self.n_anchor_points = int(np.prod(self.grid_shape))
        self.values = np.ascontiguousarray(anchor_scores.reshape(self.n_anchor_points, -1))
        # C-order strides (in rows of self.values) of the grid dimensions
        self.strides = np.array([int(np.prod(self.grid_shape[d + 1:])) for d in range(self.n_dims)],
                                dtype=int)
        # all corners of the unit hypercube, shape (2**n_dims, n_dims)
        self.corners = np.array([[(i >> (self.n_dims - 1 - d)) & 1 for d in range(self.n_dims)]
                                 for i in range(2 ** self.n_dims)], dtype=int)
        self.corner_offsets = self.corners @ self.strides

    def _locate(self, z):
        """Return the grid cell containing z: index of its lower anchor, the fractional position
        within the cell and the cell width, per dimension.

        At an interior anchor the cell to the right is used, at the upper edge the last cell.
        Dimensions with a single anchor get lower index 0, fraction 0 and width 0.
        """
        z = np.asarray(z, dtype=float).ravel()
        if len(z) != self.n_dims:
            raise ValueError("Expected a point with %d coordinates, got %s" % (self.n_dims, z))
        lower = np.zeros(self.n_dims, dtype=int)
        frac = np.zeros(self.n_dims)
        width = np.zeros(self.n_dims)
        for d, (g, x) in enumerate(zip(self.grid_arrays, z)):
            if not g[0] <= x <= g[-1]:
                raise ValueError("Point %s is out of bounds in dimension %d" % (z, d))
            if len(g) == 1:
                continue
            i = min(max(int(np.searchsorted(g, x, side='right')) - 1, 0), len(g) - 2)
            lower[d] = i
            width[d] = g[i + 1] - g[i]
            frac[d] = (x - g[i]) / width[d]
        return lower, frac, width

    @staticmethod
    def _product_weights(factors, factor_derivatives):
        """Weights that are products of one factor per dimension, and their derivatives.

        :param factors: (n_terms, n_dims) array, factors[j, d] is the factor of term j in dimension d
        :param factor_derivatives: same shape, derivatives of the factors w.r.t. z_d
        :return: weights (n_terms), weight_gradients (n_terms, n_dims), by the product rule
        """
        weights = np.prod(factors, axis=1)
        weight_gradients = np.empty(factors.shape)
        for k in range(factors.shape[1]):
            others = factors.copy()
            others[:, k] = 1.
            weight_gradients[:, k] = factor_derivatives[:, k] * np.prod(others, axis=1)
        return weights, weight_gradients

    def corner_weights(self, z, gradient=True):
        """Return the anchor rows, weights and weight derivatives used to interpolate at z.

        The interpolated value is sum_c weights[c] * self.values[rows[c]], and its derivative
        with respect to z_k is sum_c weight_gradients[c, k] * self.values[rows[c]] (the one-sided
        derivative in the cell to the right of z at an interior anchor, to the left at the upper
        edge of the grid). Corners whose weight and weight derivatives are all zero are omitted.

        :param gradient: if False, skip the weight derivatives (returns None for them) and drop
                         all zero-weight corners
        :return: rows (int array, n_corners), weights (n_corners), weight_gradients
                 (n_corners, n_dims)
        """
        lower, frac, width = self._locate(z)
        with np.errstate(divide='ignore'):
            inv_width = np.where(width > 0, 1. / width, 0.)

        # per-dimension weights (frac or 1 - frac) and their derivatives for all corners at once
        upper = self.corners == 1
        factors = np.where(upper, frac, 1. - frac)
        if not gradient:
            weights = np.prod(factors, axis=1)
            keep = weights != 0
            return int(np.dot(lower, self.strides)) + self.corner_offsets[keep], weights[keep], None
        factor_derivatives = np.where(upper, inv_width, -inv_width)
        weights, weight_gradients = self._product_weights(factors, factor_derivatives)
        # drop corners that contribute nothing (incl. those beyond single-anchor dimensions)
        keep = (weights != 0) | np.any(weight_gradients != 0, axis=1)
        rows = int(np.dot(lower, self.strides)) + self.corner_offsets[keep]
        return rows, weights[keep], weight_gradients[keep]

    def __call__(self, z):
        rows, weights, _ = self.corner_weights(z, gradient=False)
        return self.combine(rows, weights).reshape(self.output_shape)

    def value_and_gradient(self, z):
        """Return the interpolated value at z and its derivatives, shape (n_dims, *output_shape)."""
        rows, weights, weight_gradients = self.corner_weights(z)
        value = self.combine(rows, weights).reshape(self.output_shape)
        gradient = np.stack([self.combine(rows, weight_gradients[:, k])
                             for k in range(self.n_dims)])
        return value, gradient.reshape((self.n_dims,) + self.output_shape)

    def combine(self, rows, weights):
        """sum_c weights[c] * self.values[rows[c]], skipping zero weights (flat array)."""
        result = None
        for row_i, w in zip(rows, weights):
            if w == 0:
                continue
            row = self.values[row_i]
            if result is None:
                result = w * row  # fresh array: callers may modify the result in place
                buffer = np.empty_like(result)
            else:
                # result += w * row without temporaries. Plain ufuncs rather than BLAS (daxpy): BLAS
                # calls this small get multi-threaded and thrash, esp. with numpy and scipy each
                # shipping their own OpenBLAS.
                np.multiply(row, w, out=buffer)
                np.add(result, buffer, out=result)
        if result is None:
            # e.g. a derivative that vanishes in this cell
            result = np.zeros(self.values.shape[1])
        return result


def pchip_slopes(grid, values, axis=0):
    """Slopes at the grid points of the PCHIP interpolant of values along axis.

    PCHIP (Fritsch-Carlson, as scipy.interpolate.PchipInterpolator) chooses the slopes such that the
    cubic Hermite interpolant is monotone between neighbouring grid points: it never overshoots
    the values at the grid points. Each output (e.g. each bin) gets its own slopes.
    """
    grid = np.asarray(grid, dtype=float)
    if len(grid) == 1:
        return np.zeros_like(values)
    return PchipInterpolator(grid, values, axis=axis).derivative()(grid)


class CubicHermiteGridInterpolator(MultilinearGridInterpolator):
    """C1-smooth interpolation on a regular grid: tensor-product cubic Hermite with PCHIP slopes.

    In one dimension, between neighbouring anchors z_i and z_i+1 = z_i + h, with t = (z - z_i) / h:

        f(z) = h00(t) f_i + h01(t) f_i+1 + h * (h10(t) f'_i + h11(t) f'_i+1)

    with the cubic Hermite basis functions h00 = 2t^3 - 3t^2 + 1, h01 = -2t^3 + 3t^2,
    h10 = t^3 - 2t^2 + t, h11 = t^3 - t^2 and slopes f' at the anchors from PCHIP. The interpolant
    passes through the anchor values, its derivative is continuous (no kinks at the anchors, unlike
    multilinear interpolation), and it is monotone between anchors, so non-negative anchor scores
    give non-negative interpolated scores.

    In several dimensions the tensor product needs, at every anchor, the derivatives for every
    combination of derivative orders (0 or 1 per dimension): the value, d/dz_0, d/dz_1,
    d2/dz_0dz_1, ... They are computed once, by applying the PCHIP slope rule along each
    differentiated dimension in turn, and stored in self.values: block j of n_anchor_points rows
    holds the derivative orders self.derivative_orders[j] (block 0 holds the anchor values).
    The interpolated score is then a weighted sum of 4**n_dims rows (2**n_dims cell corners x
    2**n_dims derivative orders), which gives the same corner_weights interface (including weight
    derivatives for analytic gradients) as MultilinearGridInterpolator. With more than one
    dimension, the interpolant is still C1 but no longer guaranteed to be non-negative.

    PCHIP slopes are a non-linear function of the scores, chosen for each output separately, so
    the slopes of a sum over outputs are not the sum of the slopes: sums over the outputs (e.g. the
    normalization of a pmf) are not preserved between anchors (preserves_sums is False).
    """

    preserves_sums = False

    def __init__(self, grid_arrays, anchor_scores):
        super().__init__(grid_arrays, anchor_scores)
        scores = self.values.reshape(self.grid_shape + (-1,))
        # derivative orders, one 0/1 per dimension; same enumeration as the cell corners
        self.derivative_orders = self.corners.copy()
        blocks = []
        for orders in self.derivative_orders:
            block = scores
            for d, order in enumerate(orders):
                if order:
                    block = pchip_slopes(self.grid_arrays[d], block, axis=d)
            blocks.append(block.reshape(self.n_anchor_points, -1))
        self.values = np.ascontiguousarray(np.concatenate(blocks))

        # Every term of the interpolation is a (derivative orders, cell corner) combination, in
        # the row order of self.values: its derivative orders, its corner and its row offset
        # from the lower corner of the cell
        n_blocks, n_corners = len(self.derivative_orders), len(self.corners)
        self._term_orders = np.repeat(self.derivative_orders, n_corners, axis=0)
        self._term_corners = np.tile(self.corners, (n_blocks, 1))
        self._term_row_offsets = (np.repeat(np.arange(n_blocks), n_corners) * self.n_anchor_points
                                  + np.tile(self.corner_offsets, n_blocks))

    @staticmethod
    def _hermite_basis(t, h):
        """Hermite basis at fractional position t in a cell of width h, and derivatives w.r.t. z.

        :return: two (2, 2) arrays indexed [derivative order of the anchor quantity, corner]:
                 [[h00, h01], [h * h10, h * h11]] and its derivative with respect to z
        """
        t2, t3 = t * t, t * t * t
        basis = np.array([[2 * t3 - 3 * t2 + 1, -2 * t3 + 3 * t2],
                          [h * (t3 - 2 * t2 + t), h * (t3 - t2)]])
        # d/dz = (1 / h) d/dt
        basis_derivative = np.array([[(6 * t2 - 6 * t) / h, (-6 * t2 + 6 * t) / h],
                                     [3 * t2 - 4 * t + 1, 3 * t2 - 2 * t]])
        return basis, basis_derivative

    @inherit_docstring_from(MultilinearGridInterpolator)
    def corner_weights(self, z, gradient=True):
        lower, frac, width = self._locate(z)
        orders, corners = self._term_orders, self._term_corners
        factors = np.empty(orders.shape)
        factor_derivatives = np.empty(orders.shape)
        for d in range(self.n_dims):
            if width[d] == 0:
                # single anchor in this dimension: only the anchor value contributes
                basis, basis_derivative = np.array([[1., 0.], [0., 0.]]), np.zeros((2, 2))
            else:
                basis, basis_derivative = self._hermite_basis(frac[d], width[d])
            factors[:, d] = basis[orders[:, d], corners[:, d]]
            factor_derivatives[:, d] = basis_derivative[orders[:, d], corners[:, d]]
        if not gradient:
            weights = np.prod(factors, axis=1)
            keep = weights != 0
            rows = int(np.dot(lower, self.strides)) + self._term_row_offsets[keep]
            return rows, weights[keep], None
        weights, weight_gradients = self._product_weights(factors, factor_derivatives)
        rows = int(np.dot(lower, self.strides)) + self._term_row_offsets
        keep = (weights != 0) | np.any(weight_gradients != 0, axis=1)
        return rows[keep], weights[keep], weight_gradients[keep]


class GridInterpolator(Morpher):

    @inherit_docstring_from(Morpher)
    def __init__(self, config, shape_parameters):
        super().__init__(config, shape_parameters)
        # Compute the regular grid of anchor models at the specified anchor points
        self.anchor_z_arrays = [np.array(list(sorted(anchors.keys())))
                                for setting_name, (anchors, _, _) in shape_parameters.items()]
        self.anchor_z_grid = arrays_to_grid(self.anchor_z_arrays)

    @inherit_docstring_from(Morpher)
    def get_anchor_points(self, bounds, n_models=None):
        return [zs for _, zs in self._anchor_grid_iterator()]

    @inherit_docstring_from(Morpher)
    def make_interpolator(self, f, extra_dims, anchor_models):
        return MultilinearGridInterpolator(self.anchor_z_arrays,
                                           self._anchor_scores(f, extra_dims, anchor_models))

    def _anchor_scores(self, f, extra_dims, anchor_models):
        """Return f evaluated at all anchor models, shape (*anchor grid shape, *extra_dims)."""
        extra_dims = list(extra_dims)
        # Allocate an array which will hold the scores at each anchor model
        anchor_scores = np.zeros(list(self.anchor_z_grid.shape)[:-1] + extra_dims)

        # Iterate over the anchor grid points
        for anchor_grid_index, _zs in self._anchor_grid_iterator():
            # Compute f at this point, and store it in anchor_scores
            anchor_scores[tuple(anchor_grid_index + [slice(None)] * len(extra_dims))] = \
                f(anchor_models[tuple(_zs)])
        return anchor_scores

    def _anchor_grid_iterator(self):
        """Iterates over the anchor grid, yielding index, z-values"""
        fake_grid = np.zeros(list(self.anchor_z_grid.shape)[:-1])
        it = np.nditer(fake_grid, flags=['multi_index'])
        while not it.finished:
            anchor_grid_index = list(it.multi_index)
            yield (anchor_grid_index,
                   tuple(self.anchor_z_grid[tuple(anchor_grid_index + [slice(None)])]))
            it.iternext()


class CubicHermiteInterpolator(GridInterpolator):
    """C1-smooth morpher on the same anchor grid as GridInterpolator.

    Interpolates with CubicHermiteGridInterpolator (tensor-product cubic Hermite with PCHIP
    slopes) instead of multilinearly, so the likelihood has no kinks at the anchors, and analytic
    gradients and gradient-based minimizers behave well. In a binned likelihood it requires
    source_wise_interpolation, which also keeps the interpolated pmfs normalized.
    """

    preserves_sums = False

    @inherit_docstring_from(Morpher)
    def make_interpolator(self, f, extra_dims, anchor_models):
        return CubicHermiteGridInterpolator(self.anchor_z_arrays,
                                            self._anchor_scores(f, extra_dims, anchor_models))


class RadialInterpolator(Morpher):
    """This morpher is highly experimental!!"""

    @inherit_docstring_from(Morpher)
    def __init__(self, config, shape_parameters):
        defaults = dict(r_sample_points=5,
                        hypercube_shuffle_steps=500,
                        decay_response_to_density='constant')
        config = combine_dicts(defaults, config)
        super().__init__(config, shape_parameters)

    @inherit_docstring_from(Morpher)
    def get_anchor_points(self, bounds, n_models=10):
        # Sample a Latin hypercube of models
        zs_list = latin(n_models, len(self.shape_parameters), box=bounds,
                        shuffle_steps=self.config['hypercube_shuffle_steps'])
        zs_list = list(map(tuple, zs_list))

        # Get the bounds needed to scale the zs
        bounds = np.array(bounds)
        self._mins = bounds[:, 0]
        self._lengths = bounds[:, 1] - bounds[:, 0]

        # Rescale the zs to the bounds. It's fine if the zs are outside the bounds, but we need something
        # to scale the different dimensions to similar ranges so norms make sense.
        # Notice zs_list is redefined here to be the list of zs of *all* models, not just the new ones
        self._normed_model_zs = [(np.array(_zs) - self._mins) / self._lengths for _zs in zs_list]

        # Get the average distance to the five closest points
        self._r0s = KDTree(self._normed_model_zs).query(self._normed_model_zs,
                                                        self.config['r_sample_points'])[0].mean(axis=1)
        decay_response = self.config['decay_response_to_density']
        if decay_response == 'constant':
            self._r0s = np.ones_like(self._r0s) * self._r0s.mean()
        elif decay_response == 'proportional':
            pass
        else:
            raise NotImplementedError(decay_response)

        return zs_list

    @inherit_docstring_from(Morpher)
    def make_interpolator(self, f, extra_dims, anchor_models):
        anchor_scores = np.array([f(m) for m in anchor_models.values()])

        def interpolator(zs):
            # Compute the distance between the current point and each model
            normed_zs = (zs - self._mins) / self._lengths
            # print("Normed zs for this point: ", normed_zs)
            rs = np.sqrt([np.dot(normed_zs - _nzs, normed_zs - _nzs)
                          for _nzs in self._normed_model_zs])
            # print("Distances to models: ", rs)

            # Compute the weight of each model: exponential decay
            # Note we use a normalized exponential, so models with small radius of influence (i.e. in dense regions)
            # should have a higher weight when we get close to them than models with a large radius of influence.
            r_of_influence = self._r0s * self.config.get('decay_multiplier', 5)
            weights = np.exp(-rs / r_of_influence) / r_of_influence

            weights /= np.sum(weights)
            # print("Weights of models: ", weights)
            # print("Data scores at models: ", anchor_scores)

            return np.average(anchor_scores, weights=weights, axis=0)

        return interpolator


def latin(n, d, box=None, shuffle_steps=500):
    """Creates a latin hypercube of n points in d dimensions
    Stolen from https://github.com/paulknysh/blackbox
    """
    # starting with diagonal shape
    pts = np.ones((n, d))

    for i in range(n):
        pts[i] = pts[i] * i / (n-1.)

    # spread function
    def spread(p):
        s = 0.
        for i in range(n):
            for j in range(n):
                if i > j:
                    s = s + 1. / np.linalg.norm(np.subtract(p[i], p[j]))
        return s

    # minimizing spread function by shuffling
    currminspread = spread(pts)

    for m in tqdm(range(shuffle_steps), desc='Shuffling latin hypercube'):

        p1 = np.random.randint(n)
        p2 = np.random.randint(n)
        k = np.random.randint(d)

        newpts = np.copy(pts)
        newpts[p1, k], newpts[p2, k] = newpts[p2, k], newpts[p1, k]
        newspread = spread(newpts)

        if newspread < currminspread:
            pts = np.copy(newpts)
            currminspread = newspread

    if box is None:
        return pts

    for i in range(len(box)):
        pts[:, i] = box[i][0] + pts[:, i] * (box[i][1] - box[i][0])

    return pts
