"""Morphers: interpolate multidimensional functions of models
"""
import numpy as np
from scipy.interpolate import RegularGridInterpolator  # noqa: F401 (kept for importers)
from scipy.spatial import KDTree
from tqdm import tqdm

from .exceptions import NoShapeParameters
from .utils import arrays_to_grid, inherit_docstring_from, combine_dicts

__all__ = ['Morpher', 'GridInterpolator', 'RadialInterpolator', 'latin']


class Morpher(object):

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
    """

    def __init__(self, grid_arrays, anchor_scores):
        self.grid_arrays = [np.asarray(g, dtype=float) for g in grid_arrays]
        self.n_dims = len(self.grid_arrays)
        anchor_scores = np.asarray(anchor_scores, dtype=float)
        grid_shape = anchor_scores.shape[:self.n_dims]
        if tuple(len(g) for g in self.grid_arrays) != grid_shape:
            raise ValueError("Shape of anchor_scores %s does not match the grid %s"
                             % (anchor_scores.shape, [len(g) for g in self.grid_arrays]))
        for g in self.grid_arrays:
            if len(g) > 1 and not np.all(np.diff(g) > 0):
                raise ValueError("Grid points must be strictly ascending")
        self.output_shape = anchor_scores.shape[self.n_dims:]
        self.values = np.ascontiguousarray(anchor_scores.reshape(int(np.prod(grid_shape)), -1))
        # C-order strides (in rows of self.values) of the grid dimensions
        self.strides = np.array([int(np.prod(grid_shape[d + 1:])) for d in range(self.n_dims)], dtype=int)
        # all corners of the unit hypercube, shape (2**n_dims, n_dims)
        self.corners = np.array([[(i >> (self.n_dims - 1 - d)) & 1 for d in range(self.n_dims)]
                                 for i in range(2 ** self.n_dims)], dtype=int)
        self.corner_offsets = self.corners @ self.strides

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
        z = np.asarray(z, dtype=float).ravel()
        if len(z) != self.n_dims:
            raise ValueError("Expected a point with %d coordinates, got %s" % (self.n_dims, z))
        lower = np.zeros(self.n_dims, dtype=int)
        frac = np.zeros(self.n_dims)
        inv_width = np.zeros(self.n_dims)
        for d, (g, x) in enumerate(zip(self.grid_arrays, z)):
            if not g[0] <= x <= g[-1]:
                raise ValueError("Point %s is out of bounds in dimension %d" % (z, d))
            if len(g) == 1:
                continue
            i = min(max(int(np.searchsorted(g, x, side='right')) - 1, 0), len(g) - 2)
            lower[d] = i
            inv_width[d] = 1. / (g[i + 1] - g[i])
            frac[d] = (x - g[i]) * inv_width[d]

        # per-dimension weights (frac or 1 - frac) and their derivatives for all corners at once
        upper = self.corners == 1
        w_d = np.where(upper, frac, 1. - frac)
        dw_d = np.where(upper, inv_width, -inv_width)
        weights = np.prod(w_d, axis=1)
        if not gradient:
            keep = weights != 0
            return int(np.dot(lower, self.strides)) + self.corner_offsets[keep], weights[keep], None
        weight_gradients = np.empty((len(self.corners), self.n_dims))
        for k in range(self.n_dims):
            others = w_d.copy()
            others[:, k] = 1.
            weight_gradients[:, k] = dw_d[:, k] * np.prod(others, axis=1)
        # drop corners that contribute nothing (incl. those beyond single-anchor dimensions)
        keep = (weights != 0) | np.any(weight_gradients != 0, axis=1)
        rows = int(np.dot(lower, self.strides)) + self.corner_offsets[keep]
        return rows, weights[keep], weight_gradients[keep]

    def __call__(self, z):
        rows, weights, _ = self.corner_weights(z, gradient=False)
        return self.combine(rows, weights).reshape(self.output_shape)

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
            # all weights zero cannot happen for points inside the grid; keep the shape anyway
            result = np.zeros(self.values.shape[1])
        return result


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
        # Allocate an array which will hold the scores at each anchor model
        anchor_scores = np.zeros(list(self.anchor_z_grid.shape)[:-1] + extra_dims)

        # Iterate over the anchor grid points
        for anchor_grid_index, _zs in self._anchor_grid_iterator():
            # Compute f at this point, and store it in anchor_scores
            anchor_scores[tuple(anchor_grid_index + [slice(None)] * len(extra_dims))] = \
                f(anchor_models[tuple(_zs)])

        return MultilinearGridInterpolator(self.anchor_z_arrays, anchor_scores)

    def _anchor_grid_iterator(self):
        """Iterates over the anchor grid, yielding index, z-values"""
        fake_grid = np.zeros(list(self.anchor_z_grid.shape)[:-1])
        it = np.nditer(fake_grid, flags=['multi_index'])
        while not it.finished:
            anchor_grid_index = list(it.multi_index)
            yield (anchor_grid_index,
                   tuple(self.anchor_z_grid[tuple(anchor_grid_index + [slice(None)])]))
            it.iternext()


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
