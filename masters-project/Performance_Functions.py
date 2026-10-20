"""
Performance functions shared by every script in this project.

ONE CONVENTION EVERYWHERE:
    g(x) takes an (n, d) array (n points in d dimensions) and returns an (n,) array.
    Failure  <=>  g(x) > YF.
    SuS and NSuS sample in standard normal space, so x ~ N(0, I_d).

To try a new function, write it here (or anywhere) following the convention and
pass it to subset_simulation, run_nsus, fit_gpe or ec_on_grid.
"""

import numpy as np
from scipy.stats import chi2, multivariate_normal, norm


def linear_sum(x):                                # one failure mode, any d
    return np.sum(x, axis=1)


def sum_of_squares(x):                            # failure outside a sphere, any d
    return np.sum(x ** 2, axis=1)


def two_gaussians(x):                             # two failure modes, d = 2
    return np.atleast_1d(0.3 * multivariate_normal.pdf(x, mean=[-5, 0])
                         + 0.7 * multivariate_normal.pdf(x, mean=[5, 0]))


def wave(x):                                      # 1D test function for the GPE
    return np.sin(3 * x[:, 0]) * np.cos(5 * x[:, 0]) - np.exp(-x[:, 0] ** 2 - 0.3)


def random_bumps(n_bumps=100, d=2, seed=None, low=0.1, high=0.9):
    """A random landscape: a sum of Gaussian bumps with centres in [low, high]^d."""
    rng = np.random.default_rng(seed)
    centres = rng.uniform(low, high, size=(n_bumps, d))
    heights = rng.uniform(-1, 1, size=n_bumps)
    widths = rng.uniform(0.15, 0.3, size=n_bumps)

    def g(x):
        sq_dist = np.sum((x[:, None, :] - centres) ** 2, axis=2)       # (n, n_bumps)
        return np.sum(heights * np.exp(-sq_dist / (2 * widths ** 2)), axis=1)
    return g


def true_pf(g, YF, d):
    """Exact failure probability where it is known, otherwise None."""
    if g is linear_sum:
        return norm.sf(YF / np.sqrt(d))
    if g is sum_of_squares:
        return chi2.sf(YF, d)
    return None


def pointwise(g):
    """Adapt g to take a single point (a tuple or 1D array), as Kinnear's NSuS needs."""
    return lambda x: float(g(np.asarray(x, dtype=float).reshape(1, -1))[0])
