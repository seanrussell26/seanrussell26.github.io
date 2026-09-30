"""
Subset Simulation for a Linear Reliability Problem -- VECTORISED version

Python translation of the MATLAB code by K.M. Zuev (Institute for Risk &
Uncertainty, University of Liverpool).

    Performance function: g(x) = x1 + ... + xd
    Input variables x1, ..., xd are i.i.d. N(0, 1)
    Failure  <=>  g(x) > YF

This version runs the same algorithm as the MATLAB script, but replaces
the loops over samples, chains and dimensions with NumPy array operations,
so it is about 10x faster. Use it for experiments; see
subset_simulation_loops.py for the line-by-line MATLAB version.

The performance function is defined ONCE, in `performance_function` below.
In the MATLAB script `sum(...)` was written out separately wherever g was
evaluated (lines 19 and 62, and also sum(q) on line 46). Here every
evaluation calls the same function, so to try a new model you only edit it
in one place.

HOW TO RUN: set the parameters just below, then run this file. The results
are printed, and when d = 2 the level plot (in the style of Figure 6) opens in a window.
Works for any dimension d; the plot is only drawn for d = 2.
"""

import time

import numpy as np
from scipy.stats import norm

# ===========================================================================
# PARAMETERS -- edit these (original MATLAB values in brackets)
# ===========================================================================
d = 2          # dimension of the input space            [MATLAB: 1000]
YF = 9.0       # critical threshold (failure <=> g(x) > YF) [MATLAB: 200]
n = 1000       # number of samples per level             [MATLAB: 3000]
p = 0.1        # level probability                       [MATLAB: 0.1]
SEED = None    # random seed: an integer for repeatable results, None for random
SAVE_PLOT = None   # e.g. "levels.png" to also save the plot to a file

# ---------------------------------------------------------------------------
# Performance function -- the ONLY place the model is defined.
# ---------------------------------------------------------------------------
def performance_function(x):
    """System response y = g(x) = x1 + ... + xd.

    `x` has shape (d, n): one column per sample, as in the MATLAB code.
    Returns an array of shape (n,).
    """
    return np.sum(x, axis=0)



# ---------------------------------------------------------------------------
# Subset simulation
# ---------------------------------------------------------------------------
def subset_simulation(d, YF, n=3000, p=0.1, g=performance_function, rng=None):
    """Estimate pF = P(g(x) > YF) for x ~ N(0, I_d).

    Parameters
    ----------
    d   : dimension of the input space
    YF  : critical threshold (failure <=> g(x) > YF)
    n   : number of samples per level
    p   : level probability
    g   : performance function taking a (d, n) array and returning shape (n,)
    rng : seed or np.random.Generator

    Returns
    -------
    dict with keys
        'pF_SS' : subset simulation estimate of the failure probability
        'N'     : total number of samples (performance function evaluations)
        'L'     : number of conditional levels used
        'Y'     : intermediate thresholds Y_1, ..., Y_L
        'nF'    : number of failure samples at each level 0, ..., L
        'x'     : list of (d, n) sample arrays, one per level 0, ..., L
        'y'     : list of (n,) response arrays, one per level 0, ..., L
    """
    rng = np.random.default_rng(rng)
    nc = int(round(n * p))                       # number of Markov chains
    ns = int(round((1 - p) / p))                 # number of states in each chain
    if nc * (ns + 1) != n:
        raise ValueError("n*p and 1/p must be integers")

    # --- Level 0: Monte Carlo ------------------------------------------------
    L = 0                                        # current (unconditional) level
    x = [rng.standard_normal((d, n))]            # Monte Carlo samples
    y = [g(x[0])]                                # system response y = g(x)
    nF = [int(np.sum(y[0] > YF))]                # number of failure samples
    Y = []

    while nF[L] / n < p:                         # stopping criterion
        L += 1                                   # next conditional level is needed
        ind = np.argsort(-y[L - 1])              # sort 'descend'
        y[L - 1] = y[L - 1][ind]                 # renumbered responses
        x[L - 1] = x[L - 1][:, ind]              # renumbered samples
        Y.append((y[L - 1][nc - 1] + y[L - 1][nc]) / 2)  # L-th intermediate threshold

        # --- Modified Metropolis algorithm for sampling from pi(x | F_L) ---
        # z has shape (d, nc, ns+1): all nc chains are advanced together,
        # which is equivalent to the MATLAB loop over j = 1..nc.
        z = np.empty((d, nc, ns + 1))
        z[:, :, 0] = x[L - 1][:, :nc]            # Markov chain "seeds"
        for m in range(ns):
            # Step 1: component-wise proposal
            a = z[:, :, m] + rng.standard_normal((d, nc))          # Step 1(a)
            r = np.minimum(1, norm.pdf(a) / norm.pdf(z[:, :, m]))  # Step 1(b)
            q = np.where(rng.random((d, nc)) < r, a, z[:, :, m])   # Step 1(c)
            # Step 2: accept q only if it belongs to F_L
            in_FL = g(q) > Y[L - 1]
            z[:, :, m + 1] = np.where(in_FL, q, z[:, :, m])

        # samples from pi(x | F_L), ordered chain by chain as in the MATLAB code
        x.append(z.reshape(d, nc * (ns + 1)))

        y.append(g(x[L]))                        # system response y = g(x)
        nF.append(int(np.sum(y[L] > YF)))        # number of failure samples at level L

    pF_SS = p ** L * nF[L] / n                   # SS estimate
    N = n + n * (1 - p) * L                      # total number of samples

    return {"pF_SS": pF_SS, "N": int(round(N)), "L": L, "Y": Y,
            "nF": nF, "x": x, "y": y}


# ---------------------------------------------------------------------------
# 2D plot of the levels (in the style of Figure 6)
# ---------------------------------------------------------------------------
# One colour per level, in the same spectrum order as Figure 6:
# red (Monte Carlo), magenta, blue, green, cyan, olive, lime, yellow, navy,
# orange-red. If there are more levels than colours, the list starts again.
LEVEL_COLOURS = ["#ff0000", "#e000e0", "#0000ff", "#007f00", "#00c0c0",
                 "#b0b000", "#00e000", "#ffff00", "#1f2f7f", "#e03000"]


def _point_nearest(cs, target, lo, hi):
    """Point on the contour line(s) in `cs` closest to `target`, keeping
    away from the plot edges (between lo and hi) where possible."""
    pts = [seg for seg in cs.allsegs[0] if len(seg)]
    if not pts:
        return None                            # curve not visible in the plot
    pts = np.vstack(pts)
    inside = np.all((pts > lo) & (pts < hi), axis=1)
    if inside.any():
        pts = pts[inside]
    return pts[np.argmin(np.sum((pts - target) ** 2, axis=1))]


def plot_levels_2d(result, YF, g=performance_function, lim=None,
                   filename=None, legend=True):
    """Plot the samples of every level (one colour per level), the
    intermediate thresholds g(x) = Y_L as dashed lines labelled F_1, F_2, ...
    and the failure boundary g(x) = YF as a solid line labelled F."""
    import matplotlib.pyplot as plt

    x, Y = result["x"], result["Y"]
    if x[0].shape[0] != 2:
        raise ValueError("plot_levels_2d needs d = 2")

    # Axis range: a whole-number square that contains every sample
    if lim is None:
        allx = np.hstack(x)
        lim = (np.floor(allx.min() - 0.2), np.ceil(allx.max() + 0.2))

    # Evaluate g on a grid so the threshold curves can be drawn as contours
    t = np.linspace(lim[0], lim[1], 400)
    X1, X2 = np.meshgrid(t, t)
    G = g(np.vstack([X1.ravel(), X2.ravel()])).reshape(X1.shape)

    fig, ax = plt.subplots(figsize=(7, 7))

    # Samples: level 0 (Monte Carlo) first, then each conditional level
    for L, xL in enumerate(x):
        colour = LEVEL_COLOURS[L % len(LEVEL_COLOURS)]
        label = "Monte Carlo (level 0)" if L == 0 else f"Level {L}"
        ax.scatter(xL[0], xL[1], s=5, color=colour, linewidths=0,
                   label=label, zorder=2)

    # Labels go at the top-left end of each curve, away from the samples
    span = lim[1] - lim[0]
    lo, hi = lim[0] + 0.08 * span, lim[1] - 0.08 * span   # label-safe area
    corner = np.array([lo, hi])

    # Intermediate thresholds: boundary of F_L = {x : g(x) > Y_L}
    for L, YL in enumerate(Y, start=1):
        cs = ax.contour(X1, X2, G, levels=[YL], colors="k", linewidths=0.8,
                        linestyles="--", zorder=3)
        pt = _point_nearest(cs, corner, lo, hi)
        if pt is not None:
            ax.clabel(cs, fmt={YL: f"$F_{{{L}}}$"}, fontsize=12, colors="k",
                      manual=[pt])

    # Failure domain F = {x : g(x) > YF}, labelled just inside the domain
    cs = ax.contour(X1, X2, G, levels=[YF], colors="k", linewidths=1.5,
                    zorder=3)
    pt = _point_nearest(cs, corner, lo, hi)
    if pt is not None:
        h = 1e-4 * span                        # numerical gradient of g at pt
        grad = np.array([
            g(np.array([[pt[0] + h], [pt[1]]]))[0] - g(np.array([[pt[0] - h], [pt[1]]]))[0],
            g(np.array([[pt[0]], [pt[1] + h]]))[0] - g(np.array([[pt[0]], [pt[1] - h]]))[0]])
        if np.linalg.norm(grad) > 0:
            # step from the boundary into F (direction of increasing g)
            pos = np.clip(pt + 0.06 * span * grad / np.linalg.norm(grad),
                          lim[0] + 0.03 * span, lim[1] - 0.03 * span)
            ax.text(*pos, "$F$", fontsize=15, ha="center", va="center")

    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title(f"Subset simulation: {len(Y)} levels, "
                 f"$p_F^{{SS}}$ = {result['pF_SS']:.3e}")
    if legend:
        ax.legend(loc="best", fontsize=8, markerscale=2.5, framealpha=0.9)
    fig.tight_layout()

    if filename:
        fig.savefig(filename, dpi=150)
        print(f"Saved plot to {filename}")
    plt.show()
    return fig, ax


# ===========================================================================
# Run
# ===========================================================================
if __name__ == "__main__":
    start = time.perf_counter()               # start the timer
    res = subset_simulation(d, YF, n=n, p=p, rng=SEED)
    run_time = time.perf_counter() - start    # seconds the simulation took

    pF = 1 - norm.cdf(YF / np.sqrt(d))    # true value (valid for g = sum only)
    print(f"Conditional levels L : {res['L']}")
    print(f"Thresholds Y_L       : {np.round(res['Y'], 3).tolist()}")
    print(f"Failure samples nF   : {res['nF']}")
    print(f"Total samples N      : {res['N']}")
    print(f"pF (subset sim)      : {res['pF_SS']:.4e}")
    print(f"pF (true value)      : {pF:.4e}")
    print(f"Run time             : {run_time:.2f} s")

    if d == 2:
        plot_levels_2d(res, YF, filename=SAVE_PLOT)
