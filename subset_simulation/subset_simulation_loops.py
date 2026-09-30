"""
Subset Simulation for a Linear Reliability Problem -- LOOP version

Python translation of the MATLAB code by K.M. Zuev (Institute for Risk &
Uncertainty, University of Liverpool).

    Performance function: g(x) = x1 + ... + xd
    Input variables x1, ..., xd are i.i.d. N(0, 1)
    Failure  <=>  g(x) > YF

This version keeps the same loops as the MATLAB script, with the MATLAB
line numbers in the comments, so it is the closest to the original code.
It is slow in high dimension (about 40 s for d = 1000); see
subset_simulation_vectorised.py for a fast version of the same algorithm.

The performance function is defined ONCE, in `performance_function` below.
In the MATLAB script `sum(...)` was written out separately wherever g was
evaluated (lines 19 and 62, and also sum(q) on line 46). Here every
evaluation calls the same function, so to try a new model you only edit it
in one place.

HOW TO RUN: set the parameters just below, then run this file. The results
are printed, and when d = 2 the level plot (in the style of Figure 6) opens in a window.
Works for any dimension d; the plot is only drawn for d = 2.
"""

import math

import numpy as np
from scipy.stats import norm

SQRT_2PI = math.sqrt(2 * math.pi)

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
# Subset simulation (mirrors the MATLAB script line by line)
# ---------------------------------------------------------------------------
def _normpdf(t):
    """Scalar standard normal pdf (MATLAB normpdf)."""
    return math.exp(-0.5 * t * t) / SQRT_2PI


def subset_simulation_loops(d, YF, n=3000, p=0.1, g=performance_function,
                            rng=None):
    """Estimate pF = P(g(x) > YF) for x ~ N(0, I_d).

    Written with the same loops as the MATLAB code so each line can be
    compared directly. Numbers in the comments are the MATLAB line numbers.
    Indices are 0-based (MATLAB i = 1 is Python i = 0).

    Returns a dict with keys 'pF_SS', 'N', 'L', 'Y' (intermediate
    thresholds), 'nF' (failure counts per level), 'x' and 'y' (samples and
    responses per level).
    """
    rng = np.random.default_rng(rng)
    nc = int(round(n * p))                         # 12: number of Markov chains
    ns = int(round((1 - p) / p))                   # 13: number of states in each chain
    if nc * (ns + 1) != n:
        raise ValueError("n*p and 1/p must be integers")

    L = 0                                          # 15: current (unconditional) level
    x = [rng.standard_normal((d, n))]              # 16: Monte Carlo samples
    y = [np.zeros(n)]
    nF = [0]                                       # 17: number of failure samples
    for i in range(n):                             # 18
        y[0][i] = g(x[0][:, i:i + 1])[0]           # 19: system response y=g(x)
        if y[0][i] > YF:                           # 20: x(:,i) is a failure sample
            nF[0] += 1                             # 21
    Y = []
    while nF[L] / n < p:                           # 24: stopping criterion
        L += 1                                     # 25: next conditional level is needed
        ind = np.argsort(-y[L - 1], kind="stable") # 26: sort 'descend'
        y[L - 1] = y[L - 1][ind]                   # 26: renumbered responses
        x[L - 1] = x[L - 1][:, ind]                # 27: renumbered samples
        Y.append((y[L - 1][nc - 1] + y[L - 1][nc]) / 2)  # 28: L-th intermediate threshold
        z = np.zeros((d, nc, ns + 1))
        z[:, :, 0] = x[L - 1][:, :nc]              # 29: Markov chain "seeds"
        # 31: Modified Metropolis algorithm for sampling from pi(x | F_L)
        q = np.zeros(d)
        for j in range(nc):                        # 32
            for m in range(ns):                    # 33
                # Step 1:
                for k in range(d):                 # 35
                    a = z[k, j, m] + rng.standard_normal()                # 36: Step 1(a)
                    r = min(1.0, _normpdf(a) / _normpdf(z[k, j, m]))      # 37: Step 1(b)
                    if rng.random() < r:           # 39: Step 1(c)
                        q[k] = a                   # 40
                    else:
                        q[k] = z[k, j, m]          # 42
                # Step 2:
                if g(q[:, None])[0] > Y[L - 1]:    # 46: q belongs to F_L
                    z[:, j, m + 1] = q             # 47
                else:
                    z[:, j, m + 1] = z[:, j, m]    # 49
        x.append(np.zeros((d, n)))
        for j in range(nc):                        # 53
            for m in range(ns + 1):                # 54
                x[L][:, j * (ns + 1) + m] = z[:, j, m]   # 55: samples from pi(x | F_L)
        del z                                      # 58
        nF.append(0)                               # 60
        y.append(np.zeros(n))
        for i in range(n):                         # 61
            y[L][i] = g(x[L][:, i:i + 1])[0]       # 62: system response y=g(x)
            if y[L][i] > YF:                       # 63: failure sample
                nF[L] += 1                         # 64: number of failure samples at level L
    pF_SS = p ** L * nF[L] / n                     # 68: SS estimate
    N = n + n * (1 - p) * L                        # 69: total number of samples

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
    res = subset_simulation_loops(d, YF, n=n, p=p, rng=SEED)

    pF = 1 - norm.cdf(YF / np.sqrt(d))    # true value (valid for g = sum only)
    print(f"Conditional levels L : {res['L']}")
    print(f"Thresholds Y_L       : {np.round(res['Y'], 3).tolist()}")
    print(f"Failure samples nF   : {res['nF']}")
    print(f"Total samples N      : {res['N']}")
    print(f"pF (subset sim)      : {res['pF_SS']:.4e}")
    print(f"pF (true value)      : {pF:.4e}")

    if d == 2:
        plot_levels_2d(res, YF, filename=SAVE_PLOT)
