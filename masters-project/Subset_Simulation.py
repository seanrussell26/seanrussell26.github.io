"""
Subset Simulation for a Reliability Problem

    Performance function: g(x)
    Failure  <=>  g(x) > YF

HOW TO RUN: set the parameters just below, then run this file. Any function from
Performance_Functions.py (or your own, taking an (n, d) array) can be used as g.
This code was made from my Subset Simulation code (adapted from M. Zuev's SuS Tutorial using arrays instead of for loops).
"""

import numpy as np
from scipy.stats import norm
import time

from Performance_Functions import linear_sum, true_pf


# ===========================================================================
# PARAMETERS
# ===========================================================================
d = 2          # dimension of the input space            
YF = 9.0       # critical threshold (failure <=> g(x) > YF) 
n = 3000       # number of samples per level           
p = 0.1        # level probability                      
SEED = None    # random seed: an integer for repeatable results, None for random
SAVE_PLOT = None   # option to save the plot to a file
g = linear_sum     # performance function (see Performance_Functions.py)

# ---------------------------------------------------------------------------
# Subset simulation
# ---------------------------------------------------------------------------
def subset_simulation(d, YF, n=3000, p=0.1, g=linear_sum, rng=None, max_levels=50):
    """Estimate pF = P(g(x) > YF) for x ~ N(0, I_d).

    Parameters
    ----------
    d   : dimension of the input space
    YF  : critical threshold (failure <=> g(x) > YF)
    n   : number of samples per level
    p   : level probability
    g   : performance function taking an (n, d) array and returning shape (n,)
    rng : seed or np.random.Generator
    max_levels : give up after this many levels (YF is probably unreachable)

    Returns
    -------
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
    y = [g(x[0].T)]                              # system response y = g(x)
    nF = [int(np.sum(y[0] > YF))]                # number of failure samples
    Y = []

    while nF[L] / n < p:                         # stopping criterion
        if L == max_levels:
            raise RuntimeError(f"No failures after {L} levels: is YF = {YF} reachable?")
        L += 1                                   # next conditional level is needed
        ind = np.argsort(-y[L - 1])              # sort 'descend'
        y[L - 1] = y[L - 1][ind]                 # renumbered responses
        x[L - 1] = x[L - 1][:, ind]              # renumbered samples
        Y.append((y[L - 1][nc - 1] + y[L - 1][nc]) / 2)  # L-th intermediate threshold

        # --- Modified Metropolis algorithm for sampling from pi(x | F_L) ---
        z = np.empty((d, nc, ns + 1))
        z[:, :, 0] = x[L - 1][:, :nc]            # Markov chain "seeds"
        for m in range(ns):
            # Step 1: component-wise proposal
            a = z[:, :, m] + rng.standard_normal((d, nc))          # Step 1(a)
            r = np.minimum(1, norm.pdf(a) / norm.pdf(z[:, :, m]))  # Step 1(b)
            q = np.where(rng.random((d, nc)) < r, a, z[:, :, m])   # Step 1(c)
            # Step 2: accept q only if it belongs to F_L
            in_FL = g(q.T) > Y[L - 1]
            z[:, :, m + 1] = np.where(in_FL, q, z[:, :, m])

        # samples from pi(x | F_L), ordered chain by chain
        x.append(z.reshape(d, nc * (ns + 1)))

        y.append(g(x[L].T))                      # system response y = g(x)
        nF.append(int(np.sum(y[L] > YF)))        # number of failure samples at level L

    pF_SS = p ** L * nF[L] / n                   # SS estimate
    N = n + n * (1 - p) * L                      # total number of samples

    return {"pF_SS": pF_SS, "N": int(round(N)), "L": L, "Y": Y,
            "nF": nF, "x": x, "y": y}


# ---------------------------------------------------------------------------
# 2D plot of the levels (in the style of Figure 6)
# ---------------------------------------------------------------------------
LEVEL_COLOURS = ["#ff0000", "#e000e0", "#0000ff", "#007f00", "#00c0c0",
                 "#b0b000", "#00e000", "#ffff00", "#1f2f7f", "#e03000"]


def _point_nearest(cs, target, lo, hi):
    pts = [seg for seg in cs.allsegs[0] if len(seg)]
    if not pts:
        return None                            # curve not visible in the plot
    pts = np.vstack(pts)
    inside = np.all((pts > lo) & (pts < hi), axis=1)
    if inside.any():
        pts = pts[inside]
    return pts[np.argmin(np.sum((pts - target) ** 2, axis=1))]


def plot_levels_2d(result, YF, g=linear_sum, lim=None,
                   filename=None, legend=True):
    import matplotlib.pyplot as plt

    x, Y = result["x"], result["Y"]
    if x[0].shape[0] != 2:
        raise ValueError("plot_levels_2d needs d = 2")

    # Axis range:
    if lim is None:
        allx = np.hstack(x)
        lim = (np.floor(allx.min() - 0.2), np.ceil(allx.max() + 0.2))

    # Evaluate g on a grid so the threshold curves can be drawn as contours
    t = np.linspace(lim[0], lim[1], 400)
    X1, X2 = np.meshgrid(t, t)
    G = g(np.column_stack([X1.ravel(), X2.ravel()])).reshape(X1.shape)

    fig, ax = plt.subplots(figsize=(7, 7))

    # Samples:
    for L, xL in enumerate(x):
        colour = LEVEL_COLOURS[L % len(LEVEL_COLOURS)]
        label = "Monte Carlo (level 0)" if L == 0 else f"Level {L}"
        ax.scatter(xL[0], xL[1], s=5, color=colour, linewidths=0,
                   label=label, zorder=2)

    # Labels:
    span = lim[1] - lim[0]
    lo, hi = lim[0] + 0.08 * span, lim[1] - 0.08 * span  
    corner = np.array([lo, hi])

    # Intermediate thresholds: boundary of F_L = {x : g(x) > Y_L}
    for L, YL in enumerate(Y, start=1):
        cs = ax.contour(X1, X2, G, levels=[YL], colors="k", linewidths=0.8,
                        linestyles="--", zorder=3)
        pt = _point_nearest(cs, corner, lo, hi)
        if pt is not None:
            ax.clabel(cs, fmt={YL: f"$F_{{{L}}}$"}, fontsize=12, colors="k",
                      manual=[pt])

    # Failure domain F = {x : g(x) > YF}:
    cs = ax.contour(X1, X2, G, levels=[YF], colors="k", linewidths=1.5,
                    zorder=3)
    pt = _point_nearest(cs, corner, lo, hi)
    if pt is not None:
        h = 1e-4 * span                        # numerical gradient of g at pt
        grad = np.array([
            g(np.array([[pt[0] + h, pt[1]]]))[0] - g(np.array([[pt[0] - h, pt[1]]]))[0],
            g(np.array([[pt[0], pt[1] + h]]))[0] - g(np.array([[pt[0], pt[1] - h]]))[0]])
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
    start = time.perf_counter()
    res = subset_simulation(d, YF, n=n, p=p, g=g, rng=SEED)
    run_time = time.perf_counter() - start    # seconds the simulation took

    pF = true_pf(g, YF, d)                    # None when the true value is unknown
    print(f"Conditional levels L : {res['L']}")
    print(f"Thresholds Y_L       : {np.round(res['Y'], 3).tolist()}")
    print(f"Failure samples nF   : {res['nF']}")
    print(f"Total samples N      : {res['N']}")
    print(f"pF (subset sim)      : {res['pF_SS']:.4e}")
    if pF is not None:
        print(f"pF (true value)      : {pF:.4e}")
    print(f"Run time             : {run_time:.2f} s")

    if d == 2:
        plot_levels_2d(res, YF, g=g, filename=SAVE_PLOT, legend=False)
