"""
Subset Simulation for a Linear Reliability Problem  (Python translation)

Translated from the MATLAB code by K.M. Zuev (Institute for Risk & Uncertainty,
University of Liverpool).

    Performance function: g(x) = x1 + ... + xd
    Input variables x1, ..., xd are i.i.d. N(0, 1)
    Failure  <=>  g(x) > YF

The performance function is defined ONCE, in `performance_function` below.
In the MATLAB script `sum(...)` was written out separately wherever g was
evaluated (lines 19 and 62, and also sum(q) on line 46), so changing the model
meant editing the script in several places. Here every evaluation calls the
same function, so to try a new model you edit it in one place only (or pass
your own function to `subset_simulation`).

Two versions of the algorithm are provided:
    subset_simulation        - vectorised (fast); use this to experiment
    subset_simulation_loops  - same loops as the MATLAB, line by line (slow)
Both work for any dimension d. The level plot is drawn only when d = 2.

Usage examples:
    python subset_simulation.py                           # MATLAB settings (d=1000, YF=200, n=3000)
    python subset_simulation.py --d 2 --YF 5 --n 1000     # 2D: also plots the levels
    python subset_simulation.py --d 50 --YF 30 --loops    # loop-for-loop version
    python subset_simulation.py --help                    # all parameters
"""

import argparse
import math

import numpy as np
from scipy.stats import norm

SQRT_2PI = math.sqrt(2 * math.pi)


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
# Loop-for-loop version (mirrors the MATLAB script line by line)
# ---------------------------------------------------------------------------
def _normpdf(t):
    """Scalar standard normal pdf (MATLAB normpdf)."""
    return math.exp(-0.5 * t * t) / SQRT_2PI


def subset_simulation_loops(d, YF, n=3000, p=0.1, g=performance_function,
                            rng=None):
    """Same algorithm as `subset_simulation`, written with the same loops as
    the MATLAB code so each line can be compared directly. Numbers in the
    comments are the MATLAB line numbers. Indices are 0-based (MATLAB i = 1
    is Python i = 0). Much slower than `subset_simulation`; the output has
    the same form.
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
# 2D plot of the levels (cf. Figure 6)
# ---------------------------------------------------------------------------
def plot_levels_2d(result, YF, g=performance_function, lim=None,
                   filename=None):
    """Scatter the samples of every level, with the intermediate thresholds
    g(x) = Y_L as dashed lines and the failure boundary g(x) = YF."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgb

    x, Y = result["x"], result["Y"]
    if x[0].shape[0] != 2:
        raise ValueError("plot_levels_2d needs d = 2")

    if lim is None:
        lim = max(4.0, np.abs(np.hstack(x)).max() + 0.5)

    # Ordered blue ramp for the levels (light = level 0, dark = deepest level)
    ramp = ["#86b6ef", "#5598e7", "#2a78d6", "#256abf",
            "#1c5cab", "#184f95", "#104281", "#0d366b"]
    idx = np.linspace(0, len(ramp) - 1, len(x)).round().astype(int)
    colours = [ramp[i] for i in idx]
    fail_colour = "#eb6834"
    ink, muted = "#2b2b2b", "#8a8a85"
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]

    t = np.linspace(-lim, lim, 300)
    X1, X2 = np.meshgrid(t, t)
    G = g(np.vstack([X1.ravel(), X2.ravel()])).reshape(X1.shape)

    fig, ax = plt.subplots(figsize=(7, 7))
    for L, (xL, c) in enumerate(zip(x, colours)):
        label = "Monte Carlo samples" if L == 0 else f"Level {L} samples"
        ax.scatter(xL[0], xL[1], s=14, color=c, alpha=0.85, linewidths=0,
                   marker=markers[L % len(markers)], label=label, zorder=2 + L)

    # Intermediate thresholds g(x) = Y_L
    for L, (YL, c) in enumerate(zip(Y, colours[1:]), start=1):
        cs = ax.contour(X1, X2, G, levels=[YL], colors=[c], linewidths=1.5,
                        linestyles="--", zorder=20)
        for txt in ax.clabel(cs, fmt={YL: f"$Y_{L}$={YL:.2f}"}, fontsize=9,
                             colors=[ink]):
            txt.set_bbox(dict(facecolor="white", edgecolor="none", pad=1.5))
            txt.set_zorder(30)

    # Failure domain g(x) > YF
    ax.contour(X1, X2, G, levels=[YF], colors=[fail_colour], linewidths=2,
               zorder=21)
    ax.contourf(X1, X2, G, levels=[YF, G.max() + 1],
                colors=[(*to_rgb(fail_colour), 0.10)], zorder=0)
    ax.plot([], [], color=fail_colour, lw=2,
            label=f"Failure boundary g(x) = $Y_F$ = {YF:g}")
    ax.plot([], [], color=muted, lw=1.5, ls="--",
            label="Intermediate thresholds g(x) = $Y_L$")

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("$x_1$", color=ink)
    ax.set_ylabel("$x_2$", color=ink)
    ax.set_title(f"Subset simulation levels  "
                 f"($p_F^{{SS}}$ ≈ {result['pF_SS']:.3e})", color=ink)
    ax.grid(color="#e6e5e0", lw=0.6, zorder=-1)
    for s in ax.spines.values():
        s.set_color(muted)
    ax.tick_params(colors=muted)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.9)
    fig.tight_layout()

    if filename:
        fig.savefig(filename, dpi=150)
        print(f"Saved plot to {filename}")
    else:
        plt.show()
    return fig, ax


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Subset simulation, linear problem")
    ap.add_argument("--d", type=int, default=1000,
                    help="dimension of the input space")
    ap.add_argument("--YF", type=float, default=200.0,
                    help="critical threshold, failure <=> g(x) > YF")
    ap.add_argument("--n", type=int, default=3000,
                    help="number of samples per level")
    ap.add_argument("--p", type=float, default=0.1, help="level probability")
    ap.add_argument("--seed", type=int, default=None, help="random seed")
    ap.add_argument("--save", type=str, default=None,
                    help="save the 2D plot to this file instead of showing it")
    ap.add_argument("--no-plot", action="store_true",
                    help="skip the plot when d = 2")
    ap.add_argument("--loops", action="store_true",
                    help="use the loop-for-loop MATLAB version (slow)")
    args = ap.parse_args()

    run = subset_simulation_loops if args.loops else subset_simulation
    res = run(args.d, args.YF, n=args.n, p=args.p, rng=args.seed)

    # true value of the failure probability (valid for g = sum only)
    pF = 1 - norm.cdf(args.YF / np.sqrt(args.d))
    print(f"Conditional levels L : {res['L']}")
    print(f"Thresholds Y_L       : {np.round(res['Y'], 3).tolist()}")
    print(f"Failure samples nF   : {res['nF']}")
    print(f"Total samples N      : {res['N']}")
    print(f"pF (subset sim)      : {res['pF_SS']:.4e}")
    print(f"pF (true value)      : {pF:.4e}")

    if args.d == 2 and not args.no_plot:
        plot_levels_2d(res, args.YF, filename=args.save)


if __name__ == "__main__":
    main()
