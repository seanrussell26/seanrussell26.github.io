"""
Subset Simulation (Au & Beck, 2001) for estimating small failure probabilities.

The problem is posed in standard normal space: u ~ N(0, I) of dimension `dim`,
and failure is the event g(u) <= 0.

The limit-state function is defined ONCE (see `limit_state` below) and is passed
into `subset_simulation`, so the same function is used both for the initial
Monte Carlo level and inside the Markov chains. To try a different model you
only edit it in one place (or pass your own function in).

Usage examples:
    python subset_simulation.py                       # defaults, 2D, with plot
    python subset_simulation.py --beta 4 --N 2000     # rarer event, more samples
    python subset_simulation.py --dim 10 --no-plot    # higher dimension
"""

import argparse

import numpy as np
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Limit-state function -- the ONLY place the model is defined.
# ---------------------------------------------------------------------------
def limit_state(u, beta=3.0):
    """Linear limit state g(u) = beta - sum(u) / sqrt(d).

    `u` has shape (n_samples, dim). Failure is g(u) <= 0.
    Exact answer for checking: Pf = Phi(-beta).
    """
    u = np.atleast_2d(u)
    d = u.shape[1]
    return beta - u.sum(axis=1) / np.sqrt(d)


# ---------------------------------------------------------------------------
# Subset simulation
# ---------------------------------------------------------------------------
def subset_simulation(g, dim, N=1000, p0=0.1, proposal_sd=1.0,
                      max_levels=20, rng=None):
    """Estimate Pf = P(g(u) <= 0) for u ~ N(0, I_dim).

    Parameters
    ----------
    g : callable
        Limit-state function taking an (n, dim) array, returning shape (n,).
    dim : int
        Number of random variables.
    N : int
        Samples per level.
    p0 : float
        Conditional probability of each intermediate level (N*p0 and 1/p0
        must be integers).
    proposal_sd : float
        Std. dev. of the component-wise Gaussian proposal in modified Metropolis.
    max_levels : int
        Safety cap on the number of levels.
    rng : np.random.Generator, optional

    Returns
    -------
    dict with keys
        'pf'         : failure probability estimate
        'cov'        : approximate coefficient of variation of the estimate
        'thresholds' : intermediate thresholds b_1, b_2, ... (last is 0)
        'samples'    : list of (N, dim) arrays, one per level
        'g_values'   : list of (N,) arrays, one per level
        'acceptance' : mean acceptance rate at each conditional level
        'n_calls'    : total number of limit-state evaluations
    """
    rng = np.random.default_rng(rng)
    Nc = int(round(N * p0))        # number of seeds (chains)
    Ns = int(round(1 / p0))        # samples per chain
    if Nc * Ns != N:
        raise ValueError("N*p0 and 1/p0 must be integers with Nc*Ns == N")

    # Level 0: crude Monte Carlo
    u = rng.standard_normal((N, dim))
    gu = g(u)
    n_calls = N

    samples, g_values = [u], [gu]
    thresholds, acceptance, delta2 = [], [], []
    pf = None

    for level in range(max_levels):
        order = np.argsort(gu)
        # Intermediate threshold: the p0-quantile of g, but not below 0
        b = max(0.5 * (gu[order[Nc - 1]] + gu[order[Nc]]), 0.0)
        thresholds.append(b)
        indicator = gu <= b
        p_level = indicator.mean()

        # c.o.v. contribution of this level
        if level == 0:
            delta2.append((1 - p_level) / (N * p_level))
        else:
            gamma = _correlation_factor(indicator.reshape(Nc, Ns), p_level)
            delta2.append((1 - p_level) / (N * p_level) * (1 + gamma))

        if b <= 0.0:                       # reached the failure domain
            pf = p0 ** level * p_level
            break

        # Seeds for the next level
        seeds_u = u[order[:Nc]]
        seeds_g = gu[order[:Nc]]
        u, gu, acc = _modified_metropolis(g, seeds_u, seeds_g, b, Ns,
                                          proposal_sd, rng)
        n_calls += Nc * (Ns - 1)
        samples.append(u)
        g_values.append(gu)
        acceptance.append(acc)
    else:
        # Hit max_levels without reaching g <= 0
        pf = p0 ** max_levels
        print("Warning: max_levels reached; estimate is an upper bound.")

    return {
        "pf": pf,
        "cov": float(np.sqrt(np.sum(delta2))),
        "thresholds": thresholds,
        "samples": samples,
        "g_values": g_values,
        "acceptance": acceptance,
        "n_calls": n_calls,
    }


def _modified_metropolis(g, seeds_u, seeds_g, b, Ns, proposal_sd, rng):
    """Grow Ns-long chains from each seed, conditional on g(u) <= b.

    Returned samples are ordered chain-by-chain, i.e. rows
    [c*Ns : (c+1)*Ns] belong to chain c (seed first).
    """
    Nc, dim = seeds_u.shape
    chains_u = np.empty((Nc, Ns, dim))
    chains_g = np.empty((Nc, Ns))
    chains_u[:, 0], chains_g[:, 0] = seeds_u, seeds_g
    n_accept = 0

    cur_u, cur_g = seeds_u.copy(), seeds_g.copy()
    for k in range(1, Ns):
        # 1) component-wise Metropolis step w.r.t. the standard normal pdf
        cand = cur_u + proposal_sd * rng.standard_normal((Nc, dim))
        ratio = np.exp(-0.5 * (cand ** 2 - cur_u ** 2))
        keep = rng.random((Nc, dim)) < np.minimum(1.0, ratio)
        cand = np.where(keep, cand, cur_u)

        # 2) accept the candidate only if it stays in the current subset
        moved = np.any(keep, axis=1)
        cand_g = cur_g.copy()
        if moved.any():
            cand_g[moved] = g(cand[moved])
        in_subset = moved & (cand_g <= b)

        cur_u = np.where(in_subset[:, None], cand, cur_u)
        cur_g = np.where(in_subset, cand_g, cur_g)
        chains_u[:, k], chains_g[:, k] = cur_u, cur_g
        n_accept += in_subset.sum()

    acc = n_accept / (Nc * (Ns - 1))
    return chains_u.reshape(Nc * Ns, dim), chains_g.reshape(Nc * Ns), acc


def _correlation_factor(I, p):
    """Au & Beck correlation factor gamma for Nc chains of length Ns."""
    Nc, Ns = I.shape
    N = Nc * Ns
    R0 = p * (1 - p)
    if R0 == 0:
        return 0.0
    gamma = 0.0
    for k in range(1, Ns):
        Rk = np.sum(I[:, :Ns - k] * I[:, k:]) / (N - k * Nc) - p ** 2
        gamma += 2 * (1 - k * Nc / N) * Rk / R0
    return gamma


# ---------------------------------------------------------------------------
# 2D plot of the levels
# ---------------------------------------------------------------------------
def plot_levels_2d(result, g, lim=None, filename=None):
    """Scatter the samples of every level and the threshold contours g = b_j."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgb

    samples, thresholds = result["samples"], result["thresholds"]
    if samples[0].shape[1] != 2:
        raise ValueError("plot_levels_2d needs dim = 2")

    if lim is None:
        allu = np.vstack(samples)
        lim = max(4.0, np.abs(allu).max() + 0.5)

    # Ordered blue ramp for the levels (light = level 0, dark = deepest level)
    ramp = ["#86b6ef", "#5598e7", "#2a78d6", "#256abf",
            "#1c5cab", "#184f95", "#104281", "#0d366b"]
    idx = np.linspace(0, len(ramp) - 1, len(samples)).round().astype(int)
    colours = [ramp[i] for i in idx]
    fail_colour = "#eb6834"
    ink, muted = "#2b2b2b", "#8a8a85"

    x = np.linspace(-lim, lim, 300)
    X1, X2 = np.meshgrid(x, x)
    G = g(np.column_stack([X1.ravel(), X2.ravel()])).reshape(X1.shape)

    fig, ax = plt.subplots(figsize=(7, 7))
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    for j, (u, c) in enumerate(zip(samples, colours)):
        ax.scatter(u[:, 0], u[:, 1], s=14, color=c, alpha=0.85,
                   marker=markers[j % len(markers)], linewidths=0,
                   label=f"Level {j} samples", zorder=2 + j)

    # Intermediate thresholds g(u) = b_j
    for j, (b, c) in enumerate(zip(thresholds[:-1], colours)):
        cs = ax.contour(X1, X2, G, levels=[b], colors=[c], linewidths=1.5,
                        linestyles="--", zorder=20)
        for t in ax.clabel(cs, fmt={b: f"$b_{j+1}$={b:.2f}"}, fontsize=9,
                           colors=[ink]):
            t.set_bbox(dict(facecolor="white", edgecolor="none", pad=1.5))
            t.set_zorder(30)
    # Failure boundary g(u) = 0
    ax.contour(X1, X2, G, levels=[0], colors=[fail_colour], linewidths=2,
               zorder=21)
    ax.contourf(X1, X2, G, levels=[G.min() - 1, 0],
                colors=[(*to_rgb(fail_colour), 0.10)], zorder=0)
    ax.plot([], [], color=fail_colour, lw=2, label="Failure boundary g(u)=0")
    ax.plot([], [], color=muted, lw=1.5, ls="--",
            label="Intermediate thresholds g(u)=$b_j$")

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("$u_1$", color=ink)
    ax.set_ylabel("$u_2$", color=ink)
    ax.set_title(f"Subset simulation levels  (P$_f$ ≈ {result['pf']:.3e})",
                 color=ink)
    ax.grid(color="#e6e5e0", lw=0.6, zorder=-1)
    for s in ax.spines.values():
        s.set_color(muted)
    ax.tick_params(colors=muted)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
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
    p = argparse.ArgumentParser(description="Subset simulation demo")
    p.add_argument("--dim", type=int, default=2, help="number of variables")
    p.add_argument("--beta", type=float, default=3.0,
                   help="reliability index in the example limit state")
    p.add_argument("--N", type=int, default=1000, help="samples per level")
    p.add_argument("--p0", type=float, default=0.1,
                   help="level probability")
    p.add_argument("--sd", type=float, default=1.0,
                   help="proposal std. dev. for modified Metropolis")
    p.add_argument("--seed", type=int, default=None, help="random seed")
    p.add_argument("--save", type=str, default=None,
                   help="save the 2D plot to this file instead of showing it")
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args()

    def g(u):
        return limit_state(u, beta=args.beta)

    res = subset_simulation(g, args.dim, N=args.N, p0=args.p0,
                            proposal_sd=args.sd, rng=args.seed)

    exact = norm.cdf(-args.beta)
    print(f"Levels            : {len(res['samples'])}")
    print(f"Thresholds b_j    : {np.round(res['thresholds'], 4).tolist()}")
    print(f"Acceptance rates  : {np.round(res['acceptance'], 3).tolist()}")
    print(f"g evaluations     : {res['n_calls']}")
    print(f"Pf (subset sim)   : {res['pf']:.4e}   (c.o.v. ≈ {res['cov']:.2f})")
    print(f"Pf (exact)        : {exact:.4e}")

    if args.dim == 2 and not args.no_plot:
        plot_levels_2d(res, g, filename=args.save)


if __name__ == "__main__":
    main()
