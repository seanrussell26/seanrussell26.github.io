"""
Niching Subset Simulation for a Reliability Problem

    Performance function: g(x)
    Failure  <=>  g(x) > YF

HOW TO RUN: set the parameters just below, then run this file. NSuS itself is
H. Kinnear's code, kept unchanged in Kinnear_NSuS.py; this file only sets it up,
runs it next to Subset_Simulation.py, and plots the results.
"""

import time

import matplotlib.pyplot as plt
import numpy as np

from Kinnear_NSuS import HillValleyGraphNSuS, exceedance_probability, partitioner_figures
from Performance_Functions import pointwise, true_pf, two_gaussians
from Subset_Simulation import plot_levels_2d, subset_simulation

# ===========================================================================
# PARAMETERS
# ===========================================================================
d = 2              # dimension of the input space
YF = 0.03          # critical threshold (failure <=> g(x) > YF)
n = 3000           # number of samples per level
p = 0.1            # level probability
SEED = None        # random seed
SAVE_PLOT = None   # option to save the plot to a file
g = two_gaussians  # performance function (see Performance_Functions.py)

GRAPH_SIZE = 15    # NSuS: samples in each hill valley graph
MAX_BRANCHES = 4   # NSuS: maximum number of branches (1 = SuS)
SCALE = 1.0        # NSuS: proposal std of the Markov chains (None = adaptive)


def run_nsus(d, YF, n=3000, p=0.1, g=two_gaussians, seed=None, graph_size=15,
             max_branches=4, scale=1.0, verbose=False):
    """Run Kinnear's hill valley graph NSuS on g (an (n, d) -> (n,) function)."""
    nss = HillValleyGraphNSuS(
        performance_function=pointwise(g),
        level_probability=p,
        level_size=n,
        dimension=d,
        seed=seed,
        threshold=YF,
        graph_size=graph_size,
        max_branches=max_branches,
        scale=scale,
        verbose=verbose)
    nss.run()
    return nss


# ---------------------------------------------------------------------------
# NSuS plots: the samples by niche, and the tree of branches
# ---------------------------------------------------------------------------
BRANCH_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                  "#e87ba4", "#008300", "#6250d6", "#e34948"]
SHARED_COLOUR = "#8a8a85"                      # levels before a split


def branch_number(nss):
    number = {}
    for k, leaf in enumerate(nss.leaves, start=1):
        level = leaf
        while level.parent is not None and len(level.parent.children) == 1:
            number[level] = k
            level = level.parent
        if level.parent is not None:
            number[level] = k
    for level in nss.all_levels:
        number.setdefault(level, 0)
    return number


def colour(k):
    return SHARED_COLOUR if k == 0 else BRANCH_COLOURS[(k - 1) % len(BRANCH_COLOURS)]

def plot_nsus_levels_2d(nss, YF, g=two_gaussians, lim=None, filename=None): #samples coloured by niche

    number = branch_number(nss)
    if lim is None:
        allx = np.array([s.array for s in nss.all_samples])
        lim = (np.floor(allx.min() - 0.2), np.ceil(allx.max() + 0.2))
    t = np.linspace(lim[0], lim[1], 200)
    X1, X2 = np.meshgrid(t, t)
    pts = np.column_stack([X1.ravel(), X2.ravel()])

    fig, ax = plt.subplots(figsize=(7, 7))

    for level in nss.all_levels:
        xL = np.array([s.array for s in level.sample_list]).T
        ax.scatter(xL[0], xL[1], s=5, color=colour(number[level]),
                   linewidths=0, zorder=2)

    for level in nss.all_levels:
        if len(level._buds) < 2:
            continue
        labels = level.extra_info['classifier'].predict(pts).astype(float)
        inside = np.array([level.indicator.partition_indicator(x) for x in pts]) == 1
        labels[~inside] = np.nan
        ax.contour(X1, X2, labels.reshape(X1.shape),
                   levels=np.arange(len(level._buds) - 1) + 0.5,
                   colors="k", linewidths=1.5, zorder=4)

    G = g(pts).reshape(X1.shape)
    ax.contour(X1, X2, G, levels=[YF], colors="k", linewidths=1.5,
               linestyles="--", zorder=3)

    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title(f"Niching subset simulation: {len(nss.leaves)} niches, "
                 f"$p_F^{{NSuS}}$ = {exceedance_probability(nss, YF):.3e}")
    fig.tight_layout()
    if filename:
        fig.savefig(filename, dpi=150)
        print(f"Saved plot to {filename}")
    plt.show()
    return fig, ax


def plot_branch_tree(nss, filename=None):
    col = {}
    for k, leaf in enumerate(nss.leaves):
        col[leaf] = k
    for level in sorted(nss.all_levels, key=lambda lv: -lv.depth):
        if level.children:
            col[level] = np.mean([col[c] for c in level.children])

    plt.figure()
    for level in nss.all_levels:
        for child in level.children:
            plt.plot([col[level], col[child]], [level.depth, child.depth], color='black')
        plt.text(col[level], level.depth, level.name, ha='center', va='center',
                 bbox=dict(facecolor='white', edgecolor='black'))
    plt.gca().invert_yaxis()
    plt.axis('off')
    if filename:
        plt.savefig(filename)
    plt.show()


# ===========================================================================
# Run
# ===========================================================================
if __name__ == "__main__":
    # --- Original Subset Simulation ---
    start = time.perf_counter()
    res = subset_simulation(d, YF, n=n, p=p, g=g, rng=SEED)
    run_time = time.perf_counter() - start
    print("SUBSET SIMULATION")
    print(f"Conditional levels L : {res['L']}")
    print(f"Total samples N      : {res['N']}")
    print(f"pF (subset sim)      : {res['pF_SS']:.4e}")
    print(f"Run time             : {run_time:.2f} s\n")

    # --- Niching Subset Simulation ---
    start = time.perf_counter()
    nss = run_nsus(d, YF, n=n, p=p, g=g, seed=SEED, graph_size=GRAPH_SIZE,
                   max_branches=MAX_BRANCHES, scale=SCALE, verbose=True)
    run_time = time.perf_counter() - start
    pF_NSuS = exceedance_probability(nss, YF)
    print("\nNICHING SUBSET SIMULATION")
    print(f"Levels / branches    : {len(nss.all_levels)} / {len(nss.leaves)}")
    print(f"Total samples N      : {nss.performance_function.eval_count}"
          f"  ({nss.partitioner.eval_count} for hill valley graphs)")
    print(f"pF (NSuS)            : {pF_NSuS:.4e}")
    print(f"Run time             : {run_time:.2f} s\n")
    pF = true_pf(g, YF, d)
    if pF is not None:
        print(f"pF (true value)      : {pF:.4e}")

    if d == 2:
        plot_levels_2d(res, YF, g=g, filename=SAVE_PLOT, legend=False)
        plot_nsus_levels_2d(nss, YF, g=g)
        plot_branch_tree(nss)
        split = [lv for lv in nss.all_levels if len(lv._buds) > 1]
        if split:
            partitioner_figures(nss, split[0].name)
