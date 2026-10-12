"""
Niching Subset Simulation: worked examples

    Input variables x1, ..., xd are i.i.d. N(0, 1)
    Failure  <=>  g(x) >= THRESHOLD

HOW TO RUN: pick an EXAMPLE and set the parameters just below, then run this
file. It runs ordinary subset simulation and NSuS on the same problem, prints
both results next to the reference value, and (for d = 2) plots them.

Set COMPARE_RUNS to e.g. 20 to repeat both algorithms with seeds 0..19 and
print a summary table in the style of Tables 1-4 of the paper.
"""

import time
import numpy as np
from scipy.linalg import eigh
from scipy.stats import chi2, norm

import nsus
import nsus_plot

# ===========================================================================
# PARAMETERS
# ===========================================================================
EXAMPLE = "pwl"      # "pwl", "high_pwl", "mix", "two_dof", "ring", "linear"
n = None             # samples per level (None = the value used in the paper)
p = 0.1              # level probability
GRAPH_SIZE = None    # HVG graph size n_g (None = paper value)
MAX_BRANCHES = None  # maximum number of branches (None = paper value)
SCALE = 1.0          # proposal std of the Markov chains (None = adaptive)
HVG_HISTORY = True   # hill valley tests use the search history (paper default);
                     # False = always evaluate g at the midpoint of each pair
SEED = None          # integer for repeatable results, None for random
VERBOSE = True       # print what NSuS is doing level by level
COMPARE_RUNS = 0     # > 0: repeat both algorithms this many times and tabulate
SAVE_PLOT = None     # e.g. "pwl.png" to also save the plot to a file


# ===========================================================================
# Performance functions (each takes ONE sample x of shape (d,))
# ===========================================================================
def pwl(x):
    """Piecewise linear function, Eq. (9). SuS is drawn towards +x2, but the
    design point is in the +x1 direction."""
    if x[0] > 3.5:
        g_1 = 4 - x[0]
    else:
        g_1 = 0.85 - (0.1 * x[0])
    if x[1] > 2:
        g_2 = 0.5 - (0.1 * x[1])
    else:
        g_2 = 2.3 - x[1]
    return -min([g_1, g_2])


def high_pwl(x):
    """100-dimensional version, Eq. (20): pwl of two normalised sums."""
    z1 = np.sum(x[:50]) / np.sqrt(50)
    z2 = np.sum(x[50:]) / np.sqrt(50)
    return pwl(np.array([z1, z2]))


def mix(x):
    """Mixture of four normal densities, Eq. (21). Only the bump at (3, 3)
    reaches failure; the three others are local maxima that trap SuS."""
    means = np.array([[3, 3],
                      [2, -2],
                      [-2, 2],
                      [-2, -2]])
    weights = [0.4, 0.2, 0.2, 0.2]
    diffs = x - means
    exps = -0.5 * np.einsum('ij,ij->i', diffs, diffs)
    vals = (1 / (2 * np.pi)) * np.exp(exps)
    return np.dot(weights, vals) - 0.04


def two_dof(x):
    """Two-degree-of-freedom mass spring system, Eq. (23): peak displacement
    of the first mass minus 0.024. Two disjoint failure sets."""
    m1 = 2000
    m2 = 2000
    M = np.diag([m1, m2])
    sigma = np.sqrt(np.log(1.04))
    mu = np.log(2.5 * 10**5) - 0.5 * sigma**2
    k = np.exp(mu + sigma * np.array(x))
    k1, k2 = k[0], k[1]
    K = np.array([[k1 + k2, -k2], [-k2, k2]])
    OME = 11
    tm = np.arange(0, 20.01, 0.01)

    evals, evecs = eigh(K, M)
    nat_freq = np.sqrt(evals)
    sort_ind = np.argsort(nat_freq)
    nat_freq = nat_freq[sort_ind]
    evecs = evecs[:, sort_ind]
    mass_norm_evec = evecs / np.sqrt(np.diag(evecs.T @ M @ evecs))

    mod_resp = np.full((len(tm), 2), np.nan)
    for i in range(2):
        w = nat_freq[i]
        r = OME / w
        eta = 0.02
        tanth = 2 * eta * r / (1 - r**2)
        sinth = np.sqrt(1 - 1 / (1 + tanth**2))
        costh = np.sign(tanth) * np.sqrt(1 / (1 + tanth**2))
        wd = w * np.sqrt(1 - eta**2)
        A2 = sinth * (2000 * mass_norm_evec[1, i]) / (w**2 * np.sqrt((1 - r**2)**2 + (2 * eta * r)**2))
        A1 = 1 / wd * (A2 * eta * w - (2000 * mass_norm_evec[1, i]) * OME * costh /
                       (w**2 * np.sqrt((1 - r**2)**2 + (2 * eta * r)**2)))
        pp = np.exp(-eta * w * tm) * (A1 * np.sin(wd * tm) + A2 * np.cos(wd * tm)) + \
            (2000 * mass_norm_evec[1, i]) * (np.sin(OME * tm) * costh - np.cos(OME * tm) * sinth) / \
            (w**2 * np.sqrt((1 - r**2)**2 + (2 * eta * r)**2))
        mod_resp[:, i] = np.real(pp)
    mass_resp = mass_norm_evec @ mod_resp.T
    return np.max(mass_resp[0, :]) - 0.024


def ring(x):
    """Your Subset_Simulation.py function: g = x1^2 + ... + xd^2.
    The failure region (outside a circle) has no single design point."""
    return np.sum(x**2, axis=0)


def linear(x):
    """The function in your file's docstring: g = x1 + ... + xd. A half-space:
    convex, one niche, so NSuS should behave just like SuS."""
    return np.sum(x, axis=0)


# ---------------------------------------------------------------------------
# Degeneracy indicators (Section 5): True if the run never put a sample in
# an important part of the failure region.
# ---------------------------------------------------------------------------
def _arrays_perfs(ss):
    arrays = np.array([s.array for s in ss.all_samples])
    perfs = np.array([s.performance for s in ss.all_samples])
    return arrays, perfs


def degenerate_pwl(ss):
    arrays, _ = _arrays_perfs(ss)
    if arrays.shape[1] == 100:                      # high_pwl: map to (z1, z2)
        arrays = np.column_stack([arrays[:, :50].sum(1), arrays[:, 50:].sum(1)]) / np.sqrt(50)
    return not ((arrays[:, 0] >= 4) & (arrays[:, 1] <= 2)).any()


def degenerate_mix(ss):
    arrays, perfs = _arrays_perfs(ss)
    return not ((arrays[:, 0] >= 0) & (arrays[:, 1] >= 0) & (perfs > 0)).any()


def degenerate_two_dof(ss):
    arrays, perfs = _arrays_perfs(ss)
    return not ((arrays[:, 0] < 0) & (perfs > 0)).any()


# ---------------------------------------------------------------------------
# Problem catalogue. Reference probabilities are the paper's direct Monte
# Carlo values; ring and linear are exact. n / graph_size / max_branches
# are the paper's settings.
# ---------------------------------------------------------------------------
RING_YF = 20.0      # ring:   failure <=> x1^2 + x2^2 >= 20
LINEAR_YF = 5.0     # linear: failure <=> x1 + x2 >= 5

PROBLEMS = {
    "pwl":      dict(g=pwl, d=2, threshold=0, ref=3.18e-5, n=1000,
                     graph_size=15, max_branches=2, degenerate=degenerate_pwl),
    "high_pwl": dict(g=high_pwl, d=100, threshold=0, ref=3.18e-5, n=1000,
                     graph_size=15, max_branches=2, degenerate=degenerate_pwl),
    "mix":      dict(g=mix, d=2, threshold=0, ref=2.19e-4, n=750,
                     graph_size=25, max_branches=6, degenerate=degenerate_mix),
    "two_dof":  dict(g=two_dof, d=2, threshold=0, ref=2.48e-5, n=750,
                     graph_size=30, max_branches=2, degenerate=degenerate_two_dof),
    "ring":     dict(g=ring, d=2, threshold=RING_YF, ref=chi2.sf(RING_YF, 2), n=1000,
                     graph_size=15, max_branches=4, degenerate=None),
    "linear":   dict(g=linear, d=2, threshold=LINEAR_YF, ref=norm.sf(LINEAR_YF / np.sqrt(2)),
                     n=1000, graph_size=15, max_branches=2, degenerate=None),
}


# ===========================================================================
# Running
# ===========================================================================
def settings(problem):
    return dict(n=n or problem["n"],
                graph_size=GRAPH_SIZE or problem["graph_size"],
                max_branches=MAX_BRANCHES or problem["max_branches"])


def run_sus(problem, seed, verbose=False):
    s = settings(problem)
    return nsus.SubsetSimulation(performance_function=problem["g"],
                                 dimension=problem["d"],
                                 level_size=s["n"],
                                 level_probability=p,
                                 threshold=problem["threshold"],
                                 scale=SCALE,
                                 seed=seed,
                                 verbose=verbose).run()


def run_nsus(problem, seed, verbose=False):
    s = settings(problem)
    return nsus.HillValleyGraphNSuS(performance_function=problem["g"],
                                    dimension=problem["d"],
                                    level_size=s["n"],
                                    level_probability=p,
                                    threshold=problem["threshold"],
                                    graph_size=s["graph_size"],
                                    max_branches=s["max_branches"],
                                    scale=SCALE,
                                    seed=seed,
                                    verbose=verbose,
                                    is_cache=HVG_HISTORY).run()


def report(name, ss, problem, run_time):
    res = nsus.summarise(ss, problem["threshold"])
    print(f"--- {name} ---")
    print(f"Levels / branches    : {res['n_levels']} / {res['n_leaves']}")
    print(f"g evaluations N      : {res['N'] - res['N_hvg']} + {res['N_hvg']} (HVG)")
    print(f"pF estimate          : {res['pF']:.4e}")
    print(f"CoV estimate         : {res['cov']:.3f}")
    if problem["degenerate"] is not None:
        print(f"Degenerate run?      : {problem['degenerate'](ss)}")
    if len(res["branches"]) > 1:
        for k, br in enumerate(res["branches"]):
            print(f"  branch {k + 1}: levels {'-'.join(br['levels'])}, "
                  f"pF = {br['pF']:.3e} ({br['n_fail']} failure samples)")
    print(f"Run time             : {run_time:.2f} s")


def compare(problem, runs):
    """Repeat both algorithms with seeds 0..runs-1 (cf. Tables 1-4)."""
    print(f"\nRepeating each algorithm {runs} times ...")
    print(f"{'Method':7}{'Mean pF':>12}{'CoV pF':>9}{'Mean N':>16}{'Degenerate':>12}")
    for name, runner in [("SuS", run_sus), ("NSuS", run_nsus)]:
        pfs, evals, hv, degen = [], [], [], []
        for seed in range(runs):
            ss = runner(problem, seed)
            pfs.append(nsus.exceedance_probability(ss, problem["threshold"]))
            evals.append(ss.performance_function.eval_count)
            hv.append(getattr(ss.partitioner, "eval_count", 0))
            if problem["degenerate"] is not None:
                degen.append(problem["degenerate"](ss))
        pfs = np.array(pfs)
        n_str = f"{np.mean(evals) - np.mean(hv):.0f} + {np.mean(hv):.0f}"
        d_str = f"{np.mean(degen):.2f}" if degen else "-"
        print(f"{name:7}{pfs.mean():12.3e}{pfs.std() / pfs.mean():9.2f}{n_str:>16}{d_str:>12}")
    print(f"Reference pF = {problem['ref']:.3e}")


if __name__ == "__main__":
    problem = PROBLEMS[EXAMPLE]
    print(f"Example {EXAMPLE!r}: d = {problem['d']}, failure <=> g(x) >= {problem['threshold']}, "
          f"settings {settings(problem)}")

    start = time.perf_counter()
    sus = run_sus(problem, SEED)
    report("Subset simulation", sus, problem, time.perf_counter() - start)

    start = time.perf_counter()
    nss = run_nsus(problem, SEED, verbose=VERBOSE)
    report("Niching subset simulation", nss, problem, time.perf_counter() - start)
    print(f"Reference pF         : {problem['ref']:.4e}\n")
    nsus.print_tree(nss, problem["threshold"])

    if COMPARE_RUNS:
        compare(problem, COMPARE_RUNS)

    if problem["d"] == 2:
        import matplotlib.pyplot as plt
        res = 60 if EXAMPLE == "two_dof" else 150      # two_dof is slow to grid
        nsus_plot.plot_sus_vs_nsus(sus, nss, problem["threshold"], res=res,
                                   filename=SAVE_PLOT)
        # The first hill valley graph that led to a split, if there was one
        split = [lv for lv in nss.all_levels if len(lv._buds) > 1]
        if split:
            nsus_plot.plot_hill_valley_graph(nss, split[0].name)
        plt.show()
