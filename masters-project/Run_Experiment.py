"""
One experiment using every script together.

    1. Subset simulation on g gives the intermediate thresholds Y_1, ..., Y_L (then YF).
    2. The Euler characteristic of each level's domain {g > Y} is computed on a grid,
       for the true g and for a Gaussian process emulator trained on a few runs of g.
    3. Niching subset simulation is run, so its number of branches can be compared with the EC.

HOW TO RUN: set the parameters just below, then run this file.
"""

import matplotlib.pyplot as plt
import numpy as np

from Euler_Characteristic import ec_curve, grid_values, plot_ec_curve
from Gaussian_Process_Emulator import emulator, fit_gpe
from Kinnear_NSuS import exceedance_probability
from Niching_Subset_Simulation import run_nsus
from Performance_Functions import true_pf, two_gaussians
from Subset_Simulation import subset_simulation

# ===========================================================================
# PARAMETERS
# ===========================================================================
d = 2                  # dimension of the input space
YF = 0.03              # critical threshold (failure <=> g(x) > YF)
n = 3000               # number of samples per level
p = 0.1                # level probability
SEED = None            # random seed
g = two_gaussians      # performance function (see Performance_Functions.py)

LOWER, UPPER = -8, 8   # box covered by the GPE training runs and the EC grid
N_TRAIN = 60           # runs of g used to train the GPE
GRID = 100             # EC grid points per dimension (GRID ** d points in total)
MAX_CELL_DIM = 2       # 1 = graph chi (V - E), d = chi of the region (components - holes)
USE_EMULATOR = False   # True: run SuS and NSuS on the GPE instead of g


if __name__ == "__main__":
    gp = fit_gpe(g, d, LOWER, UPPER, N_TRAIN, seed=SEED)
    g_run = emulator(gp) if USE_EMULATOR else g

    sus = subset_simulation(d, YF, n=n, p=p, g=g_run, rng=SEED)
    nss = run_nsus(d, YF, n=n, p=p, g=g_run, seed=SEED)

    thresholds = np.append(sus["Y"], YF)
    shape = (GRID,) * d
    chi_true = ec_curve(grid_values(g, LOWER, UPPER, shape), thresholds,
                        superlevel=True, max_cell_dim=MAX_CELL_DIM)
    chi_gpe = ec_curve(grid_values(emulator(gp), LOWER, UPPER, shape), thresholds,
                       superlevel=True, max_cell_dim=MAX_CELL_DIM)

    print(f"{'level':>5}  {'threshold':>10}  {'chi (g)':>7}  {'chi (GPE)':>9}")
    for L, (Y, c_true, c_gpe) in enumerate(zip(thresholds, chi_true, chi_gpe), start=1):
        name = "F" if L == len(thresholds) else str(L)
        print(f"{name:>5}  {Y:10.3e}  {c_true:7d}  {c_gpe:9d}")
    print(f"\nNSuS branches        : {len(nss.leaves)}")
    print(f"pF (subset sim)      : {sus['pF_SS']:.4e}")
    print(f"pF (NSuS)            : {exceedance_probability(nss, YF):.4e}")
    if true_pf(g, YF, d) is not None:
        print(f"pF (true value)      : {true_pf(g, YF, d):.4e}")

    ax = plot_ec_curve(thresholds, chi_true, xlabel="threshold $Y$", label="true $g$")
    plot_ec_curve(thresholds, chi_gpe, xlabel="threshold $Y$", ax=ax, label="GPE", color="#2a78d6",
                  title=r"EC of the subset simulation domains $\{g > Y\}$")
    ax.axvline(YF, color="k", ls="--", lw=0.8)
    ax.legend()
    plt.show()
