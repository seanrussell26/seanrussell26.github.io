# Rethinking multimodal sampling via graph topology

Every script takes a performance function `g` with the same convention:
`g(x)` gets an `(n, d)` array of points and returns an `(n,)` array, with failure meaning `g(x) > YF`.
Each file has its parameters at the top and runs on its own.

| File | What it does |
| --- | --- |
| `Performance_Functions.py` | The test functions (`linear_sum`, `sum_of_squares`, `two_gaussians`, `wave`, `random_bumps`) and `true_pf` |
| `Subset_Simulation.py` | `subset_simulation(d, YF, n, p, g)` and the 2D level plot |
| `Kinnear_NSuS.py` | H. Kinnear's NSuS code, copied verbatim. Don't edit it. |
| `Niching_Subset_Simulation.py` | `run_nsus(d, YF, n, p, g, ...)` plus the niche and branch-tree plots |
| `Gaussian_Process_Emulator.py` | `fit_gpe(g, d, ...)`. `emulator(gp)` turns the fitted GP into a `g` you can use anywhere. |
| `Euler_Characteristic.py` | `grid_values`, `ec_curve` and `euler_characteristic` (any dimension) plus the filtration plots |
| `Run_Experiment.py` | Runs everything together: SuS thresholds, then the EC of `{g > Y}` for the true g and the GPE, then NSuS branches |

```
pip install numpy scipy scikit-learn networkx anytree matplotlib
python Run_Experiment.py
```
