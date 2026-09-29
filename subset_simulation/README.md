# Subset Simulation (Python)

Python translation of K.M. Zuev's MATLAB subset simulation code for the linear
reliability problem: g(x) = x1 + ... + xd, x ~ N(0, I), failure <=> g(x) > YF.

Requires `numpy`, `scipy`, `matplotlib`.

Two versions are provided, both for any dimension `d`:

- `subset_simulation` - vectorised (fast); use it for experiments.
- `subset_simulation_loops` - the same loops as the MATLAB script, with the
  MATLAB line numbers in the comments (slow: ~40 s for the original settings).

The level plot (cf. Figure 6) is drawn only when `d = 2`.

```bash
python subset_simulation.py                                  # MATLAB settings: d=1000, YF=200, n=3000
python subset_simulation.py --d 2 --YF 5 --n 1000 --seed 1   # 2D: also plots the levels
python subset_simulation.py --d 50 --YF 30 --p 0.2           # any other parameters
python subset_simulation.py --loops                          # loop-for-loop version
python subset_simulation.py --help                           # all parameters
```

The performance function is defined once, in `performance_function()`. In the
MATLAB script `sum(...)` appeared separately at lines 19 and 62 (and as
`sum(q)` at line 46). Here every evaluation calls this one function, so to try
a new model you only edit it in one place. You can also pass your own:

```python
import numpy as np
from subset_simulation import subset_simulation, plot_levels_2d

def my_g(x):                     # x has shape (d, n): one column per sample
    return x[0] + x[1] - 0.2 * x[0]**2

res = subset_simulation(d=2, YF=4.0, n=1000, p=0.1, g=my_g, rng=0)
print(res["pF_SS"], res["Y"], res["N"])
plot_levels_2d(res, YF=4.0, g=my_g)
```

(The "true value" printed by the script, 1 - Phi(YF/sqrt(d)), is only
valid for the default sum performance function.)
