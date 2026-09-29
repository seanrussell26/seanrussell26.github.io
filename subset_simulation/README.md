# Subset Simulation (Python)

Python translation of K.M. Zuev's MATLAB subset simulation code for the linear
reliability problem: g(x) = x1 + ... + xd, x ~ N(0, I), failure <=> g(x) > YF.

Requires `numpy`, `scipy`, `matplotlib`.

```bash
python subset_simulation.py --seed 1 --save levels_2d.png                # 2D demo + level plot (cf. Figure 6)
python subset_simulation.py --d 1000 --YF 200 --n 3000 --no-plot         # original MATLAB settings
python subset_simulation.py --help                                       # all parameters
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
