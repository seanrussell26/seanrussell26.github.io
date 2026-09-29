# Subset Simulation (Python)

Subset simulation (Au & Beck, 2001) with the modified Metropolis algorithm,
posed in standard normal space. Failure is `g(u) <= 0`.

Requires `numpy`, `scipy`, `matplotlib`.

```bash
python subset_simulation.py --seed 1 --beta 3.5 --save levels_2d.png   # 2D + level plot
python subset_simulation.py --dim 50 --beta 4 --no-plot                # high dimension
python subset_simulation.py --help                                     # all parameters
```

The limit-state function is defined once in `limit_state()`. The same function
is used for the level-0 Monte Carlo samples and inside the Markov chains, so
you only need to change the model in one place. You can also pass any function
of your own:

```python
from subset_simulation import subset_simulation, plot_levels_2d

def my_g(u):                      # u has shape (n, dim)
    return 3 - u[:, 0]**2 / 4 - u[:, 1]

res = subset_simulation(my_g, dim=2, N=1000, p0=0.1, rng=0)
print(res["pf"], res["cov"], res["thresholds"])
plot_levels_2d(res, my_g)
```
