# Subset Simulation (Python)

Python translation of K.M. Zuev's MATLAB subset simulation code for the linear
reliability problem: g(x) = x1 + ... + xd, x ~ N(0, I), failure <=> g(x) > YF.

| File | What it is |
|---|---|
| `subset_simulation_loops.py` | Same loops as the MATLAB script, with MATLAB line numbers in the comments. Closest to the original; slow in high dimension (~40 s for d = 1000). |
| `subset_simulation_vectorised.py` | Same algorithm with the loops replaced by NumPy array operations (~10x faster). Use it for experiments. |

Both files are standalone, work for any dimension `d`, and draw the level plot
(cf. Figure 6) when `d = 2`.

## Running

1. Install the packages once: `pip install numpy scipy matplotlib`
2. Open either file and edit the **PARAMETERS** block at the top
   (`d`, `YF`, `n`, `p`, `SEED`, `SAVE_PLOT`). The original MATLAB values are
   shown in brackets.
3. Run the file (the Run button in Visual Studio / VS Code, or
   `python subset_simulation_loops.py`). The results are printed and, for
   `d = 2`, the plot opens in a separate window. Set `SAVE_PLOT = "levels.png"`
   to also save it.

## Changing the model

The performance function is defined once, in `performance_function()`. In the
MATLAB script `sum(...)` appeared separately at lines 19 and 62 (and as
`sum(q)` at line 46). Here every evaluation calls this one function, so a new
model only needs editing in one place. Note that the printed "true value",
1 - Phi(YF/sqrt(d)), is only valid for the default sum.
