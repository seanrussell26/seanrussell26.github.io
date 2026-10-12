# From subset simulation to niching subset simulation

This folder holds:

| File | What it is |
|---|---|
| `Subset_Simulation.py` | Your original code, unchanged. |
| `nsus.py` | A port of Hugh Kinnear's NSuS package ([HughKinnear/nsus_paper](https://github.com/HughKinnear/nsus_paper), folder `nsus/`) in a single file. The classes and logic are the same, and so is the order of random draws. |
| `nsus_plot.py` | 2D plots: levels coloured by branch, partition boundaries, the hill valley graph. |
| `examples.py` | The paper's test problems plus your two functions, with a parameter block at the top like your script. |

Install the dependencies with `pip install -r requirements.txt`. Then set `EXAMPLE` at the top of `examples.py` and run it.

The guide below covers how your code works, how Kinnear's code does the same thing, and what NSuS adds. Equation numbers refer to Kinnear & DiazDelaO (2025), *Niching subset simulation*, Prob. Eng. Mech. 79, 103729.

---

## 1. Subset simulation in one paragraph

We want $p_F = P(g(x) \ge b)$ for $x \sim N(0, I_d)$. When $p_F$ is tiny (say $10^{-6}$), plain Monte Carlo needs about $10^{8}$ samples to see any failures. Subset simulation (SuS) writes the rare event as a chain of nested, less rare events:

$$F_1 \supset F_2 \supset \dots \supset F_m = F, \qquad F_k = \{g(x) \ge b_k\},$$

and uses the product rule: $p_F = P(F_1)\,P(F_2\mid F_1)\cdots P(F\mid F_{m-1})$. The thresholds $b_k$ are chosen *adaptively* so each factor is about $p = 0.1$. Each factor is then easy to estimate. The hard part is drawing samples *conditional* on being in $F_{k}$, and that is what the Markov chains (modified Metropolis) are for.

## 2. Your code, line by line

`subset_simulation` in `Subset_Simulation.py`:

| Lines | What happens |
|---|---|
| 59–62 | `nc = n p` chains per level, each with `ns = (1-p)/p` new states. The seed plus `ns` states gives `nc (ns+1) = n` samples per level. |
| 66–68 | **Level 0**: `n` plain Monte Carlo samples, their responses `y = g(x)`, and the number of failures `nF`. |
| 71 | Keep going while fewer than a fraction `p` of the level fails. When at least `n p` samples fail you can estimate the last factor directly. |
| 73–75 | Sort the level by response, largest first. |
| 76 | Intermediate threshold `Y_L` = midway between the `nc`-th and `(nc+1)`-th largest responses, so exactly `nc` samples (a fraction `p`) lie above it. |
| 79–80 | The `nc` best samples become the **seeds** of `nc` Markov chains. They are already distributed as $N(0,I)$ conditioned on $F_L$ (perfect sampling, so no burn-in). |
| 83–85 | **Modified Metropolis step 1**: for every chain and every component separately, propose `a = z + N(0,1)` and accept with probability `min(1, φ(a)/φ(z))`. This step alone leaves $N(0,1)$ invariant. |
| 87–88 | **Step 2**: keep the candidate only if `g(q) > Y_L`, otherwise the chain repeats its state. Together these give a chain whose stationary distribution is $N(0,I)$ restricted to $F_L$. |
| 91–94 | All chain states form the next level. Evaluate `g` and count failures. |
| 96 | $\hat p_F = p^L \cdot nF_L / n$: each conditional factor is exactly `p` by construction, and the last one is counted. |
| 97 | Cost: `n` for level 0 plus `n(1-p)` new samples per conditional level. |

Three things I noticed while reading it. I did not change your file.

* The docstring says $g = x_1 + \dots + x_d$, but `performance_function` computes $x_1^2 + \dots + x_d^2$. The "true value" on line 206 uses the formula for the sum. For the sum of squares the exact answer is `scipy.stats.chi2.sf(YF, d)` (for d = 2 that is $e^{-YF/2}$). With `YF = 9` that gives 0.0111, while line 206 prints about $10^{-10}$.
* The simulation runs twice: once at line 200, outside `if __name__ == "__main__"`, and again at line 204. `run_time` times the first run, but the second run is the one printed and plotted.
* If the chains get stuck and the threshold stops increasing, the `while` loop never ends. Kinnear's code adds a second stopping rule for this case (`ThresholdConverge`, §4).

## 3. Why SuS can fail: ergodicity

SuS is **greedy**. At each level it keeps only the globally best `n p` samples. Suppose the region that matters most at the end (around the *design point*, the most probable failure point) is not where `g` is largest early on. Then the seeds all go somewhere else, and no chain ever reaches the important region.

The piecewise linear function (`pwl`, Eq. 9) is built to show this. Near the origin, `g` increases fastest in the $+x_2$ direction, so SuS climbs upward. The design point, however, is at $x_1 = 4$, $x_2 \le 2$, which you reach by moving in $+x_1$. About two thirds of SuS runs never sample near the design point. Those runs underestimate $p_F$ by orders of magnitude, so the estimator has a bimodal distribution (paper Fig. 2).

`mix` is worse. Three of the four bumps are local maxima that never reach failure, and SuS often returns $p_F = 0$.

Run `examples.py` with `EXAMPLE = "pwl"` and `COMPARE_RUNS = 20` to see this. I got:

```
Method      Mean pF   CoV pF          Mean N  Degenerate
SuS       4.081e-05     3.01        5176 + 0        0.65
NSuS      3.655e-05     1.07        4972 + 4        0.00
Reference pF = 3.180e-05
```

Paper Table 1 reports SuS degenerate weight 0.65 and NSuS CoV 1.10 with 4934 + 4 evaluations, so these match.

## 4. Your algorithm written in Kinnear's framework

Kinnear's code doesn't write SuS as one loop. It builds SuS from replaceable parts and puts them in a general framework, `NSuS`. The class `SubsetSimulation` in `nsus.py` is ordinary SuS assembled from those parts. Here is how your lines map onto it:

| Your code | Kinnear's code (`nsus.py`) |
|---|---|
| `x[L]`, `y[L]` arrays | `Level` object: a list of chains, each a list of `Sample(array, performance)` |
| line 66, level 0 | `LevelCreator.create_initial` |
| lines 73–80, sort and choose seeds | `BudCreator.create` + `seeds_prob_exclusive` |
| line 76, threshold `Y_L` | `threshold = seeds[0].performance` (the worst seed) |
| line 87, `g(q) > Y_L` | `Indicator`: `g(x) >= threshold` (and inside the partition set, see §5) |
| lines 81–88 | `ModifiedMetropolis.update` |
| chain length `ns + 1` | `SubsetAllocator.allocate`, which returns `1/p` |
| line 71, stopping | `Failure` (worst seed already fails, i.e. ≥ `n p` failures) and `ThresholdConverge` (threshold stopped rising) |
| line 96, estimate | `exceedance_probability` |
| line 97, cost `N` | `PerformanceFunction.eval_count` (a cache: counts **distinct** points evaluated) |

There are some small differences from your version.

1. **Threshold.** Yours is the midpoint between the `nc`-th and `(nc+1)`-th responses, with `>`. Kinnear's is the `nc`-th response itself, with `>=`. Both put the seeds inside the next region.
2. **Estimate.** You use $p^L$, assuming each factor is exactly `p`. Kinnear recounts each factor as the actual fraction of the parent level inside the child's region. That can be slightly above `p`, because Markov-chain levels contain repeated points (rejected moves), so ties at the threshold are common.
3. **Cost.** The cache means a rejected move (chain stays put) costs no new evaluation. In a branch, a proposal outside the partition set is rejected before `g` is evaluated (see `Indicator.indicator`). So `N` is usually below your $n + n(1-p)L$.
4. **Proposal spread.** `scale=1` is your $N(0,1)$ proposal. `scale=None` uses the standard deviation of the seeds in each direction (`AdaptiveScaler`).
5. **Coefficient of variation.** `cov` computes the usual SuS CoV, including the chain-correlation factor $\gamma_k$ of Au & Beck (`level_gamma`, Eqs. 6–8). Your code doesn't compute a CoV.

## 5. What NSuS adds

The idea: **at each level, split the input space into regions (niches) and run a separate SuS in each one.** Each branch picks its own *locally* best samples as seeds, so a niche that looks poor early on (the $+x_1$ direction in `pwl`) still gets explored.

### 5.1 The pieces

| Piece | Class | Role |
|---|---|---|
| Level | `Level` | A node of a **tree** (paper Fig. 3). SuS gives a chain of levels; NSuS gives a tree. Levels are named 1, 2, 3, … in the order they were created. Level 1 is the paper's $X^0$. |
| Bud | `Bud` | A level that could be grown next. It stores the seeds, the region indicator and the proposal scale. Each level gets a trivial bud (plain SuS, no split). If the partitioner splits the level into *k* sets, it gets *k* buds instead. |
| Indicator | `Indicator` | Region of a bud: $A \cap \{g \ge b\}$, where $A$ is the partition set. With nested splits, $A$ is the **product** of all the set indicators along the branch (`indicator_combine`), so a chain stays inside every split above it. |
| Partitioner | `NoPartitioner` / `HillValleyGraphPartitioner` | Given a level, return a list of set-indicator functions. A list of length 1 means "don't split". |
| Allocator | `BranchingAllocator` | Splits the budget. Each step of the algorithm creates $n(1-p)$ new samples, as SuS does. They are divided equally among all open buds in the tree, then among the seeds: $n_s = \lfloor n(1-p)/(N_p n_c) \rfloor + 1$ (Eq. 10). |
| Chooser | `DepthChooser` | Picks which open bud to grow next: the shallowest one (breadth-first). |
| Stopper | `Failure`, `ThresholdConverge` | Applied to each new level. A stopped level's bud is never grown, so that branch ends. |

### 5.2 The main loop (`NSuS.run`, Algorithm 3)

```
create level 1 by Monte Carlo
loop:
    give the new level its trivial bud            (best n·p samples, new threshold)
    check stopping conditions on the new level
    if not stopped:
        ask the partitioner for a partition
        if it returned more than one set:
            replace the trivial bud by one bud per set
            (seeds = best samples *within that set*, threshold = worst of those)
    if no open, unstopped buds remain: finish
    choose a bud (shallowest first) and grow a level from it with modified Metropolis
```

`SubsetSimulation` uses `NoPartitioner`, which never splits. The tree is then a single chain, and the loop is exactly your SuS.

Here is a real `pwl` run (`examples.py`, seed 5), printed with `nsus.print_tree`:

```
level 1: 1000 samples, b = -inf
└── level 2: 1000 samples, b = -0.678
    ├── level 3: 500 samples, b = -0.2661          ← the +x2 niche (where SuS goes)
    │   └── 5 → 7 → 9 → 10 → level 11  [Failure]   pF contribution 1.1e-07
    └── level 4: 500 samples, b = -0.6116          ← the +x1 niche (design point)
        └── 6 → level 8  [Failure]                 pF contribution 1.1e-05
```

Level 2 was split into two niches. Each branch got half the budget, so it holds 500 samples per level. Level 4's threshold is lower than level 3's because it was chosen from the best samples **in its own set**. In SuS those samples would have been thrown away.

### 5.3 Estimating $p_F$ from a tree (§3.3, `estimate.py` part of `nsus.py`)

Each leaf is the end of a branch, and that branch is a SuS run inside its partition set $A_i$. It estimates $P(A_i \cap F)$ as a product of conditional fractions down the branch (`estimate_trimmed_leaf_probability`). The partition sets are disjoint, so

$$\hat p_F = \sum_{\text{leaves } i} \hat P_i \quad \text{(Eq. 12, } \texttt{exceedance\_probability}\text{)}.$$

**Trimming** (`trimmed_leaves`) handles thresholds other than the one the run stopped at. Walking down from the root, if any child of a level already has an intermediate threshold above the $b$ you asked for, that level becomes a leaf for this estimate. This also lets you compute the whole CCDF $P(g \ge b)$ for any $b$ from one run.

**CoV** (`cov`, Eq. 13) is $\sqrt{\sum_{i,j} w_{ij}\,\delta_{ij}^2}$ with $w_{ij} \propto \hat P_i \hat P_j$. The term $\delta_{ij}$ is the CoV of the part of the tree shared by branches *i* and *j* (their common ancestor). Two branch estimates are correlated only through the levels they share.

## 6. The hill valley graph (HVG) partitioner (§4)

This is how NSuS decides where the niches are. It has three stages (paper Fig. 4).

**Choose graph samples** (`choose_samples`). Take `graph_size` samples from the level, divided among the open branches. Samples are spread over *chain groups*, meaning chains that started from the same seed and so probably share a niche: shuffle the groups and take one sample from each in turn.

**Build the graph** (`HillValleyGraph`). For each pair $x_i, x_j$, ask whether there is a "valley" between them, i.e. a point on the segment with $g < \min(g(x_i), g(x_j))$. If there isn't, both probably lie in the same connected piece of the region, so connect them. Two versions are implemented:

* `fit` (`is_cache=False`): evaluate `g` at the midpoint of each pair, which costs $n_g(n_g-1)/2$ evaluations.
* `cache_fit` (default): first look in the **search history**, every point at which `g` has already been evaluated. If some old points lie in the box spanned by the pair, use them: a valley exists if any of them is below $\min(g_i, g_j)$. Only pairs whose box contains no old point get a midpoint evaluation. In 2D this is almost free; the paper reports 4 evaluations for `pwl`. In 100D, boxes are almost always empty, so it falls back to midpoints (about 225).

**Find communities** with asynchronous label propagation (`asyn_lpa_communities` in networkx). This finds groups of nodes that are densely connected to each other and sparsely connected to the rest. It is random, so it runs 100 times. If more than half of the runs find a single community, the level is not split. Otherwise any labelling with too many communities has random pairs merged (`shrink_partition`) until the total number of branches stays within `max_branches`.

**Classify** (`LSVCCreator`, `PipeLSVC`). A community labelling only covers the graph samples, but the Markov chains need to know which set *any* point $x$ belongs to. So a linear SVM is trained on the labelled graph samples. The pipeline is standardise → PCA → L1-penalised `LinearSVC`, with standardisation and PCA fitted on the whole level. It is trained once per candidate labelling, and the labelling the SVM reproduces best (balanced accuracy) wins. The partition sets are then "the SVM says class *k*" (`indicator_factory`). These are the straight lines in the plots.

The graph is built from `g` values only, with no distances between points. That is why the method still works in 100 dimensions (`EXAMPLE = "high_pwl"`).

## 7. Using it on your own problems

```python
import nsus, nsus_plot

def g(x):                       # x is ONE sample, shape (d,)
    return np.sum(x**2)         # your performance function

nss = nsus.HillValleyGraphNSuS(g, dimension=2, level_size=1000,
                               threshold=9.0, graph_size=15,
                               max_branches=4, seed=1, verbose=True).run()
res = nsus.summarise(nss, 9.0)   # pF, cov, N, branches, ...
nsus.print_tree(nss, 9.0)
nsus_plot.plot_levels_2d(nss, 9.0)
nsus_plot.plot_hill_valley_graph(nss, "2")   # the graph built on level 2
```

Your `performance_function` works unchanged, because `np.sum(x**2, axis=0)` also works on a single 1-D sample. `nsus.SubsetSimulation(...)` takes the same arguments without `graph_size` / `max_branches`.

Settings worth changing:

* `max_branches`: 1 gives SuS behaviour. More branches can find more niches, but the budget is divided between them.
* `graph_size`: a bigger graph detects niches more reliably but costs more `g` evaluations in high dimensions.
* `level_size`: samples per level before any split. After a split, each step's budget is shared, so each branch's levels are smaller (500 per branch in the `pwl` tree above).
* `scale`: proposal spread. `None` for adaptive.
* `HVG_HISTORY` / `is_cache`: see below.

Things I saw while testing that are worth trying yourself:

* **`linear`** (your docstring's $g = \sum x_i$) has a convex failure region, so ideally NSuS wouldn't split. With the default history-based test it splits in 10 of 10 runs. The box between two samples sticks out of the half-space $\{g \ge b\}$, and old points in that corner look like a valley. With `HVG_HISTORY = False` it splits in 0 of 10. In both cases the estimate is still fine, because splitting a region you didn't need to split only costs efficiency. This is a property of the approximate test in the original code, not something the port introduced.
* **`ring`** (your $\sum x_i^2$) has no single design point. NSuS cuts the circle into arcs, and each branch contributes a share of $p_F$.
* **`mix`**: NSuS finds all four bumps and stops the three that never reach failure (`ThresholdConverge`). SuS sometimes follows a wrong bump and returns 0.

## 8. How faithful the port is

When run with the same seed, `nsus.py` gives **bit-for-bit the same results** as Kinnear's package. I checked $p_F$, the number of evaluations, the number of branches, the HVG evaluations and the CoV for `pwl` and `mix`, seeds 0–2. The structure follows his files one to one; each section header in `nsus.py` names the file it came from. The deliberate changes are listed at the top of `nsus.py` and marked `PORT NOTE` in the code:

1. `anytree` is replaced by a small `TreeNode` class.
2. `g` receives a NumPy array instead of a tuple.
3. `cov` no longer crashes when a branch has zero failure probability. The original raises `ZeroDivisionError` on, e.g., `mix` with seed 0. Those terms have weight 0 in Eq. 13, so the value is unchanged.
4. Arguments have defaults, and the ALP iteration count and SVC parameters can be overridden. The defaults are the original values.
5. The midpoint-only hill valley test now counts its evaluations in `partitioner.eval_count`. The original reports 0 there.
6. Unused helpers are left out. Plotting (`nsus_plot.py`), `summarise` and `print_tree` are new.

## 9. Symbol ↔ code cheat sheet

| Paper | Code |
|---|---|
| $n$ | `level_size` |
| $p$ | `level_probability` |
| $X^k$ | a `Level` (`level.sample_list`) |
| $b_k$, $F_k$ | `level.indicator.threshold`, `level.indicator` |
| $A$ | `indicator.partition_indicator` |
| $n_c$, $n_s$ | `len(bud.seeds)`, `allocator.allocate(bud)` |
| $N_p$ | `len(level.non_used_buds)` |
| $n_g$ | `graph_size` |
| $N_p^{\max}$ | `max_branches` |
| Stop$(\cdot)$ | `Stopper` with `Failure`, `ThresholdConverge` |
| Step$_{f_s}$ | `ModifiedMetropolis.update` |
| Partition$(X)$ | `partitioner.partition(level)` |
| Choose$(T)$ | `chooser.choose(options)` |
| $\hat P_F$, $\hat\delta$ | `exceedance_probability`, `cov` |
