# Assembling Kinnear's NSuS into one program

This guide takes you through building `Niching_Subset_Simulation.py` yourself from Kinnear's `nsus/` folder. Each step says:

- **Write**: the function to create in your single file.
- **From**: which of Kinnear's files to copy from, with line numbers in his repository.
- **Change**: what to change while copying. Mostly this is turning `self.something` and classes into plain arguments and dicts.
- **Answer**: where the finished version is in `Niching_Subset_Simulation.py`.

Work from top to bottom. Each step only uses things written in earlier steps.

---

## Step 0: The top of the file

**Write:** the docstring, imports, the `PARAMETERS` block and `performance_function`. Copy the parameter block from your `Subset_Simulation.py`.

**From:** the `import` lines at the top of each of Kinnear's files. Delete every import that starts with a dot (`from .level import Level` and so on), because everything is in one file now. The ones to keep:

```python
import numpy as np
from collections import Counter                                   # hill_valley_graph_partitioner.py
import networkx as nx                                             # hill_valley_graph_partitioner.py
from networkx.algorithms.community import asyn_lpa_communities    # hill_valley_graph_partitioner.py
from scipy.stats import multivariate_normal, norm, uniform        # level_create.py, markov_chain.py
from sklearn.decomposition import PCA                             # lsvc.py
from sklearn.metrics import balanced_accuracy_score               # hill_valley_graph_partitioner.py
from sklearn.preprocessing import StandardScaler                  # lsvc.py
from sklearn.svm import LinearSVC                                 # lsvc.py
```

**Drop:**
- `anytree`: a few lines of your own replace it (Step 3).
- `abc`: you won't have abstract classes.
- `dataclasses` and `copy`: not needed.

**Change:** `performance_function` must take **one** sample `x` of shape `(d,)`. Your `np.sum(x**2, axis=0)` already does.

**Answer:** lines 1–65.

---

## Section 1: Building blocks

### Step 1: `CountedFunction`, g with a memory

**From:** `performance_function.py`, lines 3–19 (the whole file).

```python
# Kinnear
def __call__(self, x):
    tuple_x = tuple(x)
    try:
        result = self.cache[tuple_x]
    except KeyError:
        self.cache[tuple_x] = result = self.non_cache_performance_function(tuple_x)
    return result
```

**Change:**
- Pass `np.array(key)` to `g` instead of the tuple, so array maths like `x**2` works.
- `deepcopy` isn't needed.
- The `eval_count` property is just `len(G.cache)`, so you don't need it.

**Answer:** line 72.

### Step 2: `in_partition`, `in_region`, `fraction_in`, the region test

**From:** `indicator.py`, lines 1–39 (the whole file), and `estimate.py` line 12 (`probability_of_indicator`).

His `Indicator` object holds a `threshold`, a `partition_indicator` (a chain of nested lambdas built by `indicator_combine`, line 25) and the performance function. The test is lines 8–16:

```python
# Kinnear
def threshold_indicator(self, x):
    return int(self.performance_function(x) >= self.threshold)
def indicator(self, x):
    if self.partition_indicator(x) == 1:
        if self.threshold_indicator(x) == 1:
            return 1
    return 0
```

**Change:** replace the object with a small dict, `region = {'threshold': b, 'parts': [(classifier, label), ...]}`. The nested lambdas become a plain list of partition sets you loop over:

| Kinnear | You |
|---|---|
| `indicator_combine(a, b)` (line 25) | `parts + [new_part]` |
| `child_indicator(b, part)` (line 30) | `{'threshold': b, 'parts': parent_parts + [part]}` |
| `new_threshold(b)` (line 37) | `{'threshold': b, 'parts': same_parts}` |
| `probability_of_indicator(ind, level)` (estimate.py:12) | `fraction_in(region, level, G)` |

Keep `>=` and return a plain `int`. Your old code used `>`, and returning a NumPy bool changes the CoV in the 16th digit.

**Answer:** lines 92–112.

### Step 3: `new_level`, `level_buds`, `preorder`, `unused_buds`, the tree

**From:** `level.py`, lines 5–107, and `level_create.py` lines 12–14 (`Sample`).

His `Level` subclasses `anytree.AnyNode`. The library supplies `parent`, `children`, `depth`, `root` and `PreOrderIter`. Each `Level` stores `chain_list`, a list of chains, each a list of `Sample(array, performance)` objects.

**Change:** a level becomes a dict. Store the samples as one `(d, n)` array like your `x[L]`, ordered chain after chain, plus a `y` array.

| Kinnear (`level.py`) | You (dict key or function) |
|---|---|
| `chain_list` / `sample_list` (line 62) | `level['x']` (d, n) and `level['y']` (n,) |
| `len(self.chain_list)` | `level['n_chains']` |
| `level_size` (line 54) | `level['n']` |
| `indicator` (line 25) | `level['region']` |
| `AnyNode(parent=...)` | `level['parent']`, and append to `parent['children']` |
| `depth` (anytree) | `level['depth'] = parent['depth'] + 1` |
| `name = str(root.size)` (line 24) | `level['name'] = len(levels) + 1` (the creation number) |
| `trivial_bud`, `_buds`, `buds` (lines 21–33) | `level['trivial_bud']`, `level['buds']`, `level_buds(level)` |
| `stop_info`, `is_stop` (line 74) | `level['stopped_by']`, a list of reasons |
| `extra_info` | `level['info']` |
| `PreOrderIter(root)` | `preorder(level)`, a 3-line recursive generator |
| `non_used_buds` (line 80) | `unused_buds(root)` |

Leave out these unused members: `sorted_list`, `unique_list`, `budget`, `acceptance_rate`, `non_used_non_stop_buds` (it becomes part of Step 9).

**Answer:** lines 115–161.

---

## Section 2: The SuS machinery

### Step 4: `make_bud`, seeds and the next threshold

**From:**
- `bud_create.py`: `Bud` (lines 4–26), `BudCreator.create` (lines 41–58), `sorted_candidate_seeds` (lines 83–87), `seeds_prob_exclusive` (lines 95–98).
- `scale.py` (lines 11–23, both scalers).

```python
# Kinnear, bud_create.py
sorted_candidate_seeds = [s for s in level.sample_list if partition_indicator(s.array)]  # sorted by performance
seed_number = int(self.level_probability * level.level_size)
seeds = sorted_candidate_seeds[-seed_number:]
threshold = seeds[0].performance
indicator = level.indicator.child_indicator(threshold, partition_indicator)
scale = self.scaler.compute_scale(seeds)
```

**Change:**
- `Bud` becomes a dict: `{'parent', 'seeds', 'region', 'scale', 'child'}`.
- Seeds are *indices* into the level, not `Sample` objects.
- The trivial bud is `part=None`. The original passes `lambda x: 1`.
- Both scalers fit in two lines: `np.array([scale]*d)`, or `np.std(seeds, axis=0)` when `scale is None`.
- Sort with Python's `sorted(..., key=lambda k: y[k])`. It is stable, like his, so ties come out in the same order.
- Leave out `seeds_prob_inclusive` (lines 89–93); it is never used.

This replaces your lines 73–80. The threshold is the worst seed's `g`, not the midpoint you used.

**Answer:** line 167.

### Step 5: `modified_metropolis`

**From:** `markov_chain.py`, `ModifiedMetropolis.update` (lines 54–72) and `scale_convert` (lines 74–77).

```python
# Kinnear
loguniform_samps = np.log(uniform.rvs(size=sample_shape, random_state=self.random_state))
proposal_samps = norm.rvs(size=sample_shape, scale=conversion, random_state=self.random_state)
state_list = [np.array([chain[-1] for chain in chain_data.chain_list])]
for i in range(length):
    current_state = state_list[-1]
    prop_state = current_state + proposal_samps[:, :, i]
    alpha = norm.logpdf(prop_state) - norm.logpdf(current_state)
    accept = alpha >= loguniform_samps[:, :, i]
    new_state = np.where(accept, prop_state, current_state)
    ind_accept = np.array([[bool(self.indicator(state))] for state in new_state])
    state_list.append(np.where(ind_accept, new_state, current_state))
```

**Change:**
- `self.indicator` becomes `in_region(c, region, G)`.
- `self.scale` and `self.random_state` become arguments.
- Return `np.stack(states, axis=1)` with shape `(nc, length+1, d)` instead of appending to `ChainData`. The `ChainData` class (lines 5–42) disappears.
- **Keep the order of the two `.rvs` calls**, all uniforms first and then all normals. Otherwise your random numbers won't match his.

This is the same algorithm as your lines 81–88, with log-probabilities and a `scale`.

**Answer:** line 201.

### Step 6: `chain_length`

**From:** `allocate.py`, `BranchingAllocator.allocate` (lines 28–30) and `budget` from `Allocator.__init__` (line 13).

```python
# Kinnear
self.budget = int((1 - self.level_probability) * self.level_size)
split_budget = self.budget / len(bud.parent_level.non_used_buds)
return int(split_budget / len(bud.seeds)) + 1
```

**Change:** none apart from making it a function, `int(int((1-p)*n) / len(unused_buds(root)) / len(bud['seeds'])) + 1`. You don't need `SubsetAllocator` (line 22, `1/p`): with one branch the formula above already gives `1/p`.

**Answer:** line 227.

### Step 7: `grow_level`

**From:** `level_create.py`. `LevelCreator.create` (lines 53–69) grows a level from a bud. `create_initial` (lines 36–51) makes level 1; you'll put that straight into the main function in Step 13.

**Change:**
- Start the chains from `bud['parent']['x'][:, bud['seeds']].T`.
- Run `modified_metropolis` with `length - 1` steps.
- Reshape `(nc, ns, d)` to `(d, nc*ns)` and call `new_level`.
- Set `bud['child'] = level`. His `Level.__init__` does this at `level.py` line 26.

This replaces your lines 91–94.

**Answer:** line 236.

### Step 8: `check_stop`

**From:** `stop.py`, `Failure.__call__` (lines 52–54) and `ThresholdConverge.__call__` (lines 61–63). The `Stopper` and `Condition` classes (lines 5–38) are only bookkeeping.

```python
# Kinnear
return bud.seeds[0].performance >= self.threshold                          # Failure
return bud.seeds[0].performance <= bud.parent_level.indicator.threshold    # ThresholdConverge
```

**Change:** two `if`s, returning a list of reason strings. Leave out `NoMovement` (line 66); it is never used.

This replaces your `while nF/n < p` test (line 71) and adds the protection against getting stuck.

**Answer:** line 248.

### Step 9: `choose_bud`

**From:** `choose.py`, `DepthChooser.choose` (lines 11–13), with `level.py` `non_used_non_stop_buds` (lines 89–93) and `nsus.py` `options` / `is_stop` (lines 106–114).

**Change:** filter `unused_buds(root)` to buds whose parent level has not stopped. Pick the first with the smallest depth. Return `None` when nothing is left, which is the signal to end the main loop.

**Answer:** line 263.

---

## Section 3: The niching part

### Step 10: `LinearClassifier`

**From:** `lsvc.py`: the three `Barebones…` classes (lines 16–38), `PipeLSVC` (lines 40–61) and `LSVCCreator.create` (lines 69–78). The SVM settings are in `implementation.py` line 81: `{'penalty': 'l1', 'C': 1, 'dual': 'auto', 'max_iter': 1000000}`.

**Change:**
- Fold the scaler, PCA and SVM into one small class with `fit`, `predict` and `predict_single`.
- Use plain sklearn objects instead of subclasses.
- Copy the maths of the three `transform_single` / `predict_single` methods exactly into your `predict_single`. Using sklearn's `transform` on one point rounds slightly differently and can flip a borderline point.
- Skip `BarebonesPipeline` (line 8); it is never used.

**Answer:** line 278.

### Step 11: `choose_graph_samples`, `hill_valley_graph`, `shrink_partition`

**From:** `hill_valley_graph_partitioner.py`:

| Kinnear | Lines | Becomes |
|---|---|---|
| `choose_samples` | 167–190 | `choose_graph_samples` (returns indices, not `Sample`s) |
| `HillValleyGraph.cache_fit` + `create_history_data` + `create_within_range` + `history_and_within_range` | 15–27, 52–71 | the `use_history=True` branch of `hill_valley_graph` |
| `HillValleyGraph.fit` | 29–35 | the `use_history=False` branch |
| `trivial_graph`, `adj_info_to_graph` | 37–50 | the first and last few lines of `hill_valley_graph` |
| `level_to_hvg` | 152–165 | merged into `hvg_partition` (Step 12) |
| `shrink_partition` | 192–198 | `shrink_partition`, unchanged |

**Change:**
- The `HillValleyGraph` class becomes one function that returns `(graph, number_of_new_evaluations)`.
- `perf_function.cache` becomes `G.cache`.
- In `choose_graph_samples`, keep both `rng.shuffle` calls and their order: shuffle each group, then the list of groups.

**Answer:** lines 307–391.

### Step 12: `hvg_partition`

**From:** `HillValleyGraphPartitioner.partition` (lines 95–132), plus `indicator_factory` (lines 134–138) and `fit_classifier` (lines 143–150).

**Change:**
- `self.max_branches`, `self.graph_size` and `self.random_state` become arguments.
- Instead of `indicator_factory` lambdas, return a list of `(classifier, label)` pairs. These are the `parts` your regions use (Step 2).
- Return `[None]` for "don't split". His version returns `[self.trivial_indicator]`.
- `fit_classifier` becomes a small inner `fit` function.

Keep these four details exactly:
1. The early return when `len(unused_buds) >= max_branches`.
2. `graph_size // n_open`.
3. The "more than 50% single-community" rule.
4. `max_parts = max_branches - n_open + 1`.

**Answer:** line 394.

---

## Section 4: The estimate

### Step 13 (part 1): `branch`, `leaf_probability`, `trimmed_leaves`

**From:** `estimate.py`: `exceedance_probability` (lines 15–17), `estimate_trimmed_leaf_probability` (lines 30–35), `trimmed_leaves` (lines 37–52). `Level.branch` is `level.py` line 58.

```python
# Kinnear
return (np.prod([probability_of_indicator(level.indicator, level.parent) for level in branch[1:]])
        * probability_of_indicator(leaf.indicator.new_threshold(threshold), leaf))
```

**Change:** `probability_of_indicator` becomes `fraction_in`, and `.indicator` becomes `['region']`. `branch(level)` walks `parent` links up to the root and reverses the list. Keep `trimmed_leaves`' stack (`pop` / `extend`) as is, so leaves are summed in the same order.

This replaces your line 96.

**Answer:** lines 443–472.

### Step 13 (part 2, optional): the CoV

**From:** `estimate.py`: `cov` (lines 54–79), `find_common_ancestor` (81–84), `leaf_cov` (86–91), `level_cov` (93–101), `level_gamma` (103–126).

**Change:**
- `level.order == 1` becomes `level['parent'] is None`.
- Skip terms whose weight is 0. Without this, `cov` crashes with a division by zero when a branch never reaches failure, as on `mix`.

Leave out `trim_leaf`, `estimate_leaf_probability`, `level_ess`, `efficiency`, `leaf_efficiency` and the reliability-index functions. Your program doesn't need them.

**Answer:** lines 475–528.

---

## Section 5: The main function

### Step 14: `niching_subset_simulation`

**From:**
- `nsus.py`: `NSuS.run` (lines 59–73).
- `implementation.py`: `HillValleyGraphNSuS.__init__` (lines 68–143), which says which parts to plug in.
- `level_create.py`: `create_initial` (lines 36–51).

```python
# Kinnear, nsus.py
def run(self):
    while not self.is_stop:
        if self.is_initial:
            self.initial_level = self.level_creator.create_initial()
            level = self.initial_level
        else:
            bud = self.chooser.choose(self.options)
            level = self.level_creator.create(bud)
        self.bud_creator.create_trivial(level)
        self.stopper.stop(level)
        if not level.is_stop:
            partition_indicators = self.partitioner.partition(level)
            if len(partition_indicators) > 1:
                self.bud_creator.create_inidicator_buds(level, partition_indicators)
```

**Change:** write it as your function, with your parameters as arguments:

1. `rng = np.random.default_rng(seed)`, then **immediately** `svc_seed = rng.integers(0, 1000000)`. This is `LSVCCreator.__init__`, `lsvc.py` line 67, which runs when his object is built. Skip it when `max_branches == 1`, to match his `SubsetSimulation`.
2. Level 1: `multivariate_normal.rvs(mean=zeros, cov=I, size=n, random_state=rng)`, transposed to `(d, n)`. Use this, not `rng.standard_normal`, or the numbers won't match.
3. The loop, in Kinnear's order: trivial bud, stop check, partition if not stopped, buds per partition set, then `choose_bud`, ending with `break` when it returns `None`, then `grow_level`.
4. After the loop: `pF = sum(leaf_probability(leaf) for leaf in trimmed_leaves(root, YF))`, `N = len(G.cache)`. Return a dict like your `subset_simulation` does.

You can leave out these parts of `nsus.py`: `bulk` (optional post-processing), `find`, `all_samples`, and the `verbose` decorator from `utils.py` (an `if verbose: print(...)` does the same).

**Answer:** line 534.

---

## Step 15: Output

**From:** nothing you need. His `plot.py` only scatters the levels. Adapt your own `plot_levels_2d`: colour the samples by branch, and draw each split level's classifier boundary with `ax.contour` on `classifier.predict(grid)`.

**Answer:** `print_tree` (line 622), `plot_levels_2d` (line 644) and the run block (line 732).

---

## Files you never need to open

| File | Why |
|---|---|
| `__init__.py` | empty |
| `utils.py` | one `verbose` print helper |
| `partition.py` | `NoPartitioner` is the same as `max_branches = 1` |
| `plot.py` | basic plots |
| `implementation.py` → `SubsetSimulation` | ordinary SuS, the same as `max_branches = 1` (handy for checking against, Step 6) |

## Checking as you go

After Step 9 your program is ordinary SuS (`max_branches=1`). Compare it with his `SubsetSimulation`. After Step 14, compare with his `HillValleyGraphNSuS`. Use the same seed for both:

```python
import sys; sys.path.insert(0, "path/to/nsus_paper")
from nsus.implementation import HillValleyGraphNSuS
import nsus.estimate as est

his = HillValleyGraphNSuS(performance_function=lambda x: g(np.array(x)),  # he passes a tuple
                          dimension=2, level_size=1000, threshold=0, level_probability=0.1,
                          seed=1, graph_size=15, max_branches=2, scale=1, verbose=False)
his.run()
mine = niching_subset_simulation(2, 0.0, n=1000, g=g, graph_size=15, max_branches=2, rng=1)
print(est.exceedance_probability(his, 0), mine['pF_NSuS'])
print(his.performance_function.eval_count, mine['N'])
```

If you followed every "keep" above, p_F and N agree to the last digit.
