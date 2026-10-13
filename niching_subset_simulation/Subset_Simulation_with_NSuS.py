"""
Subset Simulation for a Reliability Problem

    Performance function: g(x) = x1 + ... + xd
    Input variables x1, ..., xd are i.i.d. N(0, 1)
    Failure  <=>  g(x) > YF

HOW TO RUN: set the parameters just below, then run this file. If using a different performance function, 
and if true value is unknown, disable that result on lines 201/207) 
"""

import numpy as np
from scipy.stats import norm, multivariate_normal, uniform
import time
from collections import Counter
import networkx as nx
from networkx.algorithms.community import asyn_lpa_communities
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


# ===========================================================================
# PARAMETERS
# ===========================================================================
d = 2          # dimension of the input space            
YF = 20.0      # critical threshold (failure <=> g(x) > YF) 
n = 1000       # number of samples per level           
p = 0.1        # level probability                      
SEED = None    # random seed: an integer for repeatable results, None for random
SAVE_PLOT = None   # e.g. "levels.png" to also save the plot to a file
GRAPH_SIZE = 15    # NSuS: samples in each hill valley graph
MAX_BRANCHES = 4   # NSuS: maximum number of branches (1 = plain SuS)
SCALE = 1.0        # NSuS: proposal std of the Markov chains (None = adaptive)
HVG_HISTORY = True # NSuS: hill valley tests reuse earlier g evaluations

# ---------------------------------------------------------------------------
# Performance function
# ---------------------------------------------------------------------------
def performance_function(x):
    return np.sum(x**2, axis=0)

# ---------------------------------------------------------------------------
# Subset simulation
# ---------------------------------------------------------------------------
def subset_simulation(d, YF, n=3000, p=0.1, g=performance_function, rng=None):
    """Estimate pF = P(g(x) > YF) for x ~ N(0, I_d).

    Parameters
    ----------
    d   : dimension of the input space
    YF  : critical threshold (failure <=> g(x) > YF)
    n   : number of samples per level
    p   : level probability
    g   : performance function taking a (d, n) array and returning shape (n,)
    rng : seed or np.random.Generator

    Returns
    -------
        'pF_SS' : subset simulation estimate of the failure probability
        'N'     : total number of samples (performance function evaluations)
        'L'     : number of conditional levels used
        'Y'     : intermediate thresholds Y_1, ..., Y_L
        'nF'    : number of failure samples at each level 0, ..., L
        'x'     : list of (d, n) sample arrays, one per level 0, ..., L
        'y'     : list of (n,) response arrays, one per level 0, ..., L
    """
    rng = np.random.default_rng(rng)
    nc = int(round(n * p))                       # number of Markov chains
    ns = int(round((1 - p) / p))                 # number of states in each chain
    if nc * (ns + 1) != n:
        raise ValueError("n*p and 1/p must be integers")

    # --- Level 0: Monte Carlo ------------------------------------------------
    L = 0                                        # current (unconditional) level
    x = [rng.standard_normal((d, n))]            # Monte Carlo samples
    y = [g(x[0])]                                # system response y = g(x)
    nF = [int(np.sum(y[0] > YF))]                # number of failure samples
    Y = []

    while nF[L] / n < p:                         # stopping criterion
        L += 1                                   # next conditional level is needed
        ind = np.argsort(-y[L - 1])              # sort 'descend'
        y[L - 1] = y[L - 1][ind]                 # renumbered responses
        x[L - 1] = x[L - 1][:, ind]              # renumbered samples
        Y.append((y[L - 1][nc - 1] + y[L - 1][nc]) / 2)  # L-th intermediate threshold

        # --- Modified Metropolis algorithm for sampling from pi(x | F_L) ---
        z = np.empty((d, nc, ns + 1))
        z[:, :, 0] = x[L - 1][:, :nc]            # Markov chain "seeds"
        for m in range(ns):
            # Step 1: component-wise proposal
            a = z[:, :, m] + rng.standard_normal((d, nc))          # Step 1(a)
            r = np.minimum(1, norm.pdf(a) / norm.pdf(z[:, :, m]))  # Step 1(b)
            q = np.where(rng.random((d, nc)) < r, a, z[:, :, m])   # Step 1(c)
            # Step 2: accept q only if it belongs to F_L
            in_FL = g(q) > Y[L - 1]
            z[:, :, m + 1] = np.where(in_FL, q, z[:, :, m])

        # samples from pi(x | F_L), ordered chain by chain
        x.append(z.reshape(d, nc * (ns + 1)))

        y.append(g(x[L]))                        # system response y = g(x)
        nF.append(int(np.sum(y[L] > YF)))        # number of failure samples at level L

    pF_SS = p ** L * nF[L] / n                   # SS estimate
    N = n + n * (1 - p) * L                      # total number of samples

    return {"pF_SS": pF_SS, "N": int(round(N)), "L": L, "Y": Y,
            "nF": nF, "x": x, "y": y}


# ===========================================================================
# 1. Building blocks
# ===========================================================================

class CountedFunction:
    """[performance_function.py] g with a memory.

    Every point g has been evaluated at is stored in self.cache. Asking again
    for a stored point costs nothing (e.g. a Markov chain that stayed put).
    len(cache) is therefore the number of g evaluations, and the cache doubles
    as the "search history" used by the hill valley graph.
    """

    def __init__(self, g):
        self.g = g
        self.cache = {}

    def __call__(self, x):
        key = tuple(x)
        if key not in self.cache:
            self.cache[key] = self.g(np.array(key))
        return self.cache[key]


def in_partition(x, parts):
    """[indicator.py] Is x inside every partition set along its branch?
    parts is a list of (classifier, label): "the classifier says label"."""
    for classifier, label in parts:
        if classifier.predict_single(x) != label:
            return False
    return True


def in_region(x, region, G):
    """[indicator.py] Indicator of an intermediate failure region:
           A ∩ {g(x) >= b},   region = {'threshold': b, 'parts': A}.
    The partition is checked first, so g is never evaluated outside A.
    Returns a plain int 0/1, as the original does."""
    return int(in_partition(x, region['parts']) and G(x) >= region['threshold'])


def fraction_in(region, level, G):
    """[estimate.py] Fraction of a level's samples inside a region."""
    return sum(in_region(level['x'][:, k], region, G)
               for k in range(level['n'])) / level['n']


def new_level(x, n_chains, region, parent, levels, G):
    """[level.py] A level is one node of the NSuS tree.

    x        : (d, n) samples, stored chain after chain (like your x[L])
    y        : (n,)   responses g(x)
    n_chains : number of Markov chains (n for the Monte Carlo level)
    region   : the region the chains were confined to
    buds     : the possible next levels grown from this one (see make_bud)
    """
    level = {'x': x,
             'y': np.array([G(x[:, k]) for k in range(x.shape[1])]),
             'n': x.shape[1],
             'n_chains': n_chains,
             'region': region,
             'parent': parent,
             'children': [],
             'depth': 0 if parent is None else parent['depth'] + 1,
             'name': len(levels) + 1,      # levels are numbered 1, 2, 3, ...
             'trivial_bud': None,
             'buds': [],
             'stopped_by': [],
             'info': {}}
    if parent is not None:
        parent['children'].append(level)
    levels.append(level)
    return level


def level_buds(level):
    """[level.py] The level's buds: one per partition set if it was split,
    otherwise just the trivial (plain SuS) bud."""
    return level['buds'] if level['buds'] else [level['trivial_bud']]


def preorder(level):
    """Every level of the tree below (and including) `level`, parents first."""
    yield level
    for child in level['children']:
        yield from preorder(child)


def unused_buds(root):
    """[level.py] Buds anywhere in the tree that have not grown a level yet.
    Note: buds of stopped levels count too; this matters for the budget."""
    return [bud for level in preorder(root) for bud in level_buds(level)
            if bud['child'] is None]


# ===========================================================================
# 2. Subset simulation machinery (your code, split into pieces)
# ===========================================================================

def make_bud(level, part, p, scale):
    """[bud_create.py + scale.py] Choose seeds and the next threshold.

    part = None  -> trivial bud: candidates are all samples of the level
                    (this is your lines 73-80)
    part = (classifier, label) -> only samples inside that partition set

    Seeds are the best int(p * n_level) candidates, the new threshold b is
    the performance of the worst seed (your code uses the midpoint between the
    nc-th and (nc+1)-th response instead). If there are fewer candidates than
    that, all of them are seeds: n_c = min(floor(n p), |A ∩ X|), Eq. (10).
    """
    in_set = [k for k in range(level['n'])
              if part is None or part[0].predict_single(level['x'][:, k]) == part[1]]
    if len(in_set) == 0:
        return None
    candidates = sorted(in_set, key=lambda k: level['y'][k])      # ascending
    seed_number = int(p * level['n'])
    seeds = candidates[-seed_number:]  # (as in the original: if seed_number is 0
                                       #  this slice keeps every candidate)
    parts = level['region']['parts'] + ([part] if part is not None else [])
    seed_x = level['x'][:, seeds].T                               # (nc, d)
    if scale is None:
        prop_scale = np.std(seed_x, axis=0)       # AdaptiveScaler
    else:
        prop_scale = np.array([scale] * seed_x.shape[1])          # ConstantScaler
    return {'parent': level,
            'n_candidates': len(candidates),
            'seeds': seeds,
            'region': {'threshold': level['y'][seeds[0]], 'parts': parts},
            'scale': prop_scale,
            'child': None}


def modified_metropolis(start, length, region, scale, G, rng):
    """[markov_chain.py] Modified Metropolis, Algorithm 1 (your lines 81-88).

    start : (nc, d) seeds, one chain each
    Returns (nc, length + 1, d): every chain, seed first.
    """
    nc, dim = start.shape
    shape = (nc, dim, length)
    # All random numbers are drawn up front (this fixes the random-draw order).
    log_u = np.log(uniform.rvs(size=shape, random_state=rng))
    steps = norm.rvs(size=shape,
                     scale=np.tile(scale.reshape(1, dim, 1), (nc, 1, length)),
                     random_state=rng)
    states = [start]
    for m in range(length):
        current = states[-1]
        # Step 1: component-wise proposal, accepted against the N(0,1) density
        proposal = current + steps[:, :, m]
        accept = norm.logpdf(proposal) - norm.logpdf(current) >= log_u[:, :, m]
        candidate = np.where(accept, proposal, current)
        # Step 2: keep the candidate only if it lies in the region
        inside = np.array([[in_region(c, region, G)] for c in candidate])
        states.append(np.where(inside, candidate, current))
    return np.stack(states, axis=1)


def chain_length(bud, root, n, p):
    """[allocate.py, BranchingAllocator] Each step creates n(1-p) new samples,
    shared equally by all open buds, then by the bud's seeds (Eq. 10):
        n_s = floor(n(1-p) / (N_p n_c)) + 1   (the +1 is the seed itself).
    With a single branch this is 1/p, exactly as in your code."""
    split_budget = int((1 - p) * n) / len(unused_buds(root))
    return int(split_budget / len(bud['seeds'])) + 1


def grow_level(bud, root, levels, n, p, G, rng):
    """[level_create.py] Grow a new level from a bud: one chain per seed."""
    start = bud['parent']['x'][:, bud['seeds']].T
    length = chain_length(bud, root, n, p)
    z = modified_metropolis(start, length - 1, bud['region'], bud['scale'], G, rng)
    nc, ns, dim = z.shape
    x = z.reshape(nc * ns, dim).T              # (d, n_level), chain after chain
    level = new_level(x, nc, bud['region'], bud['parent'], levels, G)
    bud['child'] = level
    return level


def check_stop(level, YF):
    """[stop.py] Applied to every new level. Returns the reasons to stop.
    Failure           : the worst would-be seed already fails, i.e. at least
                        n_c samples have g >= YF (your `nF / n < p` test)
    Threshold Converge: the next threshold would not be higher than this
                        level's own, so the chains are stuck."""
    b_next = level['y'][level['trivial_bud']['seeds'][0]]
    reasons = []
    if b_next >= YF:
        reasons.append('Failure')
    if b_next <= level['region']['threshold']:
        reasons.append('Threshold Converge')
    return reasons


def choose_bud(root):
    """[choose.py, DepthChooser] Among open buds of unstopped levels, grow
    the shallowest one (so the branches take turns)."""
    options = [bud for bud in unused_buds(root) if not bud['parent']['stopped_by']]
    if not options:
        return None
    min_depth = min(bud['parent']['depth'] for bud in options)
    return [bud for bud in options if bud['parent']['depth'] == min_depth][0]


# ===========================================================================
# 3. The niching part: the hill valley graph partitioner
#    [hill_valley_graph_partitioner.py + lsvc.py]
# ===========================================================================

class LinearClassifier:
    """[lsvc.py] standardise -> PCA -> L1 linear SVM.

    The scaler and PCA are fitted once on ALL samples of the level; the SVM
    is refitted for each candidate labelling of the graph samples.
    predict_single is a fast one-point version used inside the Markov chains.
    """

    def __init__(self, level_x, svc_seed):
        self.scaler = StandardScaler().fit(level_x)
        self.pca = PCA().fit(self.scaler.transform(level_x))
        self.svc = LinearSVC(penalty='l1', C=1, dual='auto', max_iter=1000000,
                             random_state=svc_seed)

    def fit(self, X, labels):
        self.svc.fit(self.pca.transform(self.scaler.transform(X)), labels)

    def predict(self, X):
        return self.svc.predict(self.pca.transform(self.scaler.transform(X)))

    def predict_single(self, x):
        x = (x - self.scaler.mean_) / self.scaler.var_ ** .5
        x = np.dot(x - self.pca.mean_, self.pca.components_.T)
        scores = np.dot(self.svc.coef_, x) + self.svc.intercept_
        if len(scores) == 1:                        # two classes
            return self.svc.classes_[int((scores > 0)[0])]
        return self.svc.classes_[np.argmax(scores)]


def choose_graph_samples(level, size, rng):
    """[hill_valley_graph_partitioner.py, choose_samples]

    A chain group = the distinct samples of all chains that started from the
    same seed (they very probably share a niche). Shuffle the groups, then
    take one sample from each group in turn, so the graph samples are spread
    over as many groups as possible. Returns indices into the level.
    """
    length = level['n'] // level['n_chains']
    chains = [list(range(c * length, (c + 1) * length)) for c in range(level['n_chains'])]
    seed_array = level['x'][:, [chain[0] for chain in chains]].T
    unique_seeds = np.unique(seed_array, axis=0)
    groups = [[] for _ in range(len(unique_seeds))]
    for c in range(len(seed_array)):
        groups[np.where(np.all(unique_seeds == seed_array[c], axis=1))[0][0]].append(c)
    chain_groups = []
    for group in groups:
        members = sum([chains[c] for c in group], [])
        keep = np.unique(level['x'][:, members].T, axis=0, return_index=True)[1]
        chain_groups.append([members[k] for k in keep])
    for chain_group in chain_groups:
        rng.shuffle(chain_group)
    rng.shuffle(chain_groups)
    order = []
    while any(chain_groups):
        for chain_group in chain_groups:
            if chain_group:
                order.append(chain_group.pop())
    return order[:size]


def hill_valley_graph(X, y, G, use_history):
    """[hill_valley_graph_partitioner.py, HillValleyGraph]

    Join samples i and j unless a "valley" is found between them, i.e. a
    point between them with g < min(g(x_i), g(x_j))  (Definition 3, Eq. 19).

    use_history=False : evaluate g at the midpoint of every pair.
    use_history=True  : first look at points already evaluated (G.cache) that
                        lie in the box spanned by the pair; only pairs with
                        no such point get a midpoint evaluation.
    Returns the graph and the number of new g evaluations it needed.
    """
    if len(X) == 1:
        graph = nx.Graph()
        graph.add_nodes_from(range(1))
        return graph, 0
    i, j = np.tril_indices(len(X), k=-1)                 # all pairs i > j
    if not use_history:
        mid = np.array([G((a + b) / 2) for a, b in zip(X[i], X[j])])
        adj = (mid >= np.minimum(y[i], y[j])).astype(int)
        evals = len(mid)
    else:
        def history():
            hist_x = np.array([key for key in G.cache.keys()])
            hist_y = np.array([val for val in G.cache.values()])
            mask = ~((hist_x[:, None] == X).all(-1).any(1))   # not the graph samples
            hist_x, hist_y = hist_x[mask], hist_y[mask]
            low = np.minimum(X[i], X[j])[:, np.newaxis, :]
            high = np.maximum(X[i], X[j])[:, np.newaxis, :]
            in_box = np.all((hist_x[np.newaxis] >= low) & (hist_x[np.newaxis] <= high), axis=2)
            return in_box, hist_y                       # in_box[pair, point]
        in_box, _ = history()
        evals = 0
        for k in np.where(~np.any(in_box, axis=1))[0]:  # pairs with an empty box
            G((X[i][k] + X[j][k]) / 2)
            evals += 1
        in_box, hist_y = history()
        lower = hist_y < np.minimum(y[i], y[j]).reshape(-1, 1)
        adj = (~np.any(in_box & lower, axis=1)).astype(int)
    A = np.zeros((len(X), len(X)))
    A[i, j] = adj
    A[j, i] = adj
    return nx.from_numpy_array(A), evals


def shrink_partition(partition, max_parts, rng):
    """[hill_valley_graph_partitioner.py] Merge random pairs of communities
    until there are at most max_parts."""
    partition = list(partition)
    while len(partition) > max_parts:
        a, b = rng.choice(range(len(partition)), size=2, replace=False)
        partition[a] = partition[a].union(partition[b])
        partition.pop(b)
    return partition


def hvg_partition(level, root, graph_size, max_branches, use_history,
                  svc_seed, G, rng, iterations=100):
    """[hill_valley_graph_partitioner.py, HillValleyGraphPartitioner.partition]

    Returns a list of partition sets, each as (classifier, label), or [None]
    for "do not split". Diagnostics are stored in level['info'].
    """
    n_open = len(unused_buds(root))
    if n_open >= max_branches:                   # no room for more branches
        return [None]
    max_parts = max_branches - n_open + 1

    # (a) hill valley graph on a few samples of the level
    idx = choose_graph_samples(level, graph_size // n_open, rng)
    X = level['x'][:, idx].T
    graph, evals = hill_valley_graph(X, level['y'][idx], G, use_history)
    level['info'].update(graph=graph, graph_idx=idx, hvg_evals=evals)

    # (b) community detection: label propagation is random, so repeat it
    partitions = [list(asyn_lpa_communities(graph, seed=rng)) for _ in range(iterations)]
    if Counter(len(part) for part in partitions)[1] / iterations > 0.5:
        return [None]                            # mostly one community: no split
    partitions = [part for part in partitions if len(part) > 1]
    partitions = [shrink_partition(part, max_parts, rng) for part in partitions]

    # (c) classification: fit a linear SVM to every candidate labelling of the
    #     graph samples, keep the labelling it reproduces best
    classifier = LinearClassifier(level['x'].T, svc_seed)

    def fit(partition):
        Xc = np.array([X[node] for part in partition for node in part])
        labels = np.array([k for k, part in enumerate(partition) for _ in range(len(part))])
        classifier.fit(Xc, labels)
        return Xc, labels

    scores = []
    for partition in partitions:
        Xc, labels = fit(partition)
        scores.append(balanced_accuracy_score(labels, classifier.predict(Xc)))
    best = partitions[np.argmax(scores)]
    fit(best)
    level['info'].update(partition=best, classifier=classifier)
    return [(classifier, label) for label in range(len(best))]


# ===========================================================================
# 4. Estimating the failure probability from the tree  [estimate.py]
# ===========================================================================

def branch(level):
    """The path from the root down to `level`."""
    path = []
    while level is not None:
        path.append(level)
        level = level['parent']
    return path[::-1]


def leaf_probability(leaf, YF, G):
    """SuS estimate along one branch (Eq. 4): product of the conditional
    fractions down the branch, times the failing fraction of the leaf."""
    path = branch(leaf)
    last = {'threshold': YF, 'parts': leaf['region']['parts']}
    return (np.prod([fraction_in(level['region'], level['parent'], G) for level in path[1:]])
            * fraction_in(last, leaf, G))


def trimmed_leaves(root, YF):
    """Leaves of the tree trimmed at YF (Section 3.3): stop descending at a
    level that already has a child with an intermediate threshold above YF."""
    leaves, stack = [], [root]
    while stack:
        level = stack.pop()
        if not level['children'] or any(child['region']['threshold'] > YF
                                        for child in level['children']):
            leaves.append(level)
        else:
            stack.extend(level['children'])
    return leaves


def level_gamma(level, region, G):
    """Correlation factor gamma of the Markov chains (Au & Beck 2001)."""
    n_c, n_l = level['n_chains'], level['n']
    n_s = int(n_l // n_c)
    prob = fraction_in(region, level, G)
    ind = [int(in_region(level['x'][:, k], region, G)) for k in range(n_l)]
    chains = [ind[c * n_s:(c + 1) * n_s] for c in range(n_c)]
    r = []
    for i in range(n_s):
        sums = sum([sum([chain[k] * chain[k + i] for k in range(n_s - i)]) for chain in chains])
        r.append((sums / (n_l - (i * n_c))) - (prob ** 2))
    if r[0] > 0:
        return 2 * (sum([(1 - (i / n_s)) * (r[i + 1] / r[0]) for i in range(n_s - 1)]))
    return np.inf


def level_cov(level, region, G):
    """CoV of one factor: Eq. (6) for the Monte Carlo level, Eq. (7) after."""
    prob = fraction_in(region, level, G)
    if prob == 1:
        return 0
    if level['parent'] is None:
        return np.sqrt((1 - prob) / (level['n'] * prob))
    return np.sqrt(((1 - prob) / (level['n'] * prob)) * (1 + level_gamma(level, region, G)))


def branch_cov(level, threshold, G):
    """CoV of a SuS estimator along one branch, Eq. (8)."""
    squares = [level_cov(lv['parent'], lv['region'], G) ** 2 for lv in branch(level)[1:]]
    last = {'threshold': threshold, 'parts': level['region']['parts']}
    squares.append(level_cov(level, last, G) ** 2)
    return np.sqrt(sum(squares))


def nsus_cov(root, YF, G):
    """CoV of the NSuS estimate, Eq. (13). Branches are correlated only
    through the levels they share (their common ancestor)."""
    leaves = trimmed_leaves(root, YF)
    P = [leaf_probability(leaf, YF, G) for leaf in leaves]
    w_total = sum(P[i] * P[j] for i in range(len(P)) for j in range(len(P)))
    if w_total == 0:
        return np.nan                            # nothing failed anywhere
    total = []
    for i in range(len(P)):
        for j in range(len(P)):
            w = P[i] * P[j] / w_total
            if w == 0:
                continue                         # (the original crashes here)
            names_j = [lv['name'] for lv in branch(leaves[j])]
            common = [lv for lv in branch(leaves[i]) if lv['name'] in names_j][-1]
            b = YF if i == j else common['region']['threshold']
            total.append(w * branch_cov(common, b, G) ** 2)
    return np.sqrt(sum(total))


# ===========================================================================
# 5. Niching subset simulation  [nsus.py + implementation.py]
# ===========================================================================

def niching_subset_simulation(d, YF, n=1000, p=0.1, g=performance_function,
                              graph_size=15, max_branches=2, scale=1.0,
                              use_history=True, rng=None, verbose=False):
    """Estimate pF = P(g(x) >= YF) for x ~ N(0, I_d)  (Algorithm 3).

    Parameters
    ----------
    d            : dimension of the input space
    YF           : critical threshold (failure <=> g(x) >= YF)
    n            : number of samples per level (before any split)
    p            : level probability
    g            : performance function taking ONE sample of shape (d,)
    graph_size   : samples per hill valley graph
    max_branches : maximum number of branches (1 = plain subset simulation)
    scale        : proposal std of the Markov chains (None = adaptive)
    use_history  : hill valley tests reuse earlier g evaluations
    rng          : seed or np.random.Generator

    Returns
    -------
        'pF_NSuS'  : NSuS estimate of the failure probability (Eq. 12)
        'cov'      : estimated coefficient of variation (Eq. 13)
        'N'        : total number of g evaluations
        'N_hvg'    : how many of those were spent on hill valley graphs
        'levels'   : all levels in order of creation (dicts, see new_level)
        'leaves'   : the last level of each branch
        'pF_leaf'  : each leaf's contribution to pF_NSuS
        'root'     : the initial level (root of the tree)
    """
    rng = np.random.default_rng(rng)
    G = CountedFunction(g)
    # The SVM's own seed is drawn once, here, as in the original. (The
    # original SubsetSimulation does not draw it, so skip it for 1 branch.)
    svc_seed = rng.integers(0, 1000000) if max_branches > 1 else None

    # --- Level 1: Monte Carlo  [level_create.py] ----------------------------
    levels = []
    x0 = multivariate_normal.rvs(mean=np.zeros(d), cov=np.identity(d),
                                 size=n, random_state=rng).T          # (d, n)
    root_region = {'threshold': -np.inf, 'parts': []}
    level = new_level(x0, n, root_region, None, levels, G)
    hvg_evals = 0

    while True:
        # Plain SuS bud: best n p samples of the whole level
        level['trivial_bud'] = make_bud(level, None, p, scale)
        level['stopped_by'] = check_stop(level, YF)
        if verbose:
            print(f"Level {level['name']} created ({level['n']} samples, "
                  f"b = {level['region']['threshold']:.4g})"
                  + (f", stopped: {', '.join(level['stopped_by'])}" if level['stopped_by'] else ""))

        # Niching: try to split the level, one bud per partition set
        if not level['stopped_by'] and max_branches > 1:
            parts = hvg_partition(level, levels[0], graph_size, max_branches,
                                  use_history, svc_seed, G, rng)
            hvg_evals += level['info'].get('hvg_evals', 0)
            if len(parts) > 1:
                for part in parts:
                    bud = make_bud(level, part, p, scale)
                    if bud is not None:            # empty partition sets are dropped
                        level['buds'].append(bud)
                if verbose and len(level['buds']) > 1:
                    print(f"Level {level['name']} partitioned into {len(level['buds'])}.")

        # Grow the next level, or stop when every branch has stopped
        bud = choose_bud(levels[0])
        if bud is None:
            break
        level = grow_level(bud, levels[0], levels, n, p, G, rng)

    root = levels[0]
    leaves = [lv for lv in preorder(root) if not lv['children']]
    final = trimmed_leaves(root, YF)
    pF_leaf = [leaf_probability(leaf, YF, G) for leaf in final]
    return {'pF_NSuS': sum(pF_leaf),
            'cov': nsus_cov(root, YF, G),
            'N': len(G.cache),
            'N_hvg': hvg_evals,
            'levels': levels,
            'leaves': leaves,
            'pF_leaf': dict(zip([leaf['name'] for leaf in final], pF_leaf)),
            'root': root}


# ---------------------------------------------------------------------------
# Text picture of the tree (cf. Fig. 3 of the paper)
# ---------------------------------------------------------------------------
def print_tree(result):
    def visit(level, prefix, last, is_root):
        info = f"level {level['name']}: {level['n']} samples, b = {level['region']['threshold']:.4g}"
        if level['name'] in result['pF_leaf']:
            info += f", pF contribution = {result['pF_leaf'][level['name']]:.3e}"
        if level['stopped_by']:
            info += "  [" + ", ".join(level['stopped_by']) + "]"
        print(info if is_root else prefix + ("└── " if last else "├── ") + info)
        child_prefix = "" if is_root else prefix + ("    " if last else "│   ")
        for k, child in enumerate(level['children']):
            visit(child, child_prefix, k == len(level['children']) - 1, False)
    visit(result['root'], "", True, True)


# ---------------------------------------------------------------------------
# 2D plot of the levels (in the style of Figure 6)
# ---------------------------------------------------------------------------
LEVEL_COLOURS = ["#ff0000", "#e000e0", "#0000ff", "#007f00", "#00c0c0",
                 "#b0b000", "#00e000", "#ffff00", "#1f2f7f", "#e03000"]


def _point_nearest(cs, target, lo, hi):
    pts = [seg for seg in cs.allsegs[0] if len(seg)]
    if not pts:
        return None                            # curve not visible in the plot
    pts = np.vstack(pts)
    inside = np.all((pts > lo) & (pts < hi), axis=1)
    if inside.any():
        pts = pts[inside]
    return pts[np.argmin(np.sum((pts - target) ** 2, axis=1))]


def plot_levels_2d(result, YF, g=performance_function, lim=None,
                   filename=None, legend=True):
    import matplotlib.pyplot as plt

    x, Y = result["x"], result["Y"]
    if x[0].shape[0] != 2:
        raise ValueError("plot_levels_2d needs d = 2")

    # Axis range:
    if lim is None:
        allx = np.hstack(x)
        lim = (np.floor(allx.min() - 0.2), np.ceil(allx.max() + 0.2))

    # Evaluate g on a grid so the threshold curves can be drawn as contours
    t = np.linspace(lim[0], lim[1], 400)
    X1, X2 = np.meshgrid(t, t)
    G = g(np.vstack([X1.ravel(), X2.ravel()])).reshape(X1.shape)

    fig, ax = plt.subplots(figsize=(7, 7))

    # Samples:
    for L, xL in enumerate(x):
        colour = LEVEL_COLOURS[L % len(LEVEL_COLOURS)]
        label = "Monte Carlo (level 0)" if L == 0 else f"Level {L}"
        ax.scatter(xL[0], xL[1], s=5, color=colour, linewidths=0,
                   label=label, zorder=2)

    # Labels:
    span = lim[1] - lim[0]
    lo, hi = lim[0] + 0.08 * span, lim[1] - 0.08 * span  
    corner = np.array([lo, hi])

    # Intermediate thresholds: boundary of F_L = {x : g(x) > Y_L}
    for L, YL in enumerate(Y, start=1):
        cs = ax.contour(X1, X2, G, levels=[YL], colors="k", linewidths=0.8,
                        linestyles="--", zorder=3)
        pt = _point_nearest(cs, corner, lo, hi)
        if pt is not None:
            ax.clabel(cs, fmt={YL: f"$F_{{{L}}}$"}, fontsize=12, colors="k",
                      manual=[pt])

    # Failure domain F = {x : g(x) > YF}:
    cs = ax.contour(X1, X2, G, levels=[YF], colors="k", linewidths=1.5,
                    zorder=3)
    pt = _point_nearest(cs, corner, lo, hi)
    if pt is not None:
        h = 1e-4 * span                        # numerical gradient of g at pt
        grad = np.array([
            g(np.array([[pt[0] + h], [pt[1]]]))[0] - g(np.array([[pt[0] - h], [pt[1]]]))[0],
            g(np.array([[pt[0]], [pt[1] + h]]))[0] - g(np.array([[pt[0]], [pt[1] - h]]))[0]])
        if np.linalg.norm(grad) > 0:
            # step from the boundary into F (direction of increasing g)
            pos = np.clip(pt + 0.06 * span * grad / np.linalg.norm(grad),
                          lim[0] + 0.03 * span, lim[1] - 0.03 * span)
            ax.text(*pos, "$F$", fontsize=15, ha="center", va="center")

    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title(f"Subset simulation: {len(Y)} levels, "
                 f"$p_F^{{SS}}$ = {result['pF_SS']:.3e}")
    if legend:
        ax.legend(loc="best", fontsize=8, markerscale=2.5, framealpha=0.9)
    fig.tight_layout()

    if filename:
        fig.savefig(filename, dpi=150)
        print(f"Saved plot to {filename}")
    plt.show()
    return fig, ax


# ---------------------------------------------------------------------------
# 2D plot of the levels (in the style of Subset_Simulation.py)
# ---------------------------------------------------------------------------
BRANCH_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                  "#e87ba4", "#008300", "#6250d6", "#e34948"]
TRUNK_COLOUR = "#8a8a85"     # levels before any split


def plot_nsus_2d(result, YF, g=performance_function, lim=None, res=150,
                   filename=None, legend=True):
    import matplotlib.pyplot as plt

    levels = result['levels']
    if levels[0]['x'].shape[0] != 2:
        raise ValueError("plot_levels_2d needs d = 2")

    # Axis range:
    if lim is None:
        allx = np.hstack([lv['x'] for lv in levels])
        lim = (np.floor(allx.min() - 0.5), np.ceil(allx.max() + 0.5))

    # g on a grid (one point at a time, so any g works)
    t = np.linspace(lim[0], lim[1], res)
    X1, X2 = np.meshgrid(t, t)
    pts = np.column_stack([X1.ravel(), X2.ravel()])
    Gr = np.array([g(pt) for pt in pts]).reshape(X1.shape)

    fig, ax = plt.subplots(figsize=(7, 7))

    # Colour group of each level: a new colour wherever the parent was split
    group, next_group = {}, 0
    for lv in levels:
        par = lv['parent']
        if par is None:
            group[lv['name']] = -1
        elif len(par['children']) > 1:
            group[lv['name']] = next_group
            next_group += 1
        else:
            group[lv['name']] = group[par['name']]

    # Samples:
    handles = {}
    for lv in levels:
        k = group[lv['name']]
        colour = TRUNK_COLOUR if k < 0 else BRANCH_COLOURS[k % len(BRANCH_COLOURS)]
        handles[k] = ax.scatter(lv['x'][0], lv['x'][1], s=5, color=colour,
                                linewidths=0, zorder=2)
    for k in sorted(handles):
        if k < 0:
            handles[k].set_label("Before any split" if len(handles) > 1 else "All levels")
        else:
            pf = sum(v for name, v in result['pF_leaf'].items() if group[name] == k)
            handles[k].set_label(f"Branch {k + 1}" + (f" (pF ≈ {pf:.2e})" if pf else ""))
    entries = [handles[k] for k in sorted(handles)]

    # Partition boundaries, drawn only inside the region each split level lives in
    for lv in levels:
        if len(lv['buds']) < 2:
            continue
        labels = lv['info']['classifier'].predict(pts).astype(float)
        inside = np.array([in_partition(pt, lv['region']['parts']) for pt in pts])
        labels[~inside] = np.nan
        ax.contour(X1, X2, labels.reshape(X1.shape),
                   levels=np.arange(len(lv['info']['partition']) - 1) + 0.5,
                   colors="k", linewidths=1.2, zorder=4)
    if any(len(lv['buds']) > 1 for lv in levels):
        entries += ax.plot([], [], "k-", lw=1.2, label="Partition boundary")

    # Failure domain F = {x : g(x) >= YF}:
    ax.contour(X1, X2, Gr, levels=[YF], colors="k", linewidths=1.5,
               linestyles="--", zorder=3)
    entries += ax.plot([], [], "k--", lw=1.5, label="Limit state $g = Y_F$")

    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title(f"Niching subset simulation: {len(levels)} levels, "
                 f"{len(result['leaves'])} branch(es), "
                 f"$p_F^{{NSuS}}$ = {result['pF_NSuS']:.3e}")
    if legend:
        ax.legend(handles=entries, loc="best", fontsize=8, markerscale=2.5, framealpha=0.9)
    fig.tight_layout()

    if filename:
        fig.savefig(filename, dpi=150)
        print(f"Saved plot to {filename}")
    plt.show()
    return fig, ax


# ===========================================================================
# Run
# ===========================================================================
if __name__ == "__main__":
    from scipy.stats import chi2
    pF = chi2.sf(YF, d)                   # true value (valid for g = sum of squares only)

    # --- your subset simulation ---
    start = time.perf_counter()
    res = subset_simulation(d, YF, n=n, p=p, rng=SEED)
    run_time = time.perf_counter() - start
    print("SUBSET SIMULATION")
    print(f"Conditional levels L : {res['L']}")
    print(f"Total samples N      : {res['N']}")
    print(f"pF (subset sim)      : {res['pF_SS']:.4e}")
    print(f"Run time             : {run_time:.2f} s\n")

    # --- niching subset simulation ---
    start = time.perf_counter()
    nres = niching_subset_simulation(d, YF, n=n, p=p, graph_size=GRAPH_SIZE,
                                     max_branches=MAX_BRANCHES, scale=SCALE,
                                     use_history=HVG_HISTORY, rng=SEED)
    run_time = time.perf_counter() - start
    print("NICHING SUBSET SIMULATION")
    print_tree(nres)
    print(f"Levels / branches    : {len(nres['levels'])} / {len(nres['leaves'])}")
    print(f"Total samples N      : {nres['N']}  ({nres['N_hvg']} for hill valley graphs)")
    print(f"pF (NSuS)            : {nres['pF_NSuS']:.4e}")
    print(f"CoV estimate         : {nres['cov']:.3f}")
    print(f"Run time             : {run_time:.2f} s\n")
    print(f"pF (true value)      : {pF:.4e}")

    if d == 2:
        plot_levels_2d(res, YF, filename=SAVE_PLOT, legend=False)
        plot_nsus_2d(nres, YF)
