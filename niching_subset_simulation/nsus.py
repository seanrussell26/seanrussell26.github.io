"""
Niching Subset Simulation (NSuS)

A single-file port of Hugh Kinnear's reference implementation
(https://github.com/HughKinnear/nsus_paper, package `nsus/`), which accompanies

    H.J. Kinnear and F.A. DiazDelaO, "Niching subset simulation",
    Probabilistic Engineering Mechanics 79 (2025) 103729.

The algorithm is kept as close to the original as possible: same classes,
same logic, same order of random-number draws (so a run with a given seed
gives the same answer as the original package). Each section below says which
original file it comes from. The only deliberate changes are listed here and
marked "PORT NOTE" in the code:

  1. The `anytree` dependency is replaced by a 30-line `TreeNode` class with
     the handful of tree operations NSuS uses (parent, children, ancestors,
     root, depth, size, leaves, pre-order iteration).
  2. The performance function is handed a NumPy array instead of a tuple, so
     functions written with array maths (like g = np.sum(x**2, axis=0)) work.
  3. `cov` skips terms whose weight is exactly zero, instead of crashing with
     ZeroDivisionError when a branch found no failure samples. Those terms are
     multiplied by zero in Eq. (13) of the paper, so the value is unchanged.
  4. Constructor arguments have defaults, and the two hard-coded settings of
     HillValleyGraphNSuS (100 ALP iterations, the LinearSVC parameters) can be
     overridden. The defaults are the original values.
  5. HillValleyGraph.fit (the no-history test, is_cache=False) now adds its
     midpoint evaluations to eval_count; the original left the count at 0.
  6. Unused helpers (ChainData.trim/chain_dict, NoMovement, plot.py) are left
     out; plotting lives in nsus_plot.py instead.

Conventions (same as the paper / original code)
    * Inputs x are d-dimensional standard normal, x ~ N(0, I_d).
    * The performance function g takes ONE sample, a 1-D array of length d,
      and returns a float.
    * Failure  <=>  g(x) >= threshold   (note: >=, not >).
"""

from abc import ABC, abstractmethod
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from itertools import product

import networkx as nx
import numpy as np
from networkx.algorithms.community import asyn_lpa_communities
from scipy.stats import multivariate_normal, norm, uniform
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


# ===========================================================================
# utils.py
# ===========================================================================
def verbose(method):
    """Only run a (printing) method if the owning object has verbose=True."""
    def wrapper(self, *args, **kwargs):
        if self.verbose:
            return method(self, *args, **kwargs)
    return wrapper


# ===========================================================================
# performance_function.py
# ===========================================================================
class PerformanceFunction:
    """Wraps g so every evaluation is cached.

    The cache is keyed on the sample's coordinates, so asking for g at a point
    that has already been evaluated (e.g. a Markov chain that rejected its
    move and stayed put) costs nothing. eval_count is therefore the number of
    *distinct* points at which g was actually evaluated. The cache doubles as
    the "search history" used by the hill valley graph.
    """

    def __init__(self, performance_function):
        self.non_cache_performance_function = deepcopy(performance_function)
        self.cache = {}

    def __call__(self, x):
        tuple_x = tuple(x)
        try:
            result = self.cache[tuple_x]
        except KeyError:
            # PORT NOTE 2: the original passes tuple_x; we pass an array.
            self.cache[tuple_x] = result = self.non_cache_performance_function(
                np.array(tuple_x))
        return result

    @property
    def eval_count(self):
        return len(self.cache)


# ===========================================================================
# level_create.py (Sample)
# ===========================================================================
@dataclass
class Sample:
    array: ...          # the point x, shape (d,)
    performance: ...    # g(x)


# ===========================================================================
# indicator.py
# ===========================================================================
class Indicator:
    """The indicator function of an intermediate failure region.

    A level's region is   {x : partition_indicator(x) = 1  and  g(x) >= threshold}
    i.e. the paper's  A ∩ F  (A = a set of the partition, F = {g >= b}).
    Branching makes partition_indicator a product of the classifiers' set
    indicators all the way up the branch.
    """

    def __init__(self, threshold, partition_indicator, performance_function):
        self.threshold = threshold
        self.partition_indicator = partition_indicator
        self.performance_function = performance_function

    def threshold_indicator(self, x):
        return int(self.performance_function(x)
                   >= self.threshold)

    def indicator(self, x):
        # The partition is checked first: if x is outside A we never evaluate g.
        if self.partition_indicator(x) == 1:
            if self.threshold_indicator(x) == 1:
                return 1
        return 0

    def __call__(self, x):
        if isinstance(x, list):
            return [self.indicator(sample.array) for sample in x]
        else:
            return self.indicator(x)

    @staticmethod
    def indicator_combine(indicator_a, indicator_b):
        def indicator(x):
            return indicator_a(x) * indicator_b(x)
        return indicator

    def child_indicator(self, threshold, partition_indicator):
        """Region for a child level: new threshold, and this region's
        partition set intersected with the new partition set."""
        combined_partition_indicator = self.indicator_combine(self.partition_indicator,
                                                              partition_indicator)
        return Indicator(threshold,
                         combined_partition_indicator,
                         self.performance_function)

    def new_threshold(self, threshold):
        """Same partition set, different threshold (used by the estimators)."""
        return Indicator(threshold,
                         self.partition_indicator,
                         self.performance_function)


# ===========================================================================
# Tree structure (PORT NOTE 1: replaces anytree.AnyNode / PreOrderIter)
# ===========================================================================
class TreeNode:

    def __init__(self, parent=None):
        self.parent = parent
        self.children = []
        if parent is not None:
            parent.children.append(self)

    @property
    def ancestors(self):
        """All nodes from the root down to (not including) this node."""
        nodes, node = [], self.parent
        while node is not None:
            nodes.append(node)
            node = node.parent
        return tuple(reversed(nodes))

    @property
    def root(self):
        return self.ancestors[0] if self.parent is not None else self

    @property
    def depth(self):
        return len(self.ancestors)

    def preorder(self):
        yield self
        for child in self.children:
            yield from child.preorder()

    @property
    def size(self):
        return sum(1 for _ in self.preorder())

    @property
    def leaves(self):
        return tuple(node for node in self.preorder() if not node.children)


# ===========================================================================
# bud_create.py
# ===========================================================================
class Bud:
    """A *potential* next level.

    A bud stores everything needed to grow a new level: the seeds, the
    indicator of the region the chains must stay in, and the proposal scale.
    Every level gets one "trivial" bud (no partition: plain SuS). If the
    partitioner splits the level into k sets, the level gets k buds instead,
    one per set. A bud becomes "used" once a level has been grown from it.
    """

    def __init__(self,
                 sorted_candidate_seeds=None,
                 seeds=None,
                 indicator=None,
                 parent_level=None,
                 scale=None):
        self.sorted_candidate_seeds = sorted_candidate_seeds
        self.seeds = seeds
        self.indicator = indicator
        self.parent_level = parent_level
        self.child_level = None
        self.scale = scale

    @property
    def is_used(self):
        return self.child_level is not None

    @property
    def name(self):
        pos = self.parent_level.buds.index(self)
        return f"{self.parent_level.name}-{pos}"


class BudCreator:

    def __init__(self,
                 seed_method,
                 level_probability,
                 scaler,
                 verbose):
        self.seed_method = getattr(self, seed_method)
        self.level_probability = level_probability
        self.scaler = scaler
        self.verbose = verbose

    def create(self, level, partition_indicator):
        # Candidates = the level's samples that lie in this partition set.
        sorted_candidate_seeds = self.sorted_candidate_seeds(level,
                                                             partition_indicator)
        if len(sorted_candidate_seeds) == 0:
            return None

        # Seeds = the best floor(n p) candidates; the new intermediate
        # threshold b is the performance of the worst seed.
        seeds = self.seed_method(level, sorted_candidate_seeds)
        threshold = seeds[0].performance
        indicator = level.indicator.child_indicator(threshold,
                                                    partition_indicator)
        scale = self.scaler.compute_scale(seeds)
        bud = Bud(sorted_candidate_seeds=sorted_candidate_seeds,
                  seeds=seeds,
                  indicator=indicator,
                  parent_level=level,
                  scale=scale
                  )
        return bud

    def create_trivial(self, level):
        level.trivial_bud = self.create(level, lambda x: 1)

    def create_inidicator_buds(self, level, partition_indicators):
        for partition_indicator in partition_indicators:
            bud = self.create(level, partition_indicator)
            if bud is None:
                self.print_empty_bud()
                level.empty_indicators.append(partition_indicator)
            else:
                level._buds.append(bud)
        self.print_partition(level)

    @verbose
    def print_partition(self, level):
        if len(level.buds) > 1:
            print(f"Level {level.name} partitioned into {len(level.buds)}.")

    @verbose
    def print_empty_bud(self):
        print("Empty bud.")

    @staticmethod
    def sorted_candidate_seeds(level, partition_indicator):
        samples = [sample
                   for sample in level.sample_list
                   if bool(partition_indicator(sample.array))]
        return sorted(samples, key=lambda x: x.performance)   # ascending

    def seeds_prob_inclusive(self, level, sorted_candidate_seeds):
        seed_number = int(self.level_probability * level.level_size)
        thresh = sorted_candidate_seeds[-seed_number].performance
        return sorted([seed for seed in sorted_candidate_seeds if seed.performance >= thresh],
                      key=lambda x: x.performance)

    def seeds_prob_exclusive(self, level, sorted_candidate_seeds):
        # n_c = min(floor(|X| p), |A ∩ X|)  (Eq. 10): slicing past the start
        # of the list simply returns every candidate.
        seed_number = int(self.level_probability * level.level_size)
        return sorted_candidate_seeds[-seed_number:]


# ===========================================================================
# level.py
# ===========================================================================
class Level(TreeNode):
    """One level of samples, i.e. one node (X, A, F) of the NSuS tree.

    chain_list is a list of Markov chains, each a list of Samples. For the
    initial (Monte Carlo) level every "chain" has length 1.
    """

    def __init__(self,
                 parent_bud,
                 chain_list,
                 ):
        super().__init__(parent=parent_bud.parent_level)

        self.parent_bud = parent_bud
        self.chain_list = chain_list
        self.empty_indicators = []
        self.trivial_bud = None
        self._buds = []
        self.extra_info = {}
        self.stop_info = None
        # Levels are named 1, 2, 3, ... in order of creation.
        self.name = str(self.root_size)
        self.indicator = self.parent_bud.indicator
        self.parent_bud.child_level = self

    @property
    def buds(self):
        if bool(self._buds):
            return self._buds
        return [self.trivial_bud]

    @property
    def root_size(self):
        return self.root.size

    @property
    def order(self):
        return int(self.name)

    @property
    def sorted_list(self):
        return sorted(self.sample_list, key=lambda x: x.performance)

    @property
    def unique_list(self):
        return [self.sample_list[ind]
                for ind in np.unique([samp.array for samp in self.sample_list],
                                     axis=0,
                                     return_index=True)[1]]

    @property
    def level_size(self):
        return len(self.sample_list)

    @property
    def branch(self):
        """Path from the root (initial level) down to this level."""
        return list(self.ancestors) + [self]

    @property
    def sample_list(self):
        return [sample
                for chain in self.chain_list
                for sample in chain]

    @property
    def budget(self):
        if self.order == 1:
            return self.level_size
        return self.level_size - len(self.chain_list)

    @property
    def is_stop(self):
        if self.stop_info is None:
            return False
        return any(self.stop_info.values())

    @property
    def non_used_buds(self):
        """Every bud in the WHOLE tree that has not yet grown a level."""
        non_used_buds = []
        for level in self.root.preorder():
            for bud in level.buds:
                if not bud.is_used:
                    non_used_buds.append(bud)
        return non_used_buds

    @property
    def non_used_non_stop_buds(self):
        return [bud
                for bud in self.non_used_buds
                if not bud.parent_level.is_stop]

    @property
    def acceptance_rate(self):
        jumps = 0
        succ_jumps = 0
        for chain in self.chain_list:
            for samp_a, samp_b in zip(chain[:-1], chain[1:]):
                jumps += 1
                if not np.array_equal(samp_a.array, samp_b.array):
                    succ_jumps += 1
        return succ_jumps / jumps


# ===========================================================================
# markov_chain.py
# ===========================================================================
class ChainData:

    def __init__(self, chain_list):
        self.chain_list = chain_list    # list of chains, each a list of arrays

    @property
    def number_of_params(self):
        return len(self.chain_list[0][0])

    @property
    def chain_number(self):
        return len(self.chain_list)


class ModifiedMetropolis:
    """Modified (component-wise) Metropolis, Algorithm 1 of the paper.

    All chains are advanced together, one step at a time. For every chain and
    every component a Gaussian proposal x_i' = x_i + scale_i * N(0, 1) is
    accepted with probability min(1, phi(x_i') / phi(x_i)). The resulting
    candidate is then kept only if it lies in the region (indicator = 1),
    otherwise the chain repeats its current state.
    """

    def __init__(self,
                 indicator,
                 scale,
                 random_state):
        self.indicator = indicator
        self.scale = scale
        self.random_state = random_state

    def update(self, chain_data, length):
        sample_shape = (chain_data.chain_number,
                        chain_data.number_of_params,
                        length)
        # All random numbers for all steps are drawn up front.
        loguniform_samps = np.log(uniform.rvs(size=sample_shape, random_state=self.random_state))
        conversion = self.scale_convert(chain_data.chain_number, length)
        proposal_samps = norm.rvs(size=sample_shape, scale=conversion, random_state=self.random_state)
        state_list = [np.array([chain[-1] for chain in chain_data.chain_list])]
        for i in range(length):
            current_state = state_list[-1]
            # Step 1: component-wise Metropolis against the N(0,1) density
            prop_state = current_state + proposal_samps[:, :, i]
            alpha = norm.logpdf(prop_state) - norm.logpdf(current_state)
            accept = alpha >= loguniform_samps[:, :, i]
            new_state = np.where(accept, prop_state, current_state)
            # Step 2: keep the candidate only if it is inside the region
            ind_accept = np.array([[bool(self.indicator(state))] for state in new_state])
            state_list.append(np.where(ind_accept, new_state, current_state))
        for state in state_list[1:]:
            for sample, chain in zip(state, chain_data.chain_list):
                chain.append(sample)

    def scale_convert(self, chain_number, length):
        reshaped_scales = self.scale.reshape(1, self.scale.shape[0], 1)
        expanded_array = np.tile(reshaped_scales, (chain_number, 1, length))
        return expanded_array


# ===========================================================================
# scale.py  -- proposal spread for the Markov chains
# ===========================================================================
class Scaler(ABC):

    @abstractmethod
    def compute_scale(self, seeds):
        pass


class ConstantScaler(Scaler):
    """Same proposal standard deviation in every direction."""

    def __init__(self, scale):
        self.scale = scale

    def compute_scale(self, seeds):
        return np.array([self.scale] * seeds[0].array.shape[0])


class AdaptiveScaler(Scaler):
    """Proposal standard deviation = spread of the seeds in each direction."""

    def compute_scale(self, seeds):
        stds = np.std([samp.array for samp in seeds], axis=0)
        return stds


# ===========================================================================
# allocate.py  -- how long each Markov chain is
# ===========================================================================
class Allocator(ABC):

    def __init__(self,
                 verbose,
                 level_probability,
                 level_size):
        self.verbose = verbose
        self.level_probability = level_probability
        self.level_size = level_size
        self.budget = int((1 - self.level_probability) * self.level_size)

    @abstractmethod
    def allocate(self, bud):
        pass


class SubsetAllocator(Allocator):
    """Plain SuS: every chain has length 1/p."""

    def allocate(self, bud):
        return int(self.level_probability ** -1)


class BranchingAllocator(Allocator):
    """NSuS: n(1-p) new samples are split evenly over all open branches,
    then over the seeds of this bud:  n_s = floor(n(1-p) / (N_p n_c)) + 1."""

    def allocate(self, bud):
        split_budget = self.budget / len(bud.parent_level.non_used_buds)
        return int(split_budget / len(bud.seeds)) + 1


# ===========================================================================
# choose.py  -- which bud to grow next
# ===========================================================================
class Chooser(ABC):

    @abstractmethod
    def choose(self, active_buds):
        pass


class DepthChooser(Chooser):
    """Grow the shallowest open branch first (breadth-first)."""

    def choose(self, active_buds):
        min_depth = min([bud.parent_level.depth for bud in active_buds])
        return [bud for bud in active_buds if bud.parent_level.depth == min_depth][0]


# ===========================================================================
# stop.py
# ===========================================================================
class Stopper:

    def __init__(self,
                 conditions,
                 verbose,
                 ):
        self.conditions = conditions
        self.verbose = verbose

    def stop(self, level):
        level.stop_info = {condition.name: condition(level)
                           for condition in self.conditions}
        self.print_stop_info(level)

    @verbose
    def print_stop_info(self, level):
        for item in level.stop_info.items():
            if item[1]:
                print(f"{item[0]} stopper has stopped level {level.name}.")


class Condition(ABC):

    @property
    def name(self):
        return self._name()

    @abstractmethod
    def _name(self):
        pass

    @abstractmethod
    def __call__(self, level):
        pass


class Failure(Condition):
    """Stop when even the worst would-be seed fails, i.e. at least n_c
    samples of the level satisfy g >= threshold (Eq. 2)."""

    def __init__(self, threshold):
        self.threshold = threshold

    def _name(self):
        return 'Failure'

    def __call__(self, level):
        bud = level.trivial_bud
        return bud.seeds[0].performance >= self.threshold


class ThresholdConverge(Condition):
    """Stop if the next intermediate threshold would not be higher than the
    current one (the chains are stuck, so the algorithm cannot progress)."""

    def _name(self):
        return 'Threshold Converge'

    def __call__(self, level):
        bud = level.trivial_bud
        return bud.seeds[0].performance <= bud.parent_level.indicator.threshold


# ===========================================================================
# level_create.py
# ===========================================================================
class LevelCreator:

    def __init__(self,
                 markov_chain,
                 dimension,
                 level_size,
                 random_state,
                 verbose,
                 performance_function,
                 allocator,
                 ):
        self.markov_chain = markov_chain
        self.dimension = dimension
        self.level_size = level_size
        self.random_state = random_state
        self.verbose = verbose
        self.performance_function = performance_function
        self.allocator = allocator

    def create_initial(self):
        """Level 1 (the paper's X^0): plain Monte Carlo from N(0, I)."""
        array = multivariate_normal.rvs(mean=np.zeros(self.dimension),
                                        cov=np.identity(self.dimension),
                                        size=self.level_size,
                                        random_state=self.random_state)
        chain_list = [[Sample(array=arrayi,
                              performance=self.performance_function(arrayi))]
                      for arrayi in array]
        initial_indicator = Indicator(threshold=-np.inf,
                                      partition_indicator=lambda x: 1,
                                      performance_function=self.performance_function)
        initial_bud = Bud(indicator=initial_indicator)
        initial_level = Level(chain_list=chain_list,
                              parent_bud=initial_bud)
        self.print_create_level(initial_level)
        return initial_level

    def create(self, bud):
        """Grow a new level from a bud: run one Markov chain from each seed."""
        chain_data = ChainData([[seed.array] for seed in bud.seeds])
        self.markov_chain.indicator = bud.indicator
        chain_length = self.allocator.allocate(bud)
        self.markov_chain.scale = bud.scale
        self.markov_chain.update(chain_data,
                                 chain_length - 1,
                                 )
        sample_chain_list = [[Sample(array=array,
                                     performance=self.performance_function(array))
                              for array in chain]
                             for chain in chain_data.chain_list]
        level = Level(chain_list=sample_chain_list,
                      parent_bud=bud)
        self.print_create_level(level)
        return level

    @verbose
    def print_create_level(self, level):
        print(f"Level {level.name} created.")


# ===========================================================================
# partition.py
# ===========================================================================
class Partitioner(ABC):

    def __init__(self,
                 verbose):
        self.verbose = verbose

    @abstractmethod
    def partition(self, level):
        pass


class NoPartitioner(Partitioner):
    """Never splits: NSuS with this partitioner IS subset simulation."""

    def partition(self, level):
        return [lambda x: 1]


# ===========================================================================
# hill_valley_graph_partitioner.py
# ===========================================================================
class HillValleyGraph:
    """Graph on the graph samples with an edge i--j when no "valley" is found
    between x_i and x_j, i.e. when the segment between them seems to stay in
    {g >= min(g(x_i), g(x_j))}  (Definition 3 / Eq. 19)."""

    def __init__(self):
        self.graph = None
        self.eval_count = 0

    def cache_fit(self, X, y, cache, perf_function):
        """Hill valley test using the search history where possible.

        For each pair, look at every previously evaluated point that lies in
        the open hypercube spanned by the pair. Pairs with no such point get
        one new evaluation at their midpoint (which is added to the history).
        The pair is connected iff no history point inside its hypercube has
        performance below min(y_i, y_j).
        """
        if self.trivial_graph(X):
            return
        i, j = np.tril_indices(len(X), k=-1)          # all pairs i > j
        within_range, _ = self.history_and_within_range(cache, X, i, j)
        no_between_index = np.where(~np.any(within_range, axis=1))[0]
        for ind in no_between_index:
            perf_function((X[i][ind] + X[j][ind]) / 2)
            self.eval_count += 1
        within_range, hist_perfs = self.history_and_within_range(cache, X, i, j)
        perf_bools = (hist_perfs < np.minimum(y[i], y[j]).reshape(-1, 1))
        adj_info = (~np.any(within_range & perf_bools, axis=1)).astype(int)
        self.adj_info_to_graph(adj_info, i, j, X)

    def fit(self, X, y, perf_function):
        """Plain hill valley test: one midpoint evaluation per pair."""
        if self.trivial_graph(X):
            return
        i, j = np.tril_indices(len(X), k=-1)
        performances = np.array([perf_function((x_i + x_j) / 2) for x_i, x_j in zip(X[i], X[j])])
        self.eval_count += len(performances)    # PORT NOTE 5: not counted in the original
        adj_info = (performances >= np.minimum(y[i], y[j])).astype(int)
        self.adj_info_to_graph(adj_info, i, j, X)

    def trivial_graph(self, X):
        cond = len(X) == 1
        if cond:
            graph = nx.Graph()
            graph.add_nodes_from(range(len(X)))
            self.graph = graph
        return cond

    def adj_info_to_graph(self, adj_info, i, j, X):
        adj_matrix = np.zeros((len(X), len(X)))
        adj_matrix[i, j] = adj_info
        adj_matrix[j, i] = adj_info
        self.graph = nx.from_numpy_array(adj_matrix)

    @staticmethod
    def create_history_data(cache, X):
        hist_arrays = np.array([tup for tup in cache.keys()])
        hist_perfs = np.array([val for val in cache.values()])
        mask = ~((hist_arrays[:, None] == X).all(-1).any(1))   # drop the graph samples themselves
        return hist_arrays[mask], hist_perfs[mask]

    @staticmethod
    def create_within_range(end_points_i, end_points_j, mask_hist_arrays):
        # within[pair, h] = history point h lies in the box between the pair
        min_points = np.minimum(end_points_i, end_points_j)
        max_points = np.maximum(end_points_i, end_points_j)
        min_points = min_points[:, np.newaxis, :]
        max_points = max_points[:, np.newaxis, :]
        hist_expanded = mask_hist_arrays[np.newaxis, :, :]
        return np.all((hist_expanded >= min_points) & (hist_expanded <= max_points), axis=2)

    def history_and_within_range(self, cache, X, i, j):
        hist_arrays, hist_perfs = self.create_history_data(cache, X)
        within_range = self.create_within_range(X[i], X[j], hist_arrays)
        return within_range, hist_perfs


class HillValleyGraphPartitioner(Partitioner):
    """HVG partitioner: hill valley graph -> community detection (ALP) ->
    classification (linear SVC). Section 4 / Fig. 4 of the paper."""

    def __init__(self,
                 classifier_creator,
                 random_state,
                 max_branches,
                 graph_size,
                 iterations,
                 is_cache,
                 *args,
                 **kwargs,
                 ):
        super().__init__(*args, **kwargs)
        self.classifier_creator = classifier_creator
        self.max_branches = max_branches
        self.graph_size = graph_size
        self.is_cache = is_cache
        self.random_state = random_state
        self.eval_count = 0
        self.iterations = iterations

    def partition(self, level):

        # Already at the maximum number of branches: do not split.
        if len(level.non_used_buds) >= self.max_branches:
            return [self.trivial_indicator]

        adjusted_max_partition = self.max_branches - len(level.non_used_buds) + 1

        # 1. Hill valley graph
        level_to_hvg(level, self.graph_size, self.random_state, self.is_cache)
        hvg = level.extra_info['hvg']
        self.eval_count += hvg.eval_count
        level.extra_info['graph'] = hvg.graph

        # 2. Community detection: ALP is random, so run it many times.
        partitions = [list(asyn_lpa_communities(hvg.graph, seed=self.random_state))
                      for _ in range(self.iterations)]
        part_len_counts = Counter([len(part) for part in partitions])
        if part_len_counts[1] / self.iterations > 0.5:      # mostly one community
            return [self.trivial_indicator]
        partitions = [part for part in partitions if len(part) > 1]
        partitions = [shrink_partition(part, adjusted_max_partition, self.random_state)
                      for part in partitions]

        # 3. Classification: fit a classifier to every candidate labelling and
        #    keep the one it reproduces best (balanced accuracy).
        classifier = self.classifier_creator.create(level)
        scores = []
        for partition in partitions:
            X, y, _ = self.fit_classifier(level, partition, classifier)
            scores.append(balanced_accuracy_score(y, classifier.predict(X)))
        best_partition = partitions[np.argmax(scores)]

        X, y, sample_partition = self.fit_classifier(level, best_partition, classifier)
        level.extra_info['partition'] = best_partition
        level.extra_info['classifier'] = classifier
        level.extra_info['X'] = X
        level.extra_info['y'] = y
        level.extra_info['sample_partition'] = sample_partition
        # One indicator per class: 1 if the classifier puts x in that class.
        return [self.indicator_factory(classifier.predict_single, label)
                for label in range(len(best_partition))]

    @staticmethod
    def indicator_factory(classifier, label):
        def indicator(x):
            return int(classifier(x) == label)
        return indicator

    @staticmethod
    def trivial_indicator(x):
        return 1

    def fit_classifier(self, level, partition, classifier):
        sample_partition = [[level.extra_info['graph_samples'][ind] for ind in part]
                            for part in partition]
        X = np.array([samp.array for part in sample_partition for samp in part])
        y = np.array([i for i, part_set in enumerate(sample_partition) for _ in range(len(part_set))])
        classifier.fit(X, y)
        return X, y, sample_partition


def level_to_hvg(level, graph_size, random_state, is_cache):
    choose_samples(level, graph_size, random_state)
    graph_samples = level.extra_info['graph_samples']
    perf_function = level.indicator.performance_function
    arrays = np.array([samp.array for samp in graph_samples])
    perfs = np.array([samp.performance for samp in graph_samples])
    hvg = HillValleyGraph()
    if is_cache:
        cache = perf_function.cache
        hvg.cache_fit(arrays, perfs, cache, perf_function)
    else:
        hvg.fit(arrays, perfs, perf_function)
    level.extra_info['hvg'] = hvg


def choose_samples(level, graph_size, random_state):
    """Pick the graph samples, spreading them over chain groups.

    A chain group = all (distinct) samples of chains that started from the
    same seed. Groups are visited round-robin in random order, taking one
    random sample from each, so the graph covers as many groups as possible.
    The graph size is shared between all open branches.
    """
    seed_array = np.array([chain[0].array for chain in level.chain_list])
    unique_seeds = np.unique(seed_array, axis=0)
    chain_group_indicies = [[] for _ in range(len(unique_seeds))]
    for i in range(len(seed_array)):
        chain_group_indicies[np.where(np.all(unique_seeds == seed_array[i], axis=1))[0][0]].append(i)
    chain_groups_non_unique = [sum([level.chain_list[ind] for ind in inds], [])
                               for inds in chain_group_indicies]
    chain_groups = []
    for chain_group in chain_groups_non_unique:
        unique_inds = np.unique([samp.array for samp in chain_group], axis=0, return_index=True)[1]
        chain_groups.append([chain_group[ind] for ind in unique_inds])
    for chain_group in chain_groups:
        random_state.shuffle(chain_group)
    random_state.shuffle(chain_groups)
    graph_sample_list = []
    while any(chain_group for chain_group in chain_groups):
        for chain_group in chain_groups:
            if chain_group:
                graph_sample_list.append(chain_group.pop())
    graph_size = graph_size // len(level.non_used_buds)
    graph_samples = graph_sample_list[:graph_size]
    level.extra_info['graph_samples'] = graph_samples


def shrink_partition(partition, n, rng):
    """Randomly merge communities until there are at most n."""
    partition = list(partition)
    while len(partition) > n:
        idxa, idxb = rng.choice(range(len(partition)), size=2, replace=False)
        partition[idxa] = partition[idxa].union(partition[idxb])
        partition.pop(idxb)
    return partition


# ===========================================================================
# lsvc.py  -- standardise -> PCA -> linear SVC, with a fast single-point predict
# ===========================================================================
class BarebonesStandardScaler(StandardScaler):

    def transform_single(self, x):
        return (x - self.mean_) / self.var_ ** .5


class BarebonesPCA(PCA):

    def transform_single(self, x):
        return np.dot(x - self.mean_, self.components_.T)


class BarebonesLinearSVC(LinearSVC):

    def predict_single(self, x):
        scores = np.dot(self.coef_, x) + self.intercept_
        if len(scores) == 1:
            return self.classes_[int((scores > 0)[0])]
        else:
            return self.classes_[np.argmax(scores)]


class PipeLSVC:
    """The scaler and PCA are fitted once on ALL samples of the level; only
    the SVC is refitted for each candidate labelling of the graph samples."""

    def __init__(self, bb_lsvc, bb_ss, bb_pca):
        self.bb_lsvc = bb_lsvc
        self.bb_ss = bb_ss
        self.bb_pca = bb_pca

    def fit(self, X, y):
        X = self.bb_ss.transform(X)
        X = self.bb_pca.transform(X)
        self.bb_lsvc.fit(X, y)

    def predict_single(self, x):
        x = self.bb_ss.transform_single(x)
        x = self.bb_pca.transform_single(x)
        return self.bb_lsvc.predict_single(x)

    def predict(self, X):
        X = self.bb_ss.transform(X)
        X = self.bb_pca.transform(X)
        return self.bb_lsvc.predict(X)


class LSVCCreator:

    def __init__(self, params, random_state):
        self.params = params
        self.params['random_state'] = random_state.integers(0, 1000000)

    def create(self, level):
        level_array = np.array([sample.array for sample in level.sample_list])
        bb_ss = BarebonesStandardScaler()
        bb_ss.fit(level_array)
        ss_array = bb_ss.transform(level_array)
        bb_pca = BarebonesPCA()
        bb_pca.fit(ss_array)
        bb_lsvc = BarebonesLinearSVC(**self.params)
        return PipeLSVC(bb_lsvc, bb_ss, bb_pca)


# ===========================================================================
# estimate.py
# ===========================================================================
def failure_to_reliability(failure):
    return -norm.ppf(failure)


def reliability_to_failure(reliability):
    return norm.cdf(-reliability)


def probability_of_indicator(indicator, level):
    """Fraction of a level's samples inside a region: one factor P_hat_i."""
    return sum(indicator(level.sample_list)) / level.level_size


def exceedance_probability(bss, threshold):
    """P(g(x) >= threshold) = sum over (trimmed) leaves of the branch
    estimates (Eq. 12). threshold need not be the critical one."""
    return sum([estimate_trimmed_leaf_probability(threshold, leaf)
                for leaf in trimmed_leaves(bss, threshold)])


def estimate_leaf_probability(thresh, leaf):
    trimmed_leaf = trim_leaf(leaf, thresh)
    return estimate_trimmed_leaf_probability(thresh, trimmed_leaf)


def trim_leaf(leaf, threshold):
    branch = leaf.branch
    i = -1
    while threshold < branch[i].indicator.threshold:
        i -= 1
    return branch[i]


def estimate_trimmed_leaf_probability(threshold, leaf):
    """SuS estimator along one branch (Eq. 4): product of the conditional
    fractions down the branch, times the fraction of the leaf above
    `threshold` (within the leaf's partition set)."""
    branch = leaf.branch
    return (np.prod([probability_of_indicator(level.indicator, level.parent)
                     for level in branch[1:]])
            * probability_of_indicator(leaf.indicator.new_threshold(threshold),
                                       leaf))


def trimmed_leaves(nss, threshold):
    """Leaves of the tree after trimming at `threshold` (Section 3.3): walk
    down from the root and stop at any level that has a child whose
    intermediate threshold already exceeds `threshold`."""
    leaves = []
    potential_leaves = [nss.initial_level]
    while potential_leaves:
        leaf = potential_leaves.pop()
        if not leaf.children:
            leaves.append(leaf)
            continue
        cond = any([child.indicator.threshold > threshold
                    for child in leaf.children])
        if cond:
            leaves.append(leaf)
        else:
            potential_leaves.extend(leaf.children)
    return leaves


def cov(bss, threshold):
    """Coefficient of variation of the NSuS estimator (Eq. 13)."""
    leaves = trimmed_leaves(bss, threshold)
    leave_it = range(len(leaves))
    prod_ij = list(product(leave_it, repeat=2))

    p = [estimate_trimmed_leaf_probability(threshold,
                                           leaf)
         for leaf in leaves]
    w_total = sum([p[i] * p[j]
                   for i, j in prod_ij])
    if w_total == 0:                     # PORT NOTE 3: nothing failed anywhere
        return np.nan
    w_dict = {i: {j: (p[i] * p[j]) / w_total for j in leave_it}
              for i in leave_it}

    common_ancestors = {i: {j: find_common_ancestor(leaves[i], leaves[j])
                            for j in leave_it}
                        for i in leave_it}

    big_sum = []
    for i, j in prod_ij:
        if w_dict[i][j] == 0:            # PORT NOTE 3: zero-weight term
            continue
        if i == j:
            temp_thresh = threshold
        else:
            temp_thresh = common_ancestors[i][j].indicator.threshold
        big_sum.append(w_dict[i][j]
                       * (leaf_cov(common_ancestors[i][j], temp_thresh)**2))
    return np.sqrt(sum(big_sum))


def find_common_ancestor(leaf_a, leaf_b):
    return [node
            for node in leaf_a.branch
            if node in leaf_b.branch][-1]


def leaf_cov(leaf, threshold):
    """delta = sqrt(sum_k delta_k^2) along a branch (Eq. 8)."""
    cov_square = [level_cov(level.parent, level.indicator) ** 2
                  for level in leaf.branch[1:]]
    last_indicator = leaf.indicator.new_threshold(threshold)
    cov_square.append(level_cov(leaf, last_indicator) ** 2)
    return np.sqrt(sum(cov_square))


def level_cov(level, indicator):
    """delta_k: Eq. (6) for the Monte Carlo level, Eq. (7) for MCMC levels."""
    prob = probability_of_indicator(indicator, level)
    if prob == 1:
        return 0
    if level.order == 1:
        return np.sqrt((1 - prob) / (level.level_size * prob))
    else:
        gamma = level_gamma(level, indicator)
        return np.sqrt(((1 - prob) / (level.level_size * prob)) * (1 + gamma))


def level_gamma(level, indicator):
    """gamma_k: correlation factor of the Markov chains (Au & Beck 2001)."""
    n_c = len(level.chain_list)
    n = level.level_size
    n_s = int(n // n_c)

    prob = probability_of_indicator(indicator, level)
    f_indicator_list = indicator(level.sample_list)
    indicator_list = [f_indicator_list[i * n_s:(i + 1) * n_s]
                      for i in range(n_c)]

    r_i_list = []
    for i in range(n_s):
        sums = sum([sum([chain[i_dash] * chain[i_dash + i]
                         for i_dash in range(n_s - i)])
                    for chain in indicator_list])
        r_i_list.append((sums / (n - (i * n_c))) - (prob ** 2))

    if r_i_list[0] > 0:
        gamma = 2 * (sum([(1 - (i / n_s)) * (r_i_list[i + 1] / r_i_list[0])
                          for i in range(n_s - 1)]))
    else:
        gamma = np.inf
    return gamma


def level_ess(level, indicator):
    if level.order == 1:
        return level.level_size
    else:
        gamma = level_gamma(level, indicator)
        return level.level_size / (1 + gamma)


def efficiency(level, indicator):
    return (1 + level_gamma(level, indicator))**-1


# ===========================================================================
# nsus.py  -- the framework
# ===========================================================================
class NSuS:
    """The general Niching Subset Simulation framework (Algorithm 3).

    It is assembled from interchangeable parts: a Markov chain, a
    partitioner, an allocator, a scaler, a chooser and stopping conditions.
    SubsetSimulation and HillValleyGraphNSuS below are two such assemblies.
    """

    def __init__(self,
                 performance_function,
                 level_probability,
                 level_size,
                 dimension,
                 stop_conditions,
                 markov_chain,
                 partitioner,
                 seed_method,
                 allocator,
                 scaler,
                 random_state,
                 verbose,
                 chooser,
                 ):
        self.level_probability = level_probability
        self.level_size = level_size
        self.dimension = dimension
        self.stop_conditions = stop_conditions
        self.partitioner = partitioner
        self.markov_chain = markov_chain
        self.random_state = random_state
        self.allocator = allocator
        self.verbose = verbose
        self.chooser = chooser
        self.scaler = scaler
        self.initial_level = None
        self.performance_function = PerformanceFunction(performance_function)
        self.stopper = Stopper(self.stop_conditions,
                               self.verbose,
                               )
        self.bud_creator = BudCreator(seed_method,
                                      self.level_probability,
                                      self.scaler,
                                      self.verbose,
                                      )
        self.level_creator = LevelCreator(self.markov_chain,
                                          self.dimension,
                                          self.level_size,
                                          self.random_state,
                                          self.verbose,
                                          self.performance_function,
                                          self.allocator,
                                          )

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
        self.print_terminate()
        return self

    def bulk(self, threshold, percent_limit):
        """Optional post-processing used in the paper's code: top up the
        levels of the important branches (those contributing more than
        percent_limit of pF) so each has n(1-p) new samples, as in SuS."""
        leaf_probs = [estimate_leaf_probability(threshold, leaf) for leaf in self.leaves]
        total = sum(leaf_probs)
        leaf_percentages = [prob / total for prob in leaf_probs]
        filter_leaves = [leaf for leaf, percentage in zip(self.leaves, leaf_percentages)
                         if percentage > percent_limit]
        levels = sum([leaf.branch for leaf in filter_leaves], [])
        budget = int(self.level_size * (1 - self.level_probability))
        for level in levels:
            if level.order == 1:
                continue
            markov_chain = self.level_creator.markov_chain
            chain_number = len(level.chain_list)
            chain_data = ChainData([[sample.array for sample in chain] for chain in level.chain_list])
            bud = level.parent_bud
            markov_chain.indicator = bud.indicator
            self.markov_chain.scale = bud.scale
            missing_budget = budget - level.budget
            update = missing_budget // chain_number
            markov_chain.update(chain_data, update)
            sample_chain_list = [[Sample(array=array,
                                         performance=self.performance_function(array))
                                  for array in chain]
                                 for chain in chain_data.chain_list]
            level.chain_list = sample_chain_list

    @property
    def is_initial(self):
        return self.initial_level is None

    @property
    def options(self):
        return self.initial_level.non_used_non_stop_buds

    @property
    def is_stop(self):
        if self.is_initial:
            return False
        return not bool(self.options)

    @verbose
    def print_terminate(self):
        print("No more options, algorithm terminated.")

    def find(self, name):
        return next(level for level in self.initial_level.preorder()
                    if level.name == str(name))

    @property
    def all_levels(self):
        all_levels = list(self.initial_level.preorder())
        return sorted(all_levels, key=lambda x: x.order)

    @property
    def all_samples(self):
        return [samp for level in self.all_levels
                for samp in level.sample_list]

    @property
    def leaves(self):
        return self.initial_level.leaves


# ===========================================================================
# implementation.py  -- two ready-made assemblies of the framework
# ===========================================================================
class SubsetSimulation(NSuS):
    """Ordinary subset simulation written inside the NSuS framework."""

    def __init__(self,
                 performance_function,
                 dimension,
                 level_size=1000,
                 level_probability=0.1,
                 threshold=0,
                 scale=1,
                 seed=None,
                 verbose=False):

        random_state = np.random.default_rng(seed)
        stop_conditions = [
            Failure(threshold=threshold),
            ThresholdConverge(),
        ]
        chooser = DepthChooser()

        if scale is None:
            scaler = AdaptiveScaler()
        else:
            scaler = ConstantScaler(scale)

        markov_chain = ModifiedMetropolis(indicator=None,
                                          scale=None,
                                          random_state=random_state)
        partitioner = NoPartitioner(verbose=verbose)
        allocator = SubsetAllocator(verbose=verbose,
                                    level_probability=level_probability,
                                    level_size=level_size)

        seed_method = 'seeds_prob_exclusive'

        super().__init__(performance_function=performance_function,
                         level_probability=level_probability,
                         level_size=level_size,
                         dimension=dimension,
                         stop_conditions=stop_conditions,
                         markov_chain=markov_chain,
                         partitioner=partitioner,
                         seed_method=seed_method,
                         allocator=allocator,
                         scaler=scaler,
                         random_state=random_state,
                         verbose=verbose,
                         chooser=chooser,
                         )


class HillValleyGraphNSuS(NSuS):
    """NSuS with the hill valley graph partitioner (the paper's algorithm)."""

    def __init__(self,
                 performance_function,
                 dimension,
                 level_size=1000,
                 level_probability=0.1,
                 threshold=0,
                 graph_size=15,
                 max_branches=2,
                 scale=1,
                 seed=None,
                 verbose=False,
                 is_cache=True,
                 iterations=100,
                 svc_params=None):

        # PORT NOTE 4: these two were hard-coded in the original.
        params = ({'penalty': 'l1', 'C': 1, 'dual': 'auto', 'max_iter': 1000000}
                  if svc_params is None else dict(svc_params))

        stop_conditions = [
            Failure(threshold=threshold),
            ThresholdConverge(),
        ]
        chooser = DepthChooser()
        random_state = np.random.default_rng(seed)

        if scale is None:
            scaler = AdaptiveScaler()
        else:
            scaler = ConstantScaler(scale)

        markov_chain = ModifiedMetropolis(indicator=None,
                                          scale=None,
                                          random_state=random_state)

        classifier_creator = LSVCCreator(params=params,
                                         random_state=random_state)

        partitioner = HillValleyGraphPartitioner(iterations=iterations,
                                                 classifier_creator=classifier_creator,
                                                 graph_size=graph_size,
                                                 max_branches=max_branches,
                                                 verbose=verbose,
                                                 is_cache=is_cache,
                                                 random_state=random_state)
        allocator = BranchingAllocator(verbose=verbose,
                                       level_probability=level_probability,
                                       level_size=level_size)

        seed_method = 'seeds_prob_exclusive'

        super().__init__(performance_function=performance_function,
                         level_probability=level_probability,
                         level_size=level_size,
                         dimension=dimension,
                         stop_conditions=stop_conditions,
                         markov_chain=markov_chain,
                         partitioner=partitioner,
                         seed_method=seed_method,
                         allocator=allocator,
                         scaler=scaler,
                         random_state=random_state,
                         verbose=verbose,
                         chooser=chooser,
                         )


# ===========================================================================
# Convenience: a results summary in the style of Subset_Simulation.py
# (not in the original package)
# ===========================================================================
def summarise(ss, threshold):
    """Collect the main outputs of a finished run into a dict.

        'pF'        : estimate of P(g(x) >= threshold)
        'cov'       : estimated coefficient of variation of pF (Eq. 13)
        'N'         : total number of distinct g evaluations (incl. HVG ones)
        'N_hvg'     : g evaluations spent building hill valley graphs
        'n_levels'  : number of levels in the tree
        'n_leaves'  : number of branches (leaves) at the end
        'branches'  : one dict per trimmed leaf: its levels, thresholds
                      and contribution to pF
    """
    branches = []
    for leaf in trimmed_leaves(ss, threshold):
        branches.append({
            'levels': [level.name for level in leaf.branch],
            'thresholds': [level.indicator.threshold for level in leaf.branch[1:]],
            'pF': estimate_trimmed_leaf_probability(threshold, leaf),
            'n_fail': sum(leaf.indicator.new_threshold(threshold)(leaf.sample_list)),
        })
    n_hvg = getattr(ss.partitioner, 'eval_count', 0)
    return {'pF': exceedance_probability(ss, threshold),
            'cov': cov(ss, threshold),
            'N': ss.performance_function.eval_count,
            'N_hvg': n_hvg,
            'n_levels': len(ss.all_levels),
            'n_leaves': len(ss.leaves),
            'branches': branches}


def print_tree(ss, threshold=None):
    """Text picture of the level tree (cf. Fig. 3 of the paper)."""
    def visit(level, prefix, last):
        b = level.indicator.threshold
        info = f"level {level.name}: {level.level_size} samples, b = {b:.4g}"
        if threshold is not None and not level.children:
            info += f", pF contribution = {estimate_leaf_probability(threshold, level):.3e}"
        if level.is_stop:
            info += "  [" + ", ".join(k for k, v in level.stop_info.items() if v) + "]"
        if level.parent is None:                        # the root
            print(info)
            child_prefix = ""
        else:
            print(prefix + ("└── " if last else "├── ") + info)
            child_prefix = prefix + ("    " if last else "│   ")
        for k, child in enumerate(level.children):
            visit(child, child_prefix, k == len(level.children) - 1)
    visit(ss.initial_level, "", True)
