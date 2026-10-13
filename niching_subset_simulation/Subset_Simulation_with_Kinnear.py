"""
Subset Simulation for a Reliability Problem

    Performance function: g(x) = x1 + ... + xd
    Input variables x1, ..., xd are i.i.d. N(0, 1)
    Failure  <=>  g(x) > YF

HOW TO RUN: set the parameters just below, then run this file. If using a different performance function, 
and if true value is unknown, disable that result on lines 201/207) 
"""

import numpy as np
from scipy.stats import norm
import time

# --- extra imports needed by Kinnear's code ---
from abc import ABC, abstractmethod
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from itertools import product
from anytree import AnyNode, PreOrderIter, findall_by_attr
import networkx as nx
from networkx.algorithms.community import asyn_lpa_communities
from scipy.stats import multivariate_normal, uniform
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import Pipeline
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
MAX_BRANCHES = 4   # NSuS: maximum number of branches
SCALE = 1          # NSuS: proposal std of the Markov chains (None = adaptive)

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
# Kinnear: nsus/utils.py  (pasted unchanged, only its import lines removed)
# ===========================================================================

def verbose(method):
    def wrapper(self, *args, **kwargs):
        if self.verbose:
            return method(self, *args, **kwargs)
    return wrapper


# ===========================================================================
# Kinnear: nsus/performance_function.py  (pasted unchanged, only its import lines removed)
# ===========================================================================

class PerformanceFunction:

    def __init__(self, performance_function):
        self.non_cache_performance_function = deepcopy(performance_function)
        self.cache  = {}

    def __call__(self, x):
        tuple_x = tuple(x)
        try:
            result = self.cache[tuple_x]
        except KeyError:
            self.cache[tuple_x] = result = self.non_cache_performance_function(tuple_x)
        return result

    @property
    def eval_count(self):
        return len(self.cache)
    
        
        
    
    






# ===========================================================================
# Kinnear: nsus/indicator.py  (pasted unchanged, only its import lines removed)
# ===========================================================================
class Indicator:

    def __init__(self, threshold, partition_indicator, performance_function):
        self.threshold = threshold
        self.partition_indicator = partition_indicator
        self.performance_function = performance_function
    
    def threshold_indicator(self, x):
        return int(self.performance_function(x)
                   >= self.threshold)
    
    def indicator(self, x):
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
    
    def child_indicator(self,threshold,partition_indicator):
        combined_partition_indicator = self.indicator_combine(self.partition_indicator,
                                                              partition_indicator)
        return Indicator(threshold,
                         combined_partition_indicator,
                         self.performance_function)
    
    def new_threshold(self, threshold):
        return Indicator(threshold,
                         self.partition_indicator,
                         self.performance_function)


# ===========================================================================
# Kinnear: nsus/markov_chain.py  (pasted unchanged, only its import lines removed)
# ===========================================================================


class ChainData:

    def __init__(self,
                 chain_list,
                 parameter_names=None):
        self.chain_list = chain_list

        if parameter_names is None:
            self.parameter_names = ['Parameter ' + str(i + 1)
                                    for i in range(self.number_of_params)]
        else:
            self.parameter_names = parameter_names

    @property
    def number_of_params(self):
        return len(self.chain_list[0][0])
    
    @property
    def chain_number(self):
        return len(self.chain_list)

    def trim(self, length):
        self.chain_list = [chain[length:] for chain in self.chain_list]

    @property
    def chain_dict(self):
        array_chain_list = [[sample for sample in chain]
                            for chain in self.chain_list]
        parameter_chain_list = [arr.T for arr in np.array(array_chain_list).T]
        chain_dict = {}
        for i in range(self.number_of_params):
            chain_dict[self.parameter_names[i]] = parameter_chain_list[i]
        return chain_dict

    @property
    def all_samples(self):
        return [item for chain in self.chain_list for item in chain]
    

class ModifiedMetropolis:
    
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
        loguniform_samps = np.log(uniform.rvs(size=sample_shape,random_state=self.random_state))
        conversion = self.scale_convert(chain_data.chain_number,length)
        proposal_samps = norm.rvs(size=sample_shape,scale=conversion,random_state=self.random_state)
        state_list = [np.array([chain[-1] for chain in chain_data.chain_list])]
        for i in range(length):
            current_state = state_list[-1]
            prop_state = current_state + proposal_samps[:,:,i]
            alpha = norm.logpdf(prop_state) - norm.logpdf(current_state)
            accept = alpha >= loguniform_samps[:,:,i]
            new_state = np.where(accept, prop_state, current_state)
            ind_accept = np.array([[bool(self.indicator(state))] for state in new_state])
            state_list.append(np.where(ind_accept, new_state, current_state))
        for state in state_list[1:]:
            for sample,chain in zip(state,chain_data.chain_list):
                chain.append(sample)

    def scale_convert(self, chain_number, length):
        reshaped_scales = self.scale.reshape(1,self.scale.shape[0],1)
        expanded_array = np.tile(reshaped_scales, (chain_number, 1, length))
        return expanded_array



        

    





# ===========================================================================
# Kinnear: nsus/scale.py  (pasted unchanged, only its import lines removed)
# ===========================================================================


class Scaler(ABC):

    @abstractmethod
    def compute_scale(self, seeds):
        pass

class ConstantScaler(Scaler):

    def __init__(self, scale):
        self.scale = scale

    def compute_scale(self, seeds):
        return np.array([self.scale]* seeds[0].array.shape[0])
    
class AdaptiveScaler(Scaler):
    
    def compute_scale(self, seeds):
        stds = np.std([samp.array for samp in seeds],axis=0)
        return stds
    


# ===========================================================================
# Kinnear: nsus/allocate.py  (pasted unchanged, only its import lines removed)
# ===========================================================================


class Allocator(ABC):
    
    def __init__(self,
                 verbose,
                 level_probability,
                 level_size):
        self.verbose = verbose
        self.level_probability = level_probability
        self.level_size = level_size
        self.budget = int((1-self.level_probability) * self.level_size)
    
    @abstractmethod
    def allocate(self, bud):
        pass


class SubsetAllocator(Allocator):
    
    def allocate(self, bud):
        return int(self.level_probability ** -1)
    

class BranchingAllocator(Allocator):
    
    def allocate(self, bud):
        split_budget = self.budget / len(bud.parent_level.non_used_buds)
        return int(split_budget / len(bud.seeds)) + 1
    
















# ===========================================================================
# Kinnear: nsus/choose.py  (pasted unchanged, only its import lines removed)
# ===========================================================================

class Chooser(ABC):

    @abstractmethod
    def choose(self, active_buds):
        pass

class DepthChooser(Chooser):

    def choose(self, active_buds):
        min_depth = min([bud.parent_level.depth for bud in active_buds])
        return [bud for bud in active_buds if bud.parent_level.depth == min_depth][0]



# ===========================================================================
# Kinnear: nsus/stop.py  (pasted unchanged, only its import lines removed)
# ===========================================================================

class Stopper:
    
    def __init__(self,
                 conditions,
                 verbose,
                 ):
        self.conditions = conditions
        self.verbose = verbose

    def stop(self, level):
        level.stop_info = {condition.name:condition(level)
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
    def __call__(self,bud):
        pass

class Failure(Condition):

    def __init__(self,
                 threshold,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.threshold = threshold

    def _name(self):
        return 'Failure'

    def __call__(self, level):
        bud = level.trivial_bud
        return bud.seeds[0].performance >= self.threshold
    
class ThresholdConverge(Condition):

    def _name(self):
        return 'Threshold Converge'

    def __call__(self, level):
        bud = level.trivial_bud
        return bud.seeds[0].performance <= bud.parent_level.indicator.threshold
    

class NoMovement(Condition):

    def _name(self):
        return 'No Movement'

    def __call__(self, level):
        return all(np.array_equal(chain[0].array,chain[-1].array) for chain in level.chain_list)



  


# ===========================================================================
# Kinnear: nsus/partition.py  (pasted unchanged, only its import lines removed)
# ===========================================================================


class Partitioner(ABC):

    def __init__(self,
                 verbose):
        self.verbose = verbose

    @abstractmethod
    def partition(self, level):
        pass
    

class NoPartitioner(Partitioner):
    
    def partition(self, level):
        return [lambda x: 1]
    






# ===========================================================================
# Kinnear: nsus/lsvc.py  (pasted unchanged, only its import lines removed)
# ===========================================================================


class BarebonesPipeline(Pipeline):

    def predict_single(self, x):
        for _, transformer in self.steps[:-1]:
            x = transformer.transform_single(x)
        return self.steps[-1][1].predict_single(x)


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

    def __init__(self,params,random_state):
        self.params = params
        self.params['random_state'] = random_state.integers(0, 1000000)

    def create(self,level):
        level_array = np.array([sample.array for sample in level.sample_list])
        bb_ss = BarebonesStandardScaler()
        bb_ss.fit(level_array)
        ss_array = bb_ss.transform(level_array)
        bb_pca = BarebonesPCA()
        bb_pca.fit(ss_array)
        bb_lsvc = BarebonesLinearSVC(**self.params)
        return PipeLSVC(bb_lsvc,bb_ss,bb_pca)


        
    



    


    





    





# ===========================================================================
# Kinnear: nsus/bud_create.py  (pasted unchanged, only its import lines removed)
# ===========================================================================


class Bud:

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

    def create(self,level,partition_indicator):
        sorted_candidate_seeds = self.sorted_candidate_seeds(level,
                                                             partition_indicator)
        if len(sorted_candidate_seeds) == 0:
            return None
        
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
    
    def create_trivial(self,level):
        level.trivial_bud = self.create(level, lambda x: 1)

    def create_inidicator_buds(self,level,partition_indicators):
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
    def sorted_candidate_seeds(level,partition_indicator):
        samples = [sample
                   for sample in level.sample_list
                   if bool(partition_indicator(sample.array))]
        return sorted(samples, key=lambda x: x.performance)
    
    def seeds_prob_inclusive(self,level,sorted_candidate_seeds):
        seed_number = int(self.level_probability * level.level_size)
        thresh = sorted_candidate_seeds[-seed_number].performance
        return sorted([seed for seed in sorted_candidate_seeds if seed.performance >= thresh],
                      key=lambda x: x.performance)

    def seeds_prob_exclusive(self,level,sorted_candidate_seeds):
        seed_number = int(self.level_probability * level.level_size)
        return sorted_candidate_seeds[-seed_number:]



# ===========================================================================
# Kinnear: nsus/level.py  (pasted unchanged, only its import lines removed)
# ===========================================================================


class Level(AnyNode):

    def __init__(self,
                 parent_bud,
                 chain_list,
                 *args,
                 **kwargs
                 ):
        super().__init__(parent=parent_bud.parent_level,
                         *args,
                         **kwargs)

        self.parent_bud = parent_bud
        self.chain_list = chain_list
        self.empty_indicators = []
        self.trivial_bud = None
        self._buds = []
        self.extra_info = {}
        self.stop_info = None
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
        non_used_buds = []
        for level in PreOrderIter(self.root):
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
            for samp_a, samp_b in zip(chain[:-1],chain[1:]):
                jumps += 1
                if not np.array_equal(samp_a.array,samp_b.array):
                    succ_jumps += 1
        return succ_jumps/jumps
    

        
    
    


# ===========================================================================
# Kinnear: nsus/level_create.py  (pasted unchanged, only its import lines removed)
# ===========================================================================


@dataclass
class Sample:
    array: ...
    performance: ...


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

    def create(self,bud):
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
# Kinnear: nsus/hill_valley_graph_partitioner.py  (pasted unchanged, only its import lines removed)
# ===========================================================================


class HillValleyGraph:

    def __init__(self):
        self.graph = None
        self.eval_count = 0

    def cache_fit(self, X, y, cache, perf_function):
        if self.trivial_graph(X):
            return 
        i,j = np.tril_indices(len(X), k=-1)
        within_range, _ = self.history_and_within_range(cache,X,i,j)
        no_between_index = np.where(~np.any(within_range,axis=1))[0]
        for ind in no_between_index:
            perf_function((X[i][ind] + X[j][ind])/2)
            self.eval_count += 1
        within_range, hist_perfs = self.history_and_within_range(cache,X,i,j)
        perf_bools = (hist_perfs < np.minimum(y[i],y[j]).reshape(-1,1))
        adj_info = (~np.any(within_range & perf_bools,axis=1)).astype(int)
        self.adj_info_to_graph(adj_info,i,j,X)

    def fit(self, X, y, perf_function):
        if self.trivial_graph(X):
            return 
        i,j = np.tril_indices(len(X), k=-1)
        performances = np.array([perf_function((x_i+ x_j)/2) for x_i,x_j in zip(X[i],X[j])])
        adj_info = (performances >= np.minimum(y[i],y[j])).astype(int)
        self.adj_info_to_graph(adj_info,i,j,X)

    def trivial_graph(self, X):
        cond = len(X) == 1
        if cond:
            graph = nx.Graph()
            graph.add_nodes_from(range(len(X)))
            self.graph = graph
        return cond
    
    def adj_info_to_graph(self,adj_info,i,j,X):
        adj_matrix = np.zeros((len(X),len(X)))
        adj_matrix[i,j] = adj_info
        adj_matrix[j,i] = adj_info
        self.graph = nx.from_numpy_array(adj_matrix)

    @staticmethod
    def create_history_data(cache,X):
        hist_arrays = np.array([tup for tup in cache.keys()])
        hist_perfs = np.array([val for val in cache.values()])
        mask = ~((hist_arrays[:,None] == X).all(-1).any(1))
        return hist_arrays[mask], hist_perfs[mask]

    @staticmethod
    def create_within_range(end_points_i,end_points_j, mask_hist_arrays):
        min_points = np.minimum(end_points_i, end_points_j)
        max_points = np.maximum(end_points_i, end_points_j)
        min_points = min_points[:, np.newaxis, :]
        max_points = max_points[:, np.newaxis, :]
        hist_expanded = mask_hist_arrays[np.newaxis, :, :]
        return np.all((hist_expanded >= min_points) & (hist_expanded <= max_points), axis=2)

    def history_and_within_range(self,cache,X,i,j):
        hist_arrays, hist_perfs = self.create_history_data(cache,X)
        within_range = self.create_within_range(X[i],X[j],hist_arrays)
        return within_range, hist_perfs


class HillValleyGraphPartitioner(Partitioner):

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

    
    def partition(self,level):

        if len(level.non_used_buds) >= self.max_branches:
            return [self.trivial_indicator]
        
        adjusted_max_partition = self.max_branches - len(level.non_used_buds) + 1

        level_to_hvg(level,self.graph_size,self.random_state,self.is_cache)
        hvg = level.extra_info['hvg']
        self.eval_count += hvg.eval_count
        level.extra_info['graph'] = hvg.graph
            

        partitions = [list(asyn_lpa_communities(hvg.graph,seed=self.random_state))
                      for _ in range(self.iterations)]
        part_len_counts = Counter([len(part) for part in partitions])
        if part_len_counts[1] / self.iterations > 0.5:
            return [self.trivial_indicator]
        partitions = [part for part in partitions if  len(part) > 1]
        partitions = [shrink_partition(part,adjusted_max_partition,self.random_state)
                      for part in partitions]
            
        classifier = self.classifier_creator.create(level)
        scores = []
        for partition in partitions:
            X,y,_ = self.fit_classifier(level,partition,classifier)
            scores.append(balanced_accuracy_score(y,classifier.predict(X)))
        best_partition = partitions[np.argmax(scores)]

        X,y,sample_partition = self.fit_classifier(level,best_partition,classifier)
        level.extra_info['partition'] = best_partition
        level.extra_info['classifier'] = classifier
        level.extra_info['X'] = X
        level.extra_info['y'] = y
        level.extra_info['sample_partition'] = sample_partition
        return [self.indicator_factory(classifier.predict_single,label)
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
        classifier.fit(X,y)
        return X,y,sample_partition
        
    
def level_to_hvg(level,graph_size,random_state,is_cache):
    choose_samples(level,graph_size,random_state)
    graph_samples = level.extra_info['graph_samples']
    perf_function = level.indicator.performance_function
    arrays = np.array([samp.array for samp in graph_samples])
    perfs = np.array([samp.performance for samp in graph_samples])
    hvg = HillValleyGraph()
    if is_cache:
        cache = perf_function.cache
        hvg.cache_fit(arrays,perfs,cache,perf_function)
    else:
        hvg.fit(arrays,perfs,perf_function)
    level.extra_info['hvg'] = hvg


def choose_samples(level,graph_size,random_state):
    seed_array = np.array([chain[0].array for chain in level.chain_list])
    unique_seeds = np.unique(seed_array, axis=0)
    chain_group_indicies = [[] for _ in range(len(unique_seeds))]
    for i in range(len(seed_array)):
        chain_group_indicies[np.where(np.all(unique_seeds == seed_array[i],axis=1))[0][0]].append(i)
    chain_groups_non_unique = [sum([level.chain_list[ind] for ind in inds],[])
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


def shrink_partition(partition, n,rng):
    partition = list(partition)
    while len(partition) > n:
        idxa, idxb = rng.choice(range(len(partition)),size=2,replace=False)
        partition[idxa] = partition[idxa].union(partition[idxb])
        partition.pop(idxb)
    return partition


# ===========================================================================
# Kinnear: nsus/estimate.py  (pasted unchanged, only its import lines removed)
# ===========================================================================


def failure_to_reliability(failure):
    return -norm.ppf(failure)

def reliability_to_failure(reliability):
    return norm.cdf(-reliability)

def probability_of_indicator(indicator, level):
    return sum(indicator(level.sample_list)) / level.level_size

def exceedance_probability(bss, threshold):
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
    branch = leaf.branch
    return (np.prod([probability_of_indicator(level.indicator, level.parent)
                     for level in branch[1:]])
            * probability_of_indicator(leaf.indicator.new_threshold(threshold),
                                       leaf))

def trimmed_leaves(nss, threshold):
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
    leaves = trimmed_leaves(bss, threshold)
    leave_it = range(len(leaves))
    prod_ij = list(product(leave_it, repeat=2))

    p = [estimate_trimmed_leaf_probability(threshold,
                                           leaf)
         for leaf in leaves]
    w_total = sum([p[i] * p[j]
                   for i, j in prod_ij])
    w_dict = {i: {j: (p[i] * p[j])/w_total for j in leave_it}
              for i in leave_it}

    common_ancestors = {i: {j: find_common_ancestor(leaves[i], leaves[j])
                            for j in leave_it}
                        for i in leave_it}

    big_sum = []
    for i, j in prod_ij:
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
    cov_square = [level_cov(level.parent, level.indicator) ** 2
                  for level in leaf.branch[1:]]
    last_indicator = leaf.indicator.new_threshold(threshold)
    cov_square.append(level_cov(leaf, last_indicator) ** 2)
    return np.sqrt(sum(cov_square))

def level_cov(level, indicator):
    prob = probability_of_indicator(indicator, level)
    if prob == 1:
        return 0
    if level.order == 1:
        return np.sqrt((1 - prob) / (level.level_size * prob))
    else:
        gamma = level_gamma(level, indicator)
        return np.sqrt(((1 - prob) / (level.level_size * prob)) * (1 + gamma))

def level_gamma(level, indicator):
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
        gamma = 2 * (sum([(1 - (i / n_s)) * (r_i_list[i+1] / r_i_list[0])
                          for i in range(n_s-1)]))
    else:
        gamma = np.inf
    return gamma

def level_ess(level, indicator):
    if level.order == 1:
        return level.level_size
    else:
        gamma = level_gamma(level, indicator)
        return level.level_size / (1 + gamma)  

def efficiency(level,indicator):
    return (1+level_gamma(level,indicator))**-1

def leaf_efficiency(leaf):
     branch = leaf.branch[1:]
     buds = [level.initial_bud for level in branch[1:]] + [leaf.trivial_bud]
     return [efficiency(level,bud.indicator) for level,bud in zip(branch,buds)]



# ===========================================================================
# Kinnear: nsus/nsus.py  (pasted unchanged, only its import lines removed)
# ===========================================================================


class NSuS:

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

    def bulk(self,threshold,percent_limit):
        leaf_probs = [estimate_leaf_probability(threshold,leaf) for leaf in self.leaves]
        total = sum(leaf_probs)
        leaf_percentages = [prob/total for prob in leaf_probs]
        filter_leaves = [leaf for leaf,percentage in zip(self.leaves,leaf_percentages)
                         if percentage > percent_limit]
        levels = sum([leaf.branch for leaf in filter_leaves],[])
        budget = int(self.level_size * (1-self.level_probability))
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
        return findall_by_attr(self.initial_level, name)[0]

    @property
    def all_levels(self):
        all_levels = list(PreOrderIter(self.initial_level))
        return sorted(all_levels, key=lambda x: x.order)

    @property
    def all_samples(self):
        return [samp for level in self.all_levels
                for samp in level.sample_list]
    
    @property
    def leaves(self):
        return self.initial_level.leaves
    


# ===========================================================================
# Kinnear: nsus/implementation.py  (pasted unchanged, only its import lines removed)
# ===========================================================================
        

class SubsetSimulation(NSuS):

    def __init__(self,
                 performance_function,
                 level_probability,
                 level_size,
                 dimension,
                 seed,
                 threshold,
                 scale,
                 verbose):

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
        
        markov_chain = ModifiedMetropolis(
                                          indicator=None,
                                          scale=None,
                                          random_state=random_state)
        partitioner = NoPartitioner(verbose=verbose)
        allocator = SubsetAllocator(verbose=verbose,
                                    level_probability=level_probability,
                                    level_size=level_size)
        
        seed_method = 'seeds_prob_exclusive'
    
        super().__init__(
                         performance_function=performance_function,
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

    def __init__(self,
                 performance_function,
                 level_probability,
                 level_size,
                 dimension,
                 seed,
                 threshold,
                 graph_size,
                 max_branches,
                 scale,
                 verbose,
                 is_cache=True):
        
        params = {'penalty': 'l1', 'C':1,'dual':'auto','max_iter':1000000}
        iterations = 100
            
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
        
        markov_chain = ModifiedMetropolis(
                                          indicator=None,
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
    
        super().__init__(
                         performance_function=performance_function,
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

    # --- Kinnear's niching subset simulation ---
    start = time.perf_counter()
    nss = HillValleyGraphNSuS(
        performance_function=lambda x: performance_function(np.array(x)),  # Kinnear passes a tuple
        level_probability=p,
        level_size=n,
        dimension=d,
        seed=SEED,
        threshold=YF,
        graph_size=GRAPH_SIZE,
        max_branches=MAX_BRANCHES,
        scale=SCALE,
        verbose=True)
    nss.run()
    run_time = time.perf_counter() - start
    pF_NSuS = exceedance_probability(nss, YF)         # from estimate.py
    print("\nNICHING SUBSET SIMULATION")
    print(f"Levels / branches    : {len(nss.all_levels)} / {len(nss.leaves)}")
    print(f"Total samples N      : {nss.performance_function.eval_count}"
          f"  ({nss.partitioner.eval_count} for hill valley graphs)")
    print(f"pF (NSuS)            : {pF_NSuS:.4e}")
    print(f"Run time             : {run_time:.2f} s\n")
    print(f"pF (true value)      : {pF:.4e}")

    if d == 2:
        plot_levels_2d(res, YF, filename=SAVE_PLOT, legend=False)
        # Kinnear's levels in the same plot style: one array per NSuS level
        nsus_res = {"x": [np.array([s.array for s in lv.sample_list]).T for lv in nss.all_levels],
                    "Y": [], "pF_SS": pF_NSuS}
        plot_levels_2d(nsus_res, YF, legend=False)
