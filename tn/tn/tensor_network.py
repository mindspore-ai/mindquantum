import math
import operator
from collections import defaultdict
from copy import copy
from functools import reduce
from typing import List, NamedTuple
import multiprocessing
import time
import random

import numpy as np
import yaml
from intbitset import intbitset

from utils import contract2_numpy, read_table, gates, PerformanceProfile, estimate_time


class TNScheme(NamedTuple):
    elems: List[List[int]]
    out: List[int]


def scheme_from_schedule(schedule):
    scheme = schedule.get('scheme', None)
    if scheme is not None:
        return TNScheme(scheme['nodes'], scheme['out'])
    if schedule.get('tensors', None) is not None:
        return TNScheme([legs for t, legs in schedule['tensors']], schedule.get('fv', schedule.get('out', [])))


class ContractionTree:
    __slots__ = ['tree', 'sliced', 'scheme']

    def __init__(self, /, file=None, tree=None, scheme=None, sliced=None):
        """
        Creates contraction tree either from tree and scheme or from schedule or from file with schedule
        :param file: file to read schedule from (ignores if schedule specified)
        :param tree: contraction tree represented by recursive list structure
        :param scheme: tensor network scheme
        :param sliced: list of sliced indices
        """
        if file is not None:
            with open(file) as f:
                schedule = yaml.safe_load(f)
            tree = schedule.get('tree', schedule.get('sch', tree))
            sliced = schedule.get('sliced', schedule.get('deleted', sliced))
            scheme = scheme or scheme_from_schedule(schedule)

        self.tree = tree
        self.scheme = scheme
        self.sliced = sliced

    def save_as_yaml(self, file):
        with open(file, 'w') as f:
            f.write(f'sliced: {str(list(self.sliced))}\n')
            f.write(f'tree: {str(self.tree)}\n')
            f.write(f'\nscheme:\n'
                    f'  out: {str(list(self.scheme.out))}\n'
                    f'  nodes:\n')
            for e in self.scheme.elems:
                f.write(f'  - {str(list(e))}\n')

    @property
    def num_slices(self):
        return 2 ** len(self.sliced)


class TensorNetwork:
    """ Represents tensor networks of tensors with legs of bond dimension 2 obtained from quantum circuits. """
    def __init__(self):
        self.indices = set()
        self.tensors, self.out = [], []

    def add(self, t, idx):
        """ Adds new tensor t with legs enumerates by indices idx """
        ten = np.array(t, dtype=np.complex128)
        dim = int(math.log2(reduce(operator.mul, ten.shape, 1)))
        ten = ten.reshape((2,) * dim)
        if dim != len(idx):
            raise ValueError(f"Tensors dimension={dim} is not equal to the number of legs={len(idx)}")
        self.tensors.append((ten, idx))
        self.indices.update(idx)

    def get_scheme(self) -> TNScheme:
        """ Creates contraction scheme for this tensor network """
        return TNScheme([i for _, i in self.tensors], self.out)

    def save_as_yaml(self, filename):
        with open(filename, 'w') as f:
            f.write(f'out: {str(self.out)}\n')
            f.write('tensors:\n')
            for t, idx in self.tensors:
                f.write(' - [' + str([[x.real, x.imag] for x in t.flatten()]) + ', ' + str(idx) + ']\n')

    def simplify_fuse(self):
        """ Simplify tensor network by fusing tensors """
        vt = defaultdict(lambda: [])
        for i in self.out:
            vt[i].append(-1)
        t1 = copy(self.tensors)
        for n, (t, idx) in enumerate(t1):
            for i in idx:
                vt[i].append(n)

        changes = 1

        def idx_after_fusion(ix, iy):
            common = set(ix) & set(iy)
            return list(set(ix) - common) + list(set(iy) - common) + [j for j in common if len(vt[j]) > 2]

        def fuse(ii, jj):
            nonlocal changes
            x, ix = t1[ii]
            y, iy = t1[jj]
            oi = idx_after_fusion(ix, iy)
            if len(oi) > max(len(ix), len(iy)):
                return False
            r, out = contract2_numpy(x, ix, y, iy, oi)
            nn = len(t1)
            t1.append((r, out))
            for kk in ix:
                vt[kk].remove(ii)
            for kk in iy:
                vt[kk].remove(jj)
            for kk in out:
                vt[kk].append(nn)
            t1[ii] = t1[jj] = None
            changes += 1
            return True

        while changes:
            changes = 0
            for k, v in vt.items():
                for i in range(max(len(v) - 1, 0)):
                    if min(v[i], v[i + 1]) >= 0 and fuse(v[i], v[i + 1]):
                        break

        self.tensors = [x for x in t1 if x is not None]


class Circuit:
    def __init__(self, qubit_num):
        """ Creates quantum circuit with specified number of qubits """
        self._vn = self.qubit_num = qubit_num
        self._v = {i: i for i in range(qubit_num)}
        self._tn = TensorNetwork()
        for i in range(qubit_num):
            self._tn.add([1, 0], [i])

    def gate(self, matrix, qubits):
        """
        Add tesnsor to tensor network corresponding to specified gate in quantum circuit
        :param matrix: matrix of quantum gate
        :param qubits: qubits of quantum circuit
        """
        idx_before = [self._v[qi] for qi in qubits]
        for qi in qubits:
            self._v[qi] = self._vn
            self._vn += 1
        idx_after = [self._v[qi] for qi in qubits]
        self._tn.add(matrix, idx_after + idx_before)

    def get_tn(self, simplify=False):
        """ Current tensor network """
        self._tn.out = [self._v[qi] for qi in range(self.qubit_num)]
        if simplify:
            self._tn.simplify_fuse()
        return self._tn


def read_qsim(fn, simplify=False):
    """ Read quantum circuit from .qsim file, returns tensor network """
    t = read_table(fn)
    c = Circuit(qubit_num=t[0][0])
    for (lvl, gn, *q) in t[1:]:
        args = ()
        if '(' in gn:
            gn, args = gn.replace(')', '').split('(')
            args = [float(x) for x in args.split(',')]
            g = gates[gn]
        else:
            g = gates[gn]
            if type(g) is tuple:
                args, q = q[g[0]:], q[:g[0]]
        if type(g) is tuple:
            c.gate(g[1](*args), q)
        else:
            c.gate(g, q)

    return c.get_tn(simplify)


# Below there is an example implementation of a contraction tree optimizer
# described in https://arxiv.org/abs/2108.05665
class Node:
    """ Binary tree structure for tensor network contraction optimization """
    __slots__ = ['left', 'right', 'index', 'cost', 'memrw', 'memp', 'cache',  'var', 'mvar', 'p', 'num']

    def __init__(self, left=None, right=None, index=None, var=0, mvar=0, number=None):
        self.left = left
        self.right = right
        self.mvar = mvar   # log2(max number of variants for tensors in whole tree)
        self.cost = 0      # (cost of contraction for this node)
        self.memrw = 0     # (memory operations for this node)
        self.memp = 0      # (memory size)^p for this node
        self.cache = 0     # (cache size for this node)
        self.num = number  # number of the tensor in the tensor network
        self.p = 1         # l_p norm parameter for max approximation
        if left:
            self.update_ivar()
        else:
            self.index = intbitset(index or [])  # set of indices of all tensors in subtree
            self.var = var  # log2(number of variants for tensors in this subtree)

    def setparams(self, p, mvar):
        self.p = p
        self.mvar = mvar
        if self.left:
            self.left.setparams(p, mvar)
            self.right.setparams(p, mvar)

    def memory_var(self):
        return 2.0**(len(self.index) + self.var)

    def memory(self):
        return 2.0**(len(self.index))

    def memory_cache(self):
        if not self.left or not self.var:
            return 0
        else:
            log_memv = len(self.index) + self.var
            ml = self.left.memory() if self.right.var and len(self.left.index)+self.left.var < log_memv else 0
            mr = self.right.memory() if self.left.var and len(self.right.index)+self.right.var < log_memv else 0
            return min(self.left.memory_var()+mr, ml+self.right.memory_var())

    def update_ivar(self):
        self.var = min(self.mvar, self.left.var + self.right.var)
        self.index = (self.left.index | self.right.index) - (self.left.index & self.right.index)
        old_cost, self.cost = self.cost, 2**(len(self.index | self.left.index | self.right.index) + self.var)
        old_memrw, self.memrw = self.memrw, self.memory_var()
        old_memp, self.memp = self.memp, self.memory()**self.p
        old_cache, self.cache = self.cache, self.memory_cache()
        return self.cost - old_cost, self.memrw - old_memrw, self.memp - old_memp, self.cache - old_cache

    def rotate_right(self, n):  # n=0 => ((a,b),c) -> ((c,b),a); n=1 => ((a,b),c) -> ((a,c),b)
        if not self.left or not self.left.left:
            return None, 0, 0, 0
        if n:
            self.left.left, self.right = self.right, self.left.left
        else:
            self.left.right, self.right = self.right, self.left.right
        dlcost, dlmemrw, dlmemp, dlcache = self.left.update_ivar()
        dcost, dmemrw, dmemp, dcache = self.update_ivar()
        return dcost + dlcost, dmemrw + dlmemrw, dmemp + dlmemp, dcache + dlcache

    def do_rotation(self, n):
        if not self.left:
            return None, 0, 0, 0
        if n & 1:  # swap left and right subtrees
            self.left, self.right = self.right, self.left
        if not self.left.left:
            return None, 0, 0, 0
        return self.rotate_right(n & 2)

    def undo_rotation(self, n):
        self.do_rotation(n & 2)

    def calc_recursive(self):  # recursively calculates cost, memrw, memp, cache, max memory
        if not self.left:
            return self.cost, self.memrw, self.memp, self.cache, self.memory()
        lcost, lmemrw, lmemp, lcache, lmem = self.left.calc_recursive()
        rcost, rmemrw, rmemp, rcache, rmem = self.right.calc_recursive()
        self.update_ivar()
        return self.cost + lcost + rcost, self.memrw + lmemrw + rmemrw, self.memp + lmemp + rmemp, \
               self.cache + lcache + rcache, max(lmem, rmem, self.memory())

    def __iter__(self):
        yield self
        if self.left:
            yield from self.left
            yield from self.right

    def leaves(self):
        return (x for x in self if not x.left)

    def set_index(self, leaves_indices, sliced):
        for leaf in self.leaves():
            leaf.index = leaves_indices[leaf.num] - sliced
        return self.calc_recursive()

    def leaves_indices(self):
        return [leaf.index for leaf in sorted(self.leaves(), key=lambda x: x.num)]

    def copy(self):
        res = copy(self)
        res.index = copy(self.index)
        if self.left:
            res.left = self.left.copy()
            res.right = self.right.copy()
        return res

    def as_folded_lists(self):
        if not self.left:
            return self.num
        return [self.left.as_folded_lists(), self.right.as_folded_lists()]


def calc_slicing_gain(root: Node, slicing_gain):
    mem, cache = root.memory(), root.memory_cache()
    for i in root.index:
        d = mem
        if root.left and root.var:
            dr, dl = 1 if i in root.right.index else 0, 1 if i in root.left.index else 0
            rdvsz, ldvsz, tdvsz = root.right.var - dr, root.left.var - dl, root.var - 1
            tlog_memv = len(root.index)+tdvsz
            ml = 2**(len(root.left.index)+ldvsz) if root.right.var and len(root.left.index)+ldvsz < tlog_memv else 0
            mr = 2**(len(root.right.index)+rdvsz) if root.left.var and len(root.right.index)+rdvsz < tlog_memv else 0
            mk = min(ml + 2**(len(root.right.index)+rdvsz), mr + 2**(len(root.left.index)+ldvsz))
            d += 0.5*(cache - mk)
        slicing_gain[i] += d

    if root.left:
        calc_slicing_gain(root.left, slicing_gain)
        calc_slicing_gain(root.right, slicing_gain)


class StateParams:
    """ Optimizer state parameters """
    __slots__ = ['cost', 'memrw', 'memp', 'cache', 'mem', 'root',
                 'min_cost', 'min_memrw', 'min_memp', 'min_cache',
                 'max_cost', 'max_memrw', 'max_memp', 'max_cache',
                 'p', 'max_log_memory', 'arith_int', 'mcoef']

    def __init__(self, root, p, max_log_memory, arith_int, mcoef):
        self.root = root  # root node of the tree
        self.p = p  # l_p norm parameter for max approximation
        self.max_log_memory = max_log_memory  # log2(max available memory)
        self.arith_int = arith_int  # lower limit for arithmetic intensity of the contraction
        self.mcoef = mcoef  # coefficient for memory cost in the cost function
        self.update()

    def update(self):
        """ Recalculate cost, memrw, memp, cache, mem for the whole tree """
        self.cost, self.memrw, self.memp, self.cache, self.mem = self.root.calc_recursive()
        # calculate min and max values for cost, memrw, memp, cache to check numerical stability
        self.min_cost = self.max_cost = self.cost
        self.min_memrw = self.max_memrw = self.memrw
        self.min_memp = self.max_memp = self.memp
        self.min_cache = self.max_cache = self.cache

    def add(self, dcost, dmemrw, dmemp, dcache):
        """ Add cost, memrw, memp, cache when tree changes
         recalculate all if accumulated numerical errors can be > 1e-7
        """
        self.cost += dcost
        self.memrw += dmemrw
        self.memp += dmemp
        self.cache += dcache
        # check numerical stability
        self.min_cost = min(self.min_cost, self.cost)
        self.min_memrw = min(self.min_memrw, self.memrw)
        self.min_memp = min(self.min_memp, self.memp)
        self.min_cache = max(1, min(self.min_cache, self.cache))
        self.max_cost = max(self.max_cost, self.cost)
        self.max_memrw = max(self.max_memrw, self.memrw)
        self.max_memp = max(self.max_memp, self.memp)
        self.max_cache = max(self.max_cache, self.cache, 1)
        if self.max_cost/self.min_cost >= 1e8 or self.max_memrw/self.min_memrw >= 1e8 or \
           self.max_memp/self.min_memp >= 1e8 or self.max_cache/self.min_cache >= 1e8:
            self.update()

    def cost_function(self, dcost, dmemrw, dmemp, dcache, del_count):
        """ Cost function for the tree """
        c, rw, mp, cache = self.cost + dcost, self.memrw + dmemrw, self.memp + dmemp, self.cache + dcache
        mp **= (1/self.p)
        return math.log2(c + rw*self.arith_int) + del_count +\
               self.mcoef*max(math.log2(2*mp+cache) - self.max_log_memory, 0)

    def get_metrics(self, sliced):
        sl_count = len(sliced)
        return self.cost*2**sl_count, self.memrw*2**sl_count, self.memp, self.cache, self.mem, copy(sliced)


# Simulated annealing for tensor network contraction optimization
def optimize_contraction(root: Node, mvar=-1, p=2, max_log_memory=30, timeout=60, arith_int=16,
                         mcoef=1.0, t0=1.0, t1=1e-4, slice_freq=1000, slicing_strategy='greedy',
                         init_slicing=None, msg_level=1):
    arith_int *= 2  # 1 memory op -> 8 bytes write and 8 bytes read (16 RW operations), 1 complex number multiplication -> 8 FLOPS
    nodes = list(node for node in root if node.left)
    leaves = list(root.leaves())
    n = len(nodes)
    if mvar < 0:
        mvar = sum(node.var for node in root.leaves())
    root.setparams(p, mvar)
    leaves_indices = [leaf.index for leaf in leaves]
    all_indices = reduce(lambda x, y: x | y, leaves_indices, intbitset())

    params = StateParams(root, p, max_log_memory, arith_int, mcoef)

    def set_sliced(d):
        for leaf, ind in zip(leaves, leaves_indices):
            leaf.index = ind - d
        params.update()

    if init_slicing:
        sliced = intbitset(init_slicing)
        set_sliced(sliced)
    else:
        sliced = intbitset()

    T = t0  # current temperature
    tm = last_msg = time.time()
    nsteps = 0

    best_value = curr_value = params.cost_function(0, 0, 0, 0, len(sliced))
    best_cost, best_memrw, best_memp, best_cache, best_mem, best_sliced = params.get_metrics(sliced)
    best_tree = root.copy()
    params.update()

    def curr_str():
        return (f'cost = {math.log2(best_cost):.4f}, arith = {best_cost/best_memrw/2:.1f}, '
                f'mE = {math.log2(2*best_memp**(1/p)+best_cache):.2f}, '
                f'mem = {math.log2(best_mem):.1f}, sliced = {len(best_sliced)}, '
                f'E = {best_value:.4f}, T = {T:.6f}')
    if msg_level:
        print(f'Initial: {curr_str()}')

    updated = False
    while T > t1:
        nsteps += 1
        if nsteps % 100 == 0:
            T = t0 * (t1 / t0) ** ((time.time() - tm) / timeout)
            if msg_level > 1 and updated and time.time() - last_msg > 1:
                print(curr_str())
                last_msg = time.time()
                updated = False
        if msg_level > 2 and nsteps % 1000000 == 0:
            print(f'nsteps = {nsteps}, del = {len(sliced)}, E = {curr_value:.4f}, T = {T}')

        if nsteps % slice_freq == 0:  # try to delete or restore a node
            if len(sliced) and random.randint(0, 1):
                # restore a node
                i = random.choice(list(sliced))
                new_del = sliced - intbitset([i])
            else:
                # delete a node
                if slicing_strategy == 'greedy':
                    slicing_gain = defaultdict(lambda: 0)
                    calc_slicing_gain(root, slicing_gain)
                    i = max(slicing_gain, key=slicing_gain.get)
                    new_del = sliced | intbitset([i])
                elif slicing_strategy == 'random':
                    i = random.choice(list(all_indices - sliced))
                    new_del = sliced | intbitset([i])
                else:
                    raise ValueError('Unknown slicing strategy')

            set_sliced(new_del)
            new_value = params.cost_function(0, 0, 0, 0, len(new_del))
            if new_value < curr_value or random.random() < math.exp((curr_value-new_value)/T):
                # accept the change
                curr_value = new_value
                sliced = new_del
                if new_value < best_value:
                    best_value = new_value
                    best_cost, best_memrw, best_memp, best_cache, best_mem, best_sliced = params.get_metrics(sliced)
                    best_tree = root.copy()
                    updated = True
            else:
                set_sliced(sliced)  # reject the change (restore the previous state)
            continue

        i = random.randint(0, n-1)
        ni = nodes[i]
        # try to apply random rotation to this node
        nrot = random.randint(0, 3)
        dcost, dmemrw, dmemp, dcache = ni.do_rotation(nrot)
        if dcost is None:  # if rotation is not possible
            continue
        new_value = params.cost_function(dcost, dmemrw, dmemp, dcache, len(sliced))
        if new_value < curr_value or random.random() < math.exp((curr_value-new_value)/T):
            # accept the change
            curr_value = new_value
            params.add(dcost, dmemrw, dmemp, dcache)
            if curr_value < best_value:
                best_value = curr_value
                params.update()
                best_cost, best_memrw, best_memp, best_cache, best_mem, best_sliced = params.get_metrics(sliced)
                best_tree = root.copy()
                updated = True
        else:
            ni.undo_rotation(nrot)  # reject the change (restore the previous state)
    if msg_level:
        print(f'Final: {curr_str()}')

    best_params = StateParams(best_tree, p, max_log_memory, arith_int, mcoef)
    bcost, bmemrw, bmemp, bcache, bmem, best_sliced = best_params.get_metrics(best_sliced)

    return best_tree, best_cost, best_memrw, best_memp, best_cache, bmem, best_sliced, best_value


def optimize_tn_scheme(scheme: TNScheme, num_amps, initial='simple',
                       p=2, max_log_memory=30, timeout=60, arith_int=None,
                       mcoef=1.0, t0=1.0, t1=1e-4, slice_freq=1000, msg_level=1):
    """ Optimize tensor network contraction for multi-amplitude calculation """
    init_slicing = None
    if type(initial) is str:
        root = build_tree(scheme.elems, scheme.out, initial)
    elif type(initial) is list:
        root = tree_from_folded_lists(scheme.elems, scheme.out, initial)
    elif isinstance(initial, ContractionTree):
        root = tree_from_folded_lists(scheme.elems, scheme.out, initial.tree)
        init_slicing = initial.sliced
    else:
        raise ValueError(f'Unknown initial tree type: {type(initial)}')

    mvar = int(math.ceil(math.log2(num_amps)))

    tree, cost, memrw, memp, cache, mem, sliced, e = optimize_contraction(root, p=p, max_log_memory=max_log_memory,
                                                                          timeout=timeout, arith_int=arith_int,
                                                                          mcoef=mcoef, t0=t0, t1=t1,
                                                                          slice_freq=slice_freq, mvar=mvar,
                                                                          init_slicing=init_slicing,
                                                                          msg_level=msg_level)

    ct = ContractionTree(tree=tree.as_folded_lists(), sliced=sliced, scheme=scheme)
    return ct, cost, memrw, memp, cache, mem, e


def optimize_parallel(scheme: TNScheme, num_amps: int, nthreads=..., initial=('simple',),
                      p=2, max_log_memory=30, timeout=60.0, arith_int=None,
                      mcoef=1.0, t0=1.0, t1=1e-4, slice_freq=1000, msg_level=1):
    if arith_int is None:
        profile = PerformanceProfile()
        arith_int = 4 * profile.flop_per_second / profile.memory_bandwidth

    kwargs = {'p': p, 'max_log_memory': max_log_memory, 'timeout': timeout, 'arith_int': arith_int,
              'mcoef': mcoef, 't0': t0, 't1': t1, 'slice_freq': slice_freq}
    if initial is None:
        initial = ('simple',)
    elif type(initial) is str:
        initial = (initial,)

    if nthreads is ...:
        nthreads = multiprocessing.cpu_count()-1
    if nthreads == 1:
        return optimize_tn_scheme(scheme, num_amps, **kwargs, initial=initial[0], msg_level=msg_level)

    pool = multiprocessing.Pool(nthreads)
    results = [pool.apply_async(optimize_tn_scheme, (scheme, num_amps),
                                {**kwargs, 'initial': initial[i % len(initial)], 'msg_level': max(msg_level-1, 0)})
               for i in range(nthreads)]
    pool.close()
    pool.join()
    results = [res.get() for res in results]
    return min(results, key=lambda x: (max(math.log2(max(x[-2], x[-3])) - max_log_memory, 0), x[-1]))


def optimize_tn_multiamp(scheme: TNScheme, num_amps, initial=('simple', 'min_fill_in'), nthreads=1,
                         p=2, max_log_memory=30, timeout=60.0, arith_int=None,
                         mcoef=1.0, t0=1.0, t1=1e-4, slice_freq=1000, msg_level=1):
    """ Optimize tensor network contraction for multi-amplitude calculation

    :param scheme: scheme of the tensor network
    :param num_amps: number of amplitudes to calculate
    :param initial: method or list of methods to generate initial contraction tree (can be str or ContractionTree)
    :param nthreads: number of processes to use for optimization
    :param p: parameter p for the cost function (max is approximated by l_p-norm)
    :param max_log_memory: maximum log2 of the memory required for the contraction
           (procedure doesn't guarantee that the actual memory will be less than 2**max_log_memory)
    :param timeout: timeout for the optimization
    :param arith_int: arithmetic intensity (flops per byte)
    :param mcoef: coefficient for the memory limit violation penalty
    :param t0: initial temperature for simulated annealing
    :param t1: final temperature for simulated annealing
    :param slice_freq: frequency of slicing updates among tree rotations (slicing update is expensive)
    :param msg_level: verbosity of messages printed during optimization
    :return: optimized contraction tree, cost, memory read/write, memory peak estimate, cache, max tensor dimension
    """
    if timeout is None:
        timeout = 2.0**15  # ~ 9 hours

    curr_timeout = (timeout-1e-7)/2**math.ceil(math.log2(timeout))
    ct = None
    while True:  # Adaptively increase timeout until estimated contraction time is less than timeout
        if msg_level:
            print(f'Optimizing with timeout={curr_timeout:.0f}')
        init_ext = initial if isinstance(initial, tuple) else (initial,)
        if ct is not None:
            init_ext += (ct,)  # Use previous contraction tree as one of initial tree variants
        ct, cost, memrw, memp, cache, mem, e = optimize_parallel(scheme, num_amps, nthreads=nthreads,
                                                                 initial=init_ext,
                                                                 p=p, max_log_memory=max_log_memory,
                                                                 timeout=curr_timeout, arith_int=arith_int,
                                                                 mcoef=mcoef, t0=t0, t1=t1,
                                                                 slice_freq=slice_freq, msg_level=max(msg_level-1, 0))
        est = estimate_time(cost, memrw)  # usually time is underestimated 2-3 times

        if msg_level:
            print(f"Best found: compl: 2^{math.log2(cost):.2f},"
                  f" memrw: 2^{math.log2(memrw):.2f},"
                  f" memory: 2^{math.log2(mem):.0f},"
                  f" cache: 2^{math.log2(cache):.2f},"
                  f" sliced = {len(ct.sliced)}, e: {e:.2f}")
            print(f"Estimated time: {est:.2f} s\n")

        curr_timeout *= 2
        if est < curr_timeout or curr_timeout > timeout:
            break

    if msg_level:
        print(f" ============== Optimization finished ===============")

    return ct, cost, memrw, memp, cache, mem


def tree_from_folded_lists(tn, out, folded_lists):
    """ Build tree from folded lists """
    out = set(out)
    tn_no_out = [[x for x in t if x not in out] for t in tn]
    leaves = [Node(number=i, index=tn_no_out[i], var=len(out & set(tn[i]))) for i in range(len(tn))]

    def flatten(ls):  # flatten folded lists
        if type(ls) is int:
            return [ls]
        return sum(map(flatten, ls), [])

    # check if folded lists are valid
    tree_leaves = flatten(folded_lists)
    if len(tree_leaves) != len(tn):
        raise ValueError(f'Invalid initial tree size = {len(tree_leaves)}, expected = {len(tn)}')
    if sorted(tree_leaves) != list(range(len(tn))):
        raise ValueError(f'Invalid initial tree leaves, should be 0, 1, 2, ..., {len(tn)-1}')

    def f2tree(f):  # build tree from folded lists
        if not hasattr(f, '__iter__'):
            return leaves[f]
        else:
            assert len(f) == 2
            return Node(left=f2tree(f[0]), right=f2tree(f[1]))

    return f2tree(folded_lists)


def min_fill_in_ordering(v: List[List[int]], out):
    """ Build initial contraction tree using min-fill-in heuristic """
    out = intbitset(out)
    nv = max(x+1 for vv in v for x in vv)
    n = len(v)
    nodes = [[intbitset(), -1, -1] for _ in range(2*n-1)]
    vs = [set() for _ in range(nv)]  # Maps vertex to the tree nodes, containing this vertex

    nodes[0][0] = intbitset(v[0])
    for i in range(1, n):
        nodes[i][0] = intbitset(v[i])
        nodes[i + n - 1][1] = i
        nodes[i + n - 1][2] = 0 if i == 1 else i + n - 2

    for i in range(n):
        for j in v[i]:
            vs[j].add(i)  # Initialize initial tensor sets, which contain index "j"

    t, ta = intbitset(), intbitset()
    vert = set()  # The set of remaining vertices
    for i in range(nv):
        if vs[i]:
            vert.add(i)
    vn = n  # The first free tree vertex position
    bv = intbitset(vert)  # The remaining vertices bitset
    tens = set(range(n))
    while vert:
        emin = n * n
        jmin = kmin = 0
        for i in vert:
            if not vs[i]:
                continue

            t.clear()
            for j in vs[i]:
                ej = len(nodes[j][0] & bv)   # The number of indices in "j"-th tensor
                for k in vs[i]:  # Iterate through all (j,k) tensor pairs, which contain index "i"
                    if j == k:
                        break
                    ek = len(nodes[k][0] & bv)
                    t = nodes[j][0] & nodes[k][0] & bv
                    ekj = len(t)
                    e = len(bv & (nodes[k][0] | nodes[j][0]))
                    elim = 0
                    for x in t:
                        if x in out:
                            elim += 0.1
                        else:
                            elim += 1 / (len(vs[x]) - 1)
                    ee = (e * (e - 1) + ekj * (ekj - 1) - ek * (ek - 1) - ej * (ej - 1))/elim  # Number of added edges
                    if ee < emin:
                        jmin, kmin, emin = j, k, ee

        if jmin == kmin:
            break

        # Perform contraction of jmin and kmin tensors by all common indices
        t = nodes[jmin][0] & nodes[kmin][0] & bv  # The set of common indices
        tens.remove(jmin)
        tens.remove(kmin)
        tens.add(vn)
        nodes[vn][0] = (nodes[jmin][0] | nodes[kmin][0]) & bv
        for vj in nodes[jmin][0]:
            vs[vj].remove(jmin)
        for vk in nodes[kmin][0]:
            vs[vk].remove(kmin)
        for x in nodes[vn][0]:
            if not vs[x]:
                vert.remove(x)  # Remove vertex from set, if nothing more to contract with it
            if x in out or vs[x]:  # In case a vertex remains, tensors shall be multiplied by this index
                vs[x].add(vn)      # In case a vertex remains, it must present in the new tree node
                continue

            nodes[vn][0].remove(x)
            bv.remove(x)

        nodes[vn][1] = jmin
        nodes[vn][2] = kmin     # The contraction result vertex
        vn += 1

    if vn < len(nodes):  # If was disconnected
        while len(tens) >= 2:
            nodes[vn][1] = tens.pop()
            nodes[vn][2] = tens.pop()
            tens.add(vn)
            vn += 1

    def to_list(vi):
        return vi if vi < n else [to_list(nodes[vi][1]), to_list(nodes[vi][2])]

    return to_list(vn-1)


def build_tree(tn, out, method='simple'):
    """ Build simple tree for multi-amplitude calculation.

    :param tn: tensor network scheme --- list of tensor indices
    :param out: output (open) indices
    :param method: method to build tree, one of: simple, min_fill_in, random
    :return: tree
    """
    n = len(tn)
    out = set(out)
    tn_no_out = [[x for x in t if x not in out] for t in tn]
    leaves = [Node(number=i, index=tn_no_out[i], var=len(out & set(tn[i]))) for i in range(n)]
    if method == 'simple':
        root = leaves[0]
        for i in range(1, n):
            root = Node(left=root, right=leaves[i])
    elif method == 'random':
        available = set(range(n))
        while len(available) > 1:
            i, j = random.sample(available, 2)
            leaves[i] = Node(left=leaves[i], right=leaves[j])
            available.remove(j)
        root = leaves[available.pop()]
    elif method == 'min_fill_in':
        tree = min_fill_in_ordering(tn_no_out, [])
        root = tree_from_folded_lists(tn, out, tree)
    else:
        raise ValueError(f'Unknown initial tree method: {method}')
    return root


def find_out_indices(tn):
    """ Find indices that occur only once in tensor network """
    all_indices = reduce(lambda x, y: x | y, map(set, tn), set())
    counts = defaultdict(lambda: 0)
    for t in tn:
        for i in t:
            counts[i] += 1
    return sorted(i for i in all_indices if counts[i] == 1)
