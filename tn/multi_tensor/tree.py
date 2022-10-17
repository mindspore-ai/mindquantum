import itertools
from typing import Optional, Union, List, NamedTuple
import numpy as np


class NodeChunk(NamedTuple):
    """ NodeChunk describes one batch tensor contraction in a contraction tree node. """
    beg: int
    end: int
    lidx: Union[List, np.ndarray]
    ridx: Union[List, np.ndarray]
    pos: int = 0

    def __len__(self): 
        """ Returns the number of batch elements in the chunk. """
        return len(self.ridx)


class NodeInfo:
    def __init__(self):
        self.chunks: List[NodeChunk] = []
        self.fv = self.idxa = self.idxb = self.idxr = None

    def flops(self): return 0 if self.idxa is None else 2 ** len(set(self.idxa + self.idxb)) * sum(
        len(c.ridx) for c in self.chunks)

    def num_amps(self): return self.chunks[-1].end


class MTNode:  
    """ 
    Tree node containing auxilliary information for
    building low-level multiamplitude contraction schedule
    """
    def __init__(self, idx, left=None, right=None, tnum=None):
        self.idx = idx  # Indices of contraction result of subtree with root = self
        self.left: Optional[MTNode] = left    # Left subtree
        self.right: Optional[MTNode] = right  # Right subtree
        self.a, self.b = set(), set()  # Here will be all indices in subtree and out of subtree, respectively
        if left is not None:
            left.p = right.p = self
        self.tnum = tnum                 # Number of tensor of given node if it is leaf (otherwise None)
        self.p: Optional[MTNode] = None  # Parent tree node (None for root of whole contraction tree)
        
        self.var = 0    # after update_vars() here will be number of output (open) indices in this subtree
        self.fvar = []  # after update_vars() here will be list of open indices in this subtree
        
        self.btc = 1    # calculated in calc_mem(); batch size after contraction of this subtree
        self.lcache = False  # calculated in calc_mem(); whether to store in cache contraction result for left subtree
        
        # self.info -- description of contractions performed in this node during multiamplitude simulation
        self.info: Optional[NodeInfo] = None  # It is calculated in prepare() -- last stage of simulation preparation

    @property
    def is_leaf(self):
        return self.left is None

    @property
    def mem_log(self):
        return len(self.idx)  # log2 of size of resulting tensor in this node

    @property
    def mem(self):
        return 2 ** len(self.idx)  # size of resulting tensor in this node
    
    @property
    def flops(self):
        """ Returns log2 of number of flops performed in this node. """
        return 0 if self.is_leaf else len(set(self.idx) | set(self.left.idx) | set(self.right.idx)) + self.var 

    def total_flops(self):
        """ Calculate subtree contraction complexity """
        if self.is_leaf:
            return 0
        return 8 * 2 ** self.flops + self.left.total_flops() + self.right.total_flops() 

    @property
    def memv_log(self):
        return len(self.idx) + self.var  # log2(total size of all tensors in this vertex)

    @property
    def memv(self):
        return 2 ** (len(self.idx) + self.var)  # total size of all tensors in this vertex

    def leaves(self):
        return (node for node in self.nodes() if node.is_leaf)

    def nodes(self, leaves=True):  # Generator enumerating all tree nodes
        if leaves or not self.is_leaf:
            yield self
        if not self.is_leaf:
            yield from self.left.nodes(leaves)
            yield from self.right.nodes(leaves)

    def update_vars(self, free, mvar):  # Recursively update var and fvar fields
        if self.is_leaf:
            self.fvar = sorted({x for x in self.idx if x in free})
        else:
            self.left.update_vars(free, mvar)
            self.right.update_vars(free, mvar)
            self.fvar = self.left.fvar + self.right.fvar
        self.var = min(mvar, len(self.fvar))

    def swap_lr(self):  # Correctly swaps left and right subtrees
        self.left, self.right = self.right, self.left
        c = self
        while c is not None:
            c.fvar = c.left.fvar + c.right.fvar
            c = c.p

    def _update_a(self):
        self.a = set(self.idx) if self.is_leaf else self.left._update_a().a | self.right._update_a().a
        return self

    def _update_b(self, b):
        self.b = b
        if not self.is_leaf:
            self.idx = sorted(self.b & self.a)
            self.left._update_b(self.b | self.right.a)
            self.right._update_b(self.b | self.left.a)

    def update_all(self, mvar, v):  
        """
        Update internal variables in this subtree.
        
        :param mvar: ceil(log2(maximum number of amplitudes))
        :param v: list of open indices which can take different values for different amplitudes
        """
        self._update_a()._update_b(set())
        self.update_vars(v, mvar)

    def calc_mem(self, maxbtc=27):  # curr mem, max mem, batch size
        """ Calculates optimal memory usage during this subtree multiamplitude contraction.
        
        :param maxbtc: log2(batch size limit), used to determine whether to split large batch into smaller batches
        :return: memory for current contraction, mamimal memory for subtree contraction, result batch size 
        """
        def calc_params(left, right, l_mem, l_max, btc_l, r_maxw):
            btc = fullbtc = btc_l * 2 ** (self.var - left.var)  # maximal batch size without split into smaller batches
            if self.memv_log > left.memv_log:
                btc = min(btc, 2 ** max(0, maxbtc - self.mem_log))
            usebtc = (btc == fullbtc)  # need not store left result in cache
            mem0curr = right.memv + l_mem + self.mem * btc      # memory during current contraction
            mem0mx = max(mem0curr, r_maxw, right.memv + l_max)  # max mem_log when calculating this subtree
            mem = self.mem * btc                                # memory for current result
            if btc != 2 ** self.var:         # in this case we store left cache
                mem += l_mem + right.memv    # left cache + all right + current result
                if usebtc:
                    mem -= btc_l * left.mem  # don't store left result
            return mem, mem0mx, btc, usebtc

        if self.is_leaf:
            self.btc = 2 ** self.var
            return self.mem, self.mem, self.btc
        lmem, lmax, btcl = self.left.calc_mem(maxbtc)
        rmem, rmax, btcr = self.right.calc_mem(maxbtc)
        rmaxw = rmax + self.right.memv  # max memory for all right amps
        lmaxw = lmax + self.left.memv   # max memory for all left amps
        mem1a, mem1a0mx, btc1a, usebtc1a = calc_params(self.left, self.right, lmem, lmax, btcl, rmaxw)
        mem2a, mem2a0mx, btc2a, usebtc2a = calc_params(self.right, self.left, rmem, rmax, btcr, lmaxw)
        if mem2a0mx < mem1a0mx:
            self.swap_lr()  # swap subtrees if it reduces memory usage
            mem1a, mem1a0mx, btc1a, usebtc1a = mem2a, mem2a0mx, btc2a, usebtc2a
        self.btc = btc1a
        self.lcache = not usebtc1a
        return mem1a, mem1a0mx, btc1a

    def _prepare_rec(self, bitstrs0, fvmap):
        bitstrs, res_ind = uniq(bitstrs0[:, [fvmap[x] for x in self.fvar]], return_inverse=True)
        fvmap1 = {x: i for i, x in enumerate(self.fvar)}
        idx, bsz, ridx, save, c = [], [], [], [True], self
        while c is not None:
            idx.append(len(c.fvar))
            bsz.append(c.btc)
            if c.is_leaf:
                ridx.append([])
                break
            save.append(c.lcache)
            ridx.append([fvmap1[x] for x in c.right.fvar])
            c.right._prepare_rec(bitstrs if len(c.right.fvar) else np.zeros((1, 0)), fvmap1)
            c = c.left
        for arr in [idx, save, bsz, ridx]:
            arr.reverse()
        save[0] = True
        chunks = split_strs(bitstrs, idx, bsz, ridx, save)
        for ch in chunks:
            c.info = NodeInfo()
            c.info.chunks = ch
            if not c.is_leaf:
                c.info.idxa = [t for t in c.left.idx if t not in c.fvar]
                c.info.idxb = [t for t in c.right.idx if t not in c.fvar]
            c.info.idxr = [t for t in c.idx if t not in c.fvar]
            c.info.fv = c.fvar
            c = c.p
        return res_ind

    def prepare(self, bitstrs, fvnum):
        """ 
        Prepare contraction "program" for this contraction tree.
        
        :param bitstrs: list of bitstrings for which amplitudes will be calculated
        :param fvnum: mapping of qubits into open indices
        :return: mapping between external and internal bitstring order 
        """
        fvmap = {x: i for (i, x) in enumerate(fvnum)}
        return self._prepare_rec(np.array(bitstrs), fvmap)


def uniq(data: np.ndarray, **kwargs):  # Wrapper of numpy.unique() that can handle some degenerate cases
    if len(data) == 0:
        return (np.array(data), *([[]] * len(kwargs))) if kwargs else np.array(data)
    if data.shape[1] == 0:
        r = list(itertools.compress([np.array(data[0:1]), [0], [0] * len(data)],
                                    [1, kwargs.get('return_index', 0), kwargs.get('return_inverse', 0)]))
        return r[0] if len(r) == 1 else tuple(r)
    return np.unique(data, axis=0, **kwargs)


def split_strs(bitstrs: np.ndarray, idx, bsz, ridx, save):
    n, m, bc = len(idx), len(bitstrs), bitstrs
    chunks, invmaps, rmaps = [[] for _ in range(n)], [np.array(0)] * n, [np.array(0)] * n
    for i in range(n - 1, 0, -1):
        _, rmaps[i] = uniq(bc[:, ridx[i]], return_inverse=True)
        bc, invmaps[i] = uniq(bc[:, :idx[i - 1]], return_inverse=True)
    _, rmaps[0] = uniq(bc[:, ridx[0]], return_inverse=True)
    invmaps[0] = np.zeros(len(bc), dtype=np.int64)
    
    pos, iprev = [0] * n, 0
    prev = chunks[0] = [NodeChunk(0, m, lidx=bc, ridx=rmaps[0], pos=0)]
    for i in (x for x in range(1, n) if save[x]):
        tp, bp = iprev + 1, bsz[iprev + 1]
        for c in prev:
            up, iip = uniq(bitstrs[c.beg:c.end, :idx[tp]], return_index=True)
            invp = invmaps[tp][pos[tp]:pos[tp] + len(up)]
            arr = [x + c.beg for x in iip[::bp]] + [c.end]
            for sp in range(len(arr) - 1):
                lind = np.array(invp[bp * sp:bp * (sp + 1)]) - invp[0]
                rind = list(rmaps[tp][pos[tp] + bp * sp:pos[tp] + min(len(up), bp * (sp + 1))])
                chunks[tp].append(NodeChunk(arr[sp], arr[sp + 1], lidx=lind, ridx=rind, pos=pos[tp] + bp * sp))
            pos[tp] += len(up)
        prev = chunks[tp]
        if tp < i:
            for c in prev:
                u = uniq(bitstrs[c.beg:c.end, :idx[i]])
                for t in range(i, iprev + 1, -1):
                    pr = idx[t - 1] if t else 0
                    lind = np.array(invmaps[t][pos[t]:pos[t] + len(u)]) - invmaps[t][pos[t]] if t else u
                    rind = list(rmaps[t][pos[t]:pos[t] + len(u)])
                    chunks[t].append(NodeChunk(c.beg, c.end, lidx=lind, ridx=rind, pos=pos[t]))
                    pos[t] += len(u)
                    u = uniq(u[:, :pr])
        iprev, prev = i, chunks[i]
    return chunks
