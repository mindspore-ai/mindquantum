import copy
import itertools
import math
import time
from typing import Dict
import numpy as np
import torch
from tree import MTNode, NodeInfo, NodeChunk
from utils import transpose_numpy, _ein_ind, read_amps
from tensor_network import ContractionTree, read_qsim, optimize_tn_multiamp


def ctree2mtree(sch, elems):
    if type(sch) is not list:
        return None if sch >= len(elems) else MTNode(elems[sch], tnum=sch)
    l, r = ctree2mtree(sch[0], elems), ctree2mtree(sch[1], elems)
    return r if l is None else l if r is None else MTNode([], l, r)


class MTNSchedule:
    """ Multiamplitude contraction schedule """
    def __init__(self, obj: ContractionTree, mvar=0):
        self.mvar = mvar
        self.bitstrs = []
        self.res_ind = None
        self.sch = ctree2mtree(obj.tree, obj.scheme.elems)
        self.fv = obj.scheme.out
        self.sliced = obj.sliced
        self.tensors = [[None, e] for e in obj.scheme.elems]
        d, fvset = set(self.sliced), set(self.fv)
        for leaf in self.sch.leaves():
            leaf.idx = sorted(set(obj.scheme.elems[leaf.tnum]) - d)
            leaf.fvar = sorted(set(leaf.idx) & fvset)

    def prepare(self, bitstrs, maxbtc=27):
        self.bitstrs = bitstrs
        mvar = int(math.ceil(math.log2(len(self.bitstrs))))
        self.sch.update_all(mvar, self.fv)
        maxmem = self.sch.calc_mem(maxbtc)[1]
        self.res_ind = self.sch.prepare(bitstrs, self.fv)
        return maxmem


def fixvars(ten, idx, vvals):
    """ Fix some legs in tensor """
    ii = [vvals.get(i, slice(None)) for i in idx]
    return ten[tuple(ii)], [x for x in idx if x not in vvals]


class TNBackend:
    class TWrap:  # auxilliary class, allows to delete tensor T from any reference to TWrap containing T
        def __init__(self, t):
            self.ten = t

    def __init__(self):
        self._sufficient_flops = None
        self._multiamp_coef = 0.5

    def contract2(self, x, xi, y, yi, out, accum=None):
        en = {s: i for (i, s) in enumerate(set(list(xi) + list(yi) + list(out)))}
        xind = ''.join([_ein_ind[en[i]] for i in xi])
        yind = ''.join([_ein_ind[en[i]] for i in yi])
        oind = ''.join([_ein_ind[en[i]] for i in out])
        res = torch.einsum(f'{xind},{yind} -> {oind}', x, y)
        if accum is not None:
            accum += res
            return accum, out
        return res, out

    def _contract_node(self, a, idxa, b, idxb, idres, chunk: NodeChunk, accum=None):
        assert sorted(chunk.lidx) == list(chunk.lidx)
        sizes = np.unique(chunk.lidx, return_counts=True)[1]
        mx = sizes.max(initial=0)

        if len(chunk) == 1:
            if accum is not None:
                accum = accum.reshape(accum.shape[1:])
            res, out = self.contract2(a[chunk.lidx[0]], idxa, b[chunk.ridx[0]], idxb, idres, accum)
            res = res.reshape((1,) + res.shape)
        elif len(sizes) == 1:
            res, out = self.contract2(a[chunk.lidx[0]], idxa, b[chunk.ridx], [-1] + idxb, [-1] + idres, accum)
        elif len(np.unique(sizes)) > 1 and sum(sizes) < mx * len(sizes) * self._multiamp_coef \
                and (mx * len(sizes) - sum(sizes)) * 2 ** len(set(idxa + idxb)) > self._sufficient_flops * len(chunk):
            res = torch.zeros((len(chunk),) + (2,) * len(idres), dtype=torch.complex64)
            addr = [0] + list(sizes.cumsum())
            for i, sz in enumerate(sizes):
                res[addr[i]:addr[i + 1]], out = self.contract2(a[chunk.lidx[addr[i]]], idxa,
                                                               b[chunk.ridx[addr[i]: addr[i + 1]]], [-1] + idxb,
                                                               [-1] + idres, accum)
        else:
            if len(np.unique(sizes)) > 1:
                addr = [0] + list(sizes.cumsum())
                ridx = np.zeros(mx * len(sizes), dtype=np.int64)
                inv = []
                for i, sz in enumerate(sizes):
                    ridx[i * mx:i * mx + sz] = chunk.ridx[addr[i]: addr[i + 1]]
                    inv += list(range(i * mx, i * mx + sz))
            else:
                ridx = chunk.ridx
                inv = None
            bx = b[ridx].reshape((len(sizes), mx) + tuple(b.shape[1:]))
            res, out = self.contract2(a[chunk.lidx[0]:chunk.lidx[-1] + 1], [-1] + idxa,
                                      bx, [-1, -2] + idxb, [-1, -2] + idres, accum)
            res = res.reshape((res.shape[0] * res.shape[1],) + res.shape[2:])
            if inv is not None:
                res = res[inv]

        return res, chunk.beg, chunk.end

    def _contract_node_gen(self, gen_l, gen_r, info: NodeInfo, accum=None):
        r = torch.stack(sum((list(x[0].ten) for x in gen_r), []), 0)
        end, left = 0, None

        for c in info.chunks:
            if c.end > end:
                left, _, end = next(gen_l)
            t, b, e = self._contract_node(left.ten, info.idxa, r, info.idxb, info.idxr, c, accum)

            if c.end == end:
                del left.ten
            if c.end == info.chunks[-1].end:
                r = None

            t = self.TWrap(t)
            yield t, b, e

    def _leaf_gen(self, ten, info: NodeInfo, vvals: Dict[int, int]):
        fv = info.fv
        v = copy.copy(vvals)
        res = []

        assert len(info.chunks) == 1
        for vec in info.chunks[0].lidx:
            v.update(zip(fv, vec))
            res.append(fixvars(*ten, v)[0])

        yield self.TWrap(torch.tensor(np.array(res), dtype=torch.complex64)), info.chunks[0].beg, info.chunks[0].end

    def _contract_gen_rec(self, tens, node, vvals):
        assert node.info is not None
        if node.is_leaf:
            return self._leaf_gen(tens[node.tnum], node.info, vvals)
        else:
            gen_l = self._contract_gen_rec(tens, node.left, vvals)
            gen_r = self._contract_gen_rec(tens, node.right, vvals)
            return self._contract_node_gen(gen_l, gen_r, node.info)

    def contract_multi(self, tensors, bitstrs, schedule: ContractionTree, coef=0.5, **kwargs):
        """ Multi-amplitude contraction
        :param tensors: input tensors
        :param bitstrs: list of bitstrings
        :param schedule: multi-tensor contraction schedule (MTNSchedule object)
        :param coef: minimal batch density (technical parameter)
        :param kwargs: parameters of multi-batch contraction
        :return: contraction results (list of tensors) """
        if len(bitstrs) == 0:
            return np.array([], dtype=np.complex128)
        sch = MTNSchedule(schedule)
        maxmem = sch.prepare(bitstrs, **kwargs)
        complexity = 2.0**len(sch.sliced)*sum(x.info.flops() for x in sch.sch.nodes(False))
        print(f'prepared sch: 2^{math.log2(complexity):.2f} flops')
        memory = max(len(x.info.idxr) for x in sch.sch.nodes(False))
        print(f'max tensor size: 2^{memory}')
        print(f'max memory: 2^{math.log2(maxmem):.2f} elements')
        self._multiamp_coef = coef
        tens = [(transpose_numpy(t, i, sorted(i)), sorted(i)) for t, i in tensors]
        n = sch.sch.info.num_amps()
        res_p = torch.zeros((n,) + (2,) * len(sch.sch.info.idxr), dtype=torch.complex64)
        ndel = len(sch.sliced)
        tm0 = time.time()
        self._sufficient_flops = sch.sch.total_flops() / 8 / len(tens) / n

        for i, vals in enumerate(itertools.product([0, 1], repeat=ndel)):
            gen = self._contract_gen_rec(tens, sch.sch,  dict(zip(sch.sliced, vals)))
            for t, b, e in gen:
                res_p[b:e] += t.ten
                del t.ten
            print(f'step {i + 1}. {(i + 1) / 2 ** ndel:5.0%} completed,    time = {time.time() - tm0:.3f},'
                  f'   estimated = {(time.time() - tm0) * 2 ** ndel / (i + 1):.1f}')

        return np.array(res_p[sch.res_ind])


def main(optimize=False):
    # This is simple test for multi-amplitude contraction:
    # 1. read Sycamore elided circuit of depth 14
    # 2. calculate first 1000 amplitudes
    # 3. compare with Google's results

    fcirc = 'data/circuit_n53_m14_s0_e12_pABCDCDAB.qsim'
    tn = read_qsim(fcirc, simplify=True)  # read circuit and convert to tensor network
    tn.save_as_yaml('test_circ.yml')  # tensor network can be saved in yaml format

    amp_num = 1000  # maximum number of amplitudes to be calculated
    # read bitstring with amplitudes from file
    file_amps = read_amps('data/amplitudes_n53_m14_s0_e12_pABCDCDAB.txt', limit=amp_num)
    bitstrings = [bstr for bstr, amp in file_amps]
    scheme = tn.get_scheme()
    if not optimize:
        sch = ContractionTree(file='data/sch-m14-elided-simplified.yaml')
    else:
        sch, *_ = optimize_tn_multiamp(scheme, amp_num, mcoef=0.5,
                                       timeout=None, nthreads=...,
                                       max_log_memory=29,
                                       initial=('simple', 'min_fill_in', 'random'))
        sch.save_as_yaml('optimized_sch.yml')  # contraction schedule can be saved in yaml format
    if sch.scheme != tn.get_scheme():
        raise Exception("Contraction tree was optimized for another tensor network scheme")
    bk = TNBackend()
    amps = bk.contract_multi(tn.tensors, bitstrings, sch)

    print('Comparing results with reference data:')
    delta = s = 0
    for amp, (_, file_amp) in zip(amps, file_amps):
        s += abs(file_amp)
        delta += abs(amp - file_amp)  # calculate difference between calculated amplitudes and reference data
    print(f'relative delta = {delta / s}')
    print(f'absolute delta = {delta / len(file_amps)}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run multi-amplitude quantum circuit simulation')
    parser.add_argument('--optimize', action='store_true', help='optimize contraction tree '
                                                                '(instead of using precalculated)')
    args = parser.parse_args()

    main(optimize=args.optimize)
