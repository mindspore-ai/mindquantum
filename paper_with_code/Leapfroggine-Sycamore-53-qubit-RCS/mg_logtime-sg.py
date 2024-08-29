import torch
import torch.distributed as dist
import re
import numpy as np
from copy import deepcopy
from os.path import exists, dirname, abspath
from os import makedirs
import os
import multiprocessing as mp
from math import ceil
import time
from torch.profiler import profile, record_function, ProfilerActivity
from cutensor.torch import EinsumGeneral, EinsumGeneralV2, getOutputShape

rank = int(os.environ["LOCAL_RANK"])
world_size = int(os.environ["WORLD_SIZE"])   

def init_process_group():
    backend = "nccl"
    init_method = "env://"
    dist.init_process_group(backend, init_method=init_method, rank=rank, world_size=world_size)


mgmodes = 3
split = 512
init_process_group()
torch.random.manual_seed(0)

cont_file = 'contraction/640G_reproduce_scheme_n53_m20_ABCDCDAB_3000000_einsum_10_open_3.pt'
nsch = torch.load(f'nsch640_split{split}_mg3_open10.pt')

tensors, scheme, slicing_indices, bitstrings = torch.load(cont_file)
# slicing_indices = torch.load('../mg640/640G_scheme_n53_m20_2093.pt')[2]

device=f'cuda:{rank}'

for i in range(len(nsch)):
    if 'chunk_batch' in nsch[i].keys():
        assert len(nsch[i]['chunk_batch'][0]) == split
        assert len(nsch[i]['chunk_batch'][1]) == split
        for t in range(split):
            nsch[i]['chunk_batch'][0][t] = nsch[i]['chunk_batch'][0][t].to(device)
            nsch[i]['chunk_batch'][1][t] = nsch[i]['chunk_batch'][1][t].to(device)

slicing_edges = list(slicing_indices.keys())
tensors_gpu = [tensor.to(device, dtype=torch.complex64) for tensor in tensors]
stemtensor = torch.empty([2, 2**32], dtype = torch.cfloat, device = device)

class MgTensor:
    
    @property
    def curtensor(self):
        nelem = torch.tensor(self.shape).prod().item()
        return stemtensor[self.pcurtensor,:nelem].view(self.shape)
    
    @property
    def nexttensor(self):
        ptr = 1 - self.pcurtensor
        return stemtensor[ptr]
    
    def setnewtensor(self,newshape):
        self.pcurtensor = 1 - self.pcurtensor
        self.shape = newshape
    
    def __init__(self, sgtensor):
        self.pcurtensor = 0
        self.rank = rank
        self.world_size = world_size
        assert type(sgtensor) == torch.Tensor
        sgtensor = sgtensor.flatten(end_dim = mgmodes)
        sgtensor = torch.chunk(sgtensor, 2**mgmodes)[self.rank]
        self.shape = sgtensor.shape
        self.curtensor[:] = sgtensor
    
    def einsum(self, ein, insg2, task_id):
        ein_list = re.split('->|,', ein)
        if ein_list[0][:mgmodes] != ein_list[2][:mgmodes]:
            rawtensor = self.curtensor.flatten(end_dim = mgmodes-1)
            newmgtensor = self.nexttensor
            dist.all_to_all_single(newmgtensor, rawtensor)
            self.setnewtensor(self.shape)
            
        mgchar = list(ein_list[2][:mgmodes])
        ein_new = re.sub('|'.join(mgchar), '', ein)
        # if rank == 0:
        #     print((ein_new, self.curtensor.shape, insg2.shape))
        newshape = getOutputShape(ein_new, self.curtensor, insg2)
        EinsumGeneralV2(self.nexttensor, ein_new, self.curtensor, insg2)
        self.setnewtensor(newshape)
    
    def flatsplit(self,flat, split):
        # Here I assumed that the tensors won't be larger than the splited tensor
        # It may won't work in other tensor network
        splitcurtensor = list(torch.chunk(
                            self.curtensor.flatten(end_dim=flat-mgmodes-1),
                            split // (self.world_size)))
        splitnexttensor = list(torch.chunk(
                            self.nexttensor,
                            split // (self.world_size)))
        mgtensorsplit = [MgTensorSplit(sct, snt) for sct, snt in zip(splitcurtensor, splitnexttensor)]
        return mgtensorsplit
    

class MgTensorSplit:
    @property
    def curtensor(self):
        nelem = torch.tensor(self.shape).prod().item()
        return self.ts[self.pcurtensor].flatten()[:nelem].view(self.shape)
    
    @property
    def nexttensor(self):
        ptr = 1 - self.pcurtensor
        return self.ts[ptr]
    
    def setnewtensor(self,newshape):
        self.pcurtensor = 1 - self.pcurtensor
        self.shape = newshape
        
    def __init__(self, curtensor, nexttensor):
        self.ts = [curtensor, nexttensor]
        self.pcurtensor = 0
        self.shape = curtensor.shape
        
        
    
def cont_nsch_split(tensors, nsch, task_id):
    for nstep, step in enumerate( nsch):
        # print(nstep)
        with record_function(f"step{nstep}"):
            i, j = step['index']
            if step['type'] == 1:
                # print('type ',i, ' ', j)
                flati, flatj = step['flat']
                # print(step['flat'])
                if flati > 1:
                    assert type(tensors[i]) == MgTensor
                    tensors[i] = tensors[i].flatsplit(flati,split)
                if flatj > 1:
                    lenj = len(tensors[j].shape)
                    tensors[j] = tensors[j].reshape([-1] + [2] * (lenj - flatj))

                chubi, chubj = step['chunk_batch']
                # print(len(tensors[i]))
                for chuindex in range(len(tensors[i])):
                    # torch.cuda.empty_cache()
                    # print((tensors[i][chuindex].shape,len(chubi[chuindex + split // world_size * rank])))
                    pi = tensors[i][chuindex].curtensor[chubi[chuindex + split // world_size * rank]]
                    pj = tensors[j][chubj[chuindex + split // world_size * rank]]
                    newshape = getOutputShape( step['ein_2'], pi, pj)
                    EinsumGeneralV2(tensors[i][chuindex].nexttensor, step['ein_2'], pi, pj)
                    tensors[i][chuindex].setnewtensor(newshape)
                    del pi
                    del pj            
                tensors[j] = []
            elif step['type'] == 2 or step['type'] == 3:
                if type(tensors[i]) == list:
                    for chuindex in range(len(tensors[i])):
                        # tensors[i][x] = EinsumGeneral(step['ein_2'], tensors[i][x], tensors[j])
                        pi = tensors[i][chuindex].curtensor
                        pj = tensors[j]
                        newshape = getOutputShape( step['ein_2'], pi, pj)
                        EinsumGeneralV2(tensors[i][chuindex].nexttensor, step['ein_2'], pi, pj)
                        tensors[i][chuindex].setnewtensor(newshape)
                elif 'reorder_ein' in step:
                    if type(tensors[i]) == torch.Tensor:
                        tensors[i] = EinsumGeneral(step['reorder_ein'], tensors[i], tensors[j])
                        tensors[i] = MgTensor(tensors[i])
                    else:
                        assert type(tensors[i]) == MgTensor
                        tensors[i].einsum(step['reorder_ein'], tensors[j], task_id)
                else:
                    tensors[i] = EinsumGeneral(step['ein_2'], tensors[i], tensors[j])
                tensors[j] = []
            # torch.cuda.synchronize()
            
    return torch.cat([x.curtensor for x in tensors[i]])


def calc_task(s):
    configs = list(map(int, np.binary_repr(s, len(slicing_edges))))
    sliced_tensors = tensors_gpu.copy()
    for x in range(len(slicing_edges)):
        m, n = slicing_edges[x]
        idxm_n, idxn_m = slicing_indices[(m, n)]
        sliced_tensors[m] = sliced_tensors[m].select(idxm_n, configs[x]).clone()
        sliced_tensors[n] = sliced_tensors[n].select(idxn_m, configs[x]).clone()
    
    time_begin = time.time()
    ans = cont_nsch_split(sliced_tensors, nsch, s)
    
    time_end = time.time()
    tottime = time_end - time_begin

    return ans, tottime

def gather_ans(ans):
    global stemtensor
    torch.cuda.synchronize()
    time_begin = time.time()
    del stemtensor
    ans = torch.view_as_real(ans)
    tensor_sizes = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(tensor_sizes, ans.shape)
    padlen = max([x[0] for x in tensor_sizes]) - ans.shape[0]
    padder = torch.zeros([padlen]+list(ans.shape)[1:], dtype = ans.dtype, device = ans.device)
    
    ans = torch.cat([ans, padder])
    anslist = [torch.empty_like(ans) for _ in range(dist.get_world_size())]
    dist.all_gather(anslist, ans)
    anslist = [torch.view_as_complex(anslist[i][:tensor_sizes[i][0]])
                for i in range(len(anslist))]
    torch.cuda.synchronize()
    time_end = time.time()
    tottime = time_end - time_begin
    return anslist

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    time_begin = time.time()
    time_last = time_begin
    ntask = 6
    for s in range(ntask):
        print(s,time.strftime("%H:%M:%S", time.gmtime( time.time() - time_begin)),time.time()-time_last)
        time_last = time.time()
        if s == 0:
            ans = calc_task(s)[0]
        else:
            ans += calc_task(s)[0]

    anslist = gather_ans(ans)

if rank == 0:
    prof.export_chrome_trace("trace.json")
    torch.save(anslist, f'sum_{ntask}.pt')
