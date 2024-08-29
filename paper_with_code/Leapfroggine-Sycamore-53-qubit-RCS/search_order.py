from load_circuits import QuantumCircuit
from artensor import (
    AbstractTensorNetwork, 
    ContractionTree, 
    find_order, 
    contraction_scheme,
    tensor_contraction,
    contraction_scheme_sparse,
    tensor_contraction_sparse
)
from copy import deepcopy
import numpy as np
import torch


torch.backends.cuda.matmul.allow_tf32 = False
n, m, seq, device, sc_target, seed = 53, 20, 'ABCDCDAB', 'cuda', 35, 0
max_bitstrings = 1 #3_000_000
qc = QuantumCircuit(n, m, seq=seq)
edges = []
for i in range(len(qc.neighbors)):
    for j in qc.neighbors[i]:
        if i < j:
            edges.append((i, j))
neighbors = list(qc.neighbors)
final_qubits = set(range(len(neighbors) - n, len(neighbors)))
bond_dims = {i:2.0 for i in range(len(edges))}

def read_samples(filename):
    import os
    if os.path.exists(filename):
        samples_data = []
        with open(filename, 'r') as f:
            l = f.readlines()
        f.close()
        for line in l:
            ll = line#.split()
            samples_data.append((ll))
        return samples_data
    else:
        raise ValueError("{} does not exist".format(filename))

data = read_samples('measurements_n53_m20_s0_e0_pABCDCDAB.txt')

bitstrings = [data[i][0:53] for i in range(max_bitstrings)]

amplitude_google = np.array([data[i][1] for i in range(max_bitstrings)])
tensor_bonds = {
    i:[edges.index((min(i, j), max(i, j))) for j in neighbors[i]] 
    for i in range(len(neighbors))
} # now all tensors will be included


order_slicing, slicing_bonds, ctree_new = find_order(
    tensor_bonds, bond_dims, final_qubits, seed, max_bitstrings, 
    sc_target=sc_target, trials=5, iters=50, slicing_repeat=4, # trials is number of thread，betas is inverse temperature
    betas=np.linspace(3.0, 21.0, 61), alpha = 150               # Tensor Float 32 speed/bandwidth
)

tensors = []
for x in range(len(qc.tensors)):
    if x not in final_qubits:
        tensors.append(qc.tensors[x].to(device))

scheme_sparsestate, bonds_final, bitstrings_sorted = contraction_scheme_sparse(
    ctree_new, bitstrings, sc_target=sc_target)

slicing_edges = [edges[i] for i in slicing_bonds]
slicing_indices = {}.fromkeys(slicing_edges)
tensors = []
for x in range(len(qc.tensors)):
    if x in final_qubits:
        tensors.append(
            torch.tensor([[1, 0], [0, 1]], dtype=torch.complex64, device=device)
        )
    else:
        tensors.append(qc.tensors[x].to(device))

tensors_save = [tensor.to('cpu') for tensor in tensors]

neighbors_copy = deepcopy(neighbors)
for x, y in slicing_edges:
    idxi_j = neighbors_copy[x].index(y)
    idxj_i = neighbors_copy[y].index(x)
    neighbors_copy[x].pop(idxi_j)
    neighbors_copy[y].pop(idxj_i)
    slicing_indices[(x, y)] = (idxi_j, idxj_i)

result = (tensors_save, scheme_sparsestate, slicing_indices, bitstrings_sorted)
torch.save(result, "./640G_scheme_n53_m20_test.pt")
print("time complexity, space complexity, memory complexity = ",ctree_new.tree_complexity())

