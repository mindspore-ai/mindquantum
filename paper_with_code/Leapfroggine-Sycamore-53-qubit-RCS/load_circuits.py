from os.path import join, dirname, abspath
import numpy as np
import torch
from copy import deepcopy
import sys
import cirq


def swap_seq(L, idx0, idx1):
    assert idx1 <= L - 1
    assert idx0 <= L - 1
    seq = list(range(L))
    if idx0 < idx1:
        seq[idx0 + 2:idx1 + 1] = seq[idx0 + 1:idx1]
        seq[idx0 + 1] = idx1
    else:
        seq[idx1:idx0] = seq[idx1 + 1:idx0 + 1]
        seq[idx0] = idx1

    return seq

def simplify_edges(tensors, neighbors, edges):
    for edge in edges:
        i, j = edge
        if i in neighbors[j] and j in neighbors[i]:
            flag = True
        else:
            flag = False
        if flag:
            L_i = len(neighbors[i]) - 1
            idxi_j = neighbors[i].index(j)
            idxj_i = neighbors[j].index(i)
            neighbors[i].pop(idxi_j)
            neighbors[j].pop(idxj_i)
        else:
            L_i = len(neighbors[i])
        if flag:
            tensors[i] = np.tensordot(tensors[i], tensors[j], ([idxi_j], [idxj_i]))
        else:
            tensors[i] = np.outer(tensors[i].reshape(-1), tensors[j].reshape(-1)).reshape(
                        tensors[i].shape + tensors[j].shape)

        for node in neighbors[j]:
            idxj_n = neighbors[j].index(node)
            idxn_j = neighbors[node].index(j)
            if node in neighbors[i]:
                # merge node in neighbors_i
                idxi_n = neighbors[i].index(node)
                seqi = swap_seq(len(tensors[i].shape), idxi_n, idxj_n + L_i)
                L_i -= 1
                tensors[i] = tensors[i].transpose(seqi)
                tensors[i] = tensors[i].reshape(tensors[i].shape[:idxi_n] +
                                                (-1,) + tensors[i].shape[idxi_n + 2:])

                # merge i in neighbors_node
                idxn_i = neighbors[node].index(i)
                neighbors[node].pop(idxn_j)
                seqn = swap_seq(len(tensors[node].shape), idxn_i, idxn_j)
                tensors[node] = tensors[node].transpose(seqn)
                if idxn_i < idxn_j:
                    tensors[node] = tensors[node].reshape(tensors[node].shape[:idxn_i] +
                                                          (-1,) + tensors[node].shape[idxn_i + 2:])
                else:
                    try:
                        tensors[node] = tensors[node].reshape(tensors[node].shape[:idxn_i - 1] +
                                                              (-1,) + tensors[node].shape[idxn_i + 1:])
                    except:
                        print(i, j, node, tensors[node].shape, idxn_i, idxn_j, seqn, tensors[node])
                        raise Exception('can not reshape')
            else:
                neighbors[i].append(node)
                neighbors[node][idxn_j] = i
    
        tensors[j] = []
        neighbors[j] = []

    return tensors, neighbors


class QuantumCircuit:
    def __init__(self, n=53, m=20, seed=0, elide=0, seq='ABCDCDAB', bitstring=None, fix_pairs=[],
                 twoqubit_simplify=True, fSim_simplify=None, package=torch, complex128=False):
        self.n = n
        seq = 'ABCDCDAB' if 'a' in seq or 'A' in seq else 'EFGH'
        fname = f"circuit_n{n}_m{m}_s{seed}_e{elide}_p{seq}"
        self.gates, _ = self.get_circuit(fname)

        initial_states = np.vstack([np.array([1, 0], dtype=np.complex128)] * n)
        final_states = np.empty([n, 2], dtype=np.complex128)
        if bitstring is None:
            for i in range(n):
                final_states[i] = np.array([1, 0], dtype=np.complex128)
        else:
            for i in range(n):
                final_states[i] = np.array([1, 0], dtype=np.complex128) \
                    if bitstring[i] == '0' else np.array([0, 1], dtype=np.complex128)
        for i in range(n):
            self.gates.insert(0, [initial_states[n-1-i], [n-1-i]])
            self.gates.append([final_states[i], [i]])

        self.final_qubit_id = [i for i in range(self.n)]
        
        tensors, self.neighbors, self.qubits_representation = self.gates_to_tensors(self.gates, twoqubit_simplify, fix_pairs)
        assert package in [np, torch]
        if complex128:
            self.datatype = np.complex128
        else:
            self.datatype = np.complex64
        self.tensors = []
        for i in range(len(tensors)):
            if package == torch:
                data = torch.from_numpy(tensors[i].astype(self.datatype))
            else:
                data = tensors[i].astype(self.datatype)
            self.tensors.append(data)

    def gates_to_tensors(self, gates, simplify=True, fix_pairs=[]):
        tensors = []
        labels = []
        self.chains = []
        qubits_representation = {}

        for i in range(len(gates)):
            tensors.append(gates[i][0])
            for j in gates[i][1]:
                if len(self.chains) < j + 1:
                    self.chains.append([])
                self.chains[j].append(i)
            labels.append([])
            qubits_representation[i] = [i]

        qubit_num = len(self.chains)

        for i in range(len(self.chains)):
            for j in range(len(self.chains[i])):
                if j == 0:
                    labels[self.chains[i][j]].append(self.chains[i][j + 1])
                elif j == len(self.chains[i]) - 1:
                    labels[self.chains[i][j]].append(self.chains[i][j - 1])
                else:
                    labels[self.chains[i][j]] += [self.chains[i][j + 1], self.chains[i][j - 1]]

        for i in range(len(tensors)):
            labels[i] = np.array(labels[i])
            if len(labels[i]) > 1:
                labels[i] = labels[i].reshape([-1, 2]).transpose().reshape(-1)
            label_unique, idx, counts = np.unique(labels[i], return_index=True, return_counts=True)
            idx = np.argsort(idx)
            label = label_unique[idx].tolist()
            counts = counts[idx]
            tensors[i] = tensors[i].reshape(2 ** counts)
            labels[i] = label

        if simplify:
            final_qubits_fix_id = [i for i in range(len(gates)-2*qubit_num, len(gates))]

            simplify_order = []
        
            labels_copy = deepcopy(labels)
            remove_nodes = []
            simplify_flag = False
            for i in range(len(labels)):
                if len(labels[i]) <= 2 and i not in final_qubits_fix_id:
                    simplify_flag = True

            while simplify_flag:
                for i in range(len(tensors)):
                    if i in remove_nodes or i in final_qubits_fix_id:
                        continue
                    current_node = i
                    if len(labels[current_node]) <= 2:
                        try:
                            if len(labels[current_node]) > 1 and (labels[current_node][1], labels[current_node][0]) in fix_pairs:
                                neighbor = labels[current_node][1]
                            else:
                                neighbor = labels[current_node][0]
                        except:
                            print(labels[current_node])
                            sys.exit(1)
                        if neighbor in final_qubits_fix_id:
                            source, target = current_node, neighbor
                        else:
                            source, target = neighbor, current_node
                        idx1 = labels[source].index(target)
                        idx2 = labels[target].index(source)
                        simplify_order.append((source, target))
                        if target in qubits_representation.keys():
                            value = qubits_representation.pop(target)
                            if source in qubits_representation.keys():
                                qubits_representation[source] += value
                            else:
                                qubits_representation[source] = value
                        labels[target].pop(idx2)
                        labels[source].pop(idx1)
                        for k in range(len(labels[target])):
                            node_add = labels[target][k]
                            labels[source].append(node_add)
                            idx = labels[node_add].index(target)
                            labels[node_add].pop(idx)
                            labels[node_add].insert(idx, source)
                        remove_nodes.append(target)

                for i in range(len(tensors)):
                    if i in remove_nodes or i in final_qubits_fix_id:
                        continue
                    current_node = i
                    if len(labels[current_node]) != len(list(set(labels[current_node]))):
                        nodes_add = []
                        idx1 = []
                        for k in range(len(labels[current_node])):
                            if labels[current_node].count(labels[current_node][k]) > 1:
                                neighbor = labels[current_node][k]
                                idx1.append(k)
                            else:
                                nodes_add.append(labels[current_node][k])
                        idx2 = []
                        for k in range(len(labels[neighbor])):
                            if labels[neighbor][k] == current_node:
                                idx2.append(k)
                        simplify_order.append((neighbor, current_node))
                        if current_node in qubits_representation.keys():
                            value = qubits_representation.pop(current_node)
                            if neighbor in qubits_representation.keys():
                                qubits_representation[neighbor] += value
                            else:
                                qubits_representation[neighbor] = value
                        idx2.sort(reverse=True)
                        for idx in idx2:
                            labels[neighbor].pop(idx)
                        labels[neighbor] += nodes_add
                        for node in nodes_add:
                            idx = labels[node].index(current_node)
                            labels[node].pop(idx)
                            labels[node].insert(idx, neighbor)
                        remove_nodes.append(current_node)
                simplify_flag = False
                for i in range(len(labels)):
                    if len(labels[i]) <= 2 and i not in final_qubits_fix_id and i not in remove_nodes:
                        simplify_flag = True

            tensors_tmp, labels_tmp = simplify_edges(tensors, labels_copy, simplify_order)

            remove_nodes.sort(reverse=True)
            remain_nodes = np.arange(len(tensors)).tolist()
            for i in remove_nodes:
                remain_nodes.remove(i)

            tensors_simp, labels_simp = [], []
            for i in remain_nodes:
                assert labels[i] == labels_tmp[i]
                labels_simp.append([remain_nodes.index(j) for j in labels_tmp[i]])
                tensors_simp.append(tensors_tmp[i])
        else:
            tensors_simp, labels_simp = tensors, labels
        return tensors_simp, labels_simp, qubits_representation
    
    def einsum_eq(self, group=None):
        import opt_einsum as oe
        if group is None:
            group = range(len(self.neighbors))
        edges = []
        for i in range(len(self.neighbors)):
            for j in self.neighbors[i]:
                if i < j:
                    edges.append((i, j))
        eq = []
        out_inds = ''
        sizes = []
        for i in group:
            neigh = self.neighbors[i]
            shape_tmp = []
            eq_tmp = ''
            for j in neigh:
                if j < i:
                    index = edges.index((j, i))
                else:
                    index = edges.index((i, j))
                symbol = oe.get_symbol(index)
                if j not in group:
                    out_inds += symbol
                eq_tmp += symbol
                size = self.tensors[i].shape[neigh.index(j)]
                shape_tmp.append(size)
            sizes.append(shape_tmp)
            eq.append(eq_tmp)

        return eq, sizes, out_inds

    def get_circuit(self, fname):
        import importlib
        qc = importlib.import_module(fname)

        id_map={}
        n_qubits = 0
        for qubit in qc.QUBIT_ORDER:
            id_map[(qubit.row,qubit.col)] = n_qubits
            n_qubits += 1

        gates = []
        for moment in qc.CIRCUIT:
            for gate in moment:
                qubits = gate.qubits
                mat = cirq.unitary(gate)
                if (len(qubits) == 1):
                    qubit_id = id_map[ qubits[0].row,qubits[0].col ]
                    gates.append( [mat,[qubit_id]] )
                elif (len(qubits) == 2):
                    qubit1_id = id_map[ qubits[0].row,qubits[0].col ]
                    qubit2_id = id_map[ qubits[1].row,qubits[1].col ]
                    gates.append( [mat,[qubit1_id,qubit2_id]])
                else:
                    print("unknown gates !!!")
                    print(gate.__str__)
                    sys.exit(0)
        return gates, qc