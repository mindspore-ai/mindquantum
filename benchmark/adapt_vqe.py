from copy import deepcopy
import gc
import sys
import objgraph
from mindspore import Tensor
from mindquantum.nn import generate_pqc_operator
from scipy.optimize import minimize
from mindquantum.circuit import TimeEvolution, Circuit
from mindquantum.gate import X, RX, Hamiltonian
from mindquantum.ops import QubitOperator, FermionOperator
from mindquantum.hiqfermion.transforms import Transform
from q_ham import MolInfoProduce
import numpy as np
from pool import singlet_SD, pauli_pool, singlet_GSD

class AdaptVqe(MolInfoProduce):

    def __init__(self, mole_name, geometry, basis, charge, multiplicity,
                fermion_pool, fermion_transform="jordan_wigner"):

        super(AdaptVqe, self).__init__(geometry, basis, 
                                                charge, multiplicity, fermion_transform)
        self.paras = []
        self.gradient_pqc = None
        self.e_pqc = None
        self.adapt_circuit = Circuit([X.on(i) for i in range(self.n_electrons)])
        self.encoder_circuit = RX("null").on(self.n_qubits-1)
        self.ferop_seq = []
        self.fermion_pool = fermion_pool
        self.step_energies = []
        self.gradients_seq = []

    def compute_gradients(self, 
                        fermion_pool, 
                        index,
                        delta = 1e-5
                        ):
        """
            Compute gradients for all excitation operator in the pool.

        """
        paras_left = deepcopy(self.paras)
        paras_right = deepcopy(self.paras)
        paras_left += [-delta]
        paras_right += [delta]
        
        result = []
        for ferop in self.fermion_pool:
            ferop = rename_ferop(ferop, index)
            self.generate_gradient_pqc(ferop)
            gradient = (self.adapt_energy(paras_left, self.gradients_pqc)[0] - \
                        self.adapt_energy(paras_right, self.gradients_pqc)[0]) / (2 * delta)
            
            # if abs(gradient) > 1e-6:
            result.append((abs(gradient), ferop))

        sorted_result = sorted(result, key=lambda x:x[0], reverse=True)
        return sorted_result

    def adapt_energy(self, paras, pqc):
        encoder_data = Tensor(np.array([[0]]).astype(np.float32))
        ansatz_data = Tensor(np.array(paras).astype(np.float32))

        e, _, grad = pqc(encoder_data, ansatz_data)
        return e.asnumpy()[0, 0], grad.asnumpy()[0, 0]
    
    def generate_gradient_pqc(self, ferop):
        gradients_circuit = self.adapt_circuit + \
                                TimeEvolution(Transform(ferop).jordan_wigner().imag, 1).circuit

        self.gradients_pqc = generate_pqc_operator(["null"], gradients_circuit.para_name, \
                                    self.encoder_circuit + gradients_circuit, \
                                    Hamiltonian(self.qubit_hamiltoian))

    def paras_optimize(self):
        self.paras.append(0.0)
        result = minimize(self.adapt_energy, self.paras, args=(self.pqc),method='BFGS',jac=True, tol=1e-6)
        self.step_energies.append(float(result.fun))
        if abs(result.x.tolist()[-1]) < 1e-5:
            self.paras.pop()
            return True
        else: 
            self.paras = result.x.tolist()
            return False
        

    def opti_process(self, maxiter=100, adapt_thresh=1e-3):
        for iteration in range(maxiter):
            print('\n')
            print('-----------------current_iteration {}-----------------'.format(iteration))
            print('\n')
            sorted_result = self.compute_gradients(self.fermion_pool, iteration)
            norm = norm_of_sorted_result(sorted_result)
            if norm < adapt_thresh:
                break
            print('choose fermionic operator:')
            print('\n')
            print(sorted_result[0][1])
            print('\n')
            print('corresponding gadient:{}'.format(sorted_result[0][0]))
            self.ferop_seq.append(sorted_result[0][1])
            self.gradients_seq.append(sorted_result[0][0])
            # grow ansatz by one operator with the largest gradient
            self.adapt_circuit += TimeEvolution(Transform(self.ferop_seq[-1]).jordan_wigner().imag, 1).circuit
            self.pqc = generate_pqc_operator(["null"], self.adapt_circuit.para_name, \
                                            self.encoder_circuit + self.adapt_circuit, Hamiltonian(self.qubit_hamiltoian))
            flag = self.paras_optimize()
            del self.pqc
            del self.gradients_pqc
            #del self.adapt_circuit
            del sorted_result
            gc.collect()
            objgraph.show_most_common_types(limit=10)
            print('end')
            if flag:
                self.ferop_seq.pop()
                self.gradients_seq.pop()
                break
            print('optimized energy {} hartree for iteration {}'.format(self.step_energies[-1], iteration))
        
        print('algorithm convergence because of the norm of all operators {} lowen than adapt_thresh {}'.format(norm, adapt_thresh))
        
        print('If not, the algorithm stops due to the error caused by finite difference when calculating gradients')
        print('\n')
        print('-----------algorithm convergence info:-----------')
        print('Num of total iterations:{}'.format(len(self.step_energies)))

        print('optimized parameters:')
        print(self.paras)
        print('optimized energies:')
        print(self.step_energies)

        print('hf_energy {} hartree and fci energy {} hartree'.format(self.hf_energy, self.fci_energy))
        print('absolute error compared with fci energy:')
        print('{} hartree'.format(abs(self.step_energies[-1]-self.fci_energy)))
                


def rename_ferop(ferop, index):
    renamed_ferop = FermionOperator()
    for ferop_seq, coeff in ferop.terms.items():
        renamed_ferop += coeff * FermionOperator(ferop_seq, 'p'+str(index))
    return renamed_ferop
    
def norm_of_sorted_result(sorted_result):
    norm = 0.0
    for g, _ in sorted_result:
        norm += g * g
    norm = np.sqrt(norm)
    return norm



        
if __name__=="__main__":

    n_orb, n_occ, n_vir = 4, 2, 2
    pool = singlet_SD(n_orb=n_orb, n_occ=n_occ, n_vir=n_vir)
    pool.generate_operators()
    collection_of_ferops = pool.fermi_ops

    print(len(pool.fermi_ops))

    geometry = [['H', (0, 0, 0)], ['H', (0, 0, 2.0)], ['H', (0, 0, 4.0)], ['H', (0, 0, 6.0)]]
    # geometry = [['H', (0, 0, 0)], ['Li', (0, 0, 2.0)]]
    
    adapt_H4 = AdaptVqe('H4', geometry, 'sto3g', 0, 1, collection_of_ferops)

    adapt_H4.opti_process()
    print('---------------optimization result-----------------:')
    print(adapt_H4.ferop_seq)
    # print(adapt_H4.gradients_seq)
    print(adapt_H4.hf_energy)
    print(adapt_H4.fci_energy)
    print(adapt_H4.step_energies)
    print(len(adapt_H4.ferop_seq))
    print(len(adapt_H4.step_energies))
    

    
    

