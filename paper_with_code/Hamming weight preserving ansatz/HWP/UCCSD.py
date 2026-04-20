from mindquantum.algorithm.nisq.chem import Transform
from mindquantum.algorithm.nisq.chem import get_qubit_hamiltonian
from mindquantum.algorithm.nisq.chem import uccsd_singlet_generator, uccsd_singlet_get_packed_amplitudes
from mindquantum.core import TimeEvolution
from mindquantum.framework import MQAnsatzOnlyLayer
from mindquantum.core.operators.qubit_operator import QubitOperator
from mindquantum.core.operators import Hamiltonian
from mindquantum.simulator import Simulator
from mindquantum.core.circuit import Circuit
from mindquantum.core.gates import X
from mindquantum.algorithm.nisq import generate_uccsd
import numpy as np
import openfermion
import argparse
import pickle
import math

def prepare_seed(rand_seed):
    np.random.seed(rand_seed)

def main(xargs):
    print(xargs)


    ham = openfermion.utils.load_operator(xargs.Hamiltonian_path, data_directory='')


    ucc_fermion_ops = uccsd_singlet_generator(
    xargs.n_qubits, xargs.n_electrons, anti_hermitian=True)

    ucc_qubit_ops = Transform(ucc_fermion_ops).jordan_wigner()
    hartreefock_wfn_circuit = Circuit([X.on(i) for i in range(xargs.n_electrons)])
    ansatz_circuit = TimeEvolution(ucc_qubit_ops.imag, 1.0).circuit
    ansatz_parameter_names = ansatz_circuit.params_name

    total_circuit = hartreefock_wfn_circuit + ansatz_circuit
    total_circuit.summary()
    # total_circuit
    total_circuit.svg().to_file("/mnt/nas-new/home/yange/nature_comm/pack3/circuit.svg")


    ham = Hamiltonian(QubitOperator.from_openfermion(ham))
    sim = Simulator('mqvector_gpu', total_circuit.n_qubits)
    molecule_pqc = sim.get_expectation_with_grad(ham, total_circuit)

    n_params = len(total_circuit.params_name)
    p0 = np.random.rand(n_params)
    # f, g = molecule_pqc(p0)
    # print("Energy: ", f, "\nshape: ", f.shape, '\n')
    # print("Gradient: ", g, "\nshape: ", g.shape)

    import time
    timestart = time.time()

    def fun(p0, molecule_pqc, timestart, energy_list=None):
        f, g = molecule_pqc(p0)
        f = np.real(f)[0, 0]
        g = np.real(g)[0, 0]
        if energy_list is not None:
            energy_list.append(f)
            if len(energy_list) % 1 == 0:
                print(f"Step: {len(energy_list)},\tenergy: {f},\t time:{time.time() - timestart}s")
        return f, g

    # fun(p0, molecule_pqc)
    from scipy.optimize import minimize

    energy_list = []
    res = minimize(fun, p0, args=(molecule_pqc, timestart, energy_list), method='bfgs', jac=True)

    print(f"Ground state: \n{res.fun}\n")
    Lowest_energy = pickle.load(open(xargs.Lowest_energy_path, "rb"))
    print(np.min(energy_list) - (Lowest_energy))
    A = [xargs, None, p0, res, energy_list]
    with open(xargs.Output_path, "wb") as f:
        pickle.dump(A, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("HWP")
    # parser.add_argument("--Circuit_type", type=str, default="fullconnect")  # fullconnect HAE
    # parser.add_argument("--Circuit_layers", type=int, default=5)
    # parser.add_argument("--Gate_type", type=str, default="FBS")  # RBS FBS
    parser.add_argument("--Hamiltonian_path", type=str, default='/mnt/nas-new/home/yange/nature_comm/pack/data/H2_4_2.data')
    # parser.add_argument("--Iteration", type=int, default=10000)
    # parser.add_argument("--Learningrate", type=float, default=1e-3)
    parser.add_argument("--Lowest_energy_path", type=str, default='/mnt/nas-new/home/yange/nature_comm/pack/energy/H2.pkl')
    parser.add_argument("--Randseed", type=int, default=0)
    parser.add_argument("--n_qubits", type=int, default=4)
    parser.add_argument("--n_electrons", type=int, default=2)
    parser.add_argument("--Output_path", type=str,
                        required=True)  # if enabled, the search space will be changed to Rx, Ry, Rz
    args = parser.parse_args()
    main(xargs=args)