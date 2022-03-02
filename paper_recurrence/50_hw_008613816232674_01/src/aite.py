from openfermion.chem import MolecularData
from openfermionpyscf import run_pyscf
from mindquantum.algorithm.nisq.chem import get_qubit_hamiltonian
from mindquantum import Hamiltonian
from mindquantum.simulator import Simulator

def get_system(key='LiH'):
    dist = 1.5
    if key=='LiH':
        geometry = [
            ["Li", [0.0, 0.0, 0.0 * dist]],
            ["H",  [0.0, 0.0, 1.0 * dist]],
        ]
    elif key=='H2':
        geometry = [
            ["H", [0.0, 0.0, 0.0 * dist]],
            ["H",  [0.0, 0.0, 1.0 * dist]],
        ]


    basis = "sto3g"
    spin = 0
    # print("Geometry: \n", geometry)

    molecule_of = MolecularData(
        geometry,
        basis,
        multiplicity=2 * spin + 1
    )
    molecule_of = run_pyscf(
        molecule_of,
        run_scf=1,
        run_ccsd=0,
        run_fci=1
    )

    hamiltonian_QubitOp = get_qubit_hamiltonian(molecule_of)
    return molecule_of, Hamiltonian(hamiltonian_QubitOp)


class Optimizer:
    def __init__(self, learn_rate=10, decay_rate=0.01):
        self.diff = np.zeros(1).astype(np.float32)
        self.learn_rate = learn_rate
        self.decay_rate = decay_rate
        
    def step(self, vector, grad):
        # Performing the gradient descent loop
        vector = vector.astype(np.float32)
        # grad = grad.squeeze(0).squeeze(0)
        grad = grad.astype(np.float32)

        self.diff = self.decay_rate * self.diff - self.learn_rate * grad
        vector += self.diff
        return vector


from mindquantum import Circuit, RY, RX, RZ
from mindquantum import X, Z, Y
import numpy as np
import math
from mindquantum.core import ParameterResolver
from mindquantum.core.parameterresolver import ParameterResolver as PR

class Parameter_manager:
    def __init__(self):
        self.parameters = []
        self.count = 0
    
    def init_parameter_resolver(self):
        pr = {k:np.random.randn()*2*math.pi for k in self.parameters}
        # pr = {k:0 for k in self.parameters}
        pr = ParameterResolver(pr)
        return pr

    def _replay(self):
        self.count = 0

    def create(self):
        param = 'theta_{}'.format(self.count)
        self.count += 1
        self.parameters.append(param)
        return param


def RZZ_gate(circ, i, j, P):
    circ += X.on(j, i)
    circ += RZ(P.create()).on(j)
    circ += X.on(j, i)


def layer(circ, P, n_qubits):
    for i in range(n_qubits):
        circ += RZ(P.create()).on(i)
        circ += RY(P.create()).on(i)
        circ += RX(P.create()).on(i)
        circ += RZ(P.create()).on(i)


    for i in range(0, n_qubits-1, 2):
        RZZ_gate(circ, i, i+1, P)
        RZZ_gate(circ, i, i+1, P)
        RZZ_gate(circ, i, i+1, P)

    for i in range(1, n_qubits-1, 2):
        RZZ_gate(circ, i, i+1, P)
        RZZ_gate(circ, i, i+1, P)
        RZZ_gate(circ, i, i+1, P)


from Hessian.gradients import Grad, FisherInformation
class VQE:
    def __init__(self, key='LiH'):
        molecule_of, self.ham = get_system(key=key)
        self.n_qubits = molecule_of.n_qubits
        self.fci_energy = molecule_of.fci_energy

        self.P = Parameter_manager()
        self.circ = Circuit()
        layer(self.circ, self.P, self.n_qubits)
        self.pr = self.P.init_parameter_resolver()

        self.optimizer = Optimizer(learn_rate=1.0)

    def pr2array(self, pr):
        parameters = []
        k_list = []
        for k in pr.keys():
            k_list.append(k)
            parameters.append(pr[k])

        parameters = np.array(parameters)
        return parameters, k_list

    def array2pr(self, parameters, k_list):
        _pr = {}
        for k, p in zip(k_list, parameters.tolist()):
            _pr[k] = p
        pr = PR(_pr)
        return pr

    def gradient_descent_step(self):
        parameters, k_list = self.pr2array(self.pr)
        g = Grad(self.circ, self.pr, self.ham, self.n_qubits).grad_reserveMode()
        parameters = self.optimizer.step(parameters, g).real
        self.pr = self.array2pr(parameters, k_list)


    def imaginary_time_evolution_step(self):
        parameters, k_list = self.pr2array(self.pr)

        g = Grad(self.circ, self.pr, self.ham, self.n_qubits).grad_reserveMode()
        h = FisherInformation(self.circ, self.pr, self.n_qubits).gite_preconditional()

        g = np.linalg.inv(h + np.eye(len(h))*1e-15).dot(g[:, np.newaxis]).squeeze(1) * (-1)

        parameters = self.optimizer.step(parameters, g).real
        self.pr = self.array2pr(parameters, k_list)


    def natural_gradient_step(self):
        parameters, k_list = self.pr2array(self.pr)

        g = Grad(self.circ, self.pr, self.ham, self.n_qubits).grad_reserveMode()
        h = FisherInformation(self.circ, self.pr, self.n_qubits).fisher_information()

        g = np.linalg.inv(h + np.eye(len(h))*1e-15).dot(g[:, np.newaxis]).squeeze(1) * (-1)

        parameters = self.optimizer.step(parameters, g).real
        self.pr = self.array2pr(parameters, k_list)


    def eval(self):
        sim = Simulator('projectq', self.n_qubits)
        sim.apply_circuit(self.circ, pr=self.pr)
        E = sim.get_expectation(self.ham)
        return E.real


if __name__ == '__main__':
    V = VQE(key='H2')
    for i in range(100):

        # V.gradient_descent_step()
        # V.imaginary_time_evolution_step()
        V.natural_gradient_step()

        E = V.eval()
        print(E, V.fci_energy)