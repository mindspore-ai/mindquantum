import mindquantum.core.gates as G
from mindquantum.simulator import Simulator
from mindquantum.core.operators import QubitOperator
from mindquantum import Hamiltonian
from mindquantum import Circuit
import copy
import numpy as np

from .utils import pr2array, Ising_like_ham
from .helper import Parameter_manager, layer

def example_circuit(n_qubits=5):
    circ = Circuit()
    P = Parameter_manager()
    layer(circ, P, n_qubits)
    pr = P.init_parameter_resolver()
    return circ, pr

def get_gradient_preconditional(gate):
    if isinstance(gate, G.RX):
        return G.X.on(gate.obj_qubits)
    elif isinstance(gate, G.RY):
        return G.Y.on(gate.obj_qubits)
    elif isinstance(gate, G.RZ):
        return G.Z.on(gate.obj_qubits)
    else:
        raise NotImplementedError()

J = np.complex(0,1)
def grad_circuit_symbolic_forward(circ):
    circ_list, circ_coeff_list = [], []
    for i, gate in enumerate(circ):
        if isinstance(gate, (G.RX, G.RY, G.RZ)):
            n_circ = copy.deepcopy(circ)
            n_circ.insert(i, get_gradient_preconditional(gate))
            circ_list.append(n_circ)
            circ_coeff_list.append(-1./2 * J) #for example, grad(RX) = -j X RX
    return circ_list, circ_coeff_list


PHASE_SHIFT = None

class Grad:
    def __init__(self, circ, pr, ham, n_qubits):
        self.circ, self.pr, self.ham = circ, pr, ham
        self.n_qubits = n_qubits
        self.parameters, self.k_list = pr2array(self.pr)
        self.circ_list, self.circ_coeff_list = grad_circuit_symbolic_forward(self.circ)

        assert len(self.circ_list)==len(self.k_list), '{} vs {}'.format(len(self.circ_list), len(self.k_list))

        
    def grad(self):
        if PHASE_SHIFT is None:
            raise ValueError()
        return self._grad(PHASE_SHIFT)
        
    def _grad(self, phase_shift):
        r'''
        calculate gradient using forwardMode, while also calculate Hessian with hybridMode
        '''
        jac = np.zeros(len(self.parameters)).astype(np.complex)
        hess = np.zeros((len(self.parameters), len(self.parameters))).astype(np.complex)

        for i, (circ_right, coeff) in enumerate(zip(self.circ_list, self.circ_coeff_list)):
            sim = Simulator('projectq', self.n_qubits)
            circ_right.no_grad()
            grad_ops = sim.get_expectation_with_grad(self.ham, circ_right, self.circ)
            e, g = grad_ops(self.parameters)
            jac[i] = e[0][0] * coeff #this is \partial E/ \partial circ_right
            hess[i] = g.squeeze() * coeff *(-1) * phase_shift#* J

        jac = jac * 2 * phase_shift #+ jac * J # add h.c.

        return jac.real, hess.real

    def Hess_forwardMode(self):
        r'''
        calculate Hessian using forward mode
        '''
        phase_shift = PHASE_SHIFT
            
        hess = np.zeros((len(self.parameters), len(self.parameters))).astype(np.complex)
        for i, (circ_left, coeff_left) in enumerate(zip(self.circ_list, self.circ_coeff_list)):
            for j, (circ_right, coeff_right) in enumerate(zip(self.circ_list, self.circ_coeff_list)):
                sim = Simulator('projectq', self.n_qubits)
                circ_right.no_grad()
                circ_left.no_grad()

                grad_ops = sim.get_expectation_with_grad(self.ham, circ_right, circ_left)
                e, g = grad_ops(self.parameters)
                hess[i][j] = e[0][0] * coeff_left * coeff_right * phase_shift * phase_shift #* J

        return hess.real

    def grad_reserveMode(self):
        r'''
        test method that generate gradient using backpropogation(reverse mode differentiation)
        '''
        sim = Simulator('projectq', self.n_qubits)
        grad_ops = sim.get_expectation_with_grad(self.ham, self.circ, self.circ)
        e, g = grad_ops(self.parameters)
        return g.squeeze().real


def get_phase_shift(self):
    jac, _ = self._grad(1)
    jac_reverse = self.grad_reserveMode()
    phase_shift = jac.real.mean() / jac_reverse.real.mean()
    phase_shift = phase_shift / np.abs(phase_shift)
    print('phase_shift', phase_shift)
    return phase_shift

def determine_phase_shift():
    n_qubits = 5
    circ, pr = example_circuit(n_qubits=n_qubits)
    ham = Ising_like_ham(n_qubits).local_Hamiltonian()

    class Phase_shift_getter(Grad):
        pass

    Phase_shift_getter.get_phase_shift = get_phase_shift
    ph = Phase_shift_getter(circ, pr, ham, n_qubits)
    return ph.get_phase_shift()

PHASE_SHIFT = determine_phase_shift()



class FisherInformation:
    def __init__(self, circ, pr, n_qubits):
        ham = QubitOperator('')
        ham = Hamiltonian(ham)
        self.G = Grad(circ, pr, ham, n_qubits)

        self.circ, self.pr, self.ham = circ, pr, ham
        self.n_qubits = n_qubits

    def gite_preconditional(self):
        jac, hess = self.G.grad()
        return hess

    def fisher_information(self):
        jac, hess = self.G.grad()
        jac_left = np.expand_dims(jac, axis=1)
        jac_right = np.expand_dims(jac, axis=0) * J

        matrix = hess - (jac_left * jac_right)
        matrix = 4 * matrix.real
        return matrix

    # def fisherInformation_builtin(self):
    #     sim = Simulator('projectq', self.n_qubits)
    #     # matrix = sim.get_fisher_information_matrix(circ, self.G.parameters, diagonal=False)
    #     matrix = sim.fisher_information_matrix(circ.get_cpp_obj(),
    #                                               circ.get_cpp_obj(hermitian=True),
    #                                               self.G.parameters,
    #                                               circ.params_name,
    #                                               False)
    #     return matrix

