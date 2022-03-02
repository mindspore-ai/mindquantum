import numpy as np
from mindquantum import Circuit, RY, RX, RZ
from mindquantum import X, Z, Y


from mindquantum.simulator import Simulator
from mindquantum.core import ParameterResolver
from mindquantum.core.parameterresolver import ParameterResolver as PR
import math

class Parameter_manager:
    def __init__(self, key='default'):
        self.parameters = []
        self.count = 0
        self.key = key
        self.grad_key = None
    
    def init_parameter_resolver(self):
        pr = {k:np.random.randn()*2*math.pi for k in self.parameters}
        # pr = {k:0 for k in self.parameters}
        pr = ParameterResolver(pr)
        return pr

    def _replay(self):
        self.count = 0

    def set_grad_key(self, key):
        self.grad_key = key
        self._replay()    

    def create(self):
        param = '{}_theta_{}'.format(self.key, self.count)
        self.count += 1
        self.parameters.append(param)
        if self.grad_key is None or param!=self.grad_key:
            is_grad = False
        else:
            is_grad = True
        return param, is_grad


def RY_gate(circ, i, P):
    ry, is_grad = P.create()
    if not is_grad:
        circ += RY(ry).on(i)
    else:
        circ += Y.on(i)
        circ += RY(ry).on(i)

def RX_gate(circ, i, P):
    rx, is_grad = P.create()
    if not is_grad:
        circ += RX(rx).on(i)
    else:
        circ += X.on(i)
        circ += RX(rx).on(i)

def RZ_gate(circ, i, P):
    rz, is_grad = P.create()
    if not is_grad:
        circ += RZ(rz).on(i)
    else:
        circ += Z.on(i)
        circ += RZ(rz).on(i)

def RZZ_gate(circ, i, j, P):
    circ += X.on(j, i)
    RZ_gate(circ, j, P)
    circ += X.on(j, i)

def RYY_gate(circ, i, j, P):
    circ += X.on(j, i)
    RY_gate(circ, j, P)
    circ += X.on(j, i)

def RXX_gate(circ, i, j, P):
    circ += X.on(j, i)
    RX_gate(circ, j, P)
    circ += X.on(j, i)


def layer(C, P, n_qubits):
    for i in range(n_qubits):
        # RZ_gate(C, i, P)
        # RY_gate(C, i, P)
        RX_gate(C, i, P)
    #     RZ_gate(C, i, P)

    for i in range(0, n_qubits-1, 2):
        RZZ_gate(C, i, i+1, P)
    #     RZZ_gate(C, i, i+1, P)
    #     RZZ_gate(C, i, i+1, P)

    # for i in range(1, n_qubits-1, 2):
    #     RZZ_gate(C, i, i+1, P)
    #     RZZ_gate(C, i, i+1, P)
    #     RZZ_gate(C, i, i+1, P)




from .utils import pr2array
J = np.complex(0,1)
class Gradient_test:
    def __init__(self, n_qubits):
        self.n_qubits = n_qubits
        self.P = Parameter_manager()
        self.circ = Circuit()
        layer(self.circ, self.P, self.n_qubits)
        self.pr = self.P.init_parameter_resolver()
        # _, self.k_list = pr2array(self.pr)
        self.phase_shift = None
        
    def determine_phase_shift(self, ham):
        # self.phase_shift = -1
        jac, _ = self._gradient(ham, 1)
        jac_reverse = self.grad_reserveMode(ham)
        self.phase_shift = jac.mean() / jac_reverse.mean()
        self.pahse_shift = self.phase_shift / np.abs(self.phase_shift)
        
    def gradient(self, ham):
        if self.phase_shift is None:
            self.determine_phase_shift(ham)
        return self._gradient(ham, self.phase_shift)
        
    def _gradient(self, ham, phase_shift):
        coeff = (-1./2) * J
        parameters, k_list = pr2array(self.pr)
        jac = np.zeros(len(parameters)).astype(np.complex)
        hess = np.zeros((len(parameters), len(parameters))).astype(np.complex)

        for i, ki in enumerate(k_list):

            self.P.set_grad_key(ki)
            circ_right = Circuit()
            layer(circ_right, self.P, self.n_qubits)

            sim = Simulator('projectq', self.n_qubits)
            circ_right.no_grad()

            grad_ops = sim.get_expectation_with_grad(ham, circ_right, self.circ)
            e, g = grad_ops(parameters)

            jac[i] = e[0][0] * coeff #this is \partial E/ \partial circ_right
            hess[i] = g.squeeze() * coeff * (-1) * phase_shift #* J

        jac = jac * 2 * phase_shift #+ jac * J # add h.c.
        return jac.real, hess.real

    
    def grad_reserveMode(self, ham):
        r'''
        test method that generate gradient using backpropogation(reverse mode differentiation)
        '''
        parameters, k_list = pr2array(self.pr)
        
        sim = Simulator('projectq', self.n_qubits)
        grad_ops = sim.get_expectation_with_grad(ham, self.circ, self.circ)
        e, g = grad_ops(parameters)
        return g.squeeze().real