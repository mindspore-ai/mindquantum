"""
VD.py - Virtual Distillation Algorithm Implementation
=====================================================

This module implements the virtual distillation algorithm for error mitigation in quantum circuits.

Example:
    >>> from mindquantum.core.gates import AmplitudeDampingChannel, H, CNOT, RX, Z, RY
    >>> from mindquantum.core.circuit.channel_adder import BitFlipAdder
    >>> from mindquantum.core.circuit import Circuit
    >>> from mindquantum.simulator import Simulator
    >>> from mindquantum.core.operators import Hamiltonian, QubitOperator
    >>> import numpy as np

    >>>    circ = Circuit()
    >>>    circ += RY(1).on(0)
    >>>    from mindquantum.core.gates import BitFlipChannel
    >>>    noise_circ = circ + BitFlipChannel(0.005).on(0)
    >>>    position = [0]
    >>>    vd_calculator = VDCalculator(noise_circ, position, M=3, nshots=100000, para=None)
    >>>    result = vd_calculator.calculate()
    >>>    print(result)


Note:
    This module assumes that the input observable operator is sigma_z.
"""

from mindquantum.core.gates import SWAP, Givens
from mindquantum.core.circuit import Circuit, apply
from mindquantum.simulator import Simulator
import numpy as np
from mindquantum.core.gates import  RY
from mindquantum.core.gates import BitFlipChannel

class VDCalculator:
    """
    VDCalculator class for virtual distillation.

    Attributes:
        noise_circ: The circuit of the noisy state.
        position: The position that expect operator act on.
        Mp: Power.
        nshots: The shots of the sampling.
        para: The parameters for the parameters circuit if it has.
        noise_circ.n_qubits: The qubit number of the origin circuit.
        self.prepare_circuit(): Extended circuit based on power generation.

    Methods:
        __init__(Mp, noise_circ): Initializes the VDCalculator instance.
        prepare_circuit(): Prepares the circuit for virtual distillation.
        product_swap(self, res_l, res_r): Calculates the expectation of Swap gates.
        product_with_z(self, res_l, res_r): Calculates the expectation of Sigma_Z.
        calculate(): Calculates the virtual distillation result.
    """
    def __init__(self, noise_circ,  position, Mp=2, nshots=1000000, para=None):
        self.noise_circ = noise_circ            # The circuit of the noisy state
        self.position = position                # The position that expect operator act on
        self.Mp = Mp                             # Power
        self.nshots = nshots                    # The shots of the sampling
        self.para = para                        # The parameters for the parameters circuit if it has
        self.nqubits = noise_circ.n_qubits      # The qubit number of the origin circuit
        self.circ = self.prepare_circuit()      # Extended circuit based on power generation

    def prepare_circuit(self):
        '''
        The purpose of this function is to construct a direct product state in terms of power.
        '''
        circ = Circuit()
        circ += apply(self.noise_circ, [i for i in range(self.nqubits)])
        circ = circ.remove_measure()

        if self.para is not None:
            para_name = circ.params_name
            pr = {key: value for key, value in zip(para_name, self.para)}
            circ = circ.apply_value(pr)

        temp = Circuit()
        for i in range(self.Mp):
            temp += apply(circ, [i * self.nqubits + j for j in range(self.nqubits)])
        return temp

    def product_swap(self, res_l, res_r):
        '''
        Calculate the trace of the density matrix to the M power.
        '''
        pr_l = res_l.data
        qvec_l = np.zeros(2 ** (self.nqubits * self.Mp))
        for key in pr_l:
            qvec_l[int(key, 2)] = np.sqrt(pr_l[key] / self.nshots)

        pr_r = res_r.data
        qvec_r = np.zeros(2 ** (self.nqubits * self.Mp))

        for key in pr_r:
            qvec_r[int(key, 2)] = np.sqrt(pr_r[key] / self.nshots)

        for key in pr_r:
            for j in range(self.Mp-1):
                for i in range(self.nqubits):
                    if (int(key[i+j*self.nqubits], 2) == 1 and int(key[i + (j+1)*self.nqubits], 2) == 0):
                        qvec_r[int(key, 2)] = - qvec_r[int(key, 2)]

        return np.inner(qvec_l, qvec_r)

    def product_with_z(self, res_l, res_r, position):
        '''
        Calculate the expectation with observable Pauli_Z.
        '''
        pr_l = res_l.data
        qvec_l = np.zeros(2 ** (self.nqubits * self.Mp))
        for key in pr_l:
            qvec_l[int(key, 2)] = np.sqrt(pr_l[key] / self.nshots)

        pr_r = res_r.data
        qvec_r = np.zeros(2 ** (self.nqubits * self.Mp))

        for key in pr_r:
            qvec_r[int(key, 2)] = np.sqrt(pr_r[key] / self.nshots)

        for key in pr_r:
            for j in range(self.Mp-1):
                for i in range(self.nqubits):
                    if (int(key[i+j*self.nqubits], 2) == 1 and int(key[i + (j+1)*self.nqubits], 2) == 0):
                        qvec_r[int(key, 2)] = - qvec_r[int(key, 2)]

        for key in pr_r:
            for i in range(len(position)):
                if int(key[position[i]]) == 1:
                    qvec_r[int(key, 2)] = -qvec_r[int(key, 2)]
        return np.inner(qvec_l, qvec_r)

    def calculate(self):
        """
        Calulate the result of VD by using the sampling result.
        """
        circ_swap = apply(self.circ, [i for i in range(self.circ.n_qubits)])
        circ_copy = apply(self.circ, [i for i in range(self.circ.n_qubits)])

        for i in range(self.Mp - 1):
            for j in range(self.nqubits):
                circ_copy += Givens(np.pi / 4).on([j + i * self.nqubits, j + (i + 1) * self.nqubits])

        for i in range(self.Mp - 1):
            for j in range(self.nqubits):
                circ_swap += Givens(np.pi / 4).on([j + i * self.nqubits, j + (i + 1) * self.nqubits])
                circ_swap += SWAP([j + i * self.nqubits, j + (i + 1) * self.nqubits])

        sim = Simulator('mqvector', self.circ.n_qubits)
        sim.reset()
        res = sim.sampling(circ_copy.measure_all(), shots=self.nshots, seed=42)

        sim.reset()
        circ_swap = circ_swap.remove_measure()
        res_s = sim.sampling(circ_swap.measure_all(), shots=self.nshots, seed=42)


        trpm = self.product_swap(res, res_s)
        print(trpm)
        frac = 0

        for t in range(self.Mp):
            circ_o_swap = Circuit()
            circ_o_swap += apply(circ_swap, [i for i in range(circ_swap.n_qubits)])

            sim.reset()
            res_r = sim.sampling(circ_o_swap, shots=self.nshots)
            sim.reset()
            pos = [self.position[i] + t * self.nqubits for i in range(len(self.position))]
            frac += self.product_with_z(res_s, res_r, pos)

        tropm = frac / self.Mp
        exp = tropm / trpm
        return exp

# 使用示例
# noise_circ, O, position, M, nshots, para 需要根据实际情况定义
# vd_calculator = VDCalculator(noise_circ, O, position, M, nshots, para)
# result = vd_calculator.calculate()
# print(result)

circuit = Circuit()
# circ += RX(1).on(0)
# circ += CNOT.on(1,0)
circuit += RY(1).on(0)


noise_c = circuit + BitFlipChannel(0.005).on(0)


posi = [0]

vd_calculator = VDCalculator(noise_c, posi, Mp=3, nshots=100000, para=None)
result = vd_calculator.calculate()
print(result)
