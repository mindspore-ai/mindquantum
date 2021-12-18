
from mindquantum.framework import MQLayer
from mindquantum.core import Z, H, I, RX, RY, RZ, X, UN, Circuit, Hamiltonian, QubitOperator
from mindquantum import ParameterResolver
from mindquantum.simulator import Simulator
import numpy as np
import struct, scipy, math

def controlled_gate(circuit, gate, tqubit, cqubits, zero_qubit):
    tmp = []
    for i in range(len(cqubits)):
        tmp.append(cqubits[i])
        if cqubits[i] < 0 or (cqubits[i] == 0 and zero_qubit == 0):
            circuit += X.on(abs(cqubits[i]))
            tmp[i] = -tmp[i]
    
    circuit += gate.on(tqubit, tmp)

    for i in range(len(cqubits)):
        if cqubits[i] < 0 or (cqubits[i] == 0 and zero_qubit == 0):
            circuit += X.on(abs(cqubits[i]))

def encoding(x):
    '''
    intput: x is an array, whose length is 16 * 16, representing the MNIST pixels value
    
    output: c is a parameterizd circuit, and num is a parameter resolver
            
            simulator.apply_circuit(c, num).get_qs(True) will get a qubit state:
                    x0|00000000> + x1|10000000> + x2|01000000> + x3|11000000> + ... + x254|01111111> + x255|11111111>
            which is an "amplitudes encoding" for classic data (x1, ..., x255)

    '''
    c = Circuit()
    tree = []
    for i in range(len(x) - 1):
        tree.append(0)
    for i in range(len(x)):
        tree.append(x[i])
    for i in range(len(x) - 2, -1 ,-1):
        tree[i] += math.sqrt(tree[i * 2 + 1] * tree[i * 2 + 1] + tree[i * 2 + 2] * tree[i * 2 + 2])
    
    path = [[]]
    num = {}
    cnt = 0
    for i in range(1, 2 * len(x) - 1, 2):
        path.append(path[(i - 1) // 2] + [-1])
        path.append(path[(i - 1) // 2] + [1])
        
        tmp = path[(i - 1) // 2]
        controlls = []
        for j in range(len(tmp)):
            controlls.append(tmp[j] * j)
        theta = 0
        if tree[(i - 1) // 2] > 1e-10:
            amp_0 = tree[i] / tree[(i - 1) // 2]
            theta = 2 * math.acos(amp_0)
        num[f'alpha{cnt}'] = theta
        controlled_gate(c, RY(f'alpha{cnt}'), len(tmp), controlls, (0 if len(tmp) > 0 and tmp[0] == -1 else 1))
        cnt += 1

    return c, ParameterResolver(num)

class AmplitudeEncoder:
    '''
    Example:
    --------------------------
    en = AmplitudeEncoder(8)
    x = [math.sqrt(0.5), math.sqrt(0.5)]
    sim = Simulator('projectq', 8)
    sim.apply_circuit(en.circuit(), en.parameterResolver(x))
    print(sim.get_qs(True))  
    --------------------------
    output: sqrt(0.5)|00000000> + sqrt(0.5)|10000000>
    
    '''

    def circuit(self):
        return self._circuit
    
    def parameterResolver(self, params):
    
        while len(params) < (2 ** self.n_qubits):
            params.append(0)
        if len(params) > (2 ** self.n_qubits):
            params = params[ : (2 ** self.n_qubits)]
        encoder, parameterResolver = encoding(params)
        return parameterResolver

    def __init__(self, n_qubits):
        self.n_qubits = n_qubits
        encoder, params = encoding([0 for i in range(2 ** n_qubits)])
        self._circuit = encoder
