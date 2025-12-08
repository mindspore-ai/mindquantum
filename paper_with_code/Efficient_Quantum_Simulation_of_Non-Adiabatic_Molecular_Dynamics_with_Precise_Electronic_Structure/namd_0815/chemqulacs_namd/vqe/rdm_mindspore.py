# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from itertools import product
from time import time

import numpy as np
from openfermion.ops import FermionOperator
from openfermion.transforms import jordan_wigner

from mindquantum.core.operators import QubitOperator as QubitOperator_ms
from mindquantum.core.operators import Hamiltonian as Hamiltonian_ms
from mindquantum.simulator import Simulator


def get_1rdm(circuit, params, n_qubits, n_electrons, sz=0):
    """
    Compute 1-RDM of a given state using MindQuantum simulator.

    Args:
        circuit: The quantum circuit (MindQuantum circuit).
        params: The parameters for the circuit.
        n_qubits (int): Number of qubits (spin-orbitals).
        n_electrons (int)
        sz (float, optional): Defaults to 0.

    Return:
        numpy.ndarray of shape (n_qubits, n_qubits):
            1-RDM
    """
    ret = np.zeros((n_qubits, n_qubits), dtype=np.complex128)
    for i in range(n_qubits):
        for j in range(i + 1):
            one_body_op = FermionOperator(((i, 1), (j, 0)))
            qubit_op = jordan_wigner(one_body_op)
            ms_qubit_op = QubitOperator_ms.from_openfermion(qubit_op)
            ham = Hamiltonian_ms(ms_qubit_op)
            
            sim = Simulator('mqvector', n_qubits)
            sim.apply_circuit(circuit, pr=params)
            tmp = sim.get_expectation(ham)
            
            ret[i, j] = tmp
            ret[j, i] = np.conj(tmp)
    return ret


def get_2rdm(circuit, params, n_qubits, n_electrons, sz=0):
    """
    Compute 2-RDM of a given state using MindQuantum simulator.

    Args:
        circuit: The quantum circuit (MindQuantum circuit).
        params: The parameters for the circuit.
        n_qubits (int): Number of qubits (spin-orbitals).
        n_electrons (int)
        sz (float, optional): Defaults to 0.

    Return:
        numpy.ndarray of shape (n_qubits, n_qubits, n_qubits, n_qubits):
            2-RDM
    """
    ret = np.zeros(
        (n_qubits, n_qubits, n_qubits, n_qubits),
        dtype=np.complex128,
    )
    
    start = time()
    
    for i, k in product(range(n_qubits), range(n_qubits)):
        for j, l in product(range(i), range(k)):
            two_body_op = FermionOperator(((i, 1), (j, 1), (k, 0), (l, 0)))
            qubit_op = jordan_wigner(two_body_op)
            ms_qubit_op = QubitOperator_ms.from_openfermion(qubit_op)
            ham = Hamiltonian_ms(ms_qubit_op)
            
            sim = Simulator('mqvector', n_qubits)
            sim.apply_circuit(circuit, pr=params)
            tmp = sim.get_expectation(ham)
            
            ret[i, j, k, l] = tmp
            ret[i, j, l, k] = -tmp
            ret[j, i, l, k] = tmp
            ret[j, i, k, l] = -tmp
            ret[l, k, j, i] = np.conj(tmp)
            ret[k, l, j, i] = -np.conj(tmp)
            ret[k, l, i, j] = np.conj(tmp)
            ret[l, k, i, j] = -np.conj(tmp)
    
    end = time()

    
    return ret