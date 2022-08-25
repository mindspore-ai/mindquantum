from mindquantum import Circuit
from mindquantum.core import RZ, X, ISWAP, Power, apply, BARRIER
from mindquantum.core.parameterresolver import ParameterResolver
import math


def theta_operation(theta):
    """Implements the givens rotation with sqrt(iswap). theta is a string"""
    pr = ParameterResolver(theta)
    circuit = Circuit()
    circuit += Power(ISWAP, 0.5).on([0, 1])
    circuit += RZ(-pr).on(0)
    circuit += RZ(pr + math.pi).on(1)
    circuit += Power(ISWAP, 0.5).on([0, 1])
    circuit += RZ(math.pi).on(1)
    return circuit


def half_filling_ansatz_circuit(n_electrons, count):
    circuit = Circuit()
    for depth in range(n_electrons):
        n_ops = depth + 1  # the number of gate in each layer
        q = n_electrons - n_ops  # index of start qubit
        for op in range(n_ops):  # add n_ops theta_operation
            circuit += apply(theta_operation(f'theta_{count}'), [q, q+1])
            q += 2  # next theta_operation
            count += 1
        circuit += BARRIER

    for depth in range(n_electrons - 1):
        n_ops = n_electrons - depth - 1
        q = depth+1
        for op in range(n_ops):  # add n_ops theta_operation
            circuit += apply(theta_operation(f'theta_{count}'), [q, q+1])
            q += 2  # next theta_operation
            count += 1
        circuit += BARRIER
    return circuit


def non_half_filling_ansatz_circuit(n_qubits, n_electrons):
    circuit = Circuit()
    unoccupied_orbitals = n_qubits - n_electrons
    if n_electrons > unoccupied_orbitals:
        count = 1  # the number of theta_operation
        for depth in range(unoccupied_orbitals):
            n_ops = depth + 1  # the number of gate in each layer
            q = n_electrons - n_ops  # index of start qubit
            for op in range(n_ops):  # add n_ops theta_operation
                circuit += apply(theta_operation(f'theta_{count}'), [q, q + 1])
                q += 2  # next theta_operation
                count += 1
            circuit += BARRIER
        for depth in range(n_electrons - 1):
            n_ops = unoccupied_orbitals - depth - 1
            q = n_electrons - n_ops
            for op in range(n_ops):  # add n_ops theta_operation
                circuit += apply(theta_operation(f'theta_{count}'), [q, q + 1])
                q += 2  # next theta_operation
                count += 1
            circuit += BARRIER
        circuit += half_filling_ansatz_circuit(int(n_electrons/2), count)
    else:
        count = 1  # the number of theta_operation
        for depth in range(int(unoccupied_orbitals/2)):
            n_ops = depth + 1  # the number of gate in each layer
            q = n_electrons - n_ops + int(unoccupied_orbitals/2)  # index of start qubit
            for op in range(n_ops):  # add n_ops theta_operation
                circuit += apply(theta_operation(f'theta_{count}'), [q, q + 1])
                q += 2  # next theta_operation
                count += 1
            circuit += BARRIER
        for depth in range(int(unoccupied_orbitals/2) - 1):
            n_ops = int(unoccupied_orbitals/2) - 1 - depth
            q = int(unoccupied_orbitals/2) - n_ops + n_electrons
            for op in range(n_ops):  # add n_ops theta_operation
                circuit += apply(theta_operation(f'theta_{count}'), [q, q + 1])
                q += 2  # next theta_operation
                count += 1
            circuit += BARRIER
        circuit += half_filling_ansatz_circuit(n_electrons, count)
    return circuit


def ansatz_circuit(n_qubits, n_electrons):
    hartreefock_wfn_circuit = Circuit([X.on(i) for i in range(n_electrons)])
    if n_electrons == int(n_qubits/2):
        brc_ansatz_circuit = half_filling_ansatz_circuit(n_electrons, count=1)
    else:
        brc_ansatz_circuit = non_half_filling_ansatz_circuit(n_qubits, n_electrons)
    return hartreefock_wfn_circuit + brc_ansatz_circuit


if __name__ == '__main__':
    n_qubits = 10
    n_electrons = 5

    total_circuit = ansatz_circuit(n_qubits, n_electrons)
    print(total_circuit)
    total_circuit.summary()
