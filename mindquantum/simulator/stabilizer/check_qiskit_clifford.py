import numpy as np
from qiskit import *
from qiskit.quantum_info import Clifford as qsClifford
#from qiskit.quantum_info import decompose_clifford
from qiskit.synthesis import synth_clifford_full
from qiskit.quantum_info import Statevector

def check_qiskit_clifford(num_qubits=2):
    # generate clifford and try decomposition
    qc = QuantumCircuit(2)
    qc.s(0)
    #qc.cx(0,1)
    qc.s(1)
    qc.h(1)
    qc.cx(0,1)
    cli = qsClifford(qc)
    qc_decomp1 = synth_clifford_full(cli, method='AG')
    qc_decomp2 = synth_clifford_full(cli, method='greedy')
    qc_decomp3 = synth_clifford_full(cli, method=None) # Bravyi & Maslov (n<=3)

    # check circuit
    print("original circuit:")
    print(qc.draw())
    print("AG decomposition:")
    print(qc_decomp1.draw())
    print("greedy decomposition:")
    print(qc_decomp2.draw())
    print("Bravyi & Maslov decomposition:")
    print(qc_decomp3.draw())

    # check tableau
    cli_decomp1 = qsClifford(qc_decomp1)
    cli_decomp2 = qsClifford(qc_decomp2)
    cli_decomp3 = qsClifford(qc_decomp3)
    assert np.all(cli.tableau==cli_decomp1.tableau)
    assert np.all(cli.tableau==cli_decomp2.tableau)
    assert np.all(cli.tableau==cli_decomp3.tableau)

    # check matrix
    mat0 = get_qiskit_matrix(qc)
    mat1 = get_qiskit_matrix(qc_decomp1)
    mat2 = get_qiskit_matrix(qc_decomp2)
    mat3 = get_qiskit_matrix(qc_decomp3)
    print(mat0)
    print(mat1)
    print(mat2)
    print(mat3)
    #assert np.all(mat0==mat1)
    #assert np.all(mat0==mat2)
    #assert np.all(mat0==mat3)
    #assert np.all(mat1==mat2)
    
    # check state
    state0 = Statevector(qc)  # inner product < psi1 | psi2 >
    state1 = Statevector(qc_decomp1)
    state2 = Statevector(qc_decomp2)
    state3 = Statevector(qc_decomp3)
    print(state0)
    print(state1)
    print(state2)
    print(state3)
    print(state0.probabilities())
    print(state1.probabilities())
    print(state2.probabilities())
    print(state3.probabilities())


def get_qiskit_matrix(qc):
    # get unitary matrix of quantum circuits
    backend = BasicAer.get_backend('unitary_simulator')
    job = backend.run(transpile(qc, backend))
    return job.result().get_unitary(qc, decimals=6)


if __name__=="__main__":
    check_qiskit_clifford()


