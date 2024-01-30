import numpy as np
from qiskit import *
from qiskit.quantum_info import Clifford as qsClifford
#from qiskit.quantum_info import decompose_clifford
from qiskit.synthesis import synth_clifford_full
from qiskit.quantum_info import Statevector

def check_qiskit_expval(num_qubits=2):
    # generate clifford and try expectation on Pauli string
    qc = QuantumCircuit(2)
    qc.s(0)
    #qc.cx(0,1)
    qc.s(1)
    qc.h(1)
    qc.cx(0,1)
    
    for i in range(8):
        x = np.random.randint(6)
        if   x==0: qc.cx(0,1)
        elif x==1: qc.cx(1,0)
        elif x==2: qc.h(0)
        elif x==3: qc.h(1)
        elif x==4: qc.s(0)
        elif x==5: qc.s(1)

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

    # check pauli expectation  < psi | P | psi >
    P = 'ZX'
    expval0 = get_expval_pauli(cli, pauli=P)
    expval1 = get_expval_pauli(cli_decomp1, pauli=P)
    expval2 = get_expval_pauli(cli_decomp2, pauli=P)
    expval3 = get_expval_pauli(cli_decomp3, pauli=P)
    assert expval0==expval1
    assert expval0==expval2
    assert expval0==expval3

    #expval0p = get_expval_inner(qc, pauli=P)
    #assert np.abs(expval0 - expval0p)<1e-5

    for P in ['II','IX','IY','IZ',
              'XI','XX','XY','XZ',
              'YI','YX','YY','YZ',
              'ZI','ZX','ZY','ZZ']:
        expval0 = get_expval_pauli(cli, pauli=P) # by tableau
        expval0p = get_expval_inner(qc, pauli=P) # by state vector
        try:
            assert np.abs(expval0 - expval0p)<1e-5
        except:
            print(f"<psi|{P}|psi> Failed!!! exp_tableau={expval0} while exp_state={expval0p}")


def apply_pauli(qc, pauli):
    # apply pauli to quantum circuit
    qc1 = qc.copy()
    for ip in range(len(pauli)):
        if pauli[ip]=='X': qc1.x(ip)
        elif pauli[ip]=='Y': qc1.y(ip)
        elif pauli[ip]=='Z': qc1.z(ip)
    return qc1

def get_expval_inner(qc, pauli):
    # < psi | P | psi > by inner product
    qcp = apply_pauli(qc, pauli)
    psi1 = Statevector(qc)
    psi2 = Statevector(qcp)
    return np.round(psi1.inner(psi2).real, decimals=6)

def get_expval_pauli(cli, num_qubits=2, qubits=[0,1], pauli='ZZ'):
    # < psi | P | psi >
    #   see: https://github.com/Qiskit/qiskit-aer/blob/main/src/simulators/stabilizer/clifford.hpp#L642
    if qubits is None:
        assert len(pauli)==num_qubits
        qubits = [i for i in range(num_qubits)]
    n = num_qubits
    P = np.zeros(2*n, dtype=bool)
    phase = 0
    for i in range(len(qubits)):
        if   pauli[i]=='X': P[qubits[i]]=1
        elif pauli[i]=='Z': P[qubits[i]+n]=1
        elif pauli[i]=='Y': P[qubits[i]]=1; P[qubits[i]+n]=1; phase+=1

    stabilizer_table_X = cli.tableau[n:,:n]
    stabilizer_table_Z = cli.tableau[n:,n:-1]
    destabilizer_table_X = cli.tableau[:n,:n]
    destabilizer_table_Z = cli.tableau[:n,n:-1]
    stabilizer_phase = cli.tableau[n:,-1]
    destabilizer_phase = cli.tableau[:n,-1]

    for i in range(num_qubits):
        num_anti = 0
        for qubit in qubits:
            if P[qubit+n] and stabilizer_table_X[i,qubit] :
                num_anti += 1
            if P[qubit] and stabilizer_table_Z[i,qubit] :
                num_anti += 1
        if num_anti%2==1:
            return 0.0

    PZ = P[n:]
    for i in range(num_qubits):
        num_anti = 0
        for qubit in qubits:
            if P[qubit+n] and destabilizer_table_X[i,qubit] :
                num_anti += 1
            if P[qubit] and destabilizer_table_Z[i,qubit] :
                num_anti += 1
        if num_anti%2==0: continue

        phase += 2 * stabilizer_phase[i]
        for k in range(num_qubits):
            phase += stabilizer_table_Z[i,k] & stabilizer_table_X[i,k]
            phase += 2 * (PZ[k] & stabilizer_table_X[i,k])
            PZ[k] = PZ[k] ^ stabilizer_table_Z[i,k]

    return -1.0 if phase%4 else 1.0
            

if __name__=="__main__":
    check_qiskit_expval()


