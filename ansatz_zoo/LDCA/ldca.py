from openfermion.chem import MolecularData
from mindquantum.core import FermionOperator
from mindquantum.algorithm import Transform
from mindquantum.third_party.interaction_operator import InteractionOperator
from mindquantum.core import get_fermion_operator

from mindquantum.core.operators import QubitOperator
from mindquantum.core.operators import TimeEvolution
from mindquantum.core.circuit import apply, Circuit
from mindquantum.core.gates import RZ, X


def ldca_generator_circuit(mol, L=2):
    n_qubits = mol.n_qubits
    n_electrons = mol.n_electrons

    def R_i_j_a_b(i: int, j: int, sigma1: str, sigma2: str, pr):
        if sigma1 not in "XYZ" or sigma2 not in "XYZ":
            raise ValueError
        op = QubitOperator(f"{sigma1}0 {sigma2}1", pr)
        u = TimeEvolution(op).circuit
        circ = apply(u, [i, j])
        return circ
    
    def r_double_gate(i: int, j: int, name: str):
        sign = 1
        if name[0] == "-":
            sign = -1
            name = name[1:]
        return R_i_j_a_b(i, j, name[0], name[1], {name : sign})
    
    def G_i_j_k(i: int, j: int, k: int):
        circ = Circuit()
        circ += r_double_gate(i, j, f"-YX({k})")
        circ += r_double_gate(i, j, f"XY({k})")
        circ += r_double_gate(i, j, f"-YY({k})")
        circ += r_double_gate(i, j, f"XX({k})")
        return circ
    
    def U_MG_k(n_qubits: int, k: int):
        circ = Circuit()
        for i in range(n_qubits // 2):
            circ += G_i_j_k(2 * i, 2 * i + 1, k)
        for i in range(n_qubits // 2 - 1):
            circ += G_i_j_k(2 * i + 1, 2 * i + 2, k)
        return circ
    
    def U_MG_NN(n_qubits: int):
        circ = Circuit()
        for k in range(n_qubits // 2):
            circ += U_MG_k(n_qubits, k)
        return circ
    
    def U_Bog(n_qubits: int):
        circ = Circuit()
        for i in range(n_qubits):
            circ += RZ(f"theta_{i}^AB").on(i)
        circ += U_MG_NN(n_qubits)
        return circ
    
    def K_i_j_k_l(i: int, j: int, k: int, l: int):
        circ = Circuit()
        circ += r_double_gate(i, j, f"-YX({k},{l})")
        circ += r_double_gate(i, j, f"XY({k},{l})")
        circ += r_double_gate(i, j, f"ZZ({k},{l})")
        circ += r_double_gate(i, j, f"-YY({k},{l})")
        circ += r_double_gate(i, j, f"XX({k},{l})")
        return circ
    
    def U_VarMG_k_l(n_qubits: int, k: int, l: int):
        circ = Circuit()
        for i in range(n_qubits // 2):
            circ += K_i_j_k_l(2 * i, 2 * i + 1, k, l)
        for i in range(n_qubits // 2 - 1):
            circ += K_i_j_k_l(2 * i + 1, 2 * i + 2, k, l)
        return circ
    
    def U_VarMG_l(n_qubits: int, l: int):
        circ = Circuit()
        for k in range(n_qubits // 2):
            circ += U_VarMG_k_l(n_qubits, k, l)
        return circ
    
    def U_VarMG(n_qubits: int, L: int):
        circ = Circuit()
        for i in range(n_qubits):
            circ += RZ(f"theta_{i}^Z").on(i)
        for l in range(L):
            circ += U_VarMG_l(n_qubits, l)
        return circ
    
    def LDCA(n_qubits: int, L: int):
        circ = Circuit()
        # for i in range(n_qubits):
            # circ += X.on(i)
        circ += U_VarMG(n_qubits, L)
        circ += U_Bog(n_qubits).hermitian()
        return circ
    
    circ = LDCA(n_qubits, L)
    return circ


def generate_ldca(molecular, L=2):
    if isinstance(molecular, str):
        mol = MolecularData(filename=molecular)
        mol.load()
    else:
        mol = molecular
    
    ansatz = ldca_generator_circuit(mol, L)

    ham_of = mol.get_molecular_hamiltonian()
    inter_ops = InteractionOperator(*ham_of.n_body_tensors.values())
    ham_hiq = get_fermion_operator(inter_ops)
    qubit_hamiltonian = Transform(ham_hiq).jordan_wigner()
    qubit_hamiltonian.compress()

    parameters_name = ansatz.params_name
    return ansatz, \
        parameters_name, \
        qubit_hamiltonian, \
        mol.n_qubits, \
        mol.n_electrons