import numpy as np
from mindquantum import Circuit, Simulator, H, CNOT


def purity(density_matrix):
    """
    :param density_matrix: reduced density matrix to compute purity;
    :return: purity = Tr(rho^2).
    """
    rho2 = np.matmul(density_matrix, density_matrix)
    return np.trace(rho2).real


def s2(density_matrix):
    """
    :param density_matrix: reduced density matrix to compute entropy;
    :return: second Renyi entropy.
    """
    rho2 = np.matmul(density_matrix, density_matrix)
    return -np.log(np.trace(rho2)).real


def get_rs(rho: np.array, list_qubits2keep: list) -> np.array:
    """
    :param rho: density matrix of whole circuit, rho=|psi><psi|, rank(rho) = 1;

    :param list_qubits2keep: containing the indices of qubits to keep
        For example, 5 qubits with indices 0, 1, 2, 3, 4 from top to bottom:

            q0: ─────RX──────

            q1: ─────RX──────

            q2: ─────RX──────

            q3: ─────RX──────

            q4: ─────RX──────

            if I want to get the reduced state of first 2 qubits, then list_qubits2keep = [0, 1];

    :return: reduced density matrix
    """
    list_qubits2traceout = list(
        set(range(int(np.log2(len(rho))))) - set(list_qubits2keep))
    counter_qubits = int(np.log2(len(rho)))
    rho_tensor = rho.reshape((2, 2) * counter_qubits)
    for i in list_qubits2traceout:
        index2trace = i - list_qubits2traceout.index(i)
        rho_tensor = np.trace(rho_tensor,
                              axis1=counter_qubits - 1 - index2trace,
                              axis2=2 * counter_qubits - 1 - index2trace)
        counter_qubits += -1
    rho_rs = rho_tensor.reshape(2**counter_qubits, 2**counter_qubits)
    return rho_rs


def get_rs_from_sim(sim_, list_qubits2keep):
    qs = sim_.get_qs()
    rho = np.outer(qs, qs.conjugate())
    reduced_state = get_rs(rho, list_qubits2keep)
    return reduced_state


def s_page(n_subsys, n_sys) -> float:
    """
    Get the approximate page entropy in the limit dim_sys >> dim_subsys.
    :param n_subsys: the number of qubits of subsystem;
    :param n_sys: the number of qubits of system;
    :return: the page entropy.
    """
    k = n_subsys
    N = n_sys
    return k * np.log(2) - 1 / 2**(N - 2 * k + 1)


if __name__ == '__main__':
    circ = Circuit([H.on(0), CNOT.on(1, 0)])
    sim = Simulator('mqvector', 2)
    sim.apply_circuit(circ)
    rs = get_rs_from_sim(sim, [0])
    print(circ)
    print('this is the 2nd Renyi entropy:', s2(rs))
    print('this is the page entropy:', s_page(1, 2))
    print('this is purity:', purity(rs))
    print('test over')
