# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Stabilizer tableau class. A binary table represents a stabilizer circuit.
Reference:
    S. Aaronson, D. Gottesman, Improved Simulation of Stabilizer Circuits,
        Phys. Rev. A 70, 052328 (2004).  arXiv:quant-ph/0406196
"""
import numpy as np
from stabilizer_tableau import StabilizerTableau

try:
    from mindquantum import *
except:
    raise ImportError("mindquantum is NOT implemented !!!")


class Clifford(StabilizerTableau):
    def __init__(self, table=None, phase=None, num_qubits=None):
        super().__init__(table=table, phase=phase, num_qubits=num_qubits)
        # validation check, not all circuit is Clifford
        if not (table is None):
            if not self.is_valid():
                raise ValueError('Error! table should be full rank.')

    def is_valid(self):
        return np.linalg.matrix_rank(self._table) == 2 * self.num_qubits

    def copy(self):
        newcli = Clifford(num_qubits=self.num_qubits)
        newcli._table = self._table.copy()
        newcli._phase = self._phase.copy()
        return newcli

    def clifford_decomposition(self, method='AG', output='string'):
        circuit_str = []
        n = self.num_qubits
        cli_copy = self.copy()
        if method == 'AG':
            """
            Aaronson-Gottesman method
            """
            for i in range(n):
                ######  step01: make A full rank  ######
                # if Aii=1, do nothing
                flag_Aii_true = 0
                if cli_copy._table[i, i]:
                    flag_Aii_true = 1
                # if Aii=0, find X_ij=1 and use CNOT(j,i)
                for j in range(i + 1, n):
                    if flag_Aii_true:
                        break
                    if cli_copy._table[i, j]:
                        cli_copy.CNOT(j, i)
                        print("cnot 1")
                        circuit_str.append(('C', j, i))
                        flag_Aii_true = 1
                # if all X_ij=0, find X_i(j+n)=1 and use Hadamard(j)+CNOT(j,i)
                for j in range(i, n):
                    if flag_Aii_true:
                        break
                    if cli_copy._table[i, j + n]:
                        cli_copy.Hadamard(j)
                        circuit_str.append(('H', j))
                        if j != i:
                            cli_copy.CNOT(j, i)
                            print("cnot 2")
                            circuit_str.append(('C', j, i))
                        flag_Aii_true = 1

                ######  step02: make A a lower triangular  ######
                for j in range(i + 1, n):
                    if cli_copy._table[i, j]:
                        cli_copy.CNOT(i, j)
                        print("cnot 3")
                        circuit_str.append(('C', i, j))

                ######  step03: make B a lower triangular  ######
                if np.any(cli_copy._table[i, (i + n) :]):
                    print("aaa")
                    if not cli_copy._table[i, i + n]:
                        cli_copy.PhaseGate(i)
                        circuit_str.append(('P', i))
                    for j in range(i + 1, n):
                        if cli_copy._table[i, j + n]:
                            cli_copy.CNOT(j, i)
                            print("cnot 4")
                            circuit_str.append(('C', j, i))
                    cli_copy.PhaseGate(i)
                    circuit_str.append(('P', i))

                ######  step04: make D a lower triangular  ######
                if np.any(cli_copy._table[i + n, (i + n + 1) :]):
                    for j in range(i + 1, n):
                        if cli_copy._table[i + n, j + n]:
                            cli_copy.CNOT(j, i)
                            print("cnot 5")
                            circuit_str.append(('C', j, i))

                ######  step05: make C a lower triangular  ######
                cli_copy.print_tableau()
                if np.any(cli_copy._table[i + n, (i + 1) :]):
                    cli_copy.Hadamard(i)
                    circuit_str.append(('H', i))
                    cli_copy.print_tableau()
                    print(i)
                    for j in range(i + 1, n):
                        if cli_copy._table[i + n, j]:
                            cli_copy.CNOT(i, j)
                            cli_copy.print_tableau()
                            print('cnot 6')
                            circuit_str.append(('C', i, j))
                    if cli_copy._table[i + n, i + n]:
                        cli_copy.PhaseGate(i)
                        circuit_str.append(('P', i))
                    cli_copy.Hadamard(i)
                    circuit_str.append(('H', i))

            ######  step06: deal with phase  ######
            for i in range(n):
                if cli_copy._phase[i]:
                    # Z = PP
                    circuit_str.append(('Z', i))
                if cli_copy._phase[i + n]:
                    # X = HZH = HPPH
                    circuit_str.append(('X', i))

            return circuit_str[::-1]

    def apply_circuit(self, circuit_str, reverse=True):
        if circuit_str is None:
            return
        for igate in circuit_str:
            if igate[0] == 'C':
                self.CNOT(igate[1], igate[2])
            elif igate[0] == 'H':
                self.Hadamard(igate[1])
            elif igate[0] == 'P':
                if reverse:
                    self.Pdagger(igate[1])
                else:
                    self.PhaseGate(igate[1])
            elif igate[0] == 'X':
                self.XGate(igate[1])
            elif igate[0] == 'Z':
                self.ZGate(igate[1])

    def to_circuit(self, interface='mindquantum'):
        circuit_str = self.clifford_decomposition()
        if interface == 'mindquantum':
            c = Circuit()
            for igate in circuit_str:
                if igate[0] == 'C':
                    c += X.on(igate[2], igate[1])
                elif igate[0] == 'H':
                    c += H.on(igate[1])
                elif igate[0] == 'P':
                    c += S.on(igate[1]).hermitian()
                elif igate[0] == 'X':
                    c += X.on(igate[1])
                elif igate[0] == 'Z':
                    c += Z.on(igate[1])
        return c

    def __mul__(self, cli):
        if isinstance(cli, Clifford):
            # self * cli
            cli_copy = cli.copy()
            circuit_str = self.clifford_decomposition()
            cli_copy.apply_circuit(circuit_str)
            return cli_copy
        else:
            return TypeError

    def __rmul__(self, cli):
        return self * cli

    def __eq__(self, cli):
        return np.all(self._table == cli._table) and np.all(self._phase == cli._phase)


def circuit_to_clifford(cir):
    cli = Clifford(num_qubits=cir.n_qubits)
    for icir in cir:
        cli.apply_gate(icir)
    return cli.copy()


def clifford_to_id(cli):
    tid = 0
    pid = 0
    n = cli.num_qubits
    for i in range(2 * n):
        pid += cli._phase[i] * 2**i
        for j in range(2 * n):
            tid += cli._table[i, j] * 2 ** (2 * n * i + j)
    return tid, pid


def id_to_clifford(tableid, phaseid=0, num_qubits=2):
    n = num_qubits
    cli = Clifford(num_qubits=num_qubits)
    for i in range(2 * n):
        for j in range(2 * n):
            cli._table[i, j] = (tableid // (2 ** (2 * n * i + j))) % 2
    for i in range(2 * n):
        cli._phase[i] = (phaseid // (2**i)) % 2
    return cli.copy()


def get_initial_id(num_qubits=2):
    n = num_qubits
    # id0 = 0
    # for iq in range(2*n):
    #    id0 += 2**(2*n*iq+iq)
    cli_ = Clifford(num_qubits=num_qubits)
    id0, _ = clifford_to_id(cli_)
    return id0


if __name__ == "__main__":
    C = Clifford(num_qubits=3)
    C.Hadamard(0)
    C.CNOT(0, 1)
    a = C.clifford_decomposition()
    print(a)
    # Cp = C.copy()

    # C.Hadamard(1)
    # C.CNOT(1, 2)
    # C.Hadamard(0)
    # C.CNOT(0, 1)
    # C.Hadamard(0)
    # C.print_tableau()

    # circuit = C.clifford_decomposition()
    # C.print_tableau()

    # C2 = Clifford(num_qubits=3)
    # C3 = C * C2
    # print(C2 == C3)

    # print(C3.clifford_decomposition())
    # mq_cir = C3.to_circuit(interface='mindquantum')
    # mq_cir.summary()

    # C3p = circuit_to_clifford(mq_cir)
    # C3p.print_tableau()
