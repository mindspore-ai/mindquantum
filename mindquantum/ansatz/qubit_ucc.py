# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Qubit unitary coupled-cluster ansatz."""

import warnings
import itertools

import numpy
from mindquantum.gate import CNOT, X, RY
from mindquantum.circuit import Circuit
from mindquantum.parameterresolver import ParameterResolver as PR
from mindquantum.ops import QubitExcitationOperator
from mindquantum.utils import hermitian_conjugated
from ._ansatz import Ansatz


def _check_int_list(input_list, name):
    if not isinstance(input_list, list):
        raise ValueError("The input {} should be a list, \
but get {}.".format(str(name), type(input_list)))
    for i in input_list:
        if not isinstance(i, int):
            raise ValueError("The indices of {} should be integer, \
but get {}.".format(str(name), type(i)))


class QubitUCCAnsatz(Ansatz):
    r"""
    Qubit Unitary Coupled-Cluster (qUCC) ansatz is a variant of unitary
    coupled-cluster ansatz which uses qubit excitation operators instead of
    Fermion excitation operators. The Fock space spanned by qubit excitation
    operators is equivalent as Fermion operators, therefore the exact
    wave function can be approximated using qubit excitation operators at
    the expense of a higher order of Trotterization.

    The greatest advantange of qUCC is that the number of CNOT gates is much
    smaller than the original version of UCC, even using a 3rd or 4th order
    Trotterization. Also, the accuracy is greatly improved despite that the
    number of variational parameters is increased.

    Note:
        The Hartree-Fock circuit is not included.
        Currently, generalized=True is not allowed since the theory needs verification.
        Reference: Yordan S. Yordanov et al. Phys. Rev. A, 102, 062612 (2020)

    Args:
        n_qubits (int): The number of qubits (spin-orbitals) in the simulation. Default: None.
        n_electrons (int): The number of electrons of the given molecule. Default: None.
        occ_orb(list): Indices of manually assigned occupied spatial orbitals. Default: None.
        vir_orb(list): Indices of manually assigned virtual spatial orbitals. Default: None.
        generalized(bool): Whether to use generalized excitations which
            do not distinguish occupied or virtual orbitals (qUCCGSD). Currently,
            generalized=True is not allowed since the theory needs verification. Default: False.
        trotter_step (int): The number of Trotter steps. Default is one. It is
            recommended to set a value larger than or equal to 2 to achieve a
            good accuracy. Default: 1.

    Examples:
        >>> from mindquantum.ansatz import QubitUCCAnsatz
        >>> QubitUCCAnsatz().n_qubits
        0
        >>> qucc = QubitUCCAnsatz(4, 2, trotter_step=2)
        >>> qucc.circuit[:10]
        CNOT(0 <-: 2)
        RY(t_0_q_s_0|2 <-: 0)
        CNOT(0 <-: 2)
        CNOT(1 <-: 2)
        RY(t_0_q_s_1|2 <-: 1)
        CNOT(1 <-: 2)
        CNOT(0 <-: 3)
        RY(t_0_q_s_2|3 <-: 0)
        CNOT(0 <-: 3)
        CNOT(1 <-: 3)
        >>> qucc.n_qubits
        4
        >>> qucc_2 = QubitUCCAnsatz(occ_orb=[0, 1], vir_orb=[2])
        >>> qucc_2.operator_pool
        [-1.0*t_0_q_s_0 [Q0^ Q4] +
        1.0*t_0_q_s_0 [Q4^ Q0] , -1.0*t_0_q_s_1 [Q1^ Q4] +
        1.0*t_0_q_s_1 [Q4^ Q1] , -1.0*t_0_q_s_2 [Q2^ Q4] +
        1.0*t_0_q_s_2 [Q4^ Q2] , -1.0*t_0_q_s_3 [Q3^ Q4] +
        1.0*t_0_q_s_3 [Q4^ Q3] , -1.0*t_0_q_s_4 [Q0^ Q5] +
        1.0*t_0_q_s_4 [Q5^ Q0] , -1.0*t_0_q_s_5 [Q1^ Q5] +
        1.0*t_0_q_s_5 [Q5^ Q1] , -1.0*t_0_q_s_6 [Q2^ Q5] +
        1.0*t_0_q_s_6 [Q5^ Q2] , -1.0*t_0_q_s_7 [Q3^ Q5] +
        1.0*t_0_q_s_7 [Q5^ Q3] , -1.0*t_0_q_d_0 [Q1^ Q0^ Q5 Q4] +
        1.0*t_0_q_d_0 [Q5^ Q4^ Q1 Q0] , -1.0*t_0_q_d_1 [Q2^ Q0^ Q5 Q4] +
        1.0*t_0_q_d_1 [Q5^ Q4^ Q2 Q0] , -1.0*t_0_q_d_2 [Q2^ Q1^ Q5 Q4] +
        1.0*t_0_q_d_2 [Q5^ Q4^ Q2 Q1] , -1.0*t_0_q_d_3 [Q3^ Q0^ Q5 Q4] +
        1.0*t_0_q_d_3 [Q5^ Q4^ Q3 Q0] , -1.0*t_0_q_d_4 [Q3^ Q1^ Q5 Q4] +
        1.0*t_0_q_d_4 [Q5^ Q4^ Q3 Q1] , -1.0*t_0_q_d_5 [Q3^ Q2^ Q5 Q4] +
        1.0*t_0_q_d_5 [Q5^ Q4^ Q3 Q2] ]
    """
    def __init__(self,
                 n_qubits=None,
                 n_electrons=None,
                 occ_orb=None,
                 vir_orb=None,
                 generalized=False,
                 trotter_step=1):
        if n_qubits is not None and not isinstance(n_qubits, int):
            raise ValueError("The number of qubits should be integer, \
but get {}.".format(type(n_qubits)))
        if n_electrons is not None and not isinstance(n_electrons, int):
            raise ValueError("The number of electrons should be integer, \
but get {}.".format(type(n_electrons)))
        if isinstance(n_electrons, int) and n_electrons > n_qubits:
            raise ValueError("The number of electrons must be smaller than \
the number of qubits (spin-orbitals) in the ansatz!")
        if occ_orb is not None:
            _check_int_list(occ_orb, "occupied orbitals")
        if vir_orb is not None:
            _check_int_list(vir_orb, "virtual orbitals")
        if not isinstance(generalized, bool):
            raise ValueError("The parameter generalized should be bool, \
but get {}.".format(type(generalized)))
        # Although the code for generalized excitations has been written,
        # the physical underneath of such type of operators is still not clear,
        # therefore, currently it seems reasonable to make generalized qUCC unavailable.
        if isinstance(generalized, bool) and generalized is not False:
            raise NotImplementedError(
                "Generalized version of qubit-UCC not implemented!")
        if not isinstance(trotter_step, int) or trotter_step < 1:
            raise ValueError("Trotter step must be a positive integer!")
        # n_qubits is also need for _implement()
        super().__init__("Qubit UCC", n_qubits, n_qubits, n_electrons, occ_orb,
                         vir_orb, generalized, trotter_step)

    def _single_qubit_excitation_circuit(self, i, k, theta):
        """
        Implement circuit for single qubit excitation.
        k: creation
        """
        circuit_singles = Circuit()
        circuit_singles += CNOT(i, k)
        circuit_singles += RY(theta).on(k, i)
        circuit_singles += CNOT(i, k)
        return circuit_singles

    def _double_qubit_excitation_circuit(self, i, j, k, l, theta):
        """
        Implement circuit for double qubit excitation.
        k, l: creation
        """
        circuit_doubles = Circuit()
        circuit_doubles += CNOT.on(k, l)
        circuit_doubles += CNOT.on(i, j)
        circuit_doubles += CNOT.on(j, l)
        circuit_doubles += X.on(k)
        circuit_doubles += X.on(i)
        circuit_doubles += RY(theta).on(l, ctrl_qubits=[i, j, k])
        circuit_doubles += X.on(k)
        circuit_doubles += X.on(i)
        circuit_doubles += CNOT.on(j, l)
        circuit_doubles += CNOT.on(i, j)
        circuit_doubles += CNOT.on(k, l)
        return circuit_doubles

    def _implement(self,
                   n_qubits=None,
                   n_electrons=None,
                   occ_orb=None,
                   vir_orb=None,
                   generalized=False,
                   trotter_step=1):
        """
        Implement qubit UCC circuit according to the reference paper.
        """
        occ_indices = []
        vir_indices = []
        n_orb = 0
        n_orb_occ = 0
        n_orb_vir = 0
        if n_qubits is not None:
            if n_qubits % 2 != 0:
                raise ValueError('The total number of qubits (spin-orbitals) \
should be even.')
            n_orb = n_qubits // 2
        if n_electrons is not None:
            n_orb_occ = int(numpy.ceil(n_electrons / 2))
            n_orb_vir = n_orb - n_orb_occ
            occ_indices = [i for i in range(n_orb_occ)]
            vir_indices = [i + n_orb_occ for i in range(n_orb_vir)]
        warn_flag = False
        if occ_orb is not None:
            if len(set(occ_orb)) != len(occ_orb):
                raise ValueError(
                    "Indices for occupied orbitals should be unique!")
            warn_flag = True
            n_orb_occ = len(occ_orb)
            occ_indices = occ_orb
        if vir_orb is not None:
            if len(set(vir_orb)) != len(vir_orb):
                raise ValueError(
                    "Indices for virtual orbitals should be unique!")
            warn_flag = True
            n_orb_vir = len(vir_orb)
            vir_indices = vir_orb
        if set(occ_indices).intersection(vir_indices):
            raise ValueError(
                "Occupied and virtual orbitals should be different!")
        indices_tot = occ_indices + vir_indices
        max_idx = 0
        if set(indices_tot):
            max_idx = max(set(indices_tot))
        n_orb = max(n_orb, max_idx)
        if warn_flag:
            warnings.warn(
                "[Note] Override n_qubits and n_electrons with manually \
set occ_orb and vir_orb. Handle with caution!")

        if generalized:
            occ_indices = indices_tot
            vir_indices = indices_tot

        n_occ = len(occ_indices)
        if n_occ == 0:
            warnings.warn(
                "The number of occupied orbitals is zero. Ansatz may \
contain no parameters.")
        n_vir = len(vir_indices)
        if n_vir == 0:
            warnings.warn("The number of virtual orbitals is zero. Ansatz may \
contain no parameters.")

        # Convert spatial-orbital indices to spin-orbital indices
        occ_indices_spin = []
        vir_indices_spin = []
        for i in occ_indices:
            occ_indices_spin.append(i * 2)
            occ_indices_spin.append(i * 2 + 1)
        for i in vir_indices:
            vir_indices_spin.append(i * 2)
            vir_indices_spin.append(i * 2 + 1)
        indices_spin_tot = list(set(occ_indices_spin + vir_indices_spin))

        if generalized:
            occ_indices_spin = indices_spin_tot
            vir_indices_spin = indices_spin_tot

        n_occ_spin = len(occ_indices_spin)
        n_vir_spin = len(vir_indices_spin)

        ansatz_circuit = Circuit()
        # Record the perator pool, which can be used for qubit-ADAPT-VQE or other iterative ansatz.
        generator_quccsd_singles = []
        generator_quccsd_doubles = []
        for trotter_idx in range(trotter_step):
            singles_counter = 0
            for (p, q) in itertools.product(vir_indices_spin,
                                            occ_indices_spin):
                coeff_s = PR({f't_{trotter_idx}_q_s_{singles_counter}': 1})
                q_pq = QubitExcitationOperator(((p, 1), (q, 0)), 1.)
                q_pq = q_pq - hermitian_conjugated(q_pq)
                q_pq = q_pq.normal_ordered()
                if list(q_pq.terms):
                    # The operator pool do not have to contain operators
                    # from different Trottered steps.
                    if trotter_idx == 0:
                        generator_quccsd_singles.append(q_pq * coeff_s)
                    ansatz_circuit += self._single_qubit_excitation_circuit(
                        q, p, coeff_s)
                    singles_counter += 1

            doubles_counter = 0
            for pq_counter, (p_idx, q_idx) in enumerate(
                    itertools.product(range(n_vir_spin), range(n_vir_spin))):
                # Only take half of the loop to avoid repeated excitations
                if q_idx > p_idx:
                    continue
                p = vir_indices_spin[p_idx]
                q = vir_indices_spin[q_idx]
                for rs_counter, (r_idx, s_idx) in enumerate(
                        itertools.product(range(n_occ_spin),
                                          range(n_occ_spin))):
                    # Only take half of the loop to avoid repeated excitations
                    if s_idx > r_idx:
                        continue
                    r = occ_indices_spin[r_idx]
                    s = occ_indices_spin[s_idx]
                    if generalized and pq_counter > rs_counter:
                        continue
                    coeff_d = PR({f't_{trotter_idx}_q_d_{doubles_counter}': 1})
                    q_pqrs = QubitExcitationOperator(
                        ((p, 1), (q, 1), (r, 0), (s, 0)), 1.)
                    q_pqrs = q_pqrs - hermitian_conjugated(q_pqrs)
                    q_pqrs = q_pqrs.normal_ordered()
                    if list(q_pqrs.terms):
                        # The operator pool do not have to contain operators
                        # from different Trottered steps.
                        if trotter_idx == 0:
                            generator_quccsd_doubles.append(q_pqrs * coeff_d)
                        ansatz_circuit += self._double_qubit_excitation_circuit(
                            r, s, p, q, coeff_d)
                        doubles_counter += 1
        n_qubits_circuit = 0
        if list(ansatz_circuit):
            n_qubits_circuit = ansatz_circuit.n_qubits
        # If the ansatz's n_qubits is not set by user, use n_qubits_circuit.
        if self.n_qubits is None:
            self.n_qubits = n_qubits_circuit
        if self.n_qubits < n_qubits_circuit:
            raise ValueError(
                "The number of qubits in the ansatz circuit {} is larger than \
the input n_qubits {}! Please check input parameters such as occ_orb, etc.".
                format(n_qubits_circuit, n_qubits))
        self._circuit = ansatz_circuit
        self.operator_pool = generator_quccsd_singles + generator_quccsd_doubles
