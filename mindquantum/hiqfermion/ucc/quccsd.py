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
"""Generate qUCCSD operators"""

import itertools
import warnings

import numpy
from mindquantum.ops import QubitExcitationOperator
from mindquantum.parameterresolver import ParameterResolver as PR
from mindquantum.utils import hermitian_conjugated


def _check_int_list(input_list, name):
    if not isinstance(input_list, list):
        raise ValueError("The input {} should be a list, \
but get {}.".format(str(name), type(input_list)))
    for i in input_list:
        if not isinstance(i, int):
            raise ValueError("The indices of {} should be integer, \
but get {}.".format(str(name), type(i)))


def quccsd_generator(n_qubits=None,
                     n_electrons=None,
                     anti_hermitian=True,
                     occ_orb=None,
                     vir_orb=None,
                     generalized=False):
    r"""
    Generate qubit-UCCSD (qUCCSD) ansatz using qubit-excitation operators.

    Note:
        Currently, unrestricted version is implemented, i.e., excitations from the
        same spatial-orbital but with different spins will use distinct variational
        parameters.

    Args:
        n_qubits(int): Number of qubits (spin-orbitals). Default: None.
        n_electrons(int): Number of electrons (occupied spin-orbitals). Default: None.
        anti_hermitian(bool): Whether to subtract the hermitian conjugate
            to form anti-Hermitian operators. Default: True.
        occ_orb(list): Indices of manually assigned occupied spatial
            orbitals. Default: None.
        vir_orb(list): Indices of manually assigned virtual spatial
            orbitals. Default: None.
        generalized(bool): Whether to use generalized excitations which
            do not distinguish occupied or virtual orbitals (qUCCGSD). Default: False.

    Returns:
        QubitExcitationOperator: Generator of the qUCCSD operators.

    Examples:
        >>> from mindquantum.hiqfermion.ucc import quccsd_generator
        >>> quccsd_generator()
        0
        >>> quccsd_generator(4, 2)
        -1.0*q_s_0 [Q0^ Q2] +
        -1.0*q_s_2 [Q0^ Q3] +
        -1.0*q_d_0 [Q1^ Q0^ Q3 Q2] +
        -1.0*q_s_1 [Q1^ Q2] +
        -1.0*q_s_3 [Q1^ Q3] +
        1.0*q_s_0 [Q2^ Q0] +
        1.0*q_s_1 [Q2^ Q1] +
        1.0*q_s_2 [Q3^ Q0] +
        1.0*q_s_3 [Q3^ Q1] +
        1.0*q_d_0 [Q3^ Q2^ Q1 Q0]
        >>> q_op = quccsd_generator(occ_orb=[0], vir_orb=[1], generalized=True)
        >>> q_qubit_op = q_op.to_qubit_operator()
        >>> print(str(q_qubit_op)[:315])
        0.125*I*q_d_4 + 0.125*I*q_d_7 + 0.125*I*q_d_9 [X0 X1 X2 Y3] +
        0.125*I*q_d_4 - 0.125*I*q_d_7 - 0.125*I*q_d_9 [X0 X1 Y2 X3] +
        0.25*I*q_d_12 + 0.25*I*q_d_5 + 0.5*I*q_s_0 - 0.5*I*q_s_3 [X0 Y1] +
        -0.125*I*q_d_4 + 0.125*I*q_d_7 - 0.125*I*q_d_9 [X0 Y1 X2 X3] +
        0.125*I*q_d_4 + 0.125*I*q_d_7 - 0.125*I*q_d_9 [X0 Y1 Y2 Y3] +
    """
    if n_qubits is not None and not isinstance(n_qubits, int):
        raise ValueError("The number of qubits should be integer, \
but get {}.".format(type(n_qubits)))
    if n_electrons is not None and not isinstance(n_electrons, int):
        raise ValueError("The number of electrons should be integer, \
but get {}.".format(type(n_electrons)))
    if isinstance(n_electrons, int) and n_electrons > n_qubits:
        raise ValueError("The number of electrons must be smaller than \
the number of qubits (spin-orbitals) in the ansatz!")
    if not isinstance(anti_hermitian, bool):
        raise ValueError("The parameter anti_hermitian should be bool, \
but get {}.".format(type(anti_hermitian)))
    if occ_orb is not None:
        _check_int_list(occ_orb, "occupied orbitals")
    if vir_orb is not None:
        _check_int_list(vir_orb, "virtual orbitals")
    if not isinstance(generalized, bool):
        raise ValueError("The parameter generalized should be bool, \
but get {}.".format(type(generalized)))

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
            raise ValueError("Indices for occupied orbitals should be unique!")
        warn_flag = True
        n_orb_occ = len(occ_orb)
        occ_indices = occ_orb
    if vir_orb is not None:
        if len(set(vir_orb)) != len(vir_orb):
            raise ValueError("Indices for virtual orbitals should be unique!")
        warn_flag = True
        n_orb_vir = len(vir_orb)
        vir_indices = vir_orb
    if set(occ_indices).intersection(vir_indices):
        raise ValueError("Occupied and virtual orbitals should be different!")
    indices_tot = occ_indices + vir_indices
    max_idx = 0
    if set(indices_tot):
        max_idx = max(set(indices_tot))
    n_orb = max(n_orb, max_idx)
    if warn_flag:
        warnings.warn("[Note] Override n_qubits and n_electrons with manually \
set occ_orb and vir_orb. Handle with caution!")

    if generalized:
        occ_indices = indices_tot
        vir_indices = indices_tot

    n_occ = len(occ_indices)
    if n_occ == 0:
        warnings.warn("The number of occupied orbitals is zero. Ansatz may \
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

    generator_quccsd_singles = QubitExcitationOperator()
    generator_quccsd_doubles = QubitExcitationOperator()

    singles_counter = 0
    for (p, q) in itertools.product(vir_indices_spin, occ_indices_spin):
        coeff_s = PR({f'q_s_{singles_counter}': 1})
        q_pq = QubitExcitationOperator(((p, 1), (q, 0)), 1.)
        if anti_hermitian:
            q_pq = q_pq - hermitian_conjugated(q_pq)
        q_pq = q_pq.normal_ordered()
        if list(q_pq.terms):
            generator_quccsd_singles += q_pq * coeff_s
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
                itertools.product(range(n_occ_spin), range(n_occ_spin))):
            # Only take half of the loop to avoid repeated excitations
            if s_idx > r_idx:
                continue
            r = occ_indices_spin[r_idx]
            s = occ_indices_spin[s_idx]
            if generalized and pq_counter > rs_counter:
                continue
            coeff_d = PR({f'q_d_{doubles_counter}': 1})
            q_pqrs = QubitExcitationOperator(((p, 1), (q, 1), (r, 0), (s, 0)),
                                             1.)
            if anti_hermitian:
                q_pqrs = q_pqrs - hermitian_conjugated(q_pqrs)
            q_pqrs = q_pqrs.normal_ordered()
            if list(q_pqrs.terms):
                generator_quccsd_doubles += q_pqrs * coeff_d
                doubles_counter += 1

    generator_quccsd = generator_quccsd_singles + generator_quccsd_doubles

    return generator_quccsd
