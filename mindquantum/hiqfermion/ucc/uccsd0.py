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
"""Implement UCCSD0/UCCGSD0 ansatz using CCD0 excitation operators"""

import itertools
import warnings

import numpy
from mindquantum.ops import FermionOperator
from mindquantum.parameterresolver import ParameterResolver as PR
from mindquantum.utils import hermitian_conjugated, normal_ordered


def _check_int_list(input_list, name):
    if not isinstance(input_list, list):
        raise ValueError("The input {} should be a list, \
but get {}.".format(str(name), type(input_list)))
    for i in input_list:
        if not isinstance(i, int):
            raise ValueError("The indices of {} should be integer, \
but get {}.".format(str(name), type(i)))


def _pij(i: int, j: int):
    r"""
    Helper function for CCD0 ansatz.

    See :class: `mindquantum.third_party.unitary_cc.spin_adapted_t2`.
    """
    ia = i * 2 + 0
    ib = i * 2 + 1
    ja = j * 2 + 0
    jb = j * 2 + 1
    term1 = FermionOperator(((ja, 0), (ib, 0)), 1.0)
    term2 = FermionOperator(((ia, 0), (jb, 0)), 1.0)
    return numpy.sqrt(0.5) * (term1 + term2)


def _pij_dagger(i: int, j: int):
    r"""
    Helper function for CCD0 ansatz.

    See :class: `mindquantum.third_party.unitary_cc.spin_adapted_t2`.
    """
    return hermitian_conjugated(_pij(i, j))


def _qij_plus(i: int, j: int):
    r"""
    Helper function for CCD0 ansatz.

    See :class: `mindquantum.third_party.unitary_cc.spin_adapted_t2`.
    """
    ia = i * 2 + 0
    ja = j * 2 + 0
    term = FermionOperator(((ja, 0), (ia, 0)), 1.0)
    return term


def _qij_minus(i: int, j: int):
    r"""
    Helper function for CCD0 ansatz.

    See :class: `mindquantum.third_party.unitary_cc.spin_adapted_t2`.
    """
    ib = i * 2 + 1
    jb = j * 2 + 1
    term = FermionOperator(((jb, 0), (ib, 0)), 1.0)
    return term


def _qij_0(i: int, j: int):
    r"""
    Helper function for CCD0 ansatz.

    See :class: `mindquantum.third_party.unitary_cc.spin_adapted_t2`.
    """
    ia = i * 2 + 0
    ib = i * 2 + 1
    ja = j * 2 + 0
    jb = j * 2 + 1
    term1 = FermionOperator(((ja, 0), (ib, 0)), 1.0)
    term2 = FermionOperator(((ia, 0), (jb, 0)), 1.0)
    return numpy.sqrt(0.5) * (term1 - term2)


def _qij_vec(i: int, j: int):
    r"""
    Helper function for CCD0 ansatz.

    See :class: `mindquantum.third_party.unitary_cc.spin_adapted_t2`.
    """
    return [_qij_plus(i, j), _qij_minus(i, j), _qij_0(i, j)]


def _qij_vec_dagger(i: int, j: int):
    r"""
    Helper function for CCD0 ansatz.

    See :class: `mindquantum.third_party.unitary_cc.spin_adapted_t2`.
    """
    return [hermitian_conjugated(i) for i in _qij_vec(i, j)]


def _qij_vec_inner(a: int, b: int, i: int, j: int):
    r"""
    Helper function for CCD0 ansatz.

    See :class: `mindquantum.third_party.unitary_cc.spin_adapted_t2`.
    """
    vec_dagger = _qij_vec_dagger(a, b)
    vec = _qij_vec(i, j)
    sum_result = FermionOperator()
    for idx, term in enumerate(vec):
        sum_result += term * vec_dagger[idx]
    return sum_result


def spin_adapted_t1(i, j):
    r"""
    Spin-adapted single excitation operators.

    Args:
        i(int): index of the spatial orbital which the
            creation operator will act on.
        j(int): index of the spatial orbital which the
            annihilation operator will act on.

    Returns:
        tpq_list (list): Spin-adapted single excitation operators.

    Examples:
        >>> from mindquantum.hiqfermion.ucc.uccsd0 import spin_adapted_t1
        >>> spin_adapted_t1(2, 3)
        [1.0 [4^ 6] +
        1.0 [5^ 7] ]
        >>> spin_adapted_t1(1, 1)
        [1.0 [2^ 2] +
        1.0 [3^ 3] ]

    Note:
        For more information, please refer to:
            1. Scuseria, G. E. et al., J. Chem. Phys. 89, 7382 (1988)
    """
    if not isinstance(i, int) or not isinstance(j, int):
        raise ValueError("Requires integers as orbital indices, \
but get {} and {}.".format(type(i), type(j)))

    ia = i * 2 + 0
    ib = i * 2 + 1
    ja = j * 2 + 0
    jb = j * 2 + 1
    term1 = FermionOperator(((ia, 1), (ja, 0)), 1.0)
    term2 = FermionOperator(((ib, 1), (jb, 0)), 1.0)
    tpq_list = [term1 + term2]
    return tpq_list


def spin_adapted_t2(creation_list, annihilation_list):
    r"""
    Spin-adapted CCD0 double excitation operators.

    Args:
        creation_list(list): list of spatial orbital indices which the
            creation operator will act on.
        annihilation_list(list): list of spatial orbital indices which the
            annihilation operator will act on.

    Returns:
        tpqrs_list(list): Spin-adapted double excitation operators.

    Examples:
        >>> from mindquantum.hiqfermion.ucc.uccsd0 import spin_adapted_t2
        >>> spin_adapted_t2([0, 1], [2, 3])
        [0.5000000000000001 [1^ 2^ 4 7] +
        0.5000000000000001 [1^ 2^ 6 5] +
        0.5000000000000001 [3^ 0^ 4 7] +
        0.5000000000000001 [3^ 0^ 6 5] , -0.5000000000000001 [4 7 1^ 2^] +
        0.5000000000000001 [4 7 3^ 0^] +
        1.0 [6 4 0^ 2^] +
        0.5000000000000001 [6 5 1^ 2^] +
        -0.5000000000000001 [6 5 3^ 0^] +
        1.0 [7 5 1^ 3^] ]
        >>> spin_adapted_t2([0, 0], [1, 1])
        [2.0000000000000004 [1^ 0^ 2 3] , 1.0 [2 2 0^ 0^] +
        1.0 [3 3 1^ 1^] ]

    Note:
        For more information about CCD0, please refer to:
            1. Igor O. Sokolov et al. J. Chem. Phys. 152, 124107 (2020)
            2. Ireneusz W. Bulik et al. J. Chem. Theory Comput. 11, 3171 (2015)
            3. Scuseria, G. E. et al. J. Chem. Phys. 89, 7382 (1988)
    """
    _check_int_list(creation_list, "creation operators")
    _check_int_list(annihilation_list, "annihilation operators")

    if len(creation_list) != 2 or len(annihilation_list) != 2:
        raise ValueError(f"T2 excitations take exactly 2 indices for both \
creation and annihilation operators, \
but get {len(creation_list)} and {len(annihilation_list)} indices.")

    p = creation_list[0]
    r = annihilation_list[0]
    q = creation_list[1]
    s = annihilation_list[1]
    tpqrs1 = _pij_dagger(p, q) * _pij(r, s)
    tpqrs2 = _qij_vec_inner(p, q, r, s)
    tpqrs_list = [tpqrs1, tpqrs2]
    return tpqrs_list


def uccsd0_singlet_generator(n_qubits=None,
                             n_electrons=None,
                             anti_hermitian=True,
                             occ_orb=None,
                             vir_orb=None,
                             generalized=False):
    r"""
    Generate UCCSD operators using CCD0 ansatz for molecular systems.

    Note:
        Manually assigned occ_orb or vir_orb are indices of spatial orbitals
        instead of spin-orbitals. They will override n_electrons and
        n_qubits. This is to some degree similar to the active space,
        therefore can reduce the number of variational parameters. However, it
        may not reduce the number of required qubits, since Fermion
        excitation operators are non-local, i.e.,
        :math:`a_{7}^{\dagger} a_{0}` involves not only the 0th and 7th
        qubit, but also the 1st, 2nd, ... 6th qubit.

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
            do not distinguish occupied or virtual orbitals (UCCGSD). Default: False.

    Returns:
        FermionOperator, Generator of the UCCSD operators that uses CCD0 ansatz.

    Examples:
        >>> from mindquantum.hiqfermion.ucc.uccsd0 import uccsd0_singlet_generator
        >>> uccsd0_singlet_generator(4, 2)
        -1.0*d0_s_0 [0^ 2] +
        2.0*d0_d_0 [1^ 0^ 3 2] +
        -1.0*d0_s_0 [1^ 3] +
        1.0*d0_s_0 [2^ 0] +
        1.0*d0_s_0 [3^ 1] +
        -2.0*d0_d_0 [3^ 2^ 1 0]
        >>> uccsd0_singlet_generator(4, 2, generalized=True)
        1.0*d0_s_0 - 1.0*d0_s_1 [0^ 2] +
        1.0*d0_d_0 [1^ 0^ 2 1] +
        -1.0*d0_d_0 [1^ 0^ 3 0] +
        -2.0*d0_d_1 [1^ 0^ 3 2] +
        1.0*d0_s_0 - 1.0*d0_s_1 [1^ 3] +
        -1.0*d0_s_0 + 1.0*d0_s_1 [2^ 0] +
        -1.0*d0_d_0 [2^ 1^ 1 0] +
        1.0*d0_d_2 [2^ 1^ 3 2] +
        1.0*d0_d_0 [3^ 0^ 1 0] +
        -1.0*d0_d_2 [3^ 0^ 3 2] +
        -1.0*d0_s_0 + 1.0*d0_s_1 [3^ 1] +
        2.0*d0_d_1 [3^ 2^ 1 0] +
        -1.0*d0_d_2 [3^ 2^ 2 1] +
        1.0*d0_d_2 [3^ 2^ 3 0]
        >>> uccsd0_singlet_generator(6, 2, occ_orb=[0], vir_orb=[1])
        -1.0*d0_s_0 [0^ 2] +
        2.0*d0_d_0 [1^ 0^ 3 2] +
        -1.0*d0_s_0 [1^ 3] +
        1.0*d0_s_0 [2^ 0] +
        1.0*d0_s_0 [3^ 1] +
        -2.0*d0_d_0 [3^ 2^ 1 0]
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

    generator_uccsd0_singles = FermionOperator()
    generator_uccsd0_doubles = FermionOperator()

    singles_counter = 0
    for pq_counter, (p_idx, q_idx) in enumerate(
            itertools.product(range(n_vir), range(n_occ))):
        p = vir_indices[p_idx]
        q = occ_indices[q_idx]
        tpq_list = spin_adapted_t1(p, q)
        for tpq in tpq_list:
            coeff_s = PR({f'd0_s_{singles_counter}': 1})
            if anti_hermitian:
                tpq = tpq - hermitian_conjugated(tpq)
            tpq = normal_ordered(tpq)
            if list(tpq.terms):
                generator_uccsd0_singles += tpq * coeff_s
                singles_counter += 1

    doubles_counter = 0
    for pq_counter, (p_idx, q_idx) in enumerate(
            itertools.product(range(n_vir), range(n_vir))):
        # Only take half of the loop to avoid repeated excitations
        if q_idx > p_idx:
            continue
        p = vir_indices[p_idx]
        q = vir_indices[q_idx]
        for rs_counter, (r_idx, s_idx) in enumerate(
                itertools.product(range(n_occ), range(n_occ))):
            # Only take half of the loop to avoid repeated excitations
            if s_idx > r_idx:
                continue
            r = occ_indices[r_idx]
            s = occ_indices[s_idx]
            if generalized and pq_counter > rs_counter:
                continue
            tpqrs_list = spin_adapted_t2([p, q], [r, s])
            for tpqrs in tpqrs_list:
                coeff_d = PR({f'd0_d_{doubles_counter}': 1})
                if anti_hermitian:
                    tpqrs = tpqrs - hermitian_conjugated(tpqrs)
                tpqrs = normal_ordered(tpqrs)
                if list(tpqrs.terms):
                    generator_uccsd0_doubles += tpqrs * coeff_d
                    doubles_counter += 1

    generator_uccsd0 = generator_uccsd0_singles + generator_uccsd0_doubles

    return generator_uccsd0
