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
"""Tests for the UCCSD0 generator and related functions"""

import warnings

import pytest

from mindquantum.algorithm.nisq import Transform, uccsd0_singlet_generator
from mindquantum.algorithm.nisq.chem.uccsd0 import spin_adapted_t1, spin_adapted_t2
from mindquantum.core.operators import TimeEvolution, count_qubits


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_spin_adapted_t1():
    """
    Description: Test spin adapted t1
    Expectation:
    """
    t1_20 = spin_adapted_t1(2, 0)[0]
    assert str(t1_20) == '1 [4^ 0] +\n1 [5^ 1]'
    t1_00 = spin_adapted_t1(0, 0)[0]
    assert str(t1_00) == '1 [0^ 0] +\n1 [1^ 1]'


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_spin_adapted_t2():
    """
    Description: Test spin adapted t2
    Expectation:
    """
    t2_3210_list = spin_adapted_t2([3, 2], [1, 0])
    assert len(t2_3210_list) == 2
    term1 = set(t2_3210_list[0].terms)
    assert len(term1) == 4
    term1_check = {
        ((6, 1), (5, 1), (2, 0), (1, 0)),
        ((6, 1), (5, 1), (3, 0), (0, 0)),
        ((7, 1), (4, 1), (2, 0), (1, 0)),
        ((7, 1), (4, 1), (3, 0), (0, 0)),
    }
    assert term1 == term1_check
    term2 = set(t2_3210_list[1].terms)
    assert len(term2) == 6
    term2_check = {
        ((6, 1), (4, 1), (2, 0), (0, 0)),
        ((6, 1), (5, 1), (2, 0), (1, 0)),
        ((6, 1), (5, 1), (3, 0), (0, 0)),
        ((7, 1), (4, 1), (2, 0), (1, 0)),
        ((7, 1), (4, 1), (3, 0), (0, 0)),
        ((7, 1), (5, 1), (3, 0), (1, 0)),
    }
    assert term2 == term2_check


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_uccsd0():
    """
    Description: Test uccsd0
    Expectation:
    """
    # pylint: disable=too-many-locals
    h2_uccsd0 = uccsd0_singlet_generator(4, 2)
    h2_uccsd0_terms = set(h2_uccsd0.terms)
    h2_uccsd0_terms_check = {
        ((2, 0), (0, 1)),
        ((2, 1), (0, 0)),
        ((3, 0), (1, 1)),
        ((3, 0), (2, 0), (1, 1), (0, 1)),
        ((3, 1), (1, 0)),
        ((3, 1), (2, 1), (1, 0), (0, 0)),
    }
    assert h2_uccsd0_terms == h2_uccsd0_terms_check

    lih_uccsd0 = uccsd0_singlet_generator(12, 4)
    lih_uccsd0_circuit = TimeEvolution(Transform(lih_uccsd0).jordan_wigner().imag, 1).circuit
    n_params_lih = len(lih_uccsd0_circuit.params_name)
    assert n_params_lih == 44

    # cas means complete active space
    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore', category=UserWarning, message=r'\[Note\] Override n_qubits and n_electrons with manually.*'
        )
        lih_uccgsd0_cas = uccsd0_singlet_generator(12, 4, occ_orb=[1], vir_orb=[2, 3], generalized=True)
    # The max index of affected qubits in the ansatz is 7 = 8-1.
    # Does not mean the number of qubits in Hamiltonian is reduced to 8.
    assert count_qubits(lih_uccgsd0_cas) == 8
    lih_uccgsd0_cas_circuit = TimeEvolution(Transform(lih_uccgsd0_cas).jordan_wigner().imag, 1).circuit
    n_params_lih_cas = len(lih_uccgsd0_cas_circuit.params_name)
    assert n_params_lih_cas == 24

    # UCCSD with fully occupied orbitals should lead to 0 parameters
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning, message='The number of virtual orbitals is zero.*')
        he2_uccsd = uccsd0_singlet_generator(4, 4)
    he2_uccsd_circuit = TimeEvolution(Transform(he2_uccsd).jordan_wigner().imag, 1).circuit
    n_params_he2 = len(he2_uccsd_circuit.params_name)
    assert n_params_he2 == 0

    # UCCGSD will not be affected by the occupancy number
    he2_uccgsd = uccsd0_singlet_generator(4, 4, generalized=True)
    he2_uccgsd_circuit = TimeEvolution(Transform(he2_uccgsd).jordan_wigner().imag, 1).circuit
    n_params_he2_gsd = len(he2_uccgsd_circuit.params_name)
    assert n_params_he2_gsd == 5
