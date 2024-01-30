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
"""Tests for the qUCCSD generator and related functions"""

import warnings

import pytest

from mindquantum.algorithm.nisq import quccsd_generator
from mindquantum.core.operators import TimeEvolution, count_qubits
from mindquantum.core.operators._term_value import TermValue
from mindquantum.simulator.available_simulator import SUPPORTED_SIMULATOR

AVAILABLE_BACKEND = list(filter(lambda x: x != 'stabilizer', SUPPORTED_SIMULATOR))


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('config', AVAILABLE_BACKEND)
def test_quccsd(config):
    """
    Description: Test quccsd
    Expectation:
    """
    _, _ = config
    h2_quccsd = quccsd_generator(4, 2)
    h2_quccsd_terms = set(h2_quccsd.terms)
    h2_quccsd_terms_check = {
        ((3, TermValue.adg), (0, TermValue.a)),
        ((3, TermValue.adg), (1, TermValue.a)),
        ((1, TermValue.adg), (2, TermValue.a)),
        ((1, TermValue.adg), (3, TermValue.a)),
        ((2, TermValue.adg), (0, TermValue.a)),
        ((2, TermValue.adg), (1, TermValue.a)),
        ((0, TermValue.adg), (2, TermValue.a)),
        ((0, TermValue.adg), (3, TermValue.a)),
        ((3, TermValue.adg), (2, TermValue.adg), (1, TermValue.a), (0, TermValue.a)),
        ((1, TermValue.adg), (0, TermValue.adg), (3, TermValue.a), (2, TermValue.a)),
    }
    assert h2_quccsd_terms == h2_quccsd_terms_check

    lih_quccsd = quccsd_generator(12, 4)
    lih_quccsd_circuit = TimeEvolution(lih_quccsd.to_qubit_operator().imag, 1).circuit
    n_params_lih = len(lih_quccsd_circuit.params_name)
    assert n_params_lih == 200

    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore', category=UserWarning, message=r'\[Note\] Override n_qubits and n_electrons with manually.*'
        )
        lih_quccgsd_cas = quccsd_generator(12, 4, occ_orb=[1], vir_orb=[2, 3], generalized=True)
    assert count_qubits(lih_quccgsd_cas) == 8
    lih_quccgsd_cas_circuit = TimeEvolution(lih_quccgsd_cas.to_qubit_operator().imag, 1).circuit
    n_params_lih = len(lih_quccgsd_cas_circuit.params_name)
    assert n_params_lih == 135

    # qUCCSD with fully occupied orbitals should lead to zero parameters
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning, message='The number of virtual orbitals is zero.*')
        he2_quccsd = quccsd_generator(4, 4)

    he2_quccsd_circuit = TimeEvolution(he2_quccsd.to_qubit_operator().imag, 1).circuit
    n_params_he2 = len(he2_quccsd_circuit.params_name)
    assert n_params_he2 == 0

    # qUCCGSD will not be affected by the occupancy numbers
    he2_quccsd = quccsd_generator(4, 4, generalized=True)
    he2_quccsd_circuit = TimeEvolution(he2_quccsd.to_qubit_operator().imag, 1).circuit
    n_params_he2 = len(he2_quccsd_circuit.params_name)
    assert n_params_he2 == 27
