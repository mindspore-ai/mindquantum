# Copyright 2025 Huawei Technologies Co., Ltd
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
"""Smoke tests for the experimental mqmps backend."""

import numpy as np
import pytest

import mindquantum as mq
from mindquantum.core.circuit import Circuit
from mindquantum.simulator import Simulator
from mindquantum.simulator.available_simulator import SUPPORTED_SIMULATOR


if "mqmps" not in SUPPORTED_SIMULATOR.sims:
    pytest.skip("mqmps backend is not available in this build.", allow_module_level=True)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize("dtype", [mq.complex64, mq.complex128])
def test_mqmps_apply_circuit_matches_mqvector(dtype):
    """
    Description: mqmps basic gate evolution.
    Expectation: mqmps matches mqvector for a small circuit.
    """
    atol = 1e-5 if dtype == mq.complex64 else 1e-6
    n_qubits = 3
    circuit = Circuit().h(0).rx(0.123, 1).x(2, 0).ry(0.456, 2).z(1, 2).rz(0.789, 0)

    ref = Simulator("mqvector", n_qubits, dtype=dtype)
    dut = Simulator("mqmps", n_qubits, dtype=dtype)
    ref.apply_circuit(circuit)
    dut.apply_circuit(circuit)
    assert np.allclose(dut.get_qs(), ref.get_qs(), atol=atol)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize("dtype", [mq.complex64, mq.complex128])
def test_mqmps_set_qs_roundtrip_matches_mqvector(dtype):
    """
    Description: mqmps set_qs and get_qs.
    Expectation: mqmps matches mqvector after setting the same initial state.
    """
    atol = 1e-5 if dtype == mq.complex64 else 1e-6
    n_qubits = 3
    rng = np.random.default_rng(42)
    init_state = rng.random(2**n_qubits) + 1j * rng.random(2**n_qubits)

    ref = Simulator("mqvector", n_qubits, dtype=dtype)
    dut = Simulator("mqmps", n_qubits, dtype=dtype)
    ref.set_qs(init_state)
    dut.set_qs(init_state)
    assert np.allclose(dut.get_qs(), ref.get_qs(), atol=atol)
