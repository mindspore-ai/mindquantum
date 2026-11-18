# Copyright 2026 Huawei Technologies Co., Ltd
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
"""Test parameter validation before CPU simulator state mutation."""

import numpy as np
import pytest

import mindquantum as mq
from mindquantum.core import gates as G
from mindquantum.core.circuit import Circuit
from mindquantum.core.parameterresolver import ParameterResolver
from mindquantum.simulator import Simulator


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('backend', ['mqvector', 'mqmatrix'])
@pytest.mark.parametrize('dtype', [mq.complex64, mq.complex128])
@pytest.mark.parametrize('method', ['apply_circuit', 'apply_gate'])
@pytest.mark.parametrize('value', [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize('kind', ['dict', 'resolver', 'array', 'list', 'scalar'])
def test_nonfinite_parameters(backend, dtype, method, value, kind):
    """Feature: CPU simulator parameter validation.
    Description: Reject nonfinite bindings before applying any gates.
    Expectation: ValueError is raised and the state is unchanged.
    """
    sim = Simulator(backend, 2, seed=3, dtype=dtype)
    sim.apply_circuit(Circuit().ry(0.3, 0).x(1, 0))
    before = sim.get_qs()
    parameters = {
        'dict': {'a': value},
        'resolver': ParameterResolver({'a': value}),
        'array': np.array([value]),
        'list': [value],
        'scalar': value,
    }
    operation = Circuit().h(0).rx('a', 1) if method == 'apply_circuit' else G.RX('a').on(1)
    with pytest.raises(ValueError, match='finite'):
        getattr(sim, method)(operation, parameters[kind])
    np.testing.assert_array_equal(sim.get_qs(), before)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('backend', ['mqvector', 'mqmatrix'])
@pytest.mark.parametrize('method', ['apply_circuit', 'apply_gate'])
def test_finite_and_empty_parameters(backend, method):
    """Feature: CPU simulator parameter validation.
    Description: Preserve finite bindings and operations without parameters.
    Expectation: Binding a finite value agrees with a constant gate.
    """
    sim = Simulator(backend, 2)
    reference = sim.copy()
    sim.apply_circuit(Circuit(), ParameterResolver())
    np.testing.assert_array_equal(sim.get_qs(), reference.get_qs())
    gate = G.RX('a').on(1)
    operation = Circuit([gate]) if method == 'apply_circuit' else gate
    getattr(sim, method)(operation, {'a': 0.3})
    reference.apply_gate(G.RX(0.3).on(1))
    np.testing.assert_allclose(sim.get_qs(), reference.get_qs())


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('backend', ['mqvector', 'mqmatrix'])
@pytest.mark.parametrize('dtype', [mq.complex64, mq.complex128])
@pytest.mark.parametrize('method', ['apply_circuit', 'apply_gate'])
@pytest.mark.parametrize('value', [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize('kind', ['constant', 'offset', 'coefficient', 'multi_constant', 'multi_mixed'])
def test_nonfinite_gate_coefficients(backend, dtype, method, value, kind):
    """Feature: CPU simulator gate coefficient validation.
    Description: Reject nonfinite constants and coefficients, including mixed circuits and multi-parameter gates.
    Expectation: ValueError is raised before any gate mutates the state.
    """
    sim = Simulator(backend, 2, seed=3, dtype=dtype)
    sim.apply_circuit(Circuit().ry(0.3, 0).x(1, 0))
    before = sim.get_qs()
    gates = {
        'constant': G.RX(value).on(1),
        'offset': G.RX(ParameterResolver({'a': 1}, const=value)).on(1),
        'coefficient': G.RX({'a': value}).on(1),
        'multi_constant': G.U3(0.1, value, 0.2).on(1),
        'multi_mixed': G.U3('a', value, 0.2).on(1),
    }
    gate = gates[kind]
    operation = Circuit().h(0).ry('a', 0) + gate if method == 'apply_circuit' else gate
    with pytest.raises(ValueError, match='Parameter values must be finite'):
        getattr(sim, method)(operation, {'a': 0.2})
    np.testing.assert_array_equal(sim.get_qs(), before)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('backend', ['mqvector', 'mqmatrix'])
@pytest.mark.parametrize('dtype', [mq.complex64, mq.complex128])
def test_nonfinite_constant_circuit(backend, dtype):
    """Feature: CPU simulator constant circuit validation.
    Description: Reject a nonfinite gate in a circuit without symbolic parameters.
    Expectation: Each invalid circuit raises ValueError without changing the state.
    """
    sim = Simulator(backend, 2, dtype=dtype)
    before = sim.get_qs()
    for value in (np.nan, np.inf, -np.inf):
        with pytest.raises(ValueError, match='Parameter values must be finite'):
            sim.apply_circuit(Circuit().h(0).rx(value, 1))
        np.testing.assert_array_equal(sim.get_qs(), before)
