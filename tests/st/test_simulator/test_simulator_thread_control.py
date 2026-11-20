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
"""Worker budgeting and numerical invariance of CPU gradients."""

import numpy as np
import pytest

import mindquantum as mq
from mindquantum.core.operators import Hamiltonian, QubitOperator
from mindquantum.simulator import Simulator
from mindquantum.simulator.utils import _thread_balance

pytestmark = [pytest.mark.level0, pytest.mark.platform_x86_cpu]


@pytest.mark.parametrize(
    'n_prs,n_meas,worker,expected',
    [
        pytest.param(1, 1, None, (1, 1), id='single'),
        pytest.param(4, 3, None, (4, 3), id='whole-batch'),
        pytest.param(4, 3, 100, (4, 3), id='budget-larger-than-batch'),
        pytest.param(4, 3, 2, (2, 1), id='batch-limited'),
        pytest.param(3, 6, 4, (1, 4), id='measurement-limited'),
        pytest.param(8, 8, 1, (1, 1), id='serial'),
    ],
)
def test_thread_balance_uses_one_budget(n_prs, n_meas, worker, expected):
    """
    Feature: Gradient thread budgeting.

    Description: Split a batch of parameter sets and Hamiltonians over an optional worker budget.

    Expectation: The batch and measurement teams match the documented split exactly.
    """
    assert _thread_balance(n_prs, n_meas, worker) == expected


@pytest.mark.parametrize('backend', ['mqvector', 'mqmatrix'])
@pytest.mark.parametrize('parallel_worker', [2, 3, None])
def test_parallel_worker_does_not_change_gradient(backend, parallel_worker):
    """
    Feature: Thread-invariant batched gradient evaluation.

    Description: Evaluate the same gradient operator with different worker budgets on the CPU backends.

    Expectation: Values and gradients are identical to the single-threaded reference.
    """
    circuit = mq.Circuit().ry('a', 0).rz('b', 1).rx('c', 0).as_encoder()
    hamiltonian = [Hamiltonian(QubitOperator('Z0')), Hamiltonian(QubitOperator('X0'))]
    simulator = Simulator(backend, 2, seed=5)
    values = np.array([[0.1, -0.2, 0.3], [0.4, 0.5, -0.6], [-0.7, 0.8, 0.9], [0.2, 0.2, 0.2]])

    reference = simulator.get_expectation_with_grad(hamiltonian, circuit, parallel_worker=1)(values)
    candidate = simulator.get_expectation_with_grad(hamiltonian, circuit, parallel_worker=parallel_worker)(values)

    np.testing.assert_allclose(candidate[0], reference[0], atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(candidate[1], reference[1], atol=1e-10, rtol=1e-10)
