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
"""Thread scheduling of the CPU simulators (MindQuantum 0.13 behaviour change).

0.13 removed ``Simulator.set_threads_number``: the number of OpenMP threads is
now derived from the workload and capped by ``OMP_NUM_THREADS``.  Gradient
batches share one thread budget, so the caller supplied ``parallel_worker``
must only change the scheduling, never the numerical result.
"""

import numpy as np
import pytest

import mindquantum as mq
from mindquantum.core.gates import RX
from mindquantum.core.operators import Hamiltonian, QubitOperator
from mindquantum.simulator import Simulator
from mindquantum.simulator.utils import _thread_balance

pytestmark = [pytest.mark.level0, pytest.mark.platform_x86_cpu]


def test_set_threads_number_has_been_removed():
    """
    Feature: Removal of ``Simulator.set_threads_number``.

    Description: Inspect and access the historical thread setter of the public simulator.

    Expectation: The attribute is gone and accessing it raises ``AttributeError``.
    """
    simulator = Simulator('mqvector', 2)
    assert not hasattr(simulator, 'set_threads_number')
    with pytest.raises(AttributeError, match='set_threads_number'):
        simulator.set_threads_number(2)


@pytest.mark.parametrize('backend', ['mqvector', 'mqmatrix'])
def test_simulator_uses_available_threads(backend):
    """
    Feature: Adaptive OpenMP scheduling.

    Description: Run a circuit wide enough to be distributed over several threads.

    Expectation: The evolution is correct and the state stays normalized, independent of the thread count.
    """
    n_qubits = 8
    simulator = Simulator(backend, n_qubits, seed=11)
    circuit = mq.Circuit()
    for qubit in range(n_qubits):
        circuit += RX(0.1 * (qubit + 1)).on(qubit)
    simulator.apply_circuit(circuit)
    state = simulator.get_qs()
    assert np.isfinite(state).all()
    if backend == 'mqvector':
        assert np.isclose(np.linalg.norm(state), 1.0)
    else:
        assert np.isclose(np.trace(state), 1.0)


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

    Expectation: The batch and measurement teams are positive, never exceed the budget and follow the documented split.
    """
    batch_threads, measure_threads = _thread_balance(n_prs, n_meas, worker)
    assert (batch_threads, measure_threads) == expected
    assert batch_threads >= 1
    assert measure_threads >= 1
    if worker is not None:
        assert batch_threads * measure_threads <= worker


@pytest.mark.parametrize('backend', ['mqvector', 'mqmatrix'])
@pytest.mark.parametrize('parallel_worker', [1, 2, 3, None])
def test_parallel_worker_does_not_change_gradient(backend, parallel_worker):
    """
    Feature: Thread-invariant batched gradient evaluation.

    Description: Evaluate the same gradient operator with different worker budgets on the CPU backends.

    Expectation: Values and gradients are identical to the single-threaded reference.
    """
    circuit = mq.Circuit().ry('a', 0).rz('b', 1).rx('c', 0).as_encoder()
    hamiltonian = Hamiltonian(QubitOperator('Z0') + 0.5 * QubitOperator('X1'))
    simulator = Simulator(backend, 2, seed=5)
    values = np.array([[0.1, -0.2, 0.3], [0.4, 0.5, -0.6], [-0.7, 0.8, 0.9], [0.2, 0.2, 0.2]])

    reference = simulator.get_expectation_with_grad(hamiltonian, circuit, parallel_worker=1)(values)
    candidate = simulator.get_expectation_with_grad(hamiltonian, circuit, parallel_worker=parallel_worker)(values)

    np.testing.assert_allclose(candidate[0], reference[0], atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(candidate[1], reference[1], atol=1e-10, rtol=1e-10)
