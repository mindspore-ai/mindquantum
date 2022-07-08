#   Copyright 2022 <Huawei Technologies Co., Ltd>
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

# pylint: disable=no-member

"""Test gate decomposition."""

import pytest

try:
    from mindquantum.experimental import optimizer
    from mindquantum.experimental.circuit import Circuit
    from mindquantum.experimental.ops import X
except ImportError:
    pytest.skip("MindQuantum experimental C++ module not present", allow_module_level=True)

# ==============================================================================


@pytest.mark.cxx_exp_projectq
def test_gate_cancellation():
    """
    Description: Test gate cancellation
    Expectation: Success
    """
    gate_cancellation = optimizer.GateCancellation()
    ref = Circuit()
    q0 = ref.create_qubit()
    q1 = ref.create_qubit()
    ref.apply_operator(X(), [q0])
    ref.apply_operator(X(), [q0, q1])

    circuit = Circuit()
    q0 = circuit.create_qubit()
    q1 = circuit.create_qubit()
    circuit.apply_operator(X(), [q0])
    circuit.apply_operator(X(), [q0, q1])

    gate_cancellation.run_circuit(circuit)

    assert str(circuit) == str(ref)

    circuit.apply_operator(X(), [q0, q1])
    circuit.apply_operator(X(), [q0])

    gate_cancellation.run_circuit(circuit)

    print(circuit)
    assert str(circuit) != str(ref)

    empty = Circuit()
    empty.create_qubit()
    empty.create_qubit()
    assert str(circuit) == str(empty)


# ==============================================================================
