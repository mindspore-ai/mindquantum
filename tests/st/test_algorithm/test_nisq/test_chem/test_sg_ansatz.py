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
# wITHOUT wARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import pytest
from mindquantum.algorithm import SGAnsatz, SGAnsatz2D


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_sg_ansatz_basic():
    """
    Description: Test basic functionality of SGAnsatz
    Expectation: success.
    """
    # Test basic initialization
    sg = SGAnsatz(4, 2, 1)
    assert len(sg.circuit) > 0
    assert sg.nqubits == 4

    # Test with prefix and suffix
    sg_with_prefix = SGAnsatz(4, 2, 1, prefix='test_')
    assert all('test_' in param for param in sg_with_prefix.circuit.params_name)

    # Test multiple layers
    sg_multi_layer = SGAnsatz(4, 2, nlayers=2)
    assert len(sg_multi_layer.circuit) > len(sg.circuit)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_sg_ansatz_invalid_inputs():
    """
    Description: Test input validation for SGAnsatz
    Expectation: success.
    """
    with pytest.raises(TypeError):
        SGAnsatz('4', 2, 1)  # nqubits must be integer

    with pytest.raises(TypeError):
        SGAnsatz(4, '2', 1)  # k must be integer

    with pytest.raises(ValueError):
        SGAnsatz(4, 2, 0)  # nlayers must be greater than 0


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_sg_ansatz_2d_basic():
    """
    Description: Test basic functionality of SGAnsatz2D
    Expectation: success.
    """
    # Test creation from grid
    sg = SGAnsatz2D.from_grid(2, 3, 2)
    assert sg.nqubits == 6
    assert len(sg.line_set) == 2

    # Test custom line_set
    custom_line_set = [[0, 1, 2, 3, 4, 5], [5, 4, 3, 2, 1, 0]]
    sg_custom = SGAnsatz2D(6, 2, line_set=custom_line_set)
    assert len(sg_custom.circuit) > 0


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_generate_line_set():
    """
    Description: Test line set generation functionality
    Expectation: success.
    """
    # Test 2x3 grid
    line_set = SGAnsatz2D.generate_line_set(2, 3)
    assert len(line_set) == 2
    assert len(line_set[0]) == 6
    assert len(line_set[1]) == 6

    # Verify generated paths contain all nodes
    assert set(line_set[0]) == set(range(6))
    assert set(line_set[1]) == set(range(6))


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_sg_ansatz_2d_invalid_inputs():
    """
    Description: Test input validation for SGAnsatz2D
    Expectation: success.
    """
    with pytest.raises(TypeError):
        SGAnsatz2D('6', 2)  # nqubits must be integer

    with pytest.raises(ValueError):
        # Each line in line_set must have length equal to nqubits
        invalid_line_set = [[0, 1, 2], [3, 4, 5]]
        SGAnsatz2D(6, 2, line_set=invalid_line_set)

    with pytest.raises(TypeError):
        SGAnsatz2D.from_grid('2', 3, 2)  # nrow must be integer


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_circuit_generation():
    """
    Description: Test correctness of circuit generation
    Expectation: success.
    """
    # Test 1D case
    sg_1d = SGAnsatz(4, 2, 1)
    circuit_1d = sg_1d.circuit
    # Verify gate types in circuit
    assert any(gate.name in ['RX', 'RY', 'RZ'] for gate in circuit_1d)

    # Test 2D case
    sg_2d = SGAnsatz2D.from_grid(2, 2, 2)
    circuit_2d = sg_2d.circuit
    # Verify gate types in circuit
    assert any(gate.name in ['RX', 'RY', 'RZ'] for gate in circuit_2d)
