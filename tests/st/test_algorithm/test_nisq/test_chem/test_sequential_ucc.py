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
"""Test SequentialUCCAnsatz"""

import pytest

from mindquantum.algorithm.nisq import SequentialUCCAnsatz
from mindquantum.core.operators import FermionOperator


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_sequential_ucc_append_and_circuit():
    """
    Description: Test that appending operators builds the circuit correctly.
    Expectation: Circuit length and parameter names match appended operators.
    """
    ansatz = SequentialUCCAnsatz()
    ansatz.append(FermionOperator("3^ 1", "a"))
    ansatz.append(FermionOperator("4^ 2", "b"))
    ansatz.append(FermionOperator("3^ 1", "c"))
    assert len(ansatz.circuit) == 3
    assert ansatz.circuit.params_name == ['a', 'b', 'c']


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_sequential_ucc_iadd():
    """
    Description: Test the += operator for appending operators.
    Expectation: += behaves identically to append().
    """
    ansatz = SequentialUCCAnsatz()
    ansatz += FermionOperator("3^ 1", "x")
    ansatz += FermionOperator("4^ 2", "y")
    assert len(ansatz) == 2
    assert ansatz.circuit.params_name == ['x', 'y']


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_sequential_ucc_remove():
    """
    Description: Test removing an operator by index.
    Expectation: The removed operator is returned and the circuit is rebuilt.
    """
    ansatz = SequentialUCCAnsatz()
    ansatz.append(FermionOperator("3^ 1", "a"))
    ansatz.append(FermionOperator("4^ 2", "b"))
    ansatz.append(FermionOperator("5^ 3", "c"))

    removed = ansatz.remove(1)
    assert removed == FermionOperator("4^ 2", "b")
    assert len(ansatz) == 2
    assert ansatz.circuit.params_name == ['a', 'c']


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_sequential_ucc_remove_out_of_range():
    """
    Description: Test that removing with an invalid index raises IndexError.
    Expectation: IndexError is raised.
    """
    ansatz = SequentialUCCAnsatz()
    ansatz.append(FermionOperator("3^ 1", "a"))
    with pytest.raises(IndexError):
        ansatz.remove(5)
    with pytest.raises(IndexError):
        ansatz.remove(-1)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_sequential_ucc_len_and_getitem():
    """
    Description: Test __len__ and __getitem__ container protocol.
    Expectation: len() returns count and indexing returns operators.
    """
    ansatz = SequentialUCCAnsatz()
    assert len(ansatz) == 0

    op_a = FermionOperator("3^ 1", "a")
    op_b = FermionOperator("4^ 2", "b")
    ansatz.append(op_a)
    ansatz.append(op_b)

    assert len(ansatz) == 2
    assert ansatz[0] == op_a
    assert ansatz[1] == op_b


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_sequential_ucc_operators_returns_copy():
    """
    Description: Test that operators property returns detached copies.
    Expectation: Mutating returned operators does not affect ansatz internals.
    """
    ansatz = SequentialUCCAnsatz()
    op = FermionOperator("3^ 1", "a")
    ansatz.append(op)
    ops = ansatz.operators
    ops[0] += FermionOperator("4^ 2", "b")
    ops.clear()
    assert len(ansatz) == 1
    assert ansatz[0] == op


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_sequential_ucc_getitem_returns_copy():
    """
    Description: Test that __getitem__ returns detached operator objects.
    Expectation: Mutating the fetched operator does not affect ansatz internals.
    """
    ansatz = SequentialUCCAnsatz()
    ansatz.append(FermionOperator("3^ 1", "a"))

    fetched = ansatz[0]
    fetched += FermionOperator("4^ 2", "b")
    assert ansatz[0] == FermionOperator("3^ 1", "a")


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_sequential_ucc_type_error():
    """
    Description: Test that appending a non-FermionOperator raises TypeError.
    Expectation: TypeError is raised.
    """
    ansatz = SequentialUCCAnsatz()
    with pytest.raises(TypeError):
        ansatz.append("not an operator")


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_sequential_ucc_multi_term_error():
    """
    Description: Test that appending a multi-term FermionOperator raises ValueError.
    Expectation: ValueError is raised.
    """
    ansatz = SequentialUCCAnsatz()
    multi_term = FermionOperator("3^ 1", "a") + FermionOperator("4^ 2", "b")
    with pytest.raises(ValueError):
        ansatz.append(multi_term)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_sequential_ucc_identity_term_error():
    """
    Description: Test that appending an identity FermionOperator is rejected.
    Expectation: ValueError is raised.
    """
    ansatz = SequentialUCCAnsatz()
    with pytest.raises(ValueError, match="identity term"):
        ansatz.append(FermionOperator("", "a"))


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_sequential_ucc_duplicate_index_error():
    """
    Description: Test duplicate indices are rejected during append validation.
    Expectation: ValueError is raised.
    """
    ansatz = SequentialUCCAnsatz()
    with pytest.raises(ValueError, match="duplicate indices"):
        ansatz.append(FermionOperator("1^ 1", "a"))


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_sequential_ucc_n_qubits_validation():
    """
    Description: Test that n_qubits validation prevents out-of-range operators.
    Expectation: ValueError is raised when operator exceeds n_qubits.
    """
    ansatz = SequentialUCCAnsatz(n_qubits=4)
    ansatz.append(FermionOperator("3^ 1", "a"))  # max qubit 3, within 4 qubits
    with pytest.raises(ValueError, match="exceeds n_qubits"):
        ansatz.append(FermionOperator("5^ 1", "b"))  # qubit 5 exceeds n_qubits=4


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_sequential_ucc_constructor_validation():
    """
    Description: Test constructor validates system size arguments.
    Expectation: TypeError/ValueError raised for invalid initialization.
    """
    with pytest.raises(TypeError):
        SequentialUCCAnsatz(n_qubits='4')
    with pytest.raises(ValueError):
        SequentialUCCAnsatz(n_qubits=0)
    with pytest.raises(TypeError):
        SequentialUCCAnsatz(n_electrons='2')
    with pytest.raises(ValueError):
        SequentialUCCAnsatz(n_electrons=-1)
    with pytest.raises(ValueError, match="electrons must be smaller"):
        SequentialUCCAnsatz(n_qubits=2, n_electrons=3)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_sequential_ucc_no_n_qubits_allows_any():
    """
    Description: Test that without n_qubits, any operator indices are accepted.
    Expectation: No error is raised.
    """
    ansatz = SequentialUCCAnsatz()
    ansatz.append(FermionOperator("100^ 50", "a"))
    assert len(ansatz) == 1


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_sequential_ucc_append_is_defensive_copy():
    """
    Description: Test that append stores a detached operator copy.
    Expectation: Mutating original operator after append does not break ansatz.
    """
    ansatz = SequentialUCCAnsatz()
    source = FermionOperator("3^ 1", "a")
    ansatz.append(source)
    source += FermionOperator("4^ 2", "b")

    removed = ansatz.remove(0)
    assert removed == FermionOperator("3^ 1", "a")
    assert len(ansatz.circuit) == 0
