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
'''test compiler rules'''
import pytest

from mindquantum.algorithm.compiler import (
    BasicDecompose,
    SequentialCompiler,
    KroneckerSeqCompiler,
    SimpleNeighborCanceler,
    FullyNeighborCanceler,
    CXToCZ,
    CZToCX,
    CZBasedChipCompiler,
    DAGCircuit,
)
from mindquantum.algorithm.compiler.decompose.utils import is_equiv_unitary
from mindquantum.utils import random_circuit


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_basic_decompose():
    """
    Description: Test BasicDecompose
    Expectation: success
    """
    circ = random_circuit(3, 100)
    dag_circ = DAGCircuit(circ)
    rule = BasicDecompose()
    rule.do(dag_circ)
    new_circ = dag_circ.to_circuit()
    assert is_equiv_unitary(new_circ.matrix(), circ.matrix())

    circ = random_circuit(3, 100)
    dag_circ = DAGCircuit(circ)
    rule = BasicDecompose(True)
    rule.do(dag_circ)
    new_circ = dag_circ.to_circuit()
    assert is_equiv_unitary(new_circ.matrix(), circ.matrix())


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_sequential_compiler():
    """
    Description: Test SequentialCompiler
    Expectation: success
    """
    circ = random_circuit(3, 100)
    dag_circ = DAGCircuit(circ)
    rule = SequentialCompiler([BasicDecompose(), SimpleNeighborCanceler()])
    rule.do(dag_circ)
    new_circ = dag_circ.to_circuit()
    assert is_equiv_unitary(new_circ.matrix(), circ.matrix())


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_kronecker_seq_compiler():
    """
    Description: Test KroneckerSeqCompiler
    Expectation: success
    """
    circ = random_circuit(3, 100)
    dag_circ = DAGCircuit(circ)
    rule = KroneckerSeqCompiler([BasicDecompose(), SimpleNeighborCanceler()])
    rule.do(dag_circ)
    new_circ = dag_circ.to_circuit()
    assert is_equiv_unitary(new_circ.matrix(), circ.matrix())


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_fully_neighbor_canceler():
    """
    Description: Test FullyNeighborCanceler
    Expectation: success
    """
    circ = random_circuit(3, 100)
    dag_circ = DAGCircuit(circ)
    rule = FullyNeighborCanceler()
    rule.do(dag_circ)
    new_circ = dag_circ.to_circuit()
    assert is_equiv_unitary(new_circ.matrix(), circ.matrix())


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_cz_to_cx():
    """
    Description: Test CZToCX
    Expectation: success
    """
    circ = random_circuit(3, 100)
    dag_circ = DAGCircuit(circ)
    rule = CZToCX()
    rule.do(dag_circ)
    new_circ = dag_circ.to_circuit()
    assert is_equiv_unitary(new_circ.matrix(), circ.matrix())


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_cx_to_cz():
    """
    Description: Test CXToCZ
    Expectation: success
    """
    circ = random_circuit(3, 100)
    dag_circ = DAGCircuit(circ)
    rule = CXToCZ()
    rule.do(dag_circ)
    new_circ = dag_circ.to_circuit()
    assert is_equiv_unitary(new_circ.matrix(), circ.matrix())


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_cz_based_chip_compiler():
    """
    Description: Test CZBasedChipCompiler
    Expectation: success
    """
    circ = random_circuit(3, 100)
    dag_circ = DAGCircuit(circ)
    rule = CZBasedChipCompiler()
    rule.do(dag_circ)
    new_circ = dag_circ.to_circuit()
    assert is_equiv_unitary(new_circ.matrix(), circ.matrix())
