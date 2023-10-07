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
'''test DAG circuit'''
import pytest

from mindquantum.algorithm.compiler.dag import DAGCircuit
from mindquantum.algorithm.compiler.decompose.utils import is_equiv_unitary
from mindquantum.utils import random_circuit


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_dag_circuit():
    """
    Description: Test DAG circuit
    Expectation: success
    """
    circ = random_circuit(3, 100)
    dag = DAGCircuit(circ)
    new_circ = dag.to_circuit()
    assert is_equiv_unitary(circ.matrix(), new_circ.matrix())
