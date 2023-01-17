<<<<<<<< HEAD:tests/st/test_algorithm/test_compiler/test_decompose/test_fixed_decompose/test_yy_decompose.py
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

# pylint: disable=invalid-name

'''test decompose rule'''
import numpy as np
import pytest

from mindquantum.algorithm.compiler.decompose import cyy_decompose, yy_decompose
from mindquantum.config import Context
from mindquantum.core.circuit import Circuit
from mindquantum.core.gates import YY


def circuit_equal_test(gate, decompose_circ):
    """
    require two circuits are equal.
    """
    orig_circ = Circuit() + gate
    assert np.allclose(orig_circ.matrix(), decompose_circ.matrix())


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('dtype', ['float', 'double'])
def test_yy(dtype):
    """
    Description: Test yy decompose
    Expectation: success
    """
    Context.set_dtype(dtype)
    yy = YY(1).on([0, 1])
    for solution in yy_decompose(yy):
        circuit_equal_test(yy, solution)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('dtype', ['float', 'double'])
def test_cyy(dtype):
    """
    Description: Test cyy decompose
    Expectation: success
    """
    Context.set_dtype(dtype)
    cyy = YY(2).on([0, 1], [2, 3])
    for solution in cyy_decompose(cyy):
        circuit_equal_test(cyy, solution)
========
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

# pylint: disable=invalid-name

'''test decompose rule'''
import numpy as np
import pytest

from mindquantum.algorithm.compiler.decompose import cryy_decompose, ryy_decompose
from mindquantum.config import Context
from mindquantum.core.circuit import Circuit
from mindquantum.core.gates import Ryy


def circuit_equal_test(gate, decompose_circ):
    """
    require two circuits are equal.
    """
    orig_circ = Circuit() + gate
    assert np.allclose(orig_circ.matrix(), decompose_circ.matrix())


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('dtype', ['float', 'double'])
def test_ryy(dtype):
    """
    Description: Test ryy decompose
    Expectation: success
    """
    Context.set_dtype(dtype)
    ryy = Ryy(1).on([0, 1])
    for solution in ryy_decompose(ryy):
        circuit_equal_test(ryy, solution)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('dtype', ['float', 'double'])
def test_cryy(dtype):
    """
    Description: Test cryy decompose
    Expectation: success
    """
    Context.set_dtype(dtype)
    cryy = Ryy(2).on([0, 1], [2, 3])
    for solution in cryy_decompose(cryy):
        circuit_equal_test(cryy, solution)
>>>>>>>> e4807eae (xx yy zz to rxx ryy rzz):tests/st/test_algorithm/test_compiler/test_decompose/test_ryy_decompose.py
