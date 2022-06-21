# -*- coding: utf-8 -*-
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
'''test decompose rule'''
import warnings

import numpy as np

with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=UserWarning, message='MindSpore not installed.*')
    warnings.filterwarnings(
        'ignore', category=DeprecationWarning, message=r'Please use `OptimizeResult` from the `scipy\.optimize`'
    )
    from mindquantum.algorithm.compiler.decompose import ccx_decompose
    from mindquantum.core import Circuit, X


def circuit_equal_test(gate, decompose_circ):
    """
    require two circuits are equal.
    """
    orig_circ = Circuit() + gate
    assert np.allclose(orig_circ.matrix(), decompose_circ.matrix())


def test_ccx():
    """
    Description: Test ccx decompose
    Expectation: success
    """
    ccx = X.on(1, [0, 2])
    for solution in ccx_decompose(ccx):
        circuit_equal_test(ccx, solution)
