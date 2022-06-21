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
    warnings.simplefilter('ignore', category=DeprecationWarning)
    from mindquantum.algorithm.compiler.decompose import ch_decompose
    from mindquantum.core import Circuit, H


def circuit_equal_test(gate, decompose_circ):
    """
    require two circuits are equal.
    """
    orig_circ = Circuit() + gate
    assert np.allclose(orig_circ.matrix(), decompose_circ.matrix())


def test_ch():
    """
    Description: Test ch decompose
    Expectation: success
    """
    ch = H.on(1, 0)
    for solution in ch_decompose(ch):
        circuit_equal_test(ch, solution)
