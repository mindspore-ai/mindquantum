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
"""Test max_2_sat"""

import os
os.environ['OMP_NUM_THREADS'] = '8'
import numpy as np
import mindspore as ms
from mindquantum.ansatz import Max2SATAnsatz
from mindquantum.nn import MindQuantumAnsatzOnlyLayer as MAL
from mindquantum.gate import Hamiltonian

ms.context.set_context(mode=ms.context.GRAPH_MODE, device_target="CPU")


def test_max_2_sat():
    clauses = [(1, 2), (1, -2), (-1, 2), (-1, -2), (1, 3)]
    depth = 3
    max2sat = Max2SATAnsatz(clauses, depth)
    ms.set_seed(42)
    net = MAL(max2sat.circuit.para_name, max2sat.circuit,
              Hamiltonian(max2sat.hamiltonian))
    opt = ms.nn.Adagrad(net.trainable_params(), learning_rate=4e-1)
    train_net = ms.nn.TrainOneStepCell(net, opt)
    ret = 0
    for i in range(100):
        ret = train_net().asnumpy()[0, 0]
    assert np.allclose(round(ret, 3), 1)
