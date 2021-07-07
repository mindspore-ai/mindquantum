#   Copyright (c) 2020 Huawei Technologies Co.,ltd.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   You may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
"""The test projector."""

from mindquantum.gate import Projector
from mindquantum.circuit import UN
from mindquantum import H, RX, RY, RZ
from mindquantum.nn import generate_evolution_operator
from mindquantum.nn import MindQuantumAnsatzOnlyOperator
import mindspore as ms
import numpy as np


def test_projector():
    pro = Projector('II010')
    assert str(pro) == 'I2 ⊗ ¦010⟩⟨010¦'


def test_projector_checked_by_evo():
    circ = UN(H, 3) + RX('a').on(0) + RY('b').on(1) + RZ('c').on(2)
    evo = generate_evolution_operator(circ)
    a, b, c = 0.3, 0.5, 0.9
    data = ms.Tensor(np.array([a, b, c]).astype(np.float32))
    state = evo(data)
    proj = [Projector('I10'), Projector('I10')]
    poi = [int(i, 2) for i in ['010', '110']]
    pqc = MindQuantumAnsatzOnlyOperator(circ.para_name, circ, proj)
    pob = pqc(data)
    pob_exp = np.sum(np.abs(state[poi])**2)
    assert np.allclose(pob.asnumpy(), [[pob_exp], [pob_exp]])
