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
"""Basic mindquanutm neural layer with ansatz only."""

import numpy as np
from mindspore import Tensor
from mindquantum.circuit import Circuit
from .mindquantum_layer import MindQuantumLayer


class MindQuantumAnsatzOnlyOperator(MindQuantumLayer):
    """
    An Ansatz only Mindquantum operator.

    This operator only need an ansatz circuit and the parameter data for ansatz
    circuit.

    Args:
        param_names (list[str]): Parameters names of ansatz circuit.
            The order of this parameters is the same as the order of trainable
            parameters.
        circuit (Circuit): The ansatz circuit.
        measurements (Union[Hamiltonian, list[Hamiltonian], Projector, list[Projector]]):
            Hamiltonian or a list of Hamiltonian for measurement.
        n_threads (int): Number of threads for data parallel. Default: 1.

    Inputs:
        - **input** (Tensor) - Tensor of shape :math:`(E_{in}, )`,
          where :math:`E_{in}` is the number of parameters in ansatz circuit.

    Outputs:
        Tensor of shape :math:`(1, 1)`.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> from mindquantum import Circuit, RX, Hamiltonian
        >>> from mindquantum.ops import QubitOperator
        >>> from mindquantum.nn import MindQuantumAnsatzOnlyOperator
        >>> from mindspore import Tensor
        >>> import numpy as np
        >>> circuit = Circuit(RX('a').on(0))
        >>> ham = Hamiltonian(QubitOperator('Z0'))
        >>> mea = MindQuantumAnsatzOnlyOperator(circuit.para_name, circuit, ham)
        >>> data = Tensor(np.array([0.5]).astype(np.float32))
        >>> mea(data)
        Tensor(shape=[1, 1], dtype=Float32, value=
        [[ 8.77582550e-01]])
    """
    def __init__(self, param_names, circuit, measurements, n_threads=1):
        circuit, dummy_para = _add_dummy_encoder(circuit)
        super(MindQuantumAnsatzOnlyOperator,
              self).__init__([dummy_para], param_names, circuit, measurements,
                             'normal', n_threads)
        self.fake_data = Tensor(np.array([[0]]).astype(np.float32))
        del self.weight

    def construct(self, data):
        x, _, _ = self.pqc(self.fake_data, data)
        return x


def _add_dummy_encoder(circ):
    """add a dummy parameterized gate"""
    para_name = circ.para_name
    index = 0
    while True:
        name = f'_d_{index}'
        if name not in para_name:
            dummy_circ = Circuit().rx(name, 0).no_grad()
            return dummy_circ + circ, name
        index += 1


class MindQuantumAnsatzOnlyLayer(MindQuantumLayer):
    """
    An ansatz only trainable Mindquantum layer.

    A mindquantum layer simulate a parameterized quantum circuit and get the
    measurement result. The quantum circuit is construct only by an ansatz circuit.

    Args:
        param_names (list[str]): Parameters names of ansatz circuit.
            The order of this parameters is the same as the order of trainable
            parameters.
        circuit (Circuit): The ansatz circuit.
        measurements (Union[Hamiltonian, list[Hamiltonian], Projector, list[Projector]]):
            Hamiltonian or a list of Hamiltonian for measurement.
        weight_init (Union[Tensor, str, Initializer, numbers.Number]): The
            trainable weight_init parameter. The dtype is same as input x. The
            values of str refer to the function `initializer`.
            Default: 'normal'.
        n_threads (int): Number of threads for data parallel. Default: 1.

    Inputs:
        No inputs needed.

    Outputs:
        Tensor of shape :math:`(1, 1)`.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> from mindquantum import Circuit, H, RX, RY, RZ, Hamiltonian
        >>> from mindquantum.nn import MindQuantumAnsatzOnlyLayer
        >>> from mindquantum.ops import QubitOperator
        >>> from mindspore import Tensor
        >>> import mindspore.nn as nn
        >>> import numpy as np
        >>> circuit = Circuit([H.on(0), RZ(0.4).on(0), RX('a').on(0), RY('b').on(0)])
        >>> ham = Hamiltonian(QubitOperator('Z0'))
        >>> init = Tensor(np.array([0, 0]).astype(np.float32))
        >>> net = MindQuantumAnsatzOnlyLayer(circuit.para_name, circuit, ham, init)
        >>> opti = nn.Adam(net.trainable_params(), learning_rate=0.8)
        >>> train_net = nn.TrainOneStepCell(net, opti)
        >>> for i in range(1000):
        ...     train_net()
        >>> net()
        Tensor(shape=[1, 1], dtype=Float32, value=
        [[-1.00000000e+00]])
        >>> net.weight.asnumpy()
        array([-4.712389 ,  1.9707963], dtype=float32)

    """
    def __init__(self,
                 param_names,
                 circuit,
                 measurements,
                 weight_init='normal',
                 n_threads=1):
        circuit, dummy_para = _add_dummy_encoder(circuit)
        super(MindQuantumAnsatzOnlyLayer,
              self).__init__([dummy_para], param_names, circuit, measurements,
                             weight_init, n_threads)
        self.fake_data = Tensor(np.array([[0]]).astype(np.float32))

    def construct(self):
        x, _, _ = self.pqc(self.fake_data, self.weight)
        return x

    def final_state(self, measurements=None):
        """
        Get the quantum state after evolution.

        Args:
            measurements (Hamiltonian): Hamiltonian for measurement. If None, no hamiltonians
                will be used. Default: None.

        Returns:
            numpy.ndarray, the final quantum state.
        """
        fake_data = Tensor(np.array([]).astype(np.float32))
        return super().final_state(fake_data,
                                   self.weight,
                                   measurements=measurements)
