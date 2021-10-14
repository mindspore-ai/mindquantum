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
"""Basic mindquanutm neural layer."""

import numpy as np
from mindspore import Tensor
import mindspore.nn as nn
from mindspore.common.parameter import Parameter
from mindspore.common.initializer import initializer
from .pqc import generate_pqc_operator
from .evolution import generate_evolution_operator


class MindQuantumLayer(nn.Cell):
    """
    A trainable Mindquantum layer.

    A mindquantum layer simulate a parameterized quantum circuit and get the
    measurement result. The quantum circuit is construct by a encode circuit
    and a trainable ansatz circuit. The encode circuit will encode classical
    data into quantum state, and the trainable ansatz circuit apply on the
    quantum state.

    Args:
        encoder_params_names (list[str]): Parameters names of encoder circuit.
            The order of this parameters is the same as the order of data
            passed in construct.
        ansatz_params_names (list[str]): Parameters names of ansatz circuit.
            The order of this parameters is the same as the order of trainable
            parameters.
        circuit (Circuit): Quantum circuit construct by
            encode circuit and ansatz circuit.
        measurements (Union[Hamiltonian, list[Hamiltonian], Projector, list[Projector]]):
            Hamiltonian or a list of Hamiltonian for measurement.
        weight_init (Union[Tensor, str, Initializer, numbers.Number]): The
            trainable weight_init parameter. The dtype is same as input x. The
            values of str refer to the function `initializer`.
            Default: 'normal'.
        n_threads (int): Number of threads for data parallel. Default: 1.

    Inputs:
        - **input** (Tensor) - Tensor of shape :math:`(N, E_{in})`, where :math:`N` is batch
          size, :math:`E_{in}` is the number of parameters in encoder circuit.

    Outputs:
        Tensor of shape :math:`(N, H_{out})`, where :math:`H_{out}` is the
        number of hamiltonians.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> from mindquantum.ops import QubitOperator
        >>> from mindquantum.nn import MindQuantumLayer
        >>> from mindquantum import Circuit, Hamiltonian
        >>> import mindquantum.gate as G
        >>> encoder_circ = Circuit([G.RX('a').on(0)])
        >>> encoder_circ.no_grad()
        >>> ansatz = Circuit([G.RY('b').on(0)])
        >>> ham = Hamiltonian(QubitOperator('Z0'))
        >>> net = MindQuantumLayer(['a'], ['b'], encoder_circ + ansatz, ham)
        >>> res = net(Tensor(np.array([[1.0]]).astype(np.float32)))
        >>> res.asnumpy()
        array([[0.54030216]], dtype=float32)
    """
    def __init__(self,
                 encoder_params_names,
                 ansatz_params_names,
                 circuit,
                 measurements,
                 weight_init='normal',
                 n_threads=1):
        super(MindQuantumLayer, self).__init__()
        self.circuit = circuit
        self.measurements = measurements
        self.encoder_params_names = encoder_params_names
        self.ansatz_params_names = ansatz_params_names
        self.pqc = generate_pqc_operator(encoder_params_names,
                                         ansatz_params_names,
                                         circuit,
                                         measurements,
                                         n_threads=n_threads)
        self.weight = Parameter(initializer(weight_init,
                                            len(ansatz_params_names)),
                                name="weight")

    def final_state(self,
                    encoder_data,
                    ansatz_data=None,
                    circuit=None,
                    measurements=None):
        """
        Get the quantum state after evolution.

        Args:
            encoder_data (Tensor): A one dimension tensor for encoder circuit.
            ansatz_data (Tensor): A one dimension tensor for ansatz circuit. If
                ansatz_data is None, then the traind parameter will be used.
                Default: None.
            circuit (Circuit): Quantum circuit construct
                by encode circuit and ansatz circuit. If None, the circuit for
                train will be used. Default: None.
            measurements (list[Hamiltonian]): Hamiltonian or a
                list of Hamiltonian for measurement. If None, no hamiltonians
                will be used. Default: None.

        Returns:
            numpy.ndarray, the final quantum state.
        """
        if circuit is None:
            circuit = self.circuit
        if ansatz_data is None:
            ansatz_data = self.weight
        if len(encoder_data.shape) != 1:
            raise ValueError("Except a one dimension tensor for encoder_data!")
        data = np.array([])
        data = np.append(data, encoder_data.asnumpy())
        data = np.append(data, ansatz_data.asnumpy())
        data = Tensor(data.astype(np.float32))
        param_names = []
        param_names.extend(self.encoder_params_names)
        param_names.extend(self.ansatz_params_names)
        evol = generate_evolution_operator(circuit, param_names, measurements)
        state = evol(data)
        return state

    def construct(self, x):
        x, _, _ = self.pqc(x, self.weight)
        return x
