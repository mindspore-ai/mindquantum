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
"""Mindspore quantum simulator layer."""
import mindspore as ms
import mindspore.nn as nn
from mindspore.common.parameter import Parameter
from mindspore.common.initializer import initializer
from .operations import MQOps
from .operations import MQN2Ops
from .operations import MQAnsatzOnlyOps
from .operations import MQN2AnsatzOnlyOps


class MQLayer(nn.Cell):
    """
    MindQuantum trainable layer. The parameters of ansatz circuit are trainable parameters.

    Args:
        expectation_with_grad (GradOpsWrapper): a grad ops that receive encoder data and
            ansatz data and return the expectation value and gradient value of parameters
            respect to expectation.
        weight (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the
            convolution kernel. It can be a Tensor, a string, an Initializer or a number.
            When a string is specified, values from 'TruncatedNormal', 'Normal', 'Uniform',
            'HeUniform' and 'XavierUniform' distributions as well as constant 'One' and 'Zero'
            distributions are possible. Alias 'xavier_uniform', 'he_uniform', 'ones' and
            'zeros' are acceptable. Uppercase and lowercase are both acceptable. Refer to
            the values of Initializer for more details. Default: 'normal'.

    Inputs:
        - **enc_data** (Tensor) - Tensor of encoder data that you want to encode into quantum state.

    Outputs:
        Tensor, The expectation value of the hamiltonian.

    Raises:
        ValueError: If length of shape of `weight` is not equal to 1 or shape[0] of `weight`
                    is not equal to `weight_size`.

    Supported Platforms:
        ``GPU``, ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindquantum import Circuit, Hamiltonian, QubitOperator
        >>> from mindquantum import Simulator, MQLayer
        >>> import mindspore as ms
        >>> ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="CPU")
        >>> enc = Circuit().ry('a', 0)
        >>> ans = Circuit().h(0).rx('b', 0)
        >>> ham = Hamiltonian(QubitOperator('Z0'))
        >>> sim = Simulator('projectq', 1)
        >>> grad_ops = sim.get_expectation_with_grad(ham, enc+ans,
        ...                                          encoder_params_name=['a'],
        ...                                          ansatz_params_name=['b'])
        >>> enc_data = ms.Tensor(np.array([[0.1]]))
        >>> net =  MQLayer(grad_ops)
        >>> opti = ms.nn.Adam(net.trainable_params(), learning_rate=0.1)
        >>> train_net = ms.nn.TrainOneStepCell(net, opti)
        >>> for i in range(100):
        ...     train_net(enc_data)
        >>> net.weight.asnumpy()
        array([-3.1424556], dtype=float32)
        >>> net(enc_data)
        Tensor(shape=[1, 1], dtype=Float32, value=
        [[-9.98333767e-02]])
    """
    def __init__(self, expectation_with_grad, weight='normal'):
        super(MQLayer, self).__init__()
        self.evolution = MQOps(expectation_with_grad)
        weight_size = len(self.evolution.expectation_with_grad.ansatz_params_name)
        if isinstance(weight, ms.Tensor):
            if weight.ndim != 1 or weight.shape[0] != weight_size:
                raise ValueError(f"Weight init shape error, required ({weight_size}, ), but get {weight.shape}.")
        self.weight = Parameter(initializer(weight, weight_size, dtype=ms.float32), name='ansatz_weight')

    def construct(self, x):
        return self.evolution(x, self.weight)


class MQN2Layer(nn.Cell):
    """
    MindQuantum trainable layer. The parameters of ansatz circuit are trainable parameters.
    This layer will calculate the square of absolute value of expectation automatically.

    Args:
        expectation_with_grad (GradOpsWrapper): a grad ops that receive encoder data and
            ansatz data and return the square of absolute value of expectation value and
            gradient value of parameters respect to expectation.
        weight (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the
            convolution kernel. It can be a Tensor, a string, an Initializer or a number.
            When a string is specified, values from 'TruncatedNormal', 'Normal', 'Uniform',
            'HeUniform' and 'XavierUniform' distributions as well as constant 'One' and 'Zero'
            distributions are possible. Alias 'xavier_uniform', 'he_uniform', 'ones' and
            'zeros' are acceptable. Uppercase and lowercase are both acceptable. Refer to
            the values of Initializer for more details. Default: 'normal'.

    Inputs:
        - **enc_data** (Tensor) - Tensor of encoder data that you want to encode into quantum state.

    Outputs:
        Tensor, The square of absolute value of expectation value of the hamiltonian.

    Raises:
        ValueError: If length of shape of `weight` is not equal to 1 and shape[0] of `weight`
                    is not equal to `weight_size`.

    Supported Platforms:
        ``GPU``, ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindquantum import Circuit, Hamiltonian, QubitOperator
        >>> from mindquantum import Simulator, MQN2Layer
        >>> import mindspore as ms
        >>> ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="CPU")
        >>> enc = Circuit().ry('a', 0)
        >>> ans = Circuit().h(0).rx('b', 0)
        >>> ham = Hamiltonian(QubitOperator('Z0'))
        >>> sim = Simulator('projectq', 1)
        >>> grad_ops = sim.get_expectation_with_grad(ham, enc+ans,
        ...                                          encoder_params_name=['a'],
        ...                                          ansatz_params_name=['b'])
        >>> enc_data = ms.Tensor(np.array([[0.1]]))
        >>> net =  MQN2Layer(grad_ops)
        >>> opti = ms.nn.Adam(net.trainable_params(), learning_rate=0.1)
        >>> train_net = ms.nn.TrainOneStepCell(net, opti)
        >>> for i in range(100):
        ...     train_net(enc_data)
        >>> net.weight.asnumpy()
        array([-1.56476], dtype=float32)
        >>> net(enc_data)
        Tensor(shape=[1, 1], dtype=Float32, value=
        [[ 3.63158676e-07]])
    """
    def __init__(self, expectation_with_grad, weight='normal'):
        super(MQN2Layer, self).__init__()
        self.evolution = MQN2Ops(expectation_with_grad)
        weight_size = len(self.evolution.expectation_with_grad.ansatz_params_name)
        if isinstance(weight, ms.Tensor):
            if weight.ndim != 1 or weight.shape[0] != weight_size:
                raise ValueError(f"Weight init shape error, required ({weight_size}, ), but get f{weight.shape}.")
        self.weight = Parameter(initializer(weight, weight_size, dtype=ms.float32), name='ansatz_weight')

    def construct(self, x):
        return self.evolution(x, self.weight)


class MQAnsatzOnlyLayer(nn.Cell):
    """
    MindQuantum trainable layer. The parameters of ansatz circuit are trainable parameters.

    Args:
        expectation_with_grad (GradOpsWrapper): a grad ops that receive encoder data and
            ansatz data and return the expectation value and gradient value of parameters
            respect to expectation.
        weight (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the
            convolution kernel. It can be a Tensor, a string, an Initializer or a number.
            When a string is specified, values from 'TruncatedNormal', 'Normal', 'Uniform',
            'HeUniform' and 'XavierUniform' distributions as well as constant 'One' and 'Zero'
            distributions are possible. Alias 'xavier_uniform', 'he_uniform', 'ones' and
            'zeros' are acceptable. Uppercase and lowercase are both acceptable. Refer to
            the values of Initializer for more details. Default: 'normal'.

    Outputs:
        Tensor, The expectation value of the hamiltonian.

    Raises:
        ValueError: If length of shape of `weight` is not equal to 1 and shape[0] of `weight`
                    is not equal to `weight_size`.

    Supported Platforms:
        ``GPU``, ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindquantum import Circuit, Hamiltonian, QubitOperator
        >>> from mindquantum import Simulator, MQAnsatzOnlyLayer
        >>> import mindspore as ms
        >>> ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="CPU")
        >>> circ = Circuit().ry('a', 0).h(0).rx('b', 0)
        >>> ham = Hamiltonian(QubitOperator('Z0'))
        >>> sim = Simulator('projectq', 1)
        >>> grad_ops = sim.get_expectation_with_grad(ham, circ)
        >>> net =  MQAnsatzOnlyLayer(grad_ops)
        >>> opti = ms.nn.Adam(net.trainable_params(), learning_rate=0.1)
        >>> train_net = ms.nn.TrainOneStepCell(net, opti)
        >>> for i in range(100):
        ...     train_net()
        >>> net.weight.asnumpy()
        array([-1.5724511e+00,  1.3100551e-04], dtype=float32)
        >>> net()
        Tensor(shape=[1], dtype=Float32, value= [-9.99998629e-01])
    """
    def __init__(self, expectation_with_grad, weight='normal'):
        super(MQAnsatzOnlyLayer, self).__init__()
        self.evolution = MQAnsatzOnlyOps(expectation_with_grad)
        weight_size = len(self.evolution.expectation_with_grad.ansatz_params_name)
        if isinstance(weight, ms.Tensor):
            if weight.ndim != 1 or weight.shape[0] != weight_size:
                raise ValueError(f"Weight init shape error, required ({weight_size}, ), but get f{weight.shape}.")
        self.weight = Parameter(initializer(weight, weight_size, dtype=ms.float32), name='ansatz_weight')

    def construct(self):
        return self.evolution(self.weight)


class MQN2AnsatzOnlyLayer(nn.Cell):
    """
    MindQuantum trainable layer. The parameters of ansatz circuit are trainable parameters.
    This layer will calculate the square of absolute value of expectation automatically.

    Args:
        expectation_with_grad (GradOpsWrapper): a grad ops that receive encoder data and
            ansatz data and return the expectation value and gradient value of parameters
            respect to expectation.
        weight (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the
            convolution kernel. It can be a Tensor, a string, an Initializer or a number.
            When a string is specified, values from 'TruncatedNormal', 'Normal', 'Uniform',
            'HeUniform' and 'XavierUniform' distributions as well as constant 'One' and 'Zero'
            distributions are possible. Alias 'xavier_uniform', 'he_uniform', 'ones' and
            'zeros' are acceptable. Uppercase and lowercase are both acceptable. Refer to
            the values of Initializer for more details. Default: 'normal'.

    Inputs:
        - **enc_data** (Tensor) - Tensor of encoder data that you want to encode into quantum state.

    Outputs:
        Tensor, The expectation value of the hamiltonian.

    Raises:
        ValueError: If length of shape of `weight` is not equal to 1 and shape[0] of `weight`
                    is not equal to `weight_size`.

    Supported Platforms:
        ``GPU``, ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindquantum import Circuit, Hamiltonian, QubitOperator
        >>> from mindquantum import Simulator, MQN2AnsatzOnlyLayer
        >>> import mindspore as ms
        >>> ms.set_seed(43)
        >>> ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="CPU")
        >>> circ = Circuit().ry('a', 0).h(0).rx('b', 0)
        >>> ham = Hamiltonian(QubitOperator('Z0'))
        >>> sim = Simulator('projectq', 1)
        >>> grad_ops = sim.get_expectation_with_grad(ham, circ)
        >>> net =  MQN2AnsatzOnlyLayer(grad_ops)
        >>> opti = ms.nn.Adam(net.trainable_params(), learning_rate=0.1)
        >>> train_net = ms.nn.TrainOneStepCell(net, opti)
        >>> for i in range(100):
        ...     train_net()
        >>> net.weight.asnumpy()
        array([ 0.05957536, -1.5686935 ], dtype=float32)
        >>> net()
        Tensor(shape=[1], dtype=Float32, value= [ 1.56753845e-08])
    """
    def __init__(self, expectation_with_grad, weight='normal'):
        super(MQN2AnsatzOnlyLayer, self).__init__()
        self.evolution = MQN2AnsatzOnlyOps(expectation_with_grad)
        weight_size = len(self.evolution.expectation_with_grad.ansatz_params_name)
        if isinstance(weight, ms.Tensor):
            if weight.ndim != 1 or weight.shape[0] != weight_size:
                raise ValueError(f"Weight init shape error, required ({weight_size}, ), but get f{weight.shape}.")
        self.weight = Parameter(initializer(weight, weight_size, dtype=ms.float32), name='ansatz_weight')

    def construct(self):
        return self.evolution(self.weight)
