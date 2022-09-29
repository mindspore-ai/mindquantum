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
from mindspore import nn
from mindspore.common.initializer import initializer
from mindspore.common.parameter import Parameter

from .operations import MQAnsatzOnlyOps, MQN2AnsatzOnlyOps, MQN2Ops, MQOps


class MQLayer(nn.Cell):  # pylint: disable=too-few-public-methods
    """
    Quantum neural network include encoder and ansatz circuit.

    The encoder circuit encode classical data into quantum state, while the ansatz circuit act as trainable circuit.

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
        >>> import mindspore as ms
        >>> from mindquantum.core.circuit import Circuit
        >>> from mindquantum.core.operators import Hamiltonian, QubitOperator
        >>> from mindquantum.framework import MQLayer
        >>> from mindquantum.simulator import Simulator
        >>> ms.set_seed(42)
        >>> ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
        >>> enc = Circuit().ry('a', 0).as_encoder()
        >>> ans = Circuit().h(0).rx('b', 0).as_ansatz()
        >>> ham = Hamiltonian(QubitOperator('Z0'))
        >>> sim = Simulator('projectq', 1)
        >>> grad_ops = sim.get_expectation_with_grad(ham, enc + ans)
        >>> enc_data = ms.Tensor(np.array([[0.1]]))
        >>> net =  MQLayer(grad_ops)
        >>> opti = ms.nn.Adam(net.trainable_params(), learning_rate=0.1)
        >>> train_net = ms.nn.TrainOneStepCell(net, opti)
        >>> for i in range(100):
        ...     train_net(enc_data)
        >>> net.weight.asnumpy()
        array([3.1423748], dtype=float32)
        >>> net(enc_data)
        Tensor(shape=[1, 1], dtype=Float32, value=
        [[-9.98333842e-02]])
    """

    def __init__(self, expectation_with_grad, weight='normal'):
        """Initialize a MQLayer object."""
        super().__init__()
        self.evolution = MQOps(expectation_with_grad)
        weight_size = len(self.evolution.expectation_with_grad.ansatz_params_name)
        if isinstance(weight, ms.Tensor):
            if weight.ndim != 1 or weight.shape[0] != weight_size:
                raise ValueError(f"Weight init shape error, required ({weight_size}, ), but get {weight.shape}.")
        self.weight = Parameter(initializer(weight, weight_size, dtype=ms.float32), name='ansatz_weight')

    def construct(self, arg):
        """Construct a MQLayer node."""
        return self.evolution(arg, self.weight)


class MQN2Layer(nn.Cell):  # pylint: disable=too-few-public-methods
    """
    MindQuantum trainable layer.

    Quantum neural network include encoder and ansatz circuit. The encoder circuit encode classical data into quantum
    state, while the ansatz circuit act as trainable circuit.  This layer will calculate the square of absolute value of
    expectation automatically.

    Args:
        expectation_with_grad (GradOpsWrapper): a grad ops that receive encoder data and
            ansatz data and return the expectation value and
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
        >>> import mindspore as ms
        >>> from mindquantum.core.circuit import Circuit
        >>> from mindquantum.core.operators import Hamiltonian, QubitOperator
        >>> from mindquantum.framework import MQN2Layer
        >>> from mindquantum.simulator import Simulator
        >>> ms.set_seed(42)
        >>> ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
        >>> enc = Circuit().ry('a', 0).as_encoder()
        >>> ans = Circuit().h(0).rx('b', 0).as_ansatz()
        >>> ham = Hamiltonian(QubitOperator('Z0'))
        >>> sim = Simulator('projectq', 1)
        >>> grad_ops = sim.get_expectation_with_grad(ham, enc + ans)
        >>> enc_data = ms.Tensor(np.array([[0.1]]))
        >>> net =  MQN2Layer(grad_ops)
        >>> opti = ms.nn.Adam(net.trainable_params(), learning_rate=0.1)
        >>> train_net = ms.nn.TrainOneStepCell(net, opti)
        >>> for i in range(100):
        ...     train_net(enc_data)
        >>> net.weight.asnumpy()
        array([1.5646162], dtype=float32)
        >>> net(enc_data)
        Tensor(shape=[1, 1], dtype=Float32, value=
        [[ 3.80662982e-07]])
    """

    def __init__(self, expectation_with_grad, weight='normal'):
        """Initialize a MQN2Layer object."""
        super().__init__()
        self.evolution = MQN2Ops(expectation_with_grad)
        weight_size = len(self.evolution.expectation_with_grad.ansatz_params_name)
        if isinstance(weight, ms.Tensor):
            if weight.ndim != 1 or weight.shape[0] != weight_size:
                raise ValueError(f"Weight init shape error, required ({weight_size}, ), but get f{weight.shape}.")
        self.weight = Parameter(initializer(weight, weight_size, dtype=ms.float32), name='ansatz_weight')

    def construct(self, arg):
        """Construct a MQN2Layer node."""
        return self.evolution(arg, self.weight)


class MQAnsatzOnlyLayer(nn.Cell):  # pylint: disable=too-few-public-methods
    """
    MindQuantum trainable layer.

    Quantum neural network only include ansatz circuit.  The ansatz circuit act as trainable circuit.

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
        >>> import mindspore as ms
        >>> from mindquantum.core.circuit import Circuit
        >>> from mindquantum.core.operators import Hamiltonian, QubitOperator
        >>> from mindquantum.framework import MQAnsatzOnlyLayer
        >>> from mindquantum.simulator import Simulator
        >>> ms.set_seed(42)
        >>> ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
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
        array([-1.5720805e+00,  1.7390326e-04], dtype=float32)
        >>> net()
        Tensor(shape=[1], dtype=Float32, value= [-9.99999166e-01])
    """

    def __init__(self, expectation_with_grad, weight='normal'):
        """Initialize a MQAnsatzOnlyLayer object."""
        super().__init__()
        self.evolution = MQAnsatzOnlyOps(expectation_with_grad)
        weight_size = len(self.evolution.expectation_with_grad.ansatz_params_name)
        if isinstance(weight, ms.Tensor):
            if weight.ndim != 1 or weight.shape[0] != weight_size:
                raise ValueError(f"Weight init shape error, required ({weight_size}, ), but get f{weight.shape}.")
        self.weight = Parameter(initializer(weight, weight_size, dtype=ms.float32), name='ansatz_weight')

    def construct(self):
        """Construct a MQAnsatzOnlyLayer node."""
        return self.evolution(self.weight)


class MQN2AnsatzOnlyLayer(nn.Cell):  # pylint: disable=too-few-public-methods
    """
    MindQuantum trainable layer.

    Quantum neural network only include ansatz circuit. The ansatz circuit act as trainable circuit.
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
        >>> import mindspore as ms
        >>> from mindquantum.core.circuit import Circuit
        >>> from mindquantum.core.operators import Hamiltonian, QubitOperator
        >>> from mindquantum.framework import MQN2AnsatzOnlyLayer
        >>> from mindquantum.simulator import Simulator
        >>> ms.set_seed(43)
        >>> ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
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
        array([ 0.05957557, -1.5686936 ], dtype=float32)
        >>> net()
        Tensor(shape=[1], dtype=Float32, value= [ 1.56737148e-08])
    """

    def __init__(self, expectation_with_grad, weight='normal'):
        """Initialize a MQN2AnsatzOnlyLayer object."""
        super().__init__()
        self.evolution = MQN2AnsatzOnlyOps(expectation_with_grad)
        weight_size = len(self.evolution.expectation_with_grad.ansatz_params_name)
        if isinstance(weight, ms.Tensor):
            if weight.ndim != 1 or weight.shape[0] != weight_size:
                raise ValueError(f"Weight init shape error, required ({weight_size}, ), but get f{weight.shape}.")
        self.weight = Parameter(initializer(weight, weight_size, dtype=ms.float32), name='ansatz_weight')

    def construct(self):
        """Construct a MQN2AnsatzOnlyLayer node."""
        return self.evolution(self.weight)
