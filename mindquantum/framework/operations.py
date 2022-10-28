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

"""Mindspore quantum simulator operator."""

import mindspore as ms
import numpy as np
from mindspore import context, nn
from mindspore.ops import operations
from mindspore.ops.primitive import constexpr

from mindquantum.simulator import GradOpsWrapper


@constexpr
def check_enc_input_shape(data, encoder_tensor, enc_len):
    """Check encoder parameter input shape."""
    if not isinstance(data, ms.Tensor):
        raise TypeError(f"Encoder parameter requires a Tensor but get {type(data)}")
    if len(encoder_tensor) != 2 or encoder_tensor[1] != enc_len:
        raise ValueError(
            'Encoder data requires a two dimension Tensor with second'
            + f' dimension should be {enc_len}, but get shape {encoder_tensor}'
        )


@constexpr
def check_ans_input_shape(data, ansatz_tensor, ans_len):
    """Check ansatz input shape."""
    if not isinstance(data, ms.Tensor):
        raise TypeError(f"Ansatz parameter requires a Tensor but get {type(data)}")
    if len(ansatz_tensor) != 1 or ansatz_tensor[0] != ans_len:
        raise ValueError(
            f'Ansatz data requires a one dimension Tensor with shape {ans_len} ' + f'but get {ansatz_tensor}'
        )


class MQOps(nn.Cell):
    """
    MindQuantum operator.

    A quantum circuit evolution operator that include encoder and ansatz circuit, who return
    the expectation of given hamiltonian w.r.t final state of parameterized quantum circuit (PQC).
    This ops is `PYNATIVE_MODE` supported only.

    Args:
        expectation_with_grad (GradOpsWrapper): a grad ops that receive encoder data and
            ansatz data and return the expectation value and gradient value of parameters
            respect to expectation.

    Inputs:
        - **enc_data** (Tensor) - Tensor of encoder data with shape :math:`(N, M)` that
          you want to encode into quantum state, where :math:`N` means the batch size
          and :math:`M` means the number of encoder parameters.
        - **ans_data** (Tensor) - Tensor with shape :math:`N` for ansatz circuit,
          where :math:`N` means the number of ansatz parameters.

    Outputs:
        Tensor, The expectation value of the hamiltonian.

    Supported Platforms:
        ``GPU``, ``CPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore as ms
        >>> from mindquantum.core.circuit import Circuit
        >>> from mindquantum.core.operators import Hamiltonian, QubitOperator
        >>> from mindquantum.framework import MQOps
        >>> from mindquantum.simulator import Simulator
        >>> ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
        >>> enc = Circuit().ry('a', 0).as_encoder()
        >>> ans = Circuit().h(0).rx('b', 0)
        >>> ham = Hamiltonian(QubitOperator('Z0'))
        >>> sim = Simulator('mqvector', 1)
        >>> grad_ops = sim.get_expectation_with_grad(ham, enc + ans)
        >>> enc_data = np.array([[0.1]])
        >>> ans_data = np.array([0.2])
        >>> f, g_enc, g_ans = grad_ops(enc_data, ans_data)
        >>> f
        array([[0.0978434+0.j]])
        >>> net = MQOps(grad_ops)
        >>> f_ms = net(ms.Tensor(enc_data), ms.Tensor(ans_data))
        >>> f_ms
        Tensor(shape=[1, 1], dtype=Float32, value=
        [[ 9.78433937e-02]])
    """

    def __init__(self, expectation_with_grad):
        """Initialize a MQOps object."""
        super().__init__()
        _mode_check(self)
        _check_grad_ops(expectation_with_grad)
        self.expectation_with_grad = expectation_with_grad
        self.shape_ops = operations.Shape()
        self.g_enc = None
        self.g_ans = None

    def extend_repr(self):
        """Extend string representation."""
        return self.expectation_with_grad.str

    def construct(self, enc_data, ans_data):
        """Construct an MQOps node."""
        check_enc_input_shape(enc_data, self.shape_ops(enc_data), len(self.expectation_with_grad.encoder_params_name))
        check_ans_input_shape(ans_data, self.shape_ops(ans_data), len(self.expectation_with_grad.ansatz_params_name))
        fval, g_enc, g_ans = self.expectation_with_grad(enc_data.asnumpy(), ans_data.asnumpy())
        self.g_enc = np.real(g_enc)
        self.g_ans = np.real(g_ans)
        return ms.Tensor(np.real(fval), dtype=ms.float32)

    def bprop(self, enc_data, ans_data, out, dout):  # pylint: disable=unused-argument
        """Implement the bprop function."""
        dout = dout.asnumpy()
        enc_grad = np.einsum('smp,sm->sp', self.g_enc, dout)
        ans_grad = np.einsum('smp,sm->p', self.g_ans, dout)
        return ms.Tensor(enc_grad, dtype=ms.float32), ms.Tensor(ans_grad, dtype=ms.float32)


class MQN2Ops(nn.Cell):
    r"""
    MindQuantum operator.

    A quantum circuit evolution operator that include encoder and ansatz circuit, who return
    the square of absolute value of expectation of given hamiltonian w.r.t final state of
    parameterized quantum circuit (PQC). This ops is `PYNATIVE_MODE` supported only.

    .. math::

        O = \left|\left<\varphi\right| U^\dagger_l H U_r\left|\psi\right>\right|^2

    Args:
        expectation_with_grad (GradOpsWrapper): a grad ops that receive encoder data and
            ansatz data and return the expectation value and
            gradient value of parameters respect to expectation.

    Inputs:
        - **enc_data** (Tensor) - Tensor of encoder data with shape :math:`(N, M)` that
          you want to encode into quantum state, where :math:`N` means the batch size
          and :math:`M` means the number of encoder parameters.
        - **ans_data** (Tensor) - Tensor with shape :math:`N` for ansatz circuit,
          where :math:`N` means the number of ansatz parameters.

    Outputs:
        Tensor, The square of absolute value of expectation value of the hamiltonian.

    Supported Platforms:
        ``GPU``, ``CPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore as ms
        >>> from mindquantum.core.circuit import Circuit
        >>> from mindquantum.core.operators import Hamiltonian, QubitOperator
        >>> from mindquantum.framework import MQN2Ops
        >>> from mindquantum.simulator import Simulator
        >>> ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
        >>> enc = Circuit().ry('a', 0).as_encoder()
        >>> ans = Circuit().h(0).rx('b', 0)
        >>> ham = Hamiltonian(QubitOperator('Z0'))
        >>> sim = Simulator('mqvector', 1)
        >>> grad_ops = sim.get_expectation_with_grad(ham, enc + ans)
        >>> enc_data = np.array([[0.1]])
        >>> ans_data = np.array([0.2])
        >>> f, g_enc, g_ans = grad_ops(enc_data, ans_data)
        >>> np.abs(f) ** 2
        array([[0.00957333]])
        >>> net = MQN2Ops(grad_ops)
        >>> f_ms = net(ms.Tensor(enc_data), ms.Tensor(ans_data))
        >>> f_ms
        Tensor(shape=[1, 1], dtype=Float32, value=
        [[ 9.57333017e-03]])
    """

    def __init__(self, expectation_with_grad):
        """Initialize a MQN2Ops object."""
        super().__init__()
        _mode_check(self)
        _check_grad_ops(expectation_with_grad)
        self.expectation_with_grad = expectation_with_grad
        self.shape_ops = operations.Shape()
        self.f = None  # pylint: disable=invalid-name
        self.g_enc = None
        self.g_ans = None

    def extend_repr(self):
        """Extend string representation."""
        return self.expectation_with_grad.str

    def construct(self, enc_data, ans_data):
        """Construct an MQN2Ops node."""
        check_enc_input_shape(enc_data, self.shape_ops(enc_data), len(self.expectation_with_grad.encoder_params_name))
        check_ans_input_shape(ans_data, self.shape_ops(ans_data), len(self.expectation_with_grad.ansatz_params_name))
        fval, g_enc, g_ans = self.expectation_with_grad(enc_data.asnumpy(), ans_data.asnumpy())
        self.f = fval
        self.g_enc = g_enc
        self.g_ans = g_ans
        return ms.Tensor(np.abs(fval) ** 2, dtype=ms.float32)

    def bprop(self, enc_data, ans_data, out, dout):  # pylint: disable=unused-argument
        """Implement the bprop function."""
        dout = dout.asnumpy()
        enc_grad = 2 * np.real(np.einsum('smp,sm,sm->sp', self.g_enc, dout, np.conj(self.f)))
        ans_grad = 2 * np.real(np.einsum('smp,sm,sm->p', self.g_ans, dout, np.conj(self.f)))
        return ms.Tensor(enc_grad, dtype=ms.float32), ms.Tensor(ans_grad, dtype=ms.float32)


class MQAnsatzOnlyOps(nn.Cell):
    r"""
    MindQuantum operator.

    A quantum circuit evolution operator that only include ansatz circuit, who return
    the expectation of given hamiltonian w.r.t final state of parameterized quantum circuit (PQC).
    This ops is `PYNATIVE_MODE` supported only.

    Args:
        expectation_with_grad (GradOpsWrapper): a grad ops that receive encoder data and
            ansatz data and return the expectation value and gradient value of parameters
            respect to expectation.

    Inputs:
        - **ans_data** (Tensor) - Tensor with shape :math:`N` for ansatz circuit,
          where :math:`N` means the number of ansatz parameters.

    Outputs:
        Tensor, The expectation value of the hamiltonian.

    Supported Platforms:
        ``GPU``, ``CPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore as ms
        >>> from mindquantum.core.circuit import Circuit
        >>> from mindquantum.core.operators import Hamiltonian, QubitOperator
        >>> from mindquantum.framework import MQAnsatzOnlyOps
        >>> from mindquantum.simulator import Simulator
        >>> ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
        >>> circ = Circuit().ry('a', 0).h(0).rx('b', 0)
        >>> ham = Hamiltonian(QubitOperator('Z0'))
        >>> sim = Simulator('mqvector', 1)
        >>> grad_ops = sim.get_expectation_with_grad(ham, circ)
        >>> data = np.array([0.1, 0.2])
        >>> f, g = grad_ops(data)
        >>> f
        array([[0.0978434+0.j]])
        >>> net = MQAnsatzOnlyOps(grad_ops)
        >>> f_ms = net(ms.Tensor(data))
        >>> f_ms
        Tensor(shape=[1], dtype=Float32, value= [ 9.78433937e-02])
    """

    def __init__(self, expectation_with_grad):
        """Initialize a MQAnsatzOnlyOps object."""
        super().__init__()
        _mode_check(self)
        _check_grad_ops(expectation_with_grad)
        self.expectation_with_grad = expectation_with_grad
        self.shape_ops = operations.Shape()
        self.g = None  # pylint: disable=invalid-name

    def extend_repr(self):
        """Extend string representation."""
        return self.expectation_with_grad.str

    def construct(self, arg):
        """Construct a MQAnsatzOnlyOps node."""
        check_ans_input_shape(arg, self.shape_ops(arg), len(self.expectation_with_grad.ansatz_params_name))
        fval, g_ans = self.expectation_with_grad(arg.asnumpy())
        self.g = np.real(g_ans[0])
        return ms.Tensor(np.real(fval[0]), dtype=ms.float32)

    def bprop(self, arg, out, dout):  # pylint: disable=unused-argument
        """Implement the bprop function."""
        dout = dout.asnumpy()
        grad = dout @ self.g
        return ms.Tensor(grad, dtype=ms.float32)


class MQN2AnsatzOnlyOps(nn.Cell):
    r"""
    MindQuantum operator.

    A quantum circuit evolution operator that only include ansatz circuit, who return
    the square of absolute value of given hamiltonian w.r.t final state of parameterized
    quantum circuit (PQC). This ops is `PYNATIVE_MODE` supported only.

    Args:
        expectation_with_grad (GradOpsWrapper): a grad ops that receive encoder data and
            ansatz data and return the expectation value and
            gradient value of parameters respect to expectation.

    Inputs:
        - **ans_data** (Tensor) - Tensor with shape :math:`N` for ansatz circuit,
          where :math:`N` means the number of ansatz parameters.

    Outputs:
        Tensor, The square of absolute value of expectation value of the hamiltonian.

    Supported Platforms:
        ``GPU``, ``CPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore as ms
        >>> from mindquantum.core.circuit import Circuit
        >>> from mindquantum.core.operators import Hamiltonian, QubitOperator
        >>> from mindquantum.framework import MQN2AnsatzOnlyOps
        >>> from mindquantum.simulator import Simulator
        >>> ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
        >>> circ = Circuit().ry('a', 0).h(0).rx('b', 0)
        >>> ham = Hamiltonian(QubitOperator('Z0'))
        >>> sim = Simulator('mqvector', 1)
        >>> grad_ops = sim.get_expectation_with_grad(ham, circ)
        >>> data = np.array([0.1, 0.2])
        >>> f, g = grad_ops(data)
        >>> np.abs(f) ** 2
        array([[0.00957333]])
        >>> net = MQN2AnsatzOnlyOps(grad_ops)
        >>> f_ms = net(ms.Tensor(data))
        >>> f_ms
        Tensor(shape=[1], dtype=Float32, value= [ 9.57333017e-03])
    """

    def __init__(self, expectation_with_grad):
        """Initialize a MQN2AnsatzOnlyOps object."""
        super().__init__()
        _mode_check(self)
        _check_grad_ops(expectation_with_grad)
        self.expectation_with_grad = expectation_with_grad
        self.shape_ops = operations.Shape()
        # pylint: disable=invalid-name
        self.f = None
        self.g = None

    def extend_repr(self):
        """Extend string representation."""
        return self.expectation_with_grad.str

    def construct(self, arg):
        """Construct a MQN2AnsatzOnlyOps node."""
        check_ans_input_shape(arg, self.shape_ops(arg), len(self.expectation_with_grad.ansatz_params_name))
        fval, g_ans = self.expectation_with_grad(arg.asnumpy())
        self.f = fval[0]
        self.g = g_ans[0]
        return ms.Tensor(np.abs(fval[0]) ** 2, dtype=ms.float32)

    def bprop(self, arg, out, dout):  # pylint: disable=unused-argument
        """Implement the bprop function."""
        dout = dout.asnumpy()
        grad = 2 * np.real(np.einsum('m,m,mp->p', np.conj(self.f), dout, self.g))
        return ms.Tensor(grad, dtype=ms.float32)


class MQEncoderOnlyOps(nn.Cell):
    r"""
    MindQuantum operator.

    A quantum circuit evolution operator that only include encoder circuit, who return
    the square of absolute value of given hamiltonian w.r.t final state of parameterized
    quantum circuit (PQC). This ops is `PYNATIVE_MODE` supported only.

    Args:
        expectation_with_grad (GradOpsWrapper): a grad ops that receive encoder data and
            ansatz data and return the expectation value and gradient value of parameters
            respect to expectation.

    Inputs:
        - **enc_data** (Tensor) - Tensor of encoder data with shape :math:`(N, M)` that
          you want to encode into quantum state, where :math:`N` means the batch size
          and :math:`M` means the number of encoder parameters.

    Outputs:
        Tensor, The expectation value of the hamiltonian.

    Supported Platforms:
        ``GPU``, ``CPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore as ms
        >>> from mindquantum.core.circuit import Circuit
        >>> from mindquantum.core.operators import Hamiltonian, QubitOperator
        >>> from mindquantum.framework import MQEncoderOnlyOps
        >>> from mindquantum.simulator import Simulator
        >>> ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
        >>> circ = Circuit().ry('a', 0).h(0).rx('b', 0).as_encoder()
        >>> ham = Hamiltonian(QubitOperator('Z0'))
        >>> sim = Simulator('mqvector', 1)
        >>> grad_ops = sim.get_expectation_with_grad(ham, circ)
        >>> data = np.array([[0.1, 0.2], [0.3, 0.4]])
        >>> f, g = grad_ops(data)
        >>> f
        array([[0.0978434 +0.j],
               [0.27219214+0.j]])
        >>> net = MQEncoderOnlyOps(grad_ops)
        >>> f_ms = net(ms.Tensor(data))
        >>> f_ms
        Tensor(shape=[2, 1], dtype=Float32, value=
        [[ 9.78433937e-02],
         [ 2.72192121e-01]])
    """

    def __init__(self, expectation_with_grad):
        """Initialize a MQEncoderOnlyOps object."""
        super().__init__()
        _mode_check(self)
        _check_grad_ops(expectation_with_grad)
        self.expectation_with_grad = expectation_with_grad
        self.shape_ops = operations.Shape()
        self.g = None  # pylint: disable=invalid-name

    def extend_repr(self):
        """Extend string representation."""
        return self.expectation_with_grad.str

    def construct(self, arg):
        """Construct a MQEncoderOnlyOps node."""
        check_enc_input_shape(arg, self.shape_ops(arg), len(self.expectation_with_grad.encoder_params_name))
        fval, g_enc = self.expectation_with_grad(arg.asnumpy())
        self.g = np.real(g_enc)
        return ms.Tensor(np.real(fval), dtype=ms.float32)

    def bprop(self, arg, out, dout):  # pylint: disable=unused-argument
        """Implement the bprop function."""
        dout = dout.asnumpy()
        grad = np.einsum('smp,sm->sp', self.g, dout)
        return ms.Tensor(grad, dtype=ms.float32)


class MQN2EncoderOnlyOps(nn.Cell):
    r"""
    MindQuantum operator.

    A quantum circuit evolution operator that only include encoder circuit, who return
    the square of absolute value of given hamiltonian w.r.t final state of parameterized
    quantum circuit (PQC). This ops is `PYNATIVE_MODE` supported only.

    Args:
        expectation_with_grad (GradOpsWrapper): a grad ops that receive encoder data and
            ansatz data and return the square of absolute value of expectation value and
            gradient value of parameters respect to expectation.

    Inputs:
        - **ans_data** (Tensor) - Tensor with shape :math:`N` for ansatz circuit,
          where :math:`N` means the number of ansatz parameters.

    Outputs:
        Tensor, The square of absolute value of expectation value of the hamiltonian.

    Supported Platforms:
        ``GPU``, ``CPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore as ms
        >>> from mindquantum.core.circuit import Circuit
        >>> from mindquantum.core.operators import Hamiltonian, QubitOperator
        >>> from mindquantum.framework import MQN2EncoderOnlyOps
        >>> from mindquantum.simulator import Simulator
        >>> ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
        >>> circ = Circuit().ry('a', 0).h(0).rx('b', 0).as_encoder()
        >>> ham = Hamiltonian(QubitOperator('Z0'))
        >>> sim = Simulator('mqvector', 1)
        >>> grad_ops = sim.get_expectation_with_grad(ham, circ)
        >>> data = np.array([[0.1, 0.2], [0.3, 0.4]])
        >>> f, g = grad_ops(data)
        >>> np.abs(f) ** 2
        array([[0.00957333],
               [0.07408856]])
        >>> net = MQN2EncoderOnlyOps(grad_ops)
        >>> f_ms = net(ms.Tensor(data))
        >>> f_ms
        Tensor(shape=[2, 1], dtype=Float32, value=
        [[ 9.57333017e-03],
         [ 7.40885586e-02]])
    """

    def __init__(self, expectation_with_grad):
        """Initialize a MQN2EncoderOnlyOps object."""
        super().__init__()
        _mode_check(self)
        _check_grad_ops(expectation_with_grad)
        self.expectation_with_grad = expectation_with_grad
        self.shape_ops = operations.Shape()
        # pylint: disable=invalid-name
        self.f = None
        self.g = None

    def extend_repr(self):
        """Extend string representation."""
        return self.expectation_with_grad.str

    def construct(self, arg):
        """Construct a MQN2EncoderOnlyOps node."""
        check_enc_input_shape(arg, self.shape_ops(arg), len(self.expectation_with_grad.encoder_params_name))
        fval, g_enc = self.expectation_with_grad(arg.asnumpy())
        self.f = fval
        self.g = g_enc
        return ms.Tensor(np.abs(fval) ** 2, dtype=ms.float32)

    def bprop(self, arg, out, dout):  # pylint: disable=unused-argument
        """Implement the bprop function."""
        dout = dout.asnumpy()
        grad = 2 * np.real(np.einsum('smp,sm,sm->sp', self.g, dout, np.conj(self.f)))
        return ms.Tensor(grad, dtype=ms.float32)


def _mode_check(self):
    if context.get_context('mode') != context.PYNATIVE_MODE:
        raise RuntimeError(
            f'{self.__class__} is `PYNATIVE_MODE` supported only. Run command below to set context\n'
            'import mindspore as ms\n'
            'ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="CPU")'
        )


def _check_grad_ops(expectation_with_grad):
    if not isinstance(expectation_with_grad, GradOpsWrapper):
        raise TypeError(f'expectation_with_grad requires a GradOpsWrapper, but get {type(expectation_with_grad)}')
