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
"""Mindspore quantum simulator operator."""
import numpy as np
import mindspore as ms
from mindspore import context
from mindspore.ops import operations as P
import mindspore.nn as nn
from mindspore.ops.primitive import constexpr
from mindquantum.simulator import GradOpsWrapper


@constexpr
def check_enc_input_shape(data, x, enc_len):
    if not isinstance(data, ms.Tensor):
        raise TypeError(f"Encoder parameter requires a Tensor but get {type(data)}")
    if len(x) != 2 or x[1] != enc_len:
        raise ValueError(f'Encoder data requires a two dimension Tensor with second' +
                         f' dimension should be {enc_len}, but get shape {x}')


@constexpr
def check_ans_input_shape(data, x, ans_len):
    if not isinstance(data, ms.Tensor):
        raise TypeError(f"Ansatz parameter requires a Tensor but get {type(data)}")
    if len(x) != 1 or x[0] != ans_len:
        raise ValueError(f'Ansatz data requires a one dimension Tensor with shape {ans_len} ' + f'but get {x}')


class MQOps(nn.Cell):
    """
    MindQuantum operator that get the expectation of a hamiltonian on a quantum
    state evaluated by a parameterized quantum circuit (PQC). This PQC should contains
    a encoder circuit and an ansatz circuit. This ops is `PYNATIVE_MODE` supported only.

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
        >>> from mindquantum import Circuit, Hamiltonian, QubitOperator
        >>> from mindquantum import Simulator, MQOps
        >>> import mindspore as ms
        >>> ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="CPU")
        >>> enc = Circuit().ry('a', 0)
        >>> ans = Circuit().h(0).rx('b', 0)
        >>> ham = Hamiltonian(QubitOperator('Z0'))
        >>> sim = Simulator('projectq', 1)
        >>> grad_ops = sim.get_expectation_with_grad(ham, enc+ans,
        ...                                          encoder_params_name=['a'],
        ...                                          ansatz_params_name=['b'])
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
        super(MQOps, self).__init__()
        _mode_check(self)
        _check_grad_ops(expectation_with_grad)
        self.expectation_with_grad = expectation_with_grad
        self.shape_ops = P.Shape()

    def extend_repr(self):
        return self.expectation_with_grad.str

    def construct(self, enc_data, ans_data):
        check_enc_input_shape(enc_data, self.shape_ops(enc_data), len(self.expectation_with_grad.encoder_params_name))
        check_ans_input_shape(ans_data, self.shape_ops(ans_data), len(self.expectation_with_grad.ansatz_params_name))
        enc_data = enc_data.asnumpy()
        ans_data = ans_data.asnumpy()
        f, g_enc, g_ans = self.expectation_with_grad(enc_data, ans_data)
        f = ms.Tensor(np.real(f), dtype=ms.float32)
        self.g_enc = np.real(g_enc)
        self.g_ans = np.real(g_ans)
        return f

    def bprop(self, enc_data, ans_data, out, dout):
        dout = dout.asnumpy()
        enc_grad = np.einsum('smp,sm->sp', self.g_enc, dout)
        ans_grad = np.einsum('smp,sm->p', self.g_ans, dout)
        return ms.Tensor(enc_grad, dtype=ms.float32), ms.Tensor(ans_grad, dtype=ms.float32)


class MQN2Ops(nn.Cell):
    r"""
    MindQuantum operator that get the square of absolute value of expectation of a hamiltonian
    on a quantum state evaluated by a parameterized quantum circuit (PQC). This PQC should contains
    a encoder circuit and an ansatz circuit. This ops is `PYNATIVE_MODE` supported only.

    .. math::

        O = \left|\left<\varphi\right| U^\dagger_l H U_r\left|\psi\right>\right|^2

    Args:
        expectation_with_grad (GradOpsWrapper): a grad ops that receive encoder data and
            ansatz data and return the square of absolute value of expectation value and
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
        >>> from mindquantum import Circuit, Hamiltonian, QubitOperator
        >>> from mindquantum import Simulator, MQN2Ops
        >>> import mindspore as ms
        >>> ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="CPU")
        >>> enc = Circuit().ry('a', 0)
        >>> ans = Circuit().h(0).rx('b', 0)
        >>> ham = Hamiltonian(QubitOperator('Z0'))
        >>> sim = Simulator('projectq', 1)
        >>> grad_ops = sim.get_expectation_with_grad(ham, enc+ans,
        ...                                          encoder_params_name=['a'],
        ...                                          ansatz_params_name=['b'])
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
        super(MQN2Ops, self).__init__()
        _mode_check(self)
        _check_grad_ops(expectation_with_grad)
        self.expectation_with_grad = expectation_with_grad
        self.shape_ops = P.Shape()

    def extend_repr(self):
        return self.expectation_with_grad.str

    def construct(self, enc_data, ans_data):
        check_enc_input_shape(enc_data, self.shape_ops(enc_data), len(self.expectation_with_grad.encoder_params_name))
        check_ans_input_shape(ans_data, self.shape_ops(ans_data), len(self.expectation_with_grad.ansatz_params_name))
        enc_data = enc_data.asnumpy()
        ans_data = ans_data.asnumpy()
        f, g_enc, g_ans = self.expectation_with_grad(enc_data, ans_data)
        self.f = f
        f = ms.Tensor(np.abs(f)**2, dtype=ms.float32)
        self.g_enc = g_enc
        self.g_ans = g_ans
        return f

    def bprop(self, enc_data, ans_data, out, dout):
        dout = dout.asnumpy()
        enc_grad = 2 * np.real(np.einsum('smp,sm,sm->sp', self.g_enc, dout, np.conj(self.f)))
        ans_grad = 2 * np.real(np.einsum('smp,sm,sm->p', self.g_ans, dout, np.conj(self.f)))
        return ms.Tensor(enc_grad, dtype=ms.float32), ms.Tensor(ans_grad, dtype=ms.float32)


class MQAnsatzOnlyOps(nn.Cell):
    r"""
    MindQuantum operator that get the expectation of a hamiltonian
    on a quantum state evaluated by a parameterized quantum circuit (PQC). This PQC should
    contains an ansatz circuit only. This ops is `PYNATIVE_MODE` supported only.

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
        >>> from mindquantum import Circuit, Hamiltonian, QubitOperator
        >>> from mindquantum import Simulator, MQAnsatzOnlyOps
        >>> import mindspore as ms
        >>> ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="CPU")
        >>> circ = Circuit().ry('a', 0).h(0).rx('b', 0)
        >>> ham = Hamiltonian(QubitOperator('Z0'))
        >>> sim = Simulator('projectq', 1)
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
        super(MQAnsatzOnlyOps, self).__init__()
        _mode_check(self)
        _check_grad_ops(expectation_with_grad)
        self.expectation_with_grad = expectation_with_grad
        self.shape_ops = P.Shape()

    def extend_repr(self):
        return self.expectation_with_grad.str

    def construct(self, x):
        check_ans_input_shape(x, self.shape_ops(x), len(self.expectation_with_grad.ansatz_params_name))
        x = x.asnumpy()
        f, g = self.expectation_with_grad(x)
        f = ms.Tensor(np.real(f[0]), dtype=ms.float32)
        self.g = np.real(g[0])
        return f

    def bprop(self, x, out, dout):
        dout = dout.asnumpy()
        grad = dout @ self.g
        return ms.Tensor(grad, dtype=ms.float32)


class MQN2AnsatzOnlyOps(nn.Cell):
    r"""
    MindQuantum operator that get the square of absolute value of expectation of a hamiltonian
    on a quantum state evaluated by a parameterized quantum circuit (PQC). This PQC should
    contains an ansatz circuit only. This ops is `PYNATIVE_MODE` supported only.

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
        >>> from mindquantum import Circuit, Hamiltonian, QubitOperator
        >>> from mindquantum import Simulator, MQN2AnsatzOnlyOps
        >>> import mindspore as ms
        >>> ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="CPU")
        >>> circ = Circuit().ry('a', 0).h(0).rx('b', 0)
        >>> ham = Hamiltonian(QubitOperator('Z0'))
        >>> sim = Simulator('projectq', 1)
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
        super(MQN2AnsatzOnlyOps, self).__init__()
        _mode_check(self)
        _check_grad_ops(expectation_with_grad)
        self.expectation_with_grad = expectation_with_grad
        self.shape_ops = P.Shape()

    def extend_repr(self):
        return self.expectation_with_grad.str

    def construct(self, x):
        check_ans_input_shape(x, self.shape_ops(x), len(self.expectation_with_grad.ansatz_params_name))
        x = x.asnumpy()
        f, g = self.expectation_with_grad(x)
        self.f = f[0]
        f = ms.Tensor(np.abs(f[0])**2, dtype=ms.float32)
        self.g = g[0]
        return f

    def bprop(self, x, out, dout):
        dout = dout.asnumpy()
        grad = 2 * np.real(np.einsum('m,m,mp->p', np.conj(self.f), dout, self.g))
        return ms.Tensor(grad, dtype=ms.float32)


class MQEncoderOnlyOps(nn.Cell):
    r"""
    MindQuantum operator that get the expectation of a hamiltonian
    on a quantum state evaluated by a parameterized quantum circuit (PQC). This PQC should
    contains a encoder circuit only. This ops is `PYNATIVE_MODE` supported only.

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
        >>> from mindquantum import Circuit, Hamiltonian, QubitOperator
        >>> from mindquantum import Simulator, MQEncoderOnlyOps
        >>> import mindspore as ms
        >>> ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="CPU")
        >>> circ = Circuit().ry('a', 0).h(0).rx('b', 0)
        >>> ham = Hamiltonian(QubitOperator('Z0'))
        >>> sim = Simulator('projectq', 1)
        >>> grad_ops = sim.get_expectation_with_grad(ham, circ, encoder_params_name=circ.params_name)
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
        super(MQEncoderOnlyOps, self).__init__()
        _mode_check(self)
        _check_grad_ops(expectation_with_grad)
        self.expectation_with_grad = expectation_with_grad
        self.shape_ops = P.Shape()

    def extend_repr(self):
        return self.expectation_with_grad.str

    def construct(self, x):
        check_enc_input_shape(x, self.shape_ops(x), len(self.expectation_with_grad.encoder_params_name))
        x = x.asnumpy()
        f, g = self.expectation_with_grad(x)
        f = ms.Tensor(np.real(f), dtype=ms.float32)
        self.g = np.real(g)
        return f

    def bprop(self, x, out, dout):
        dout = dout.asnumpy()
        grad = np.einsum('smp,sm->sp', self.g, dout)
        return ms.Tensor(grad, dtype=ms.float32)


class MQN2EncoderOnlyOps(nn.Cell):
    r"""
    MindQuantum operator that get the square of absolute value of expectation of a hamiltonian
    on a quantum state evaluated by a parameterized quantum circuit (PQC). This PQC should
    contains a encoder circuit only. This ops is `PYNATIVE_MODE` supported only.

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
        >>> from mindquantum import Circuit, Hamiltonian, QubitOperator
        >>> from mindquantum import Simulator, MQN2EncoderOnlyOps
        >>> import mindspore as ms
        >>> ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="CPU")
        >>> circ = Circuit().ry('a', 0).h(0).rx('b', 0)
        >>> ham = Hamiltonian(QubitOperator('Z0'))
        >>> sim = Simulator('projectq', 1)
        >>> grad_ops = sim.get_expectation_with_grad(ham, circ, encoder_params_name=circ.params_name)
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
        super(MQN2EncoderOnlyOps, self).__init__()
        _mode_check(self)
        _check_grad_ops(expectation_with_grad)
        self.expectation_with_grad = expectation_with_grad
        self.shape_ops = P.Shape()

    def extend_repr(self):
        return self.expectation_with_grad.str

    def construct(self, x):
        check_enc_input_shape(x, self.shape_ops(x), len(self.expectation_with_grad.encoder_params_name))
        x = x.asnumpy()
        f, g = self.expectation_with_grad(x)
        self.f = f
        f = ms.Tensor(np.abs(f)**2, dtype=ms.float32)
        self.g = g
        return f

    def bprop(self, x, out, dout):
        dout = dout.asnumpy()
        grad = 2 * np.real(np.einsum('smp,sm,sm->sp', self.g, dout, np.conj(self.f)))
        return ms.Tensor(grad, dtype=ms.float32)


def _mode_check(self):
    if context.get_context('mode') != context.PYNATIVE_MODE:
        raise RuntimeError(f'{self.__class__} is `PYNATIVE_MODE` supported only. Run command below to set context\n\
    import mindspore as ms\n\
    ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="CPU")')


def _check_grad_ops(expectation_with_grad):
    if not isinstance(expectation_with_grad, GradOpsWrapper):
        raise TypeError(f'expectation_with_grad requires a GradOpsWrapper, but get {type(expectation_with_grad)}')
