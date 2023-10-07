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

from mindquantum.core.circuit import Circuit
from mindquantum.core.operators import Hamiltonian
from mindquantum.simulator import GradOpsWrapper, Simulator
from mindquantum.utils.type_value_check import (
    _check_input_type,
    _check_int_type,
    _check_value_should_not_less,
)


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


@constexpr
def check_state_vector_shape(data, data_shape, qubit_dim):
    """Check state vector shape."""
    if not isinstance(data, ms.Tensor):
        raise TypeError(f"Quantum state vector requires a Tensor but get {type(data)}.")
    if len(data_shape) != 2 or data_shape[1] != qubit_dim:
        raise ValueError(
            'Quantum state vector requires a two dimension Tensor with'
            f' second dimension should be {qubit_dim}, but get shape {data_shape}.'
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


# pylint: disable=too-many-instance-attributes
class QRamVecOps(nn.Cell):
    r"""
    MindQuantum vector state qram operator.

    A QRam operator with can directly encode classical data into quantum state vector.
    This ops is `PYNATIVE_MODE` supported only.

    Note:
        - For MindSpore with version less than 2.0.0, complex tensor as neural
          network cell input is not supported, so we should split quantum
          state to real and image part, and use them as input tensor. This may change when MindSpore upgrade.
        - Currently, we can not compute the gradient of the measurement result with respect to each quantum amplitude.

    Args:
        hams (Union[:class:`~.core.operators.Hamiltonian`, List[:class:`~.core.operators.Hamiltonian`]]):
            A :class:`~.core.operators.Hamiltonian` or a list of :class:`~.core.operators.Hamiltonian` that
            need to get expectation.
        circ (:class:`~.core.circuit.Circuit`): The parameterized quantum circuit.
        sim (:class:`~.simulator.Simulator`): The simulator to do simulation.
        n_thread (int): The parallel thread for evaluate a batch of initial state. If ``None``, evolution will run in
            single thread. Default: ``None``.

    Inputs:
        - **qs_r** (Tensor) - The real part of quantum state with shape :math:`(N, M)`, where :math:`N` is batch size
          and :math:`M` is the length of quantum state vector.
        - **qs_i** (Tensor) - The image part of quantum state with shape :math:`(N, M)`, where :math:`N` is batch size
          and :math:`M` is the length of quantum state vector.
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
        >>> from mindquantum.framework import QRamVecOps
        >>> from mindquantum.simulator import Simulator
        >>> from mindquantum.utils import random_state
        >>> ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
        >>> circ = Circuit().ry('a', 0).h(0).rx('b', 0).as_ansatz()
        >>> ham = Hamiltonian(QubitOperator('Z0'))
        >>> sim = Simulator('mqvector', 1)
        >>> grad_ops = sim.get_expectation_with_grad(ham, circ)
        >>> qs = random_state((3, 2), norm_axis=1, seed=42)
        >>> qs_r, qs_i = ms.Tensor(qs.real), ms.Tensor(qs.imag)
        >>> ansatz_data = np.array([1.0, 2.0])
        >>> net = QRamVecOps(ham, circ, sim)
        >>> f_ms = net(qs_r, qs_i, ms.Tensor(ansatz_data))
        >>> f_ms
        Tensor(shape=[3, 1], dtype=Float32, value=
        [[-7.97555372e-02],
         [-3.92564088e-01],
         [ 4.03987877e-02]])
        >>> for i in qs:
        ...     sim.set_qs(i)
        ...     f, g = grad_ops(ansatz_data)
        ...     print(f.real[0, 0])
        -0.07975553553458492
        -0.39256407750502403
        0.04039878594782581
    """

    def __init__(self, hams, circ, sim, n_thread=None):
        """Initialize a qram operator."""
        super().__init__()
        _mode_check(self)
        if isinstance(hams, Hamiltonian):
            hams = [hams]
        for i in hams:
            _check_input_type('hams', Hamiltonian, i)
        if n_thread is None:
            n_thread = 1
        _check_input_type('circ', Circuit, circ)
        _check_input_type('sim', Simulator, sim)
        _check_int_type('n_thread', n_thread)
        _check_value_should_not_less('n_thread', 1, n_thread)
        if circ.encoder_params_name:
            raise ValueError("circ can not have encoder parameters.")
        self.hams = hams
        self.circ = circ
        self.sim = sim.copy()
        self.n_thread = n_thread
        self.shape_ops = operations.Shape()
        self.g_ans = None
        self.g_enc = None
        self.dim = 1 << self.sim.n_qubits

    def extend_repr(self):
        """Extend string representation."""
        grad_str = f'{self.sim.n_qubits} qubit' + ('' if self.sim.n_qubits == 1 else 's')
        grad_str += f' {self.sim.backend.name} VQA Operator'
        return grad_str

    def construct(self, qs_r, qs_i, ans_data):
        """Construct an MQOps node."""
        check_state_vector_shape(qs_r, self.shape_ops(qs_r), self.dim)
        check_state_vector_shape(qs_i, self.shape_ops(qs_i), self.dim)
        check_ans_input_shape(ans_data, self.shape_ops(ans_data), len(self.circ.params_name))
        enc_data = qs_r.asnumpy() + qs_i.asnumpy() * 1j
        f = self.sim.backend.sim.qram_expectation_with_grad(
            [i.get_cpp_obj() for i in self.hams],
            self.circ.get_cpp_obj(),
            self.circ.get_cpp_obj(True),
            enc_data,
            ans_data.asnumpy(),
            self.circ.params_name,
            self.n_thread,
            len(self.hams),
        )
        f = np.array(f)
        f, g = f[:, :, 0], f[:, :, 1:]
        self.g_ans = np.real(g)
        self.g_enc = ms.Tensor(np.zeros(shape=enc_data.shape), dtype=ms.float32)
        return ms.Tensor(np.real(f), dtype=ms.float32)

    def bprop(self, enc_data_r, enc_data_i, ans_data, out, dout):  # pylint: disable=unused-argument,too-many-arguments
        """Implement the bprop function."""
        dout = dout.asnumpy()
        ans_grad = np.einsum('smp,sm->p', self.g_ans, dout)
        return self.g_enc, self.g_enc, ms.Tensor(ans_grad, dtype=ms.float32)


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
