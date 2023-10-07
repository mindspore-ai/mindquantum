"""Common quantum gates."""

from typing import List
import numpy as np
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.numpy as msnp
from mindspore import Tensor, Parameter

from .operations import ctranspose, cmatmul, creshape, get_transpose_index, get_transpose_index_for_zz
from .define import DTYPE


DEFAULT_VALUE = Tensor(0.0, DTYPE)
MINUS_ONE = Tensor(-1.0, DTYPE)
DEFAULT_PARAM_NAME = '_param_'


def state_evolution(n_qubit, mat, qs, obj_qubit:int, ctrl_qubit=None):
    """Get the end state after the matrix operate from the initial state.

    Args:
        n_qubit (int): Number of qubits.
        mat (Tensor): The matrix of the gate.
        qs (Tensor): Initial state.
        obj_qubit (int): The objective qubit(s) of gate.
        ctrl_qubit (None | int): The control qubit(s) of gate.
    """
    if ctrl_qubit is None:
        idx1, idx2 = get_transpose_index(n_qubit, obj_qubit)
        qs2 = ctranspose(cmatmul(ctranspose(qs, idx1), ctranspose(mat)), idx2)
        return qs2
    elif isinstance(ctrl_qubit, int):
        idx1, idx2 = get_transpose_index(n_qubit, obj_qubit, ctrl_qubit)
        qs2 = ctranspose(qs, idx1)
        qs2_part = (qs2[0][...,1], qs2[1][...,1])
        qs2_part = creshape(cmatmul(mat, creshape(qs2_part, (2,-1))), [2]*(n_qubit - 1))
        qs2r = ops.stack([qs2[0][..., 0], qs2_part[0]], axis=-1)
        qs2i = ops.stack([qs2[1][..., 0], qs2_part[1]], axis=-1)
        qs2 = (qs2r, qs2i)
        qs2 = ctranspose(qs2, idx2)
        return qs2
    elif isinstance(ctrl_qubit, list):
        n_ctrl = len(ctrl_qubit)
        idx1, idx2 = get_transpose_index(n_qubit, obj_qubit, ctrl_qubit)
        qs2 = list(ctranspose(qs, idx1))
        select_idx = list([...] + [1] * n_ctrl)
        qs2_part = (qs2[0][select_idx], qs2[1][select_idx])
        qs2_part = creshape(cmatmul(mat, creshape(qs2_part, (2,-1))), [2]*(n_qubit - n_ctrl))
        qs2[0][select_idx] = qs2_part[0]
        qs2[1][select_idx] = qs2_part[1]
        qs2 = ctranspose(qs2, idx2)
        return qs2


class BaseGate(nn.Cell):
    """Base class of the quantum gate.
    """
    def __init__(self):
        super(BaseGate, self).__init__()
        self.trainable = False
        self.obj_qubit = 0
        self.ctrl_qubit = None
        self.max_qubit_index = 0

    def on(self, obj_qubit, ctrl_qubit=None):
        """To set the quantum gate on specific qubit.
        
        Args:
            obj_qubit (int): objective qubit.
            ctrl_qubit (None | int): control qubit.
        """
        self.trainable = False
        self.obj_qubit = obj_qubit
        self.ctrl_qubit = ctrl_qubit
        self.max_qubit_index = self._get_max_qubit_index()
        return self

    def _get_max_qubit_index(self):
        """Get the maximum index of all qubits.
        """
        index = []
        for qubit in [self.obj_qubit, self.ctrl_qubit]:
            if isinstance(qubit, int):
                index.append(self.obj_qubit)
            elif isinstance(qubit, (List, tuple)):
                index.extend(qubit)
        return max(index)

    def construct(self, *args):
        """Get next state after the gate operation.
        """
        n_qubit, qs = args[0]
        mat = self._build_matrix()
        qs2 = state_evolution(n_qubit, mat, qs, self.obj_qubit, self.ctrl_qubit)
        return n_qubit, qs2

    def _build_matrix(self):
        return NotImplemented

    def __repr__(self):
        if self.ctrl_qubit is not None:
            return f"{self.__class__.__name__}<{self.obj_qubit}, {self.ctrl_qubit}>"
        return f"{self.__class__.__name__}<{self.obj_qubit}>"


class NoneParamGate(BaseGate):
    """The quantum gate without parameters.
    """
    def __init__(self, obj_qubit, ctrl_qubit=None):
        super(NoneParamGate, self).__init__()
        self.on(obj_qubit, ctrl_qubit)
        self.trainable = False


class WithParamGate(BaseGate):
    """The quantum gate without parameters.
    """
    def __init__(self, pr=None, init_way='normal'):
        super(WithParamGate, self).__init__()
        self.trainable = True
        self.param = None
        self.param_name = ""
        self._build_parameter(pr, init_way)

    def _build_parameter(self, pr, init_way):
        """Construct the parameter according the input.
        """
        value = DEFAULT_VALUE
        name = DEFAULT_PARAM_NAME
        if pr is None:
            pass
        elif isinstance(pr, Parameter):
            self.param = pr
        elif isinstance(pr, str):
            value = msnp.randn()[0] if init_way == 'normal' else DEFAULT_VALUE
            name = pr
        elif isinstance(pr, (int, float)):
            value = Tensor(pr, DTYPE)
        elif isinstance(pr, Tensor) and pr.size == 1:
            value = pr
        else:
            raise TypeError("Parameter `pr` should be str of Tensor(float32), "
                             "`init_way` should be str or None.")
        self.param = Parameter(value, name=name)
        self.param_name = name

    def no_grad(self):
        """No gradient for the parameter.
        """
        self.trainable = False
        if isinstance(self.param, Parameter):
            self.param.requires_grad = False

    def with_grad(self):
        """Need gradient for the parameter.
        """
        self.trainable = True
        if isinstance(self.param, Parameter):
            self.param.requires_grad = True

    def set_param_value(self, value):
        """Set the value of the parameter.
        """
        assert self.param is not None, "`param` is None."
        self.param.set_data(Tensor(value, DTYPE))

    def __repr__(self):
        if self.ctrl_qubit is not None:
            return f"{self.__class__.__name__}({self.param_name})<{self.obj_qubit}, {self.ctrl_qubit}>"
        return f"{self.__class__.__name__}({self.param_name})<{self.obj_qubit}>"


class H(NoneParamGate):
    """Hadamard gate.
    """
    def __init__(self, obj_qubit, ctrl_qubit=None):
        super(H, self).__init__(obj_qubit, ctrl_qubit)
    
    def _build_matrix(self):
        re = Tensor([[1.0, 1.0], [1.0, -1.0]], DTYPE) / Tensor(2**0.5, DTYPE)
        im = Tensor([[0.0, 0.0], [0.0, 0.0]], DTYPE)
        return re, im


class X(NoneParamGate):
    """Pauli-X gate.
    """
    def __init__(self, obj_qubit, ctrl_qubit=None):
        """
        Args:
            obj_qubit (int): objective qubit.
            ctrl_qubit (None | int): control qubit.
        """
        super(X, self).__init__(obj_qubit, ctrl_qubit)
    
    def _build_matrix(self):
        re = Tensor([[0.0, 1.0], [1.0, 0.0]], DTYPE)
        im = Tensor([[0.0, 0.0], [0.0, 0.0]], DTYPE)
        return re, im


class Y(NoneParamGate):
    """Pauli-Y gate.
    """
    def __init__(self, obj_qubit, ctrl_qubit=None):
        """
        Args:
            obj_qubit (int): objective qubit.
            ctrl_qubit (None | int): control qubit.
        """
        super(Y, self).__init__(obj_qubit, ctrl_qubit)

    def _build_matrix(self):
        re = Tensor([[0.0, 0.0], [0.0, 0.0]], DTYPE)
        im = Tensor([[0.0, -1.0], [1.0, 0.0]], DTYPE)
        return re, im


class Z(NoneParamGate):
    """Pauli-Z gate.
    """
    def __init__(self, obj_qubit, ctrl_qubit=None):
        """
        Args:
            obj_qubit (int): objective qubit.
            ctrl_qubit (None | int): control qubit.
        """
        super(Z, self).__init__(obj_qubit, ctrl_qubit)

    def _build_matrix(self):
        re = Tensor([[1.0, 0.0], [0.0, -1.0]], DTYPE)
        im = Tensor([[0.0, 0.0], [0.0, 0.0]], DTYPE)
        return re, im


class T(NoneParamGate):
    """T gate.
    """
    def __init__(self, obj_qubit, ctrl_qubit=None):
        """
        Args:
            obj_qubit (int): objective qubit.
            ctrl_qubit (None | int): control qubit.
        """
        super(T, self).__init__(obj_qubit, ctrl_qubit)

    def _build_matrix(self):
        re = Tensor([[1.0, 0.0], [0.0, 1.0 / 2**0.5]], DTYPE)
        im = Tensor([[0.0, 0.0], [0.0, 1.0 / 2**0.5]], DTYPE)
        return re, im


CNOT = X


class SWAP(NoneParamGate):
    """Swap gate, swap two qubit.
    """
    def __init__(self, obj_qubit):
        """
        Args:
            obj_qubit (int): objective qubit.
        """
        super(SWAP, self).__init__(obj_qubit)

    def construct(self, *args):
        assert len(self.obj_qubit) == 2, "The `obj_qubit` tuple should contains 2 `int`."
        n_qubit, qs = args[0]
        obj1 = self.obj_qubit[0]
        obj2 = self.obj_qubit[1]
        idx1, idx2 = get_transpose_index_for_zz(n_qubit, obj1, obj2)
        qs2 = creshape(ctranspose(creshape(qs, [2]*n_qubit), idx1), (-1, 4))
        re, im = qs2
        re = ops.stack([re[:, 0], re[:, 2], re[:, 1], re[:, 3]], axis=-1)
        im = ops.stack([im[:, 0], im[:, 2], im[:, 1], im[:, 3]], axis=-1)
        qs2 = (re, im)
        qs2 = ctranspose(creshape(qs2, [2]*n_qubit), idx2)
        return n_qubit, qs2


class ISWAP(NoneParamGate):
    """iSwap gate, swap two qubit.
    """
    def __init__(self, obj_qubit):
        """
        Args:
            obj_qubit (int): objective qubit.
        """
        super(ISWAP, self).__init__(obj_qubit)

    def construct(self, *args):
        assert len(self.obj_qubit) == 2, "The `obj_qubit` tuple should contains 2 `int`."
        n_qubit, qs = args[0]
        obj1 = self.obj_qubit[0]
        obj2 = self.obj_qubit[1]
        idx1, idx2 = get_transpose_index_for_zz(n_qubit, obj1, obj2)
        qs2 = creshape(ctranspose(creshape(qs, [2]*n_qubit), idx1), (-1, 4))
        re, im = qs2
        re2 = ops.stack([re[:, 0], -im[:, 2], -im[:, 1], re[:, 3]], axis=-1)
        im2 = ops.stack([im[:, 0], re[:, 2], re[:, 1], im[:, 3]], axis=-1)
        qs2 = (re2, im2)
        qs2 = ctranspose(creshape(qs2, [2]*n_qubit), idx2)
        return n_qubit, qs2


class RX(WithParamGate):
    """Rotate Pauli-X gate.
    """
    def __init__(self, pr=None, init_way='normal'):
        """
        Args:
            pr (float | Tensor | int | str): The value(or name) of parameter. 
            init_way (None | str): If `init_way` is 'normal' then use normal distribution value to
                initialize the parameter when without giving specific value.
        """
        super(RX, self).__init__(pr, init_way)

    def _build_matrix(self):
        angle = self.param / Tensor(2.0, DTYPE)
        re = ops.stack([ops.cos(angle), Tensor(0.0, DTYPE),
                        Tensor(0.0, DTYPE), ops.cos(angle)]).reshape((2, 2))
        im = ops.stack([Tensor(0.0, DTYPE), ops.sin(angle) * MINUS_ONE,
                        ops.sin(angle) * MINUS_ONE, Tensor(0.0, DTYPE)]).reshape((2, 2))
        return re, im


class RY(WithParamGate):
    """Rotate Pauli-Y gate.
    """
    def __init__(self, pr=None, init_way='normal'):
        """
        Args:
            pr (float | Tensor | int | str): The value(or name) of parameter. 
            init_way (None | str): If `init_way` is 'normal' then use normal distribution value to
                initialize the parameter when without giving specific value.
        """
        super(RY, self).__init__(pr, init_way)

    def _build_matrix(self):
        angle = self.param / Tensor(2.0, DTYPE)
        re = ops.stack([ops.cos(angle), ops.sin(angle) * MINUS_ONE,
                        ops.sin(angle), ops.cos(angle)]).reshape((2, 2))
        im = Tensor([[0.0, 0.0], [0.0, 0.0]], DTYPE)
        return re, im


class RZ(WithParamGate):
    """Rotate Pauli-Z gate.
    """
    def __init__(self, pr=None, init_way='normal'):
        """
        Args:
            pr (float | Tensor | int | str): The value(or name) of parameter. 
            init_way (None | str): If `init_way` is 'normal' then use normal distribution value to
                initialize the parameter when without giving specific value.
        """
        super(RZ, self).__init__(pr, init_way)

    def _build_matrix(self):
        angle = self.param / Tensor(2.0, DTYPE)
        re = ops.stack([ops.cos(angle), Tensor(0.0, DTYPE),
                        Tensor(0.0, DTYPE), ops.cos(angle)]).reshape((2, 2))
        im = ops.stack([ops.sin(angle) * MINUS_ONE, Tensor(0.0, DTYPE),
                        Tensor(0.0, DTYPE), ops.sin(angle)]).reshape((2, 2))
        return re, im


class ZZ(WithParamGate):
    """Rotate ZZ gate.
    """
    def __init__(self, pr=None, init_way='normal'):
        super(ZZ, self).__init__(pr, init_way)

    def construct(self, *args):
        assert len(self.obj_qubit) == 2, "The `obj_qubit` tuple should contains 2 `int`."
        n_qubit, qs = args[0]
        obj1 = self.obj_qubit[0]
        obj2 = self.obj_qubit[1]
        idx1, idx2 = get_transpose_index_for_zz(n_qubit, obj1, obj2)
        qs2 = creshape(ctranspose(creshape(qs, [2]*n_qubit), idx1), (-1, 4))
        re = ops.cos(self.param) * qs2[0] + ops.sin(self.param) * (Tensor([[1.0, -1.0, -1.0, 1.0]]) * qs2[1])
        im = ops.cos(self.param) * qs2[1] - ops.sin(self.param) * (Tensor([[1.0, -1.0, -1.0, 1.0]]) * qs2[0])
        qs2 = (re, im)
        qs2 = ctranspose(creshape(qs2, [2]*n_qubit), idx2)
        return n_qubit, qs2


class MRX(WithParamGate):
    """Multi-rotate Pauli-X gate, which is useful for QAOA circuit where several gates with common parameter.
    """
    def __init__(self, pr=None, init_way='normal'):
        super(MRX, self).__init__(pr, init_way)

    def construct(self, *args):
        n_qubit, qs = args[0]
        mat = self._build_matrix()
        qs2 = qs
        for obj in self.obj_qubit:
            qs2 = state_evolution(n_qubit, mat, qs2, obj, self.ctrl_qubit)
        return n_qubit, qs2

    def _build_matrix(self):
        angle = self.param / Tensor(2.0, DTYPE)
        re = ops.stack([ops.cos(angle), Tensor(0.0, DTYPE),
                        Tensor(0.0, DTYPE), ops.cos(angle)]).reshape((2, 2))
        im = ops.stack([Tensor(0.0, DTYPE), ops.sin(angle) * MINUS_ONE, 
                        ops.sin(angle) * MINUS_ONE, Tensor(0.0, DTYPE)]).reshape((2, 2))
        return re, im


class MZZ(WithParamGate):
    """Rotate ZZ gate."""
    def __init__(self, pr=None, init_way='normal'):
        super(MZZ, self).__init__(pr, init_way)

    def construct(self, *args):
        n_qubit, qs = args[0]
        qs2 = qs
        for (obj1, obj2) in self.obj_qubit:
            idx1, idx2 = get_transpose_index_for_zz(n_qubit, obj1, obj2)
            qs2 = creshape(ctranspose(creshape(qs2, [2]*n_qubit), idx1), (-1, 4))
            re = ops.cos(self.param) * qs2[0] + ops.sin(self.param) * (Tensor([[1.0, -1.0, -1.0, 1.0]]) * qs2[1])
            im = ops.cos(self.param) * qs2[1] - ops.sin(self.param) * (Tensor([[1.0, -1.0, -1.0, 1.0]]) * qs2[0])
            qs2 = (re, im)
            qs2 = ctranspose(creshape(qs2, [2]*n_qubit), idx2)
        return n_qubit, qs2

    def _build_matrix(self):
        return None

class U1(WithParamGate):
    """Universal gate with one parameters.

    Matrix:
        [[1  0],
         [0 exp(i theta)]]
    """
    def __init__(self, theta: float):
        """
        Args:
            theta (float): The value(or name) of parameter.
        """
        super(U1, self).__init__()
        self.param = Parameter(Tensor(theta, DTYPE))
        self.param_name = "u1_params"

    def _build_matrix(self):
        t = self.param
        re = ops.stack([Tensor(1.0, DTYPE),
                        Tensor(0.0, DTYPE),
                        Tensor(0.0, DTYPE),
                        ops.cos(t)]).reshape((2, 2))
        im = ops.stack([Tensor(1.0, DTYPE),
                        Tensor(0.0, DTYPE),
                        Tensor(0.0, DTYPE),
                        ops.sin(t)]).reshape((2, 2))
        return re, im


class U2(WithParamGate):
    """Universal gate with two parameters.

    Matrix:
        (1/sqrt{2}) * [[1            -exp(i_lamda)]
                       [exp(i_phi)   exp(i_lamda + i_phi)]] 
    """
    def __init__(self, phi: float, lamda: float):
        """
        Args:
            phi, lamda (float): The value(or name) of parameter. 
        """
        super(U2, self).__init__()
        self.param = Parameter(Tensor([phi, lamda], DTYPE))
        self.param_name = "u2_params"

    def _build_matrix(self):
        p, la = self.param
        re = ops.stack([Tensor(1.0, DTYPE),
                        ops.cos(la) * MINUS_ONE,
                        ops.cos(p),
                        ops.cos(p + la)]).reshape((2, 2))
        im = ops.stack([Tensor(0.0, DTYPE),
                        ops.sin(la) * MINUS_ONE,
                        ops.sin(p),
                        ops.sin(p + la)]).reshape((2, 2))
        return re, im


class U3(WithParamGate):
    """Universal gate with three parameters.
    """
    def __init__(self, theta: float, phi: float, lamda: float):
        """
        Args:
            theta, phi, lamda (float): The value(or name) of parameter. 
            init_way (None | str): If `init_way` is 'normal' then use normal distribution value to
                initialize the parameter when without giving specific value.
        """
        super(U3, self).__init__()
        self.param = Parameter(Tensor([theta, phi, lamda], DTYPE))
        self.param_name = "u3_params"

    def _build_matrix(self):
        t, p, la = self.param
        t2 = t / Tensor(2.0, DTYPE)
        re = ops.stack([ops.cos(t2),
                        ops.cos(la) * ops.sin(t2) * MINUS_ONE,
                        ops.cos(p) * ops.sin(t2),
                        ops.cos(p + la) * ops.cos(t2)]).reshape((2, 2))
        im = ops.stack([Tensor(0.0, DTYPE),
                        ops.sin(la) * ops.sin(t2) * MINUS_ONE,
                        ops.sin(p) * ops.sin(t2),
                        ops.sin(p + la) * ops.cos(t2)]).reshape((2, 2))
        return re, im
