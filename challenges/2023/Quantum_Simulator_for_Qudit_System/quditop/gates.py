"""Qudits gates."""

from typing import List, Tuple, Union, Iterable

import numpy as np
import mindspore.numpy as msnp
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, Parameter

from quditop.common import ket, bra, get_complex_tuple, check_unitary
from quditop.global_var import DTYPE, DEFAULT_VALUE, DEFAULT_PARAM_NAME
from quditop.evolution import state_evolution


def get_multi_value_controlled_gate_cmatrix(u_list: List[Tensor]):
    """Get the matrix of multi-value controlled gate by given matrix list."""
    assert len(u_list) == u_list[0].shape[0] == u_list[0].shape[1]
    dim = len(u_list)
    re = ops.zeros((dim**2, dim**2))
    im = ops.zeros((dim**2, dim**2))
    for i, u in enumerate(u_list):
        if isinstance(u, np.ndarray):
            u = Tensor(u)
        if ops.is_complex(u):
            re[i * dim:(i + 1) * dim, i * dim:(i + 1) * dim] = u.real
            im[i * dim:(i + 1) * dim, i * dim:(i + 1) * dim] = u.imag
        else:
            re[i * dim:(i + 1) * dim, i * dim:(i + 1) * dim] = u
    return re, im


def get_pauli_x_gate_cmatrix(dim: int, ind: Iterable) -> Tuple:
    """Get the matrix of extended pauli-X gate. Read the documents of this package for more information."""
    i, j = ind
    re = ket(i, dim) * bra(j, dim) + ket(j, dim) * bra(i, dim)
    im = ops.zeros((dim, dim), dtype=DTYPE)
    for k in range(dim):
        if k != i and k != j:
            re += ket(k, dim) * bra(k, dim)
    return re, im


def get_pauli_y_gate_cmatrix(dim: int, ind: Iterable) -> Tuple:
    """Get the matrix of extended pauli-Y gate. Read the documents of this package for more information."""
    i, j = ind
    re = ops.zeros((dim, dim), dtype=DTYPE)
    im = -ket(i, dim) * bra(j, dim) + ket(j, dim) * bra(i, dim)
    for k in range(dim):
        if k != i and k != j:
            re += ket(k, dim) * bra(k, dim)
    return re, im


def get_pauli_z_gate_cmatrix(dim: int, ind: Iterable) -> Tuple:
    """Get the matrix of extended pauli-Z gate. Read the documents of this package for more information."""
    i, j = ind
    re = ket(i, dim) * bra(i, dim) - ket(j, dim) * bra(j, dim)
    im = ops.zeros((dim, dim), dtype=DTYPE)
    for k in range(dim):
        if k != i and k != j:
            re += ket(k, dim) * bra(k, dim)
    return re, im


def get_rotate_x_gate_cmatrix(dim: int, ind: Iterable, pr: Parameter) -> Tuple:
    """Get the matrix of rotate-X gate. Read the documents of this package for more information."""
    re = [Tensor(0.0, DTYPE) for _ in range(dim * dim)]
    im = [Tensor(0.0, DTYPE) for _ in range(dim * dim)]
    for i in range(dim):
        for j in range(dim):
            k = i * dim + j
            if i == ind[0] and j == ind[0] or \
               i == ind[1] and j == ind[1]:
                re[k] = ops.cos(pr / 2.0)
            elif i == ind[0] and j == ind[1] or \
                    i == ind[1] and j == ind[0]:
                im[k] = -ops.sin(pr / 2.0)
            elif i == j:
                re[k] = Tensor(1.0, DTYPE)
    re = ops.stack(re).reshape((dim, dim))
    im = ops.stack(im).reshape((dim, dim))
    return re, im


def get_rotate_y_gate_cmatrix(dim: int, ind: Iterable, pr: Parameter) -> Tuple:
    """Get the matrix of rotate-Y gate. Read the documents of this package for more information."""
    re = [Tensor(0.0, DTYPE) for _ in range(dim * dim)]
    im = [Tensor(0.0, DTYPE) for _ in range(dim * dim)]
    for i in range(dim):
        for j in range(dim):
            k = i * dim + j
            if i == ind[0] and j == ind[0]:
                re[k] = ops.cos(pr / 2.0)
            elif i == ind[0] and j == ind[1]:
                re[k] = -ops.sin(pr / 2.0)
            elif i == ind[1] and j == ind[0]:
                re[k] = ops.sin(pr / 2.0)
            elif i == ind[1] and j == ind[1]:
                re[k] = ops.cos(pr / 2.0)
            elif i == j:
                re[k] = Tensor(1.0, DTYPE)
    re = ops.stack(re).reshape((dim, dim))
    im = ops.stack(im).reshape((dim, dim))
    return re, im


def get_rotate_z_gate_cmatrix(dim: int, ind: Iterable, pr: Parameter) -> Tuple:
    """Get the matrix of rotate-Y gate. Read the documents of this package for more information."""
    re = [Tensor(0.0, DTYPE) for _ in range(dim * dim)]
    im = [Tensor(0.0, DTYPE) for _ in range(dim * dim)]
    for i in range(dim):
        for j in range(dim):
            k = i * dim + j
            if i == ind[0] and j == ind[0]:
                re[k] = ops.cos(pr / 2.0)
                im[k] = -ops.sin(pr / 2.0)
            elif i == ind[1] and j == ind[1]:
                re[k] = ops.cos(pr / 2.0)
                im[k] = ops.sin(pr / 2.0)
            elif i == j:
                re[k] = Tensor(1.0, DTYPE)
    re = ops.stack(re).reshape((dim, dim))
    im = ops.stack(im).reshape((dim, dim))
    return re, im


def get_global_phase_gate_cmatrix(dim: int, pr: Parameter) -> Tuple:
    """Get the matrix of global phase gate."""
    re = [Tensor(0.0, DTYPE) for i in range(dim * dim)]
    im = [Tensor(0.0, DTYPE) for i in range(dim * dim)]
    for i in range(dim):
        k = i * dim + i
        re[k] = ops.cos(-pr)
        im[k] = ops.sin(-pr)
    re = ops.stack(re).reshape((dim, dim))
    im = ops.stack(im).reshape((dim, dim))
    return re, im


def get_increment_gate_cmatrix(dim: int) -> Tuple:
    """Get the matrix of increment gate. Read the documents of this package for more information."""
    re = Tensor(np.eye(dim, k=-1), dtype=DTYPE)
    re[dim - 1, dim - 1] = 1.0
    im = ops.zeros((dim, dim))
    return re, im


def get_hadamard_gate_cmatrix(dim: int) -> Tuple:
    """Get the matrix of Hadamard gate. Read the documents of this package for more information."""
    re = ops.zeros((dim, dim), dtype=DTYPE)
    im = ops.zeros_like(re)
    for i in range(dim):
        for j in range(dim):
            re[i, j] = ops.cos(Tensor(2.0 * msnp.pi * i * j / dim))
            im[i, j] = ops.sin(Tensor(2.0 * msnp.pi * i * j / dim))
    re /= ops.sqrt(Tensor(dim, dtype=DTYPE))
    im /= ops.sqrt(Tensor(dim, dtype=DTYPE))
    return re, im


def get_swap_gate_cmatrix(dim: int) -> Tuple:
    """Get the matrix of SWAP gate."""
    n = dim**2
    re = ops.zeros((n, n), dtype=DTYPE)
    im = ops.zeros_like(re)
    for i in range(n):
        for j in range(n):
            if j == (i * dim + i // dim) % n:
                re[i, j] = 1
    return re, im


class GateBase(nn.Cell):
    """Base class for qudit gates."""

    def __init__(
        self,
        dim,
        obj_qudits=None,
        ctrl_qudits=None,
        ctrl_states=None,
        name="GateBase",
    ):
        """Initialize a QuditGate."""
        super().__init__()
        if not isinstance(name, str):
            raise TypeError(f"Excepted string for gate name, get {type(name)}")
        if dim > 10:
            raise ValueError(
                f"The supported maximum dimension is 10, but got {dim}.")
        self.dim = dim
        self.name = name
        if obj_qudits is not None:
            self.on(obj_qudits, ctrl_qudits, ctrl_states)

    def on(self, obj_qudits, ctrl_qudits=None, ctrl_states=None):
        """Define which qudits the gate act on and control qudits."""
        if isinstance(obj_qudits, int):
            obj_qudits = [obj_qudits]
        if isinstance(ctrl_qudits, int):
            ctrl_qudits = [ctrl_qudits]
        if isinstance(ctrl_states, int):
            ctrl_states = [ctrl_states] * len(ctrl_qudits)
        if ctrl_qudits is not None and ctrl_states is None:
            ctrl_states = [self.dim - 1] * len(ctrl_qudits)
        if ctrl_qudits is None:
            ctrl_qudits = []
            ctrl_states = []
        if obj_qudits is None:
            raise ValueError("The `obj_qudits` can't be None")
        if set(obj_qudits) & set(ctrl_qudits):
            raise ValueError(
                f"{self.name} obj_qudits {obj_qudits} and ctrl_qudits {ctrl_qudits} cannot be same"
            )
        if len(set(obj_qudits)) != len(obj_qudits):
            raise ValueError(
                f"{self.name} gate obj_qudits {obj_qudits} cannot be repeated")
        if len(set(ctrl_qudits)) != len(ctrl_qudits):
            raise ValueError(
                f"{self.name} gate ctrl_qudits {ctrl_qudits} cannot be repeated"
            )
        self.obj_qudits = obj_qudits
        self.ctrl_qudits = ctrl_qudits
        self.ctrl_states = ctrl_states
        return self

    def construct(self, qs):
        """Get next state after the gate operation."""
        cmat = self._cmatrix()
        return state_evolution(cmat, qs, self.obj_qudits, self.ctrl_qudits,
                               self.ctrl_states)

    def _cmatrix(self):
        """Get the gate matrix that in Tuple format."""
        return NotImplemented

    def matrix(self):
        """Get the matrix of gate."""
        cmat = self._cmatrix()
        complex = ops.Complex()
        return complex(cmat[0], cmat[1])

    def is_unitary(self):
        """Check if this gate is unitary."""
        mat = self.matrix().numpy()
        return check_unitary(mat)


class NoneParamGate(GateBase):
    """None Parameter Gate."""

    def __init__(
        self,
        dim,
        obj_qudits=None,
        ctrl_qudits=None,
        ctrl_states=None,
        name="NoneParamGate",
    ):
        """Initialize an `NoneParamGate`.

        Args:
            dim: The dimension of qudits.
            name: The gate name.
            n_qudits: The number of qudits.
            obj_qudits: The objective qudits.
            ctrl_qudits: The control qudits.
        """
        super().__init__(dim, obj_qudits, ctrl_qudits, ctrl_states, name)

    def __repr__(self):
        """Return a string representation of the object."""
        assert self.obj_qudits is not None, "There's no object qudit."

        str_obj = " ".join(str(i) for i in self.obj_qudits)
        str_ctrl = " ".join(str(i) for i in self.ctrl_qudits)
        str_ctrl_state = " ".join(str(i) for i in self.ctrl_states)
        if str_ctrl:
            return (
                f"{self.name}({self.dim}|{str_obj} <-: {str_ctrl} - {str_ctrl_state})"
            )
        else:
            return f"{self.name}({self.dim}|{str_obj})"


class WithParamGate(GateBase):
    """Rotation qudit gate."""

    def __init__(
        self,
        dim,
        pr,
        obj_qudits=None,
        ctrl_qudits=None,
        ctrl_states=None,
        name="WithParamGate",
    ):
        """Initialize an RotationGate."""
        super().__init__(dim, obj_qudits, ctrl_qudits, ctrl_states, name)
        self.trainable = True
        self._build_parameter(pr)

    def _build_parameter(self, pr):
        """Construct the parameter according the input."""
        value = DEFAULT_VALUE
        name = DEFAULT_PARAM_NAME

        if pr is None:
            pass
        elif isinstance(pr, Parameter):
            self.param = pr
        elif isinstance(pr, str):
            value = 1.0
            name = pr
        elif isinstance(pr, (int, float, Tensor)):
            value = pr
        else:
            raise TypeError(
                f"Parameter `pr` should be str of Tensor(float32), but get {pr}"
            )
        self.param = Parameter(Tensor(value, DTYPE), name=name)
        self.param_name = name

    def __repr__(self):
        """Return a string representation of the object."""
        str_obj = " ".join(str(i) for i in self.obj_qudits)
        str_ctrl = " ".join(str(i) for i in self.ctrl_qudits)
        str_ctrl_state = " ".join(str(i) for i in self.ctrl_states)
        str_pr = self.param_name
        if str_ctrl:
            return f"{self.name}({self.dim} {str_pr}|{str_obj} <-: {str_ctrl} - {str_ctrl_state})"
        else:
            return f"{self.name}({self.dim} {str_pr}|{str_obj})"

    def no_grad_(self):
        """No gradient for the parameter."""
        self.trainable = False
        if isinstance(self.param, Parameter):
            self.param.requires_grad = False

    def with_grad_(self):
        """Need gradient for the parameter."""
        self.trainable = True
        if isinstance(self.param, Parameter):
            self.param.requires_grad = True

    def assign_param(self, value):
        """Set the value of the parameter. Note: this function won't check if the shape
        of input is reasonable.
        """
        assert self.param is not None, "`param` is None."
        self.param.set_data(Tensor(value, DTYPE))


class PauliNoneParamGate(NoneParamGate):
    """Pauli based none parameter gate. This gate contains two indexes."""

    def __init__(
        self,
        dim,
        ind: Iterable,
        obj_qudits=None,
        ctrl_qudits=None,
        ctrl_states=None,
        name="PauliNoneParamGate",
    ):
        super().__init__(dim, obj_qudits, ctrl_qudits, ctrl_states, name)
        assert (isinstance(ind, Iterable) and len(ind)
                == 2), "The `ind` should be a iterable object with 2 elements."
        assert ind[0] != ind[1], "`ind[0]` must not equal to `ind[1]`."
        assert max(ind) < dim, "Elements in `ind` should less than `dim`."
        self.ind = list(ind)

    def __repr__(self):
        str_ind = " ".join(str(i) for i in self.ind)
        str_obj = " ".join(str(i) for i in self.obj_qudits)
        str_ctrl = " ".join(str(i) for i in self.ctrl_qudits)
        str_ctrl_state = " ".join(str(i) for i in self.ctrl_states)
        if str_ctrl:
            return f"{self.name}({self.dim} [{str_ind}]|{str_obj} <-: {str_ctrl} - {str_ctrl_state})"
        return f"{self.name}({self.dim} [{str_ind}]|{str_obj})"


class X(PauliNoneParamGate):
    """Extended Pauli-X gate for qudit."""

    def __init__(
        self,
        dim,
        ind,
        obj_qudits=None,
        ctrl_qudits=None,
        ctrl_states=None,
        name="X",
    ):
        super().__init__(dim, ind, obj_qudits, ctrl_qudits, ctrl_states, name)
        self.ind = ind

    def _cmatrix(self):
        return get_pauli_x_gate_cmatrix(self.dim, self.ind)


class Y(PauliNoneParamGate):
    """Extended Pauli-Y gate for qudit."""

    def __init__(
        self,
        dim,
        ind,
        obj_qudits=None,
        ctrl_qudits=None,
        ctrl_states=None,
        name="Y",
    ):
        super().__init__(dim, ind, obj_qudits, ctrl_qudits, ctrl_states, name)
        self.ind = ind

    def _cmatrix(self):
        return get_pauli_y_gate_cmatrix(self.dim, self.ind)


class Z(PauliNoneParamGate):
    """Extended Pauli-Z gate for qudit."""

    def __init__(
        self,
        dim,
        ind,
        obj_qudits=None,
        ctrl_qudits=None,
        ctrl_states=None,
        name="Z",
    ):
        super().__init__(dim, ind, obj_qudits, ctrl_qudits, ctrl_states, name)
        self.ind = ind

    def _cmatrix(self):
        return get_pauli_z_gate_cmatrix(self.dim, self.ind)


class H(NoneParamGate):
    """Hadamard gate for qudit."""

    def __init__(self,
                 dim: int,
                 obj_qudits=None,
                 ctrl_qudits=None,
                 ctrl_states=None,
                 name="H"):
        super().__init__(dim, obj_qudits, ctrl_qudits, ctrl_states, name)

    def _cmatrix(self):
        return get_hadamard_gate_cmatrix(self.dim)


class PauliWithParamGate(WithParamGate):
    """Pauli based none parameter gate. This gate contains two indexes."""

    def __init__(
        self,
        dim,
        ind,
        pr,
        obj_qudits,
        ctrl_qudits,
        ctrl_states=None,
        name="PauliWithParamGate",
    ):
        super().__init__(dim, pr, obj_qudits, ctrl_qudits, ctrl_states, name)
        assert (isinstance(ind, Iterable) and len(ind)
                == 2), "The `ind` should be a iterable object with 2 elements."
        assert ind[0] != ind[1], "`ind[0]` must not equal to `ind[1]`."
        assert max(ind) < dim, "Elements in `ind` should less than `dim`."
        self.ind = list(ind)

    def __repr__(self):
        str_ind = " ".join(str(i) for i in self.ind)
        str_obj = " ".join(str(i) for i in self.obj_qudits)
        str_ctrl = " ".join(str(i) for i in self.ctrl_qudits)
        if str_ctrl:
            return f"{self.name}({self.dim} [{str_ind}] {self.param_name}|{str_obj} <-: {str_ctrl})"
        else:
            return f"{self.name}({self.dim} [{str_ind}] {self.param_name}|{str_obj})"


class RX(PauliWithParamGate):
    """Rotation X gate for qudit."""

    def __init__(
        self,
        dim,
        ind,
        pr=None,
        obj_qudits=None,
        ctrl_qudits=None,
        ctrl_states=None,
        name="RX",
    ):
        super().__init__(dim, ind, pr, obj_qudits, ctrl_qudits, ctrl_states,
                         name)
        self.ind = ind

    def _cmatrix(self):
        return get_rotate_x_gate_cmatrix(self.dim, self.ind, self.param)


class RY(PauliWithParamGate):
    """Rotation Y gate for qudit."""

    def __init__(
        self,
        dim,
        ind,
        pr=None,
        obj_qudits=None,
        ctrl_qudits=None,
        ctrl_states=None,
        name="RY",
    ):
        super().__init__(dim, ind, pr, obj_qudits, ctrl_qudits, ctrl_states, name)
        self.ind = ind

    def _cmatrix(self):
        return get_rotate_y_gate_cmatrix(self.dim, self.ind, self.param)


class RZ(PauliWithParamGate):
    """Rotation Z gate for qudit."""

    def __init__(
        self,
        dim,
        ind,
        pr=None,
        obj_qudits=None,
        ctrl_qudits=None,
        ctrl_states=None,
        name="RZ",
    ):
        super().__init__(dim, ind, pr, obj_qudits, ctrl_qudits, ctrl_states, name)
        self.ind = ind

    def _cmatrix(self):
        return get_rotate_z_gate_cmatrix(self.dim, self.ind, self.param)


class GP(WithParamGate):
    """Global phase gate."""

    def __init__(
        self,
        dim,
        pr=None,
        obj_qudits=None,
        ctrl_qudits=None,
        ctrl_states=None,
        name="GP",
    ):
        super().__init__(dim, pr, obj_qudits, ctrl_qudits, ctrl_states, name)

    def _cmatrix(self):
        return get_global_phase_gate_cmatrix(self.dim, self.param)


class INC(NoneParamGate):
    """Increment gate for qudit."""

    def __init__(self,
                 dim: int,
                 obj_qudits=None,
                 ctrl_qudits=None,
                 ctrl_states=None,
                 name="INC"):
        super().__init__(dim, obj_qudits, ctrl_qudits, ctrl_states, name)

    def _cmatrix(self):
        return get_increment_gate_cmatrix(self.dim)


class SWAP(NoneParamGate):
    """Swap gate for qudit."""

    def __init__(self,
                 dim: int,
                 obj_qudits=None,
                 ctrl_qudits=None,
                 ctrl_states=None,
                 name="SWAP"):
        super().__init__(dim, obj_qudits, ctrl_qudits, ctrl_states, name)

    def _cmatrix(self):
        return get_swap_gate_cmatrix(self.dim)


class MVCG(NoneParamGate):
    """Multi-value controlled gate."""

    def __init__(
        self,
        dim: int,
        u_list: List,
        obj_qudits=None,
        ctrl_qudits=None,
        ctrl_states=None,
        name="MVCG",
    ):
        assert len(
            u_list) == dim, "The length of gates should equals to the `dim`."
        super().__init__(dim, obj_qudits, ctrl_qudits, ctrl_states, name)
        self.u_list = u_list

    def _cmatrix(self):
        return get_multi_value_controlled_gate_cmatrix(self.u_list)


class UMG(NoneParamGate):
    """Universal math gate."""

    def __init__(
        self,
        dim: int,
        mat: Union[Tensor, Tuple],
        obj_qudits=None,
        ctrl_qudits=None,
        ctrl_states=None,
        name="UMG",
    ):
        super().__init__(dim, obj_qudits, ctrl_qudits, ctrl_states, name)
        self._mat = mat

    def _cmatrix(self):
        re, im = get_complex_tuple(self._mat)
        return re, im


__all__ = [
    "GateBase",
    "NoneParamGate",
    "WithParamGate",
    "PauliNoneParamGate",
    "PauliWithParamGate",
    "X",
    "Y",
    "Z",
    "RX",
    "RY",
    "RZ",
    "GP",
    "H",
    "INC",
    "MVCG",
    "UMG",
    "SWAP",
]
