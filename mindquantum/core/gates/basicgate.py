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

# pylint: disable=abstract-method,import-outside-toplevel,too-many-lines
"""Basic module for quantum gate."""

import copy
import numbers
from functools import reduce
from inspect import signature
from typing import List, Tuple

import numpy as np
from scipy.linalg import fractional_matrix_power

from mindquantum import mqbackend as mb
from mindquantum.config.config import _GLOBAL_MAT_VALUE
from mindquantum.utils.f import is_power_of_two
from mindquantum.utils.type_value_check import _check_gate_type, _check_input_type

from ..parameterresolver import ParameterResolver
from .basic import (
    BasicGate,
    FunctionalGate,
    NoneParamNonHermMat,
    NoneParamSelfHermMat,
    ParameterGate,
    ParameterOppsGate,
    ParamNonHerm,
    PauliGate,
    PauliStringGate,
    RotSelfHermMat,
)


class UnivMathGate(NoneParamNonHermMat):
    r"""
    Universal math gate.

    More usage, please see :class:`mindquantum.core.gates.XGate`.

    Args:
        name (str): the name of this gate.
        mat (np.ndarray): the matrix value of this gate.

    Examples:
        >>> from mindquantum.core.gates import UnivMathGate
        >>> x_mat=np.array([[0,1],[1,0]])
        >>> X_gate=UnivMathGate('X',x_mat)
        >>> x1=X_gate.on(0,1)
        >>> print(x1)
        X(0 <-: 1)
    """

    def __init__(self, name, matrix_value):
        """Initialize a UnivMathGate object."""
        if len(matrix_value.shape) != 2:
            raise ValueError(f"matrix_value require shape of 2, but get shape of {matrix_value.shape}")
        if matrix_value.shape[0] != matrix_value.shape[1]:
            raise ValueError(f"matrix_value need a square matrix, but get shape {matrix_value.shape}")
        if not is_power_of_two(matrix_value.shape[0]):
            raise ValueError(f"Dimension of matrix_value need should be power of 2, but get {matrix_value.shape[0]}")
        n_qubits = int(np.log2(matrix_value.shape[0]))
        super().__init__(name=name, n_qubits=n_qubits, matrix_value=matrix_value)

    def get_cpp_obj(self):
        """Get the underlying C++ object."""
        mat = mb.dim2matrix(self.matrix())
        cpp_gate = mb.basic_gate(False, self.name, 1, mat)
        cpp_gate.daggered = self.hermitianed
        cpp_gate.obj_qubits = self.obj_qubits
        cpp_gate.ctrl_qubits = self.ctrl_qubits
        cpp_gate.is_custom = True
        return cpp_gate


class HGate(NoneParamSelfHermMat):
    r"""
    Hadamard gate.

    Hadamard gate with matrix as:

    .. math::

        {\rm H}=\frac{1}{\sqrt{2}}\begin{pmatrix}1&1\\1&-1\end{pmatrix}

    More usage, please see :class:`mindquantum.core.gates.XGate`.
    """

    def __init__(self):
        """Initialize an HGate object."""
        super().__init__(
            name='H',
            n_qubits=1,
            matrix_value=_GLOBAL_MAT_VALUE['H'],
        )

    def __eq__(self, other):
        """Equality comparison operator."""
        return BasicGate.__eq__(self, other)


class XGate(PauliGate):
    r"""
    Pauli-X gate.

    Pauli X gate with matrix as:

    .. math::

        {\rm X}=\begin{pmatrix}0&1\\1&0\end{pmatrix}

    For simplicity, we define ```X``` as a instance of ```XGate()```. For more
    redefine, please refer to the functional table below.

    Note:
        For simplicity, you can do power operator on pauli gate (only works
        for pauli gate at this time). The rule is set below as:

        .. math::

            X^\theta = RX(\theta\pi)

    Examples:
        >>> from mindquantum.core.gates import X
        >>> x1 = X.on(0)
        >>> cnot = X.on(0, 1)
        >>> print(x1)
        X(0)
        >>> print(cnot)
        X(0 <-: 1)
        >>> x1.matrix()
        array([[0, 1],
               [1, 0]])
        >>> x1**2
        RX(2π)
        >>> (x1**'a').coeff
        {'a': 3.141592653589793}, const: 0.0
        >>> (x1**{'a' : 2}).coeff
        {'a': 6.283185307179586}, const: 0.0
    """

    def __init__(self):
        """Initialize an XGate object."""
        super().__init__(
            name='X',
            n_qubits=1,
            matrix_value=_GLOBAL_MAT_VALUE['X'],
        )

    def __eq__(self, other):
        """Equality comparison operator."""
        if isinstance(other, CNOTGate):
            obj = [other.obj_qubits[0]]
            ctrl = [other.obj_qubits[1]]
            ctrl.extend(other.ctrl_qubits)
            if self.obj_qubits == obj and set(self.ctrl_qubits) == set(ctrl):
                return True
            return False
        return super().__eq__(other)


class YGate(PauliGate):
    r"""
    Pauli Y gate.

    Pauli Y gate with matrix as:

    .. math::

        {\rm Y}=\begin{pmatrix}0&-i\\i&0\end{pmatrix}

    More usage, please see :class:`mindquantum.core.gates.XGate`.
    """

    def __init__(self):
        """Initialize a YGate object."""
        super().__init__(
            name='Y',
            n_qubits=1,
            matrix_value=_GLOBAL_MAT_VALUE['Y'],
        )


class ZGate(PauliGate):
    r"""
    Pauli-Z gate.

    Pauli Z gate with matrix as:

    .. math::

        {\rm Z}=\begin{pmatrix}1&0\\0&-1\end{pmatrix}

    More usage, please see :class:`mindquantum.core.gates.XGate`.
    """

    def __init__(self):
        """Initialize a ZGate object."""
        super().__init__(
            name='Z',
            n_qubits=1,
            matrix_value=_GLOBAL_MAT_VALUE['Z'],
        )


class IGate(PauliGate):
    r"""
    Identity gate.

    Identity gate with matrix as:

    .. math::

        {\rm I}=\begin{pmatrix}1&0\\0&1\end{pmatrix}

    More usage, please see :class:`mindquantum.core.gates.XGate`.
    """

    def __init__(self):
        """Initialize an IGate object."""
        super().__init__(
            name='I',
            n_qubits=1,
            matrix_value=_GLOBAL_MAT_VALUE['I'],
        )

    def __eq__(self, other):
        """Equality comparison operator."""
        _check_gate_type(other)
        return isinstance(other, IGate)


class CNOTGate(NoneParamSelfHermMat):
    r"""
    Control-X gate.

    More usage, please see :class:`mindquantum.core.gates.XGate`.
    """

    def __init__(self):
        """Initialize a CNOTGate object."""
        super().__init__(
            name='CNOT',
            n_qubits=2,
            matrix_value=_GLOBAL_MAT_VALUE['CNOT'],
        )

    def on(self, obj_qubits, ctrl_qubits=None):
        """
        Define which qubit the gate act on and the control qubit.

        Note:
            In this framework, the qubit that the gate act on is specified
            first, even for control gate, e.g. CNOT, the second arg is control
            qubits.

        Args:
            obj_qubits (int, list[int]): Specific which qubits the gate act on.
            ctrl_qubits (int, list[int]): Specific the control qubits. Default, None.

        Returns:
            Gate, Return a new gate.
        """
        if ctrl_qubits is None:
            raise ValueError("A control qubit is needed for CNOT gate!")
        if isinstance(ctrl_qubits, (int, np.int64)):
            ctrl_qubits = [ctrl_qubits]
        elif not isinstance(ctrl_qubits, list) or not ctrl_qubits:
            raise ValueError(f"ctrl_qubits requires a list, but get {type(ctrl_qubits)}")
        return super().on([obj_qubits, ctrl_qubits[0]], ctrl_qubits[1:])

    def __eq__(self, other):
        """Equality comparison operator."""
        if isinstance(other, XGate):
            return other.__eq__(self)
        return BasicGate.__eq__(self, other)

    def __decompose__(self):
        """Gate decomposition method."""
        return X.on(self.obj_qubits[0], [self.obj_qubits[1], *self.ctrl_qubits]).__decompose__()


class SWAPGate(NoneParamSelfHermMat):
    """
    SWAP gate that swap two different qubits.

    More usage, please see :class:`mindquantum.core.gates.XGate`.
    """

    def __init__(self):
        """Initialize a SWAPGate object."""
        super().__init__(
            name='SWAP',
            n_qubits=2,
            matrix_value=_GLOBAL_MAT_VALUE['SWAP'],
        )

    def __eq__(self, other):
        """Equality comparison operator."""
        _check_gate_type(other)
        if isinstance(other, SWAPGate):
            return set(self.obj_qubits) == set(other.obj_qubits) and set(self.ctrl_qubits) == set(other.ctrl_qubits)
        return False


class ISWAPGate(NoneParamNonHermMat):
    r"""
    ISWAP gate.

    ISWAP gate that swaps two different qubits and phase the :math:`\left|01\right>` and :math:`\left|10\right>`
    amplitudes by :math:`i`.

    More usage, please see :class:`mindquantum.core.gates.XGate`.
    """

    def __init__(self):
        """Initialize an ISWAPGate object."""
        super().__init__(
            name='ISWAP',
            n_qubits=2,
            matrix_value=_GLOBAL_MAT_VALUE['ISWAP'],
        )

    def __eq__(self, other):
        """Equality comparison operator."""
        _check_gate_type(other)
        if isinstance(other, ISWAPGate):
            return set(self.obj_qubits) == set(other.obj_qubits) and set(self.ctrl_qubits) == set(other.ctrl_qubits)
        return False


class TGate(NoneParamNonHermMat):
    r"""
    T gate.

    T gate with matrix as :

    .. math::
        {\rm T}=\begin{pmatrix}1&0\\0&(1+i)/\sqrt(2)\end{pmatrix}

    More usage, please see :class:`mindquantum.core.gates.XGate`.
    """

    def __init__(self):
        """Initialize a TGate object."""
        super().__init__(
            name='T',
            n_qubits=1,
            matrix_value=_GLOBAL_MAT_VALUE['T'],
        )


class SGate(NoneParamNonHermMat):
    r"""
    S gate.

    S gate with matrix as :

    .. math::
        {\rm S}=\begin{pmatrix}1&0\\0&i\end{pmatrix}

    More usage, please see :class:`mindquantum.core.gates.XGate`.
    """

    def __init__(self):
        """Initialize an SGate object."""
        super().__init__(
            name='S',
            n_qubits=1,
            matrix_value=_GLOBAL_MAT_VALUE['S'],
        )


class RX(RotSelfHermMat):
    r"""
    Rotation gate around x-axis.

    .. math::

        {\rm RX}=\begin{pmatrix}\cos(\theta/2)&-i\sin(\theta/2)\\
                       -i\sin(\theta/2)&\cos(\theta/2)\end{pmatrix}

    The rotation gate can be initialized in three different ways.

    1. If you initialize it with a single number, then it will be a non
    parameterized gate with a certain rotation angle.

    2. If you initialize it with a single str, then it will be a
    parameterized gate with only one parameter and the default
    coefficient is one.

    3. If you initialize it with a dict, e.g. `{'a':1,'b':2}`, this gate
    can have multiple parameters with certain coefficients. In this case,
    it can be expressed as:


    .. math::

        RX(a+2b)

    Args:
        pr (Union[int, float, str, dict, ParameterResolver]): the parameters of
            parameterized gate, see above for detail explanation.

    Examples:
        >>> from mindquantum.core.gates import RX
        >>> import numpy as np
        >>> rx1 = RX(0.5)
        >>> np.round(rx1.matrix(), 2)
        array([[0.97+0.j  , 0.  -0.25j],
               [0.  -0.25j, 0.97+0.j  ]])
        >>> rx2 = RX('a')
        >>> np.round(rx2.matrix({'a':0.1}), 3)
        array([[0.999+0.j  , 0.   -0.05j],
               [0.   -0.05j, 0.999+0.j  ]])
        >>> rx3 = RX({'a' : 0.2, 'b': 0.5}).on(0, 2)
        >>> print(rx3)
        RX(0.2*a + 0.5*b|0 <-: 2)
        >>> np.round(rx3.matrix({'a' : 1, 'b' : 2}), 2)
        array([[0.83+0.j  , 0.  -0.56j],
               [0.  -0.56j, 0.83+0.j  ]])
        >>> np.round(rx3.diff_matrix({'a' : 1, 'b' : 2}, about_what = 'a'), 2)
        array([[-0.06+0.j  ,  0.  -0.08j],
               [ 0.  -0.08j, -0.06+0.j  ]])
        >>> rx3.coeff
        {'a': 0.2, 'b': 0.5}
    """

    def __init__(self, pr):
        """Initialize an RX gate."""
        super().__init__(
            pr=ParameterResolver(pr),
            name='RX',
            n_qubits=1,
            core=XGate(),
        )


class RY(RotSelfHermMat):
    r"""
    Rotation gate around y-axis. More usage, please see :class:`mindquantum.core.gates.RX`.

    .. math::

        {\rm RY}=\begin{pmatrix}\cos(\theta/2)&-\sin(\theta/2)\\
                         \sin(\theta/2)&\cos(\theta/2)\end{pmatrix}

    Args:
        pr (Union[int, float, str, dict, ParameterResolver]): the parameters of
            parameterized gate, see above for detail explanation.
    """

    def __init__(self, pr):
        """Initialize an RY object."""
        super().__init__(
            pr=ParameterResolver(pr),
            name='RY',
            n_qubits=1,
            core=YGate(),
        )


class RZ(RotSelfHermMat):
    r"""
    Rotation gate around z-axis. More usage, please see :class:`mindquantum.core.gates.RX`.

    .. math::

        {\rm RZ}=\begin{pmatrix}\exp(-i\theta/2)&0\\
                         0&\exp(i\theta/2)\end{pmatrix}

    Args:
        pr (Union[int, float, str, dict, ParameterResolver]): the parameters of
            parameterized gate, see above for detail explanation.
    """

    def __init__(self, pr):
        """Initialize an RZ object."""
        super().__init__(
            pr=ParameterResolver(pr),
            name='RZ',
            n_qubits=1,
            core=ZGate(),
        )


class ZZ(RotSelfHermMat):
    r"""
    Ising ZZ  gate. More usage, please see :class:`mindquantum.core.gates.RX`.

    .. math::

        {\rm ZZ_\theta}=\cos(\theta)I\otimes I-i\sin(\theta)\sigma_Z\otimes\sigma_Z

    Args:
        pr (Union[int, float, str, dict, ParameterResolver]): the parameters of
            parameterized gate, see above for detail explanation.
    """

    def __init__(self, pr):
        """Initialize a ZZ object."""
        super().__init__(
            pr=ParameterResolver(pr),
            name='ZZ',
            n_qubits=2,
            core=PauliStringGate([Z, Z]),
        )

    def __decompose__(self):
        """Gate decomposition method."""
        from ..circuit import Circuit  # pylint: disable=cyclic-import

        out = []
        out.append(Circuit())
        out[-1] += X.on(self.obj_qubits[0], [self.obj_qubits[1], *self.ctrl_qubits])
        out[-1] += RZ(2 * self.coeff).on(self.obj_qubits[0], [*self.ctrl_qubits])
        out[-1] += X.on(self.obj_qubits[0], [self.obj_qubits[1], *self.ctrl_qubits])
        out.append(Circuit())
        out[-1] += X.on(self.obj_qubits[1], [self.obj_qubits[0], *self.ctrl_qubits])
        out[-1] += RZ(2 * self.coeff).on(self.obj_qubits[1], [*self.ctrl_qubits])
        out[-1] += X.on(self.obj_qubits[1], [self.obj_qubits[0], *self.ctrl_qubits])
        return out

    def matrix(self, pr=None, frac=1):
        """
        Get the matrix of this parameterized gate.

        Args:
            pr (Union[ParameterResolver, dict]): The parameter value for parameterized gate. Default: None.
            frac (numbers.Number): The multiple of the coefficient. Default: 1.

        Returns:
            numpy.ndarray, the matrix of this gate.
        """
        return super().matrix(pr, frac)

    def diff_matrix(self, pr=None, about_what=None, frac=1):
        """
        Differential form of this parameterized gate.

        Args:
            pr (Union[ParameterResolver, dict]): The parameter value for parameterized gate. Default: None.
            about_what (str): calculate the gradient w.r.t which parameter. Default: None.
            frac (numbers.Number): The multiple of the coefficient. Default: 1.

        Returns:
            numpy.ndarray, the differential form matrix.
        """
        return super().diff_matrix(pr, about_what, 1)


class XX(RotSelfHermMat):
    r"""
    Ising XX gate. More usage, please see :class:`mindquantum.core.gates.RX`.

    .. math::

        {\rm XX_\theta}=\cos(\theta)I\otimes I-i\sin(\theta)\sigma_x\otimes\sigma_x

    Args:
        pr (Union[int, float, str, dict, ParameterResolver]): the parameters of
            parameterized gate, see above for detail explanation.
    """

    def __init__(self, pr):
        """Initialize an XX object."""
        super().__init__(
            pr=ParameterResolver(pr),
            name='XX',
            n_qubits=2,
            core=PauliStringGate([X, X]),
        )

    def __decompose__(self):
        """Gate decomposition method."""
        from ..circuit import Circuit  # pylint: disable=cyclic-import

        out = []
        out.append(Circuit())
        out[-1] += H.on(self.obj_qubits[0], [*self.ctrl_qubits])
        out[-1] += H.on(self.obj_qubits[1], [*self.ctrl_qubits])
        out[-1] += X.on(self.obj_qubits[0], [self.obj_qubits[1], *self.ctrl_qubits])
        out[-1] += RZ(2 * self.coeff).on(self.obj_qubits[0], [*self.ctrl_qubits])
        out[-1] += X.on(self.obj_qubits[0], [self.obj_qubits[1], *self.ctrl_qubits])
        out[-1] += H.on(self.obj_qubits[0], [*self.ctrl_qubits])
        out[-1] += H.on(self.obj_qubits[1], [*self.ctrl_qubits])
        out.append(Circuit())
        out[-1] += H.on(self.obj_qubits[0], [*self.ctrl_qubits])
        out[-1] += H.on(self.obj_qubits[1], [*self.ctrl_qubits])
        out[-1] += X.on(self.obj_qubits[1], [self.obj_qubits[0], *self.ctrl_qubits])
        out[-1] += RZ(2 * self.coeff).on(self.obj_qubits[1], [*self.ctrl_qubits])
        out[-1] += X.on(self.obj_qubits[1], [self.obj_qubits[0], *self.ctrl_qubits])
        out[-1] += H.on(self.obj_qubits[0], [*self.ctrl_qubits])
        out[-1] += H.on(self.obj_qubits[1], [*self.ctrl_qubits])
        return out

    def matrix(self, pr=None, frac=1):
        """
        Get the matrix of this parameterized gate.

        Args:
            pr (Union[ParameterResolver, dict]): The parameter value for parameterized gate. Default: None.
            frac (numbers.Number): The multiple of the coefficient. Default: 1.

        Returns:
            numpy.ndarray, the matrix of this gate.
        """
        return super().matrix(pr, 1)

    def diff_matrix(self, pr=None, about_what=None, frac=1):
        """
        Differential form of this parameterized gate.

        Args:
            pr (Union[ParameterResolver, dict]): The parameter value for parameterized gate. Default: None.
            about_what (str): calculate the gradient w.r.t which parameter. Default: None.
            frac (numbers.Number): The multiple of the coefficient. Default: 1.

        Returns:
            numpy.ndarray, the differential form matrix.
        """
        return super().diff_matrix(pr, about_what, 1)


class YY(RotSelfHermMat):
    r"""
    Ising YY  gate. More usage, please see :class:`mindquantum.core.gates.RX`.

    .. math::

        {\rm YY_\theta}=\cos(\theta)I\otimes I-i\sin(\theta)\sigma_y\otimes\sigma_y

    Args:
        pr (Union[int, float, str, dict, ParameterResolver]): the parameters of
            parameterized gate, see above for detail explanation.
    """

    def __init__(self, pr):
        """Initialize an YY object."""
        super().__init__(
            pr=ParameterResolver(pr),
            name='YY',
            n_qubits=2,
            core=PauliStringGate([Y, Y]),
        )

    def __decompose__(self):
        """Gate decomposition method."""
        from ..circuit import Circuit  # pylint: disable=cyclic-import

        out = []
        out.append(Circuit())
        out[-1] += RX(np.pi / 2).on(self.obj_qubits[0], [*self.ctrl_qubits])
        out[-1] += RX(np.pi / 2).on(self.obj_qubits[1], [*self.ctrl_qubits])
        out[-1] += X.on(self.obj_qubits[0], [self.obj_qubits[1], *self.ctrl_qubits])
        out[-1] += RZ(2 * self.coeff).on(self.obj_qubits[0], [*self.ctrl_qubits])
        out[-1] += X.on(self.obj_qubits[0], [self.obj_qubits[1], *self.ctrl_qubits])
        out[-1] += RX(7 * np.pi / 2).on(self.obj_qubits[0], [*self.ctrl_qubits])
        out[-1] += RX(7 * np.pi / 2).on(self.obj_qubits[1], [*self.ctrl_qubits])
        out.append(Circuit())
        out[-1] += RX(np.pi / 2).on(self.obj_qubits[0], [*self.ctrl_qubits])
        out[-1] += RX(np.pi / 2).on(self.obj_qubits[1], [*self.ctrl_qubits])
        out[-1] += X.on(self.obj_qubits[1], [self.obj_qubits[0], *self.ctrl_qubits])
        out[-1] += RZ(2 * self.coeff).on(self.obj_qubits[1], [*self.ctrl_qubits])
        out[-1] += X.on(self.obj_qubits[1], [self.obj_qubits[0], *self.ctrl_qubits])
        out[-1] += RX(7 * np.pi / 2).on(self.obj_qubits[0], [*self.ctrl_qubits])
        out[-1] += RX(7 * np.pi / 2).on(self.obj_qubits[1], [*self.ctrl_qubits])
        return out

    def matrix(self, pr=None, frac=1):
        """
        Get the matrix of this parameterized gate.

        Args:
            pr (Union[ParameterResolver, dict]): The parameter value for parameterized gate. Default: None.
            frac (numbers.Number): The multiple of the coefficient. Default: 1.

        Returns:
            numpy.ndarray, the matrix of this gate.
        """
        return super().matrix(pr, 1)

    def diff_matrix(self, pr=None, about_what=None, frac=1):
        """
        Differential form of this parameterized gate.

        Args:
            pr (Union[ParameterResolver, dict]): The parameter value for parameterized gate. Default: None.
            about_what (str): calculate the gradient w.r.t which parameter. Default: None.
            frac (numbers.Number): The multiple of the coefficient. Default: 1.

        Returns:
            numpy.ndarray, the differential form matrix.
        """
        return super().diff_matrix(pr, about_what, 1)


class BarrierGate(FunctionalGate):
    """
    Barrier gate will separate two gate in two different layer.

    Args:
        show (bool): whether show the barrier gate. Default: True.
    """

    def __init__(self, show=True):
        """Initialize a BarrierGate object."""
        super().__init__(name='BARRIER', n_qubits=0)
        self.show = show

    def on(self, obj_qubits, ctrl_qubits=None):
        """
        Define which qubits the gate act on.

        The control qubits should always be none, since this gate can never be controlled by other qubits.

        Args:
            obj_qubits (int, list[int]): Specific which qubits the gate act on, can be
                a single qubit or a set of consecutive qubits.
            ctrl_qubits (int, list[int]): Specific the control qubits. Default, None.

        Returns:
            Gate, Return a new gate.
        """
        from mindquantum.core.circuit import Circuit

        new = super().on(obj_qubits, ctrl_qubits)
        if new.ctrl_qubits:
            raise ValueError("BarrierGate cannot have ctrl_qubits.")
        all_qubits = sorted(new.obj_qubits)
        qubits = []
        for i in all_qubits:
            if not qubits:
                qubits.append([i])
            else:
                if i - qubits[-1][-1] == 1:
                    qubits[-1].append(i)
                else:
                    qubits.append([i])
        if len(qubits) == 1:
            new.show = self.show
            return new
        return Circuit([BarrierGate(self.show).on(i) for i in qubits])


class GlobalPhase(RotSelfHermMat):
    r"""
    Global phase gate. More usage, please see :class:`mindquantum.core.gates.RX`.

    .. math::

        {\rm GlobalPhase}=\begin{pmatrix}\exp(-i\theta)&0\\
                        0&\exp(-i\theta)\end{pmatrix}

    Args:
        pr (Union[int, float, str, dict, ParameterResolver]): the parameters of
            parameterized gate, see above for detail explanation.
    """

    def __init__(self, pr):
        """Initialize a GlobalPhase object."""
        super().__init__(
            pr=ParameterResolver(pr),
            name='GP',
            n_qubits=1,
            core=IGate(),
        )

    def matrix(self, pr=None, **kwargs):  # pylint: disable=arguments-differ,unused-argument
        """
        Matrix of parameterized gate.

        Args:
            pr (Union[ParameterResolver, dict]): The parameter value for parameterized gate. Default: None.
            kwargs (dict): other key arguments.

        Returns:
            numpy.ndarray, the matrix of this gate.
        """
        return RotSelfHermMat.matrix(self, pr, 1)

    def diff_matrix(self, pr=None, about_what=None, **kwargs):  # pylint: disable=arguments-differ,unused-argument
        """
        Differential form of this parameterized gate.

        Args:
            pr (Union[ParameterResolver, dict]): The parameter value for parameterized gate. Default: None.
            about_what (str): calculate the gradient w.r.t which parameter. Default: None.
            kwargs (dict): other key arguments.

        Returns:
            numpy.ndarray, the differential form matrix.
        """
        return RotSelfHermMat.diff_matrix(self, pr, about_what, 1)


BARRIER = BarrierGate(show=False)


class PhaseShift(ParameterOppsGate):
    r"""
    Phase shift gate. More usage, please see :class:`mindquantum.core.gates.RX`.

    .. math::

        {\rm PhaseShift}=\begin{pmatrix}1&0\\
                         0&\exp(i\theta)\end{pmatrix}

    Args:
        pr (Union[int, float, str, dict, ParameterResolver]): the parameters of
            parameterized gate, see above for detail explanation.
    """

    def __init__(self, pr):
        """Initialize a PhaseShift object."""
        super().__init__(
            pr=ParameterResolver(pr),
            name='PS',
            n_qubits=1,
        )

    # pylint: disable=arguments-differ
    def matrix(self, pr=None):
        """
        Get the matrix of this parameterized gate.

        Args:
            pr (Union[ParameterResolver, dict]): The parameter value for parameterized gate. Default: None.

        Returns:
            numpy.ndarray, the matrix of this gate.
        """
        val = 0
        if self.coeff.is_const():
            val = self.coeff.const
        else:
            new_pr = self.coeff.combination(pr)
            if not new_pr.is_const():
                raise ValueError("The parameter is not set completed.")
            val = new_pr.const
        return np.array([[1, 0], [0, np.exp(1j * val)]])

    def diff_matrix(self, pr=None, about_what=None):
        """
        Differential form of this parameterized gate.

        Args:
            pr (Union[ParameterResolver, dict]): The parameter value for parameterized gate. Default: None.
            about_what (str): calculate the gradient w.r.t which parameter.

        Returns:
            numpy.ndarray, the differential form matrix.
        """
        if self.coeff.is_const():
            return np.zeros((2, 2))
        new_pr = self.coeff.combination(pr)
        if not new_pr.is_const():
            raise ValueError("The parameter is not set completed.")
        val = new_pr.const
        if about_what is None:
            if len(self.coeff) != 1:
                raise ValueError("Should specific which parameter are going to do derivation.")
            for i in self.coeff:
                about_what = i
        return np.array([[0, 0], [0, 1j * self.coeff[about_what] * np.exp(1j * val)]])


class Power(NoneParamNonHermMat):
    r"""
    Power operator on a non parameterized gate.

    Args:
        gates (:class:`mindquantum.core.gates.NoneParameterGate`): The basic gate you need to apply power operator.
        exponent (int, float): The exponent. Default: 0.5.

    Examples:
        >>> from mindquantum.core.gates import Power
        >>> import numpy as np
        >>> rx1 = RX(0.5)
        >>> rx2 = RX(1)
        >>> assert np.all(np.isclose(Power(rx2,0.5).matrix(), rx1.matrix()))
    """

    def __init__(self, gate, exponent=0.5):
        """Initialize a Power object."""
        _check_input_type('t', numbers.Number, exponent)
        name = f'{gate}^{exponent}'
        n_qubits = gate.n_qubits
        matrix_value = fractional_matrix_power(gate.matrix(), exponent)
        super().__init__(
            name=name,
            n_qubits=n_qubits,
            matrix_value=matrix_value,
        )
        self.gate = gate
        self.t = exponent  # pylint: disable=invalid-name

    def get_cpp_obj(self):
        """Get the underlying C++ object."""
        mat = mb.dim2matrix(self.matrix())
        cpp_gate = mb.basic_gate(False, self.name, 1, mat)
        cpp_gate.daggered = self.hermitianed
        cpp_gate.obj_qubits = self.obj_qubits
        cpp_gate.ctrl_qubits = self.ctrl_qubits
        cpp_gate.is_custom = True
        return cpp_gate

    def __eq__(self, other):
        """Equality comparison operator."""
        _check_gate_type(other)
        if self.obj_qubits == other.obj_qubits and set(self.ctrl_qubits) == set(other.ctrl_qubits):
            if self.gate == other and self.t == 1:
                return True
            if isinstance(other, Power):
                if self.gate == other.gate and self.t == other.t:
                    return True
        return False


def wrapper_numba(compiled_fun):
    """Wrap a compiled function with numba."""
    try:
        import numba as nb

        try:
            import importlib.metadata as importlib_metadata
        except ImportError:
            import importlib_metadata
        import packaging.version

        nb_version = importlib_metadata.version('numba')

        nb_requires = packaging.version.parse('0.53.1')
        if packaging.version.parse(nb_version) < nb_requires:
            raise ImportError(
                "To use customized parameterized gate, please install numba with 'pip install \"numba>=0.53.1\"'."
            )
    except ImportError as exc:
        raise ImportError(
            "To use customized parameterized gate, please install numba with 'pip install \"numba>=0.53.1\"'."
        ) from exc

    @nb.cfunc(nb.types.void(nb.types.double, nb.types.CPointer(nb.types.complex128)))
    def fun(theta, out_):
        """Map data to array."""
        out = nb.carray(
            out_,
            (2, 2),
        )
        matrix = compiled_fun(theta)
        for i in range(2):
            for j in range(2):
                out[i][j] = matrix[i][j]

    return fun.address


# pylint: disable=too-many-locals,too-many-statements
def gene_univ_parameterized_gate(name, matrix_generator, diff_matrix_generator):
    """
    Generate a customer parameterized gate based on the single parameter defined unitary matrix.

    Args:
        name (str): The name of this gate.
        matrix_generator (Union[FunctionType, MethodType]): A function or a method that
            take exactly one argument to generate a unitary matrix.
        diff_matrix_generator (Union[FunctionType, MethodType]): A function or a method
            that take exactly one argument to generate the derivative of this unitary matrix.

    Returns:
        _ParamNonHerm, a customer parameterized gate.

    Examples:
        >>> import numpy as np
        >>> from mindquantum.core.circuit import Circuit
        >>> from mindquantum.core.gates import gene_univ_parameterized_gate
        >>> from mindquantum.simulator import Simulator
        >>> def matrix(theta):
        ...     return np.array([[np.exp(1j * theta), 0],
        ...                      [0, np.exp(-1j * theta)]])
        >>> def diff_matrix(theta):
        ...     return 1j*np.array([[np.exp(1j * theta), 0],
        ...                         [0, -np.exp(-1j * theta)]])
        >>> TestGate = gene_univ_parameterized_gate('Test', matrix, diff_matrix)
        >>> circ = Circuit().h(0)
        >>> circ += TestGate('a').on(0)
        >>> circ
        q0: ──H────Test(a)──
        >>> circ.get_qs(pr={'a': 1.2})
        array([0.25622563+0.65905116j, 0.25622563-0.65905116j])
    """
    try:
        import numba as nb

        try:
            import importlib.metadata as importlib_metadata
        except ImportError:
            import importlib_metadata
        import packaging.version

        nb_version = importlib_metadata.version('numba')

        nb_requires = packaging.version.parse('0.53.1')
        if packaging.version.parse(nb_version) < nb_requires:
            raise ImportError(
                "To use customized parameterized gate, please install numba with 'pip install \"numba>=0.53.1\"'."
            )
    except ImportError as exc:
        raise ImportError(
            "To use customized parameterized gate, please install numba with 'pip install \"numba>=0.53.1\"'."
        ) from exc
    matrix_sig = signature(matrix_generator)
    diff_matrix_sig = signature(diff_matrix_generator)
    if len(matrix_sig.parameters) != 1:
        raise ValueError(f"matrix_generator can only have one argument, but get {len(matrix_sig.parameters)}.")
    if len(diff_matrix_sig.parameters) != 1:
        raise ValueError(f"diff_matrix_generator can only have one argument, but get {len(diff_matrix_sig.parameters)}")

    matrix = matrix_generator(0.123)
    diff_matrix = diff_matrix_generator(0.123)
    if matrix.shape != diff_matrix.shape:
        raise ValueError("matrix_generator and diff_matrix_generator should generate same shape matrix.")
    if not isinstance(matrix, np.ndarray) or matrix.dtype != complex:
        raise ValueError(f"matrix_generator should return numpy array with complex type, but get {type(matrix)}.")
    if not isinstance(diff_matrix, np.ndarray) or diff_matrix.dtype != complex:
        raise ValueError(
            f"diff_matrix_generator should return numpy array with complex type, but get {type(diff_matrix)}"
        )

    n_qubits = int(np.log2(matrix.shape[0]))
    if n_qubits not in [1, 2]:
        raise ValueError(f"Can only custom one or two qubits gate, but get {n_qubits} qubits")
    c_sig = nb.types.Array(nb.types.complex128, 2, 'C')(nb.types.double)
    c_matrix_generator = nb.cfunc(c_sig)(matrix_generator)
    c_diff_matrix_generator = nb.cfunc(c_sig)(diff_matrix_generator)

    def herm_matrix_generator(x):
        """Generate hermitian conjugate matrix."""
        return np.conj(c_matrix_generator(x)).T.copy()

    def herm_diff_matrix_generator(x):
        """Generate hermitian conjugate diff matrix."""
        return np.conj(c_diff_matrix_generator(x)).T.copy()

    matrix_addr = wrapper_numba(c_matrix_generator)
    diff_matrix_addr = wrapper_numba(c_diff_matrix_generator)
    herm_matrix_addr = wrapper_numba(nb.cfunc(c_sig)(herm_matrix_generator))
    herm_diff_matrix_addr = wrapper_numba(nb.cfunc(c_sig)(herm_diff_matrix_generator))

    class _ParamNonHerm(ParamNonHerm):
        """The customer parameterized gate."""

        def __init__(self, pr):
            super().__init__(
                pr=ParameterResolver(pr),
                name=name,
                n_qubits=n_qubits,
                matrix_generator=matrix_generator,
                diff_matrix_generator=diff_matrix_generator,
            )

        def __deepcopy__(self, memo):
            """Deep copy this gate."""
            copied_gate = _ParamNonHerm(self.coeff)
            copied_gate.obj_qubits = self.obj_qubits
            copied_gate.ctrl_qubits = self.ctrl_qubits
            copied_gate.hermitianed = self.hermitianed
            return copied_gate

        def hermitian(self):
            """Get hermitian conjugate gate."""
            hermitian_gate = _ParamNonHerm(self.coeff)
            hermitian_gate.obj_qubits = self.obj_qubits
            hermitian_gate.ctrl_qubits = self.ctrl_qubits
            hermitian_gate.hermitianed = not self.hermitianed
            if hermitian_gate.hermitianed:
                hermitian_gate.matrix_generator = herm_matrix_generator
                hermitian_gate.diff_matrix_generator = herm_diff_matrix_generator
            return hermitian_gate

        def get_cpp_obj(self):
            cpp_gate = mb.basic_gate(self.name, 1, matrix_addr, diff_matrix_addr, 1 << n_qubits)
            if self.hermitianed:
                cpp_gate = mb.basic_gate(self.name, 1, herm_matrix_addr, herm_diff_matrix_addr, 1 << n_qubits)
            cpp_gate.daggered = self.hermitianed
            cpp_gate.obj_qubits = self.obj_qubits
            cpp_gate.ctrl_qubits = self.ctrl_qubits
            cpp_gate.is_custom = True
            if not self.parameterized:
                cpp_gate.apply_value(self.coeff.const)
            else:
                cpp_gate.params = self.coeff.to_real_obj()
            return cpp_gate

    return _ParamNonHerm


class MultiParamsGate(ParameterGate):
    """Gate with multi intrinsic parameters."""

    def __init__(self, name: str, n_qubits: int, prs: List[ParameterResolver]):
        """Initialize a MultiParamsGate object."""
        super().__init__(pr=None, name=name, n_qubits=n_qubits, obj_qubits=None, ctrl_qubits=None)
        self.prs = prs

    def __type_specific_str__(self) -> str:
        """Get parameter string."""
        return ', '.join([pr.expression() for pr in self.prs])

    def __call__(self, prs: List[ParameterResolver]) -> "MultiParamsGate":
        """Generate new one with given parameters."""
        new = copy.deepcopy(self)
        new.prs = prs
        return new

    def __params_prop__(self) -> Tuple[List[str], List[str], List[str]]:
        """Get properties of all parameters.

        Returns:
            (List[str], List[str], List[str]), a tuple with first element is all
            parameters, second one is all ansatz parameters, and the third one is
            all encoder parameters.
        """

        def extend(x, y):
            """Extend element."""
            x.extend(y)
            return x

        keys = []
        ansatz_params = []
        encoder_parameters = []
        reduce(extend, [list(pr.keys()) for pr in self.prs], keys)
        reduce(extend, [list(pr.ansatz_parameters) for pr in self.prs], ansatz_params)
        reduce(extend, [list(pr.encoder_parameters) for pr in self.prs], encoder_parameters)
        return keys, ansatz_params, encoder_parameters

    @property
    def parameterized(self) -> bool:
        """Check whether this gate is a parameterized gate."""
        for pr in self.prs:
            if not pr.is_const():
                return True
        return False

    def get_parameters(self) -> List[ParameterResolver]:
        """Return a list of parameters of parameterized gate."""
        return self.prs

    def no_grad(self):
        """Set all parameters do not require gradient."""
        for pr in self.prs:
            pr.no_grad()
        return self

    def requires_grad(self):
        """Set all parameters require gradient."""
        for pr in self.prs:
            pr.requires_grad()
        return self

    def requires_grad_part(self, names: List[str]):
        """Set part of parameters require gradient."""
        for pr in self.prs:
            pr.requires_grad_part(names)
        return self

    def no_grad_part(self, names: List[str]):
        """Set part of parameters not require gradient."""
        for pr in self.prs:
            pr.no_grad_part(names)
        return self


class U3(MultiParamsGate):
    r"""
    U3 gate represent arbitrary single qubit gate.

    U3 gate with matrix as:

    .. math::

        U3(\theta, \phi, \lambda) =\begin{pmatrix}\cos(\theta/2)&-e^{i\lambda}\sin(\theta/2)\\
            e^{i\phi}\sin(\theta/2)&e^{i(\phi+\lambda)}\cos(\theta/2)\end{pmatrix}

    It can be decomposed as:

    .. math::

        U3(\theta, \phi, \lambda) = RZ(\phi) RX(-\pi/2) RZ(\theta) RX(\pi/2) RZ(\lambda)

    Args:
        theta (Union[numbers.Number, dict, ParameterResolver]): First parameter for U3 gate.
        phi (Union[numbers.Number, dict, ParameterResolver]): Second parameter for U3 gate.
        lamda (Union[numbers.Number, dict, ParameterResolver]): Third parameter for U3 gate.

    Examples:
        >>> from mindquantum.core.gates import U3
        >>> U3('theta','phi','lambda').on(0, 1)
        U3(𝜃=theta, 𝜑=phi, 𝜆=lambda|0 <-: 1)
    """

    def __init__(self, theta: ParameterResolver, phi: ParameterResolver, lamda: ParameterResolver):
        """Initialize U3 gate."""
        prs = [ParameterResolver(theta), ParameterResolver(phi), ParameterResolver(lamda)]
        super().__init__(name="U3", n_qubits=1, prs=prs)

    def __type_specific_str__(self) -> str:
        """Get parameter string."""
        return f"𝜃={self.theta.expression()}, 𝜑={self.phi.expression()}, 𝜆={self.lamda.expression()}"

    def __call__(self, theta: ParameterResolver, phi: ParameterResolver, lamda: ParameterResolver) -> "U3":
        """Call the U3 gate with new parameters."""
        theta = ParameterResolver(theta)
        phi = ParameterResolver(phi)
        lamda = ParameterResolver(lamda)
        prs = [theta, phi, lamda]
        return super().__call__(prs)

    @property
    def theta(self) -> ParameterResolver:
        """
        Get theta parameter of U3 gate.

        Returns:
            ParameterResolver, the theta.
        """
        return self.prs[0]

    @property
    def phi(self) -> ParameterResolver:
        """
        Get phi parameter of U3 gate.

        Returns:
            ParameterResolver, the phi.
        """
        return self.prs[1]

    @property
    def lamda(self) -> ParameterResolver:
        """
        Get lamda parameter of U3 gate.

        Returns:
            ParameterResolver, the lamda.
        """
        return self.prs[2]

    def hermitian(self) -> "U3":
        """
        Get hermitian form of U3 gate.

        Examples:
            >>> from mindquantum.core.gates import U3
            >>> u3 = U3('a', 'b', 0.5).on(0)
            >>> u3.hermitian()
            U3(𝜃=-a, 𝜑=-1/2, 𝜆=-b|0)
        """
        out = U3(-self.theta, -self.lamda, -self.phi)
        out.obj_qubits = self.obj_qubits
        out.ctrl_qubits = self.ctrl_qubits
        return out

    # pylint: disable=arguments-differ
    def matrix(self, pr: ParameterResolver = None) -> np.ndarray:
        """
        Get the matrix of U3 gate.

        Args:
            pr (Union[ParameterResolver, dict]): The parameter for U3 gate.
        """
        theta = self.theta
        phi = self.phi
        lamda = self.lamda
        if self.parameterized:
            if pr is None:
                raise ValueError("Parameterized gate need a parameter resolver to get matrix.")
            theta = theta.combination(pr)
            phi = phi.combination(pr)
            lamda = lamda.combination(pr)
            if not theta.is_const():
                raise ValueError("Theta not set completed.")
            if not phi.is_const():
                raise ValueError("Phi not set completed.")
            if not lamda.is_const():
                raise ValueError("Lambda not set completed.")
        theta = theta.const
        phi = phi.const
        lamda = lamda.const
        return np.array(
            [
                [np.cos(theta / 2), -np.exp(1j * lamda) * np.sin(theta / 2)],
                [np.exp(1j * phi) * np.sin(theta / 2), np.exp(1j * (phi + lamda)) * np.cos(theta / 2)],
            ]
        )

    def get_cpp_obj(self):
        """Get cpp obj."""
        return mb.u3(
            self.theta.get_cpp_obj(),
            self.phi.get_cpp_obj(),
            self.lamda.get_cpp_obj(),
            self.obj_qubits,
            self.ctrl_qubits,
        )


class FSim(MultiParamsGate):
    r"""
    FSim gate represent fermionic simulation gate.

    The matrix is:

    .. math::

        FSim(\theta, \phi) =
        \begin{pmatrix}
            1 & 0 & 0 & 0\\
            0 & \cos(\theta) & -i\sin(\theta) & 0\\
            0 & -i\sin(\theta) & \cos(\theta) & 0\\
            0 & 0 & 0 & e^{i\phi}\\
        \end{pmatrix}

    Args:
        theta (Union[numbers.Number, dict, ParameterResolver]): First parameter for FSim gate.
        phi (Union[numbers.Number, dict, ParameterResolver]): Second parameter for FSim gate.

    Examples:
        >>> from mindquantum.core.gates import FSim
        >>> fsim = FSim('a', 'b').on([0, 1])
        >>> fsim
        FSim(𝜃=a, 𝜑=b|0 1)
    """

    def __init__(self, theta: ParameterResolver, phi: ParameterResolver):
        """Initialize FSim gate."""
        prs = [ParameterResolver(theta), ParameterResolver(phi)]
        super().__init__(name="FSim", n_qubits=2, prs=prs)

    def __type_specific_str__(self) -> str:
        """Get parameter string."""
        return f"𝜃={self.theta.expression()}, 𝜑={self.phi.expression()}"

    def __call__(self, theta: ParameterResolver, phi: ParameterResolver) -> "FSim":
        """Generate a new FSim gate with given parameters."""
        theta = ParameterResolver(theta)
        phi = ParameterResolver(phi)
        prs = [theta, phi]
        return super().__call__(prs)

    @property
    def theta(self) -> ParameterResolver:
        """
        Get theta parameter of FSim gate.

        Returns:
            ParameterResolver, the theta.
        """
        return self.prs[0]

    @property
    def phi(self) -> ParameterResolver:
        """
        Get phi parameter of FSim gate.

        Returns:
            ParameterResolver, the phi.
        """
        return self.prs[1]

    def hermitian(self) -> "FSim":
        """
        Get the hermitian gate of FSim.

        Examples:
            >>> from mindquantum.core.gates import FSim
            >>> fsim = FSim('a', 'b').on([0, 1])
            >>> fsim.hermitian()
            FSim(𝜃=-a, 𝜑=-b|0 1)
        """
        out = FSim(-self.theta, -self.phi)
        out.obj_qubits = self.obj_qubits
        out.ctrl_qubits = self.ctrl_qubits
        return out

    # pylint: disable=arguments-differ
    def matrix(self, pr: ParameterResolver = None) -> np.ndarray:
        """
        Get the matrix of FSim.

        Args:
            pr (Union[ParameterResolver, dict]): The parameter of fSim gate. Default: None.
        """
        theta = self.theta
        phi = self.phi
        if self.parameterized:
            if pr is None:
                raise ValueError("Parameterized gate need a parameter resolver to get matrix.")
            theta = theta.combination(pr)
            phi = phi.combination(pr)
            if not theta.is_const():
                raise ValueError("Theta not set completed.")
            if not phi.is_const():
                raise ValueError("Phi not set completed.")
        theta = theta.const
        phi = phi.const
        ele_a = np.cos(theta)
        ele_b = -1j * np.sin(theta)
        ele_c = np.exp(1j * phi)
        return np.array([[1, 0, 0, 0], [0, ele_a, ele_b, 0], [0, ele_b, ele_a, 0], [0, 0, 0, ele_c]])

    def get_cpp_obj(self):
        """Get cpp object."""
        return mb.fsim(self.theta.get_cpp_obj(), self.phi.get_cpp_obj(), self.obj_qubits, self.ctrl_qubits)


X = XGate()
Y = YGate()
Z = ZGate()
I = IGate()  # noqa: E741
H = HGate()
T = TGate()
S = SGate()
CNOT = CNOTGate()
ISWAP = ISWAPGate()
SWAP = SWAPGate()
