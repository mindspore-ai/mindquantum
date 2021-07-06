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
"""Basic quantum gate."""

import numpy as np
import projectq.ops as pjops
from scipy.linalg import fractional_matrix_power
from mindquantum.parameterresolver import ParameterResolver as PR
from .basic import IntrinsicOneParaGate
from .basic import NoneParameterGate


class CNOTGate(NoneParameterGate):
    r"""
    Control-X gate.

    More usage, please see :class:`mindquantum.gate.XGate`.
    """
    def __init__(self):
        NoneParameterGate.__init__(self, 'CNOT')
        self.matrix_value = X.matrix_value

    def define_projectq_gate(self):
        """Define the corresponded projectq gate."""
        self.projectq_gate = pjops.CNOT


class HGate(NoneParameterGate):
    r"""
    Hadamard gate with matrix as:

    .. math::

        {\rm H}=\frac{1}{\sqrt{2}}\begin{pmatrix}1&1\\1&-1\end{pmatrix}

    More usage, please see :class:`mindquantum.gate.XGate`.
    """
    def __init__(self):
        NoneParameterGate.__init__(self, 'H')
        self.matrix_value = np.array([[1, 1], [1, -1]]) / np.sqrt(2)

    def define_projectq_gate(self):
        """Define the corresponded projectq gate."""
        self.projectq_gate = pjops.H


class IGate(NoneParameterGate):
    r"""
    Identity gate with matrix as:

    .. math::

        {\rm I}=\begin{pmatrix}1&0\\0&1\end{pmatrix}

    More usage, please see :class:`mindquantum.gate.XGate`.
    """
    def __init__(self):
        NoneParameterGate.__init__(self, 'I')
        self.matrix_value = np.array([[1, 0], [0, 1]])

    def define_projectq_gate(self):
        """Define the corresponded projectq gate."""
        self.projectq_gate = None


class XGate(NoneParameterGate):
    r"""
    Pauli X gate with matrix as:

    .. math::

        {\rm X}=\begin{pmatrix}0&1\\1&0\end{pmatrix}

    Note:
        For simplicity, you can do power operator on pauli gate (only works
        for pauli gate at this time). The rules is set below as:

        .. math::

            X^\theta = RX(\theta\pi)

    Examples:
        >>> from mindquantum.gate import X
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
        RX(6.283)
        >>> (x1**'a').coeff
        {'a': 3.141592653589793}
        >>> (x1**{'a' : 2}).coeff
        {'a': 6.283185307179586}
    """
    def __init__(self):
        NoneParameterGate.__init__(self, 'X')
        self.matrix_value = np.array([[0, 1], [1, 0]])

    def __pow__(self, coeff):
        if isinstance(coeff, (float, int, complex)):
            return RX(coeff * np.pi)
        if isinstance(coeff, str):
            return RX({coeff: np.pi})
        if isinstance(coeff, PR):
            return RX(np.pi * coeff)
        if isinstance(coeff, dict):
            return RX({i: np.pi * j for i, j in coeff.items()})
        raise TypeError(
            "Unsupported type for parameters, get {}.".format(coeff))

    def define_projectq_gate(self):
        """Define the corresponded projectq gate."""
        self.projectq_gate = pjops.X


class YGate(NoneParameterGate):
    r"""
    Pauli Y gate with matrix as:

    .. math::

        {\rm Y}=\begin{pmatrix}0&-i\\i&0\end{pmatrix}

    More usage, please see :class:`mindquantum.gate.XGate`.
    """
    def __init__(self):
        NoneParameterGate.__init__(self, 'Y')
        self.matrix_value = np.array([[0, -1j], [1j, 0]])

    def __pow__(self, coeff):
        if isinstance(coeff, (float, int, complex)):
            return RY(coeff * np.pi)
        if isinstance(coeff, str):
            return RY({coeff: np.pi})
        if isinstance(coeff, PR):
            return RY(np.pi * coeff)
        if isinstance(coeff, dict):
            return RY({i: np.pi * j for i, j in coeff.items()})
        raise TypeError(
            "Unsupported type for parameters, get {}.".format(coeff))

    def define_projectq_gate(self):
        """Define the corresponded projectq gate."""
        self.projectq_gate = pjops.Y


class ZGate(NoneParameterGate):
    r"""
    Pauli Z gate with matrix as:

    .. math::

        {\rm Z}=\begin{pmatrix}1&0\\0&-1\end{pmatrix}

    More usage, please see :class:`mindquantum.gate.XGate`.
    """
    def __init__(self):
        NoneParameterGate.__init__(self, 'Z')
        self.matrix_value = np.array([[1, 0], [0, -1]])

    def __pow__(self, coeff):
        if isinstance(coeff, (float, int, complex)):
            return RZ(coeff * np.pi)
        if isinstance(coeff, str):
            return RZ({coeff: np.pi})
        if isinstance(coeff, PR):
            return RZ(np.pi * coeff)
        if isinstance(coeff, dict):
            return RZ({i: np.pi * j for i, j in coeff.items()})
        raise TypeError(
            "Unsupported type for parameters, get {}.".format(coeff))

    def define_projectq_gate(self):
        """Define the corresponded projectq gate."""
        self.projectq_gate = pjops.Z


class UnivMathGate(NoneParameterGate):
    r"""
    Universal math gate.

    More usage, please see :class:`mindquantum.gate.XGate`.

    Args:
        name (str): the name of this gate.
        mat (np.ndarray): the matrix value of this gate.

    Examples:
        >>> from mindquantum.gate import UnivMathGate
        >>> x_mat=np.array([[0,1],[1,0]])
        >>> X_gate=UnivMathGate('X',x_mat)
        >>> x1=X_gate.on(0,1)
        >>> print(x1)
        X(0 <-: 1)
    """
    def __init__(self, name, mat):
        NoneParameterGate.__init__(self, name)
        self.matrix_value = mat

    def define_projectq_gate(self):
        """Define the corresponded projectq gate."""
        self.projectq_gate = None


class SWAPGate(NoneParameterGate):
    """
    SWAP gate that swap two different qubits.

    More usage, please see :class:`mindquantum.gate.XGate`.
    """
    def __init__(self):
        NoneParameterGate.__init__(self, 'SWAP')
        self.matrix_value = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0],
                                      [0, 0, 0, 1]])

    def define_projectq_gate(self):
        """Define the corresponded projectq gate."""
        self.projectq_gate = pjops.Swap


class RX(IntrinsicOneParaGate):
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
    coefficience is one.

    3. If you initialize it with a dict, e.g. `{'a':1,'b':2}`, this gate
    can have multiple parameters with certain coefficiences. In this case,
    it can be expressed as:


    .. math::

        RX(a+2b)

    Args:
        coeff (Union[int, float, str, dict]): the parameters of
            parameterized gate, see above for detail explanation. Default: None.

    Examples:
        >>> from mindquantum.gate import RX
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
        RX(a b|0 <-: 2)
        >>> np.round(rx3.matrix({'a' : 1, 'b' : 2}), 2)
        array([[0.83+0.j  , 0.  -0.56j],
               [0.  -0.56j, 0.83+0.j  ]])
        >>> np.round(rx3.diff_matrix({'a' : 1, 'b' : 2}, about_what = 'a'), 2)
        array([[-0.06+0.j  ,  0.  -0.08j],
               [ 0.  -0.08j, -0.06+0.j  ]])
        >>> rx3.coeff
        {'a': 0.2, 'b': 0.5}
    """
    def __init__(self, coeff=None):
        IntrinsicOneParaGate.__init__(self, 'RX', coeff)

    def _matrix(self, theta):
        return np.array([[np.cos(theta / 2), -1j * np.sin(theta / 2)],
                         [-1j * np.sin(theta / 2),
                          np.cos(theta / 2)]])

    def _diff_matrix(self, theta):
        return 0.5 * np.array([[-np.sin(theta / 2), -1j * np.cos(theta / 2)],
                               [-1j * np.cos(theta / 2), -np.sin(theta / 2)]])

    def define_projectq_gate(self):
        """Define the corresponded projectq gate."""
        self.projectq_gate = pjops.Rx(self.coeff)


class RZ(IntrinsicOneParaGate):
    r"""
    Rotation gate around z-axis.

    .. math::

        {\rm RZ}=\begin{pmatrix}\exp(-i\theta/2)&0\\
                         0&\exp(i\theta/2)\end{pmatrix}

    More usage, please see :class:`mindquantum.gate.RX`.
    """
    def __init__(self, coeff=None):
        IntrinsicOneParaGate.__init__(self, 'RZ', coeff)

    def _matrix(self, theta):
        return np.array([[np.exp(-1j * theta / 2), 0],
                         [0, np.exp(1j * theta / 2)]])

    def _diff_matrix(self, theta):
        return 0.5j * np.array([[-np.exp(-1j * theta / 2), 0],
                                [0, np.exp(1j * theta / 2)]])

    def define_projectq_gate(self):
        """Define the corresponded projectq gate."""
        self.projectq_gate = pjops.Rz(self.coeff)


class RY(IntrinsicOneParaGate):
    r"""
    Rotation gate around z-axis.

    .. math::

        {\rm RY}=\begin{pmatrix}\cos(\theta/2)&-\sin(\theta/2)\\
                         \sin(\theta/2)&\cos(\theta/2)\end{pmatrix}

    More usage, please see :class:`mindquantum.gate.RX`.
    """
    def __init__(self, coeff=None):
        IntrinsicOneParaGate.__init__(self, 'RY', coeff)

    def _matrix(self, theta):
        return np.array([[np.cos(theta / 2), -np.sin(theta / 2)],
                         [np.sin(theta / 2),
                          np.cos(theta / 2)]])

    def _diff_matrix(self, theta):
        return 0.5 * np.array([[-np.sin(theta / 2), -np.cos(theta / 2)],
                               [np.cos(theta / 2), -np.sin(theta / 2)]])

    def define_projectq_gate(self):
        """Define the corresponded projectq gate."""
        self.projectq_gate = pjops.Ry(self.coeff)


class PhaseShift(IntrinsicOneParaGate):
    r"""
    Phase shift gate.

    .. math::

        {\rm PhaseShift}=\begin{pmatrix}1&0\\
                         0&\exp(i\theta)\end{pmatrix}

    More usage, please see :class:`mindquantum.gate.RX`.
    """
    def __init__(self, coeff=None):
        IntrinsicOneParaGate.__init__(self, 'PS', coeff)

    def _matrix(self, theta):
        return np.array([[1, 0], [0, np.exp(1j * theta)]])

    def _diff_matrix(self, theta):
        return np.array([[0, 0], [0, 1j * np.exp(1j * theta)]])

    def define_projectq_gate(self):
        """Define the corresponded projectq gate."""
        self.projectq_gate = pjops.R


class XX(IntrinsicOneParaGate):
    r"""
    Ising XX gate.

    .. math::

        {\rm XX_\theta}=\cos(\theta)I\otimes I-i\sin(\theta)\sigma_x\otimes\sigma_x

    More usage, please see :class:`mindquantum.gate.RX`.
    """
    def __init__(self, coeff=None):
        IntrinsicOneParaGate.__init__(self, 'XX', coeff)

    def _matrix(self, theta):
        return np.array([[np.cos(theta), 0, 0, -1j * np.sin(theta)],
                         [0, np.cos(theta), -1j * np.sin(theta), 0],
                         [0, -1j * np.sin(theta),
                          np.cos(theta), 0],
                         [-1j * np.sin(theta), 0, 0,
                          np.cos(theta)]])

    def _diff_matrix(self, theta):
        return np.array([[-np.sin(theta), 0, 0, -1j * np.cos(theta)],
                         [0, -np.sin(theta), -1j * np.cos(theta), 0],
                         [0, -1j * np.cos(theta), -np.sin(theta), 0],
                         [-1j * np.cos(theta), 0, 0, -np.sin(theta)]])

    def define_projectq_gate(self):
        """Define the corresponded projectq gate."""
        self.projectq_gate = None


class YY(IntrinsicOneParaGate):
    r"""
    Ising YY  gate.

    .. math::

        {\rm YY_\theta}=\cos(\theta)I\otimes I-i\sin(\theta)\sigma_y\otimes\sigma_y

    More usage, please see :class:`mindquantum.gate.RX`.
    """
    def __init__(self, coeff=None):
        IntrinsicOneParaGate.__init__(self, 'YY', coeff)

    def _matrix(self, theta):
        return np.array([[np.cos(theta), 0, 0, 1j * np.sin(theta)],
                         [0, np.cos(theta), -1j * np.sin(theta), 0],
                         [0, -1j * np.sin(theta),
                          np.cos(theta), 0],
                         [1j * np.sin(theta), 0, 0,
                          np.cos(theta)]])

    def _diff_matrix(self, theta):
        return np.array([[-np.sin(theta), 0, 0, 1j * np.cos(theta)],
                         [0, -np.sin(theta), -1j * np.cos(theta), 0],
                         [0, -1j * np.cos(theta), -np.sin(theta), 0],
                         [1j * np.cos(theta), 0, 0, -np.sin(theta)]])

    def define_projectq_gate(self):
        """Define the corresponded projectq gate."""
        self.projectq_gate = None


class ZZ(IntrinsicOneParaGate):
    r"""
    Ising ZZ  gate.

    .. math::

        {\rm ZZ_\theta}=\cos(\theta)I\otimes I-i\sin(\theta)\sigma_Z\otimes\sigma_Z

    More usage, please see :class:`mindquantum.gate.RX`.
    """
    def __init__(self, coeff=None):
        IntrinsicOneParaGate.__init__(self, 'ZZ', coeff)

    def _matrix(self, theta):
        return np.array([[np.exp(-1j * theta), 0, 0, 0],
                         [0, np.exp(1j * theta), 0, 0],
                         [0, 0, np.exp(1j * theta), 0],
                         [0, 0, 0, np.exp(-1j * theta)]])

    def _diff_matrix(self, theta):
        return -1j * np.array([[np.exp(-1j * theta), 0, 0, 0],
                               [0, -np.exp(1j * theta), 0, 0],
                               [0, 0, -np.exp(1j * theta), 0],
                               [0, 0, 0, np.exp(-1j * theta)]])

    def define_projectq_gate(self):
        """Define the corresponded projectq gate."""
        self.projectq_gate = None


class Power(NoneParameterGate):
    r"""
    Power operator on a non parameterized gate.

    Args:
        gates (:class:`mindquantum.gate.NoneParameterGate`): The basic gate you need to apply power operator.
        t (int, float): The exponenet. Default: 0.5.

    Examples:
        >>> from mindquantum import Power
        >>> import numpy as np
        >>> rx1 = RX(0.5)
        >>> rx2 = RX(1)
        >>> assert np.all(np.isclose(Power(rx2,0.5).matrix(), rx1.matrix()))
    """
    def __init__(self, gate: NoneParameterGate, t=0.5):
        NoneParameterGate.__init__(self,
                                   '{}^{}'.format(gate.name, round(t, 2)))
        self.matrix_value = fractional_matrix_power(gate.matrix(), t)

    def define_projectq_gate(self):
        """Define the corresponded projectq gate."""
        self.projectq_gate = None


I = IGate()
X = XGate()
Y = YGate()
Z = ZGate()
H = HGate()
S = PhaseShift(np.pi / 2)
SWAP = SWAPGate()
CNOT = CNOTGate()
