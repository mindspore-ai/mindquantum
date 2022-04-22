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
"""Basic module for quantum gate."""

import numpy as np
from scipy.linalg import fractional_matrix_power
from mindquantum.core.parameterresolver import ParameterResolver as PR
from mindquantum.utils.f import is_power_of_two
from mindquantum.config.config import _GLOBAL_MAT_VALUE
from mindquantum.core.gates.basic import NoneParamNonHermMat
from mindquantum.core.gates.basic import NoneParamSelfHermMat
from mindquantum.core.gates.basic import PauliGate
from mindquantum.core.gates.basic import RotSelfHermMat
from mindquantum.core.gates.basic import FunctionalGate
from mindquantum.core.gates.basic import ParameterOppsGate
from mindquantum.core.gates.basic import ParamNonHerm
from mindquantum.core.gates.basic import PauliStringGate
from mindquantum import mqbackend as mb


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
        if len(matrix_value.shape) != 2:
            raise ValueError(f"matrix_value require shape of 2, but get shape of {matrix_value.shape}")
        if matrix_value.shape[0] != matrix_value.shape[1]:
            raise ValueError(f"matrix_value need a square matrix, but get shape {matrix_value.shape}")
        if not is_power_of_two(matrix_value.shape[0]):
            raise ValueError(f"Dimension of matrix_value need should be power of 2, but get {matrix_value.shape[0]}")
        n_qubits = int(np.log2(matrix_value.shape[0]))
        super().__init__(name=name, n_qubits=n_qubits, matrix_value=matrix_value)

    def get_cpp_obj(self):
        mat = mb.dim2matrix(self.matrix())
        cpp_gate = mb.basic_gate(False, self.name, 1, mat)
        cpp_gate.daggered = self.hermitianed
        cpp_gate.obj_qubits = self.obj_qubits
        cpp_gate.ctrl_qubits = self.ctrl_qubits
        return cpp_gate


class HGate(NoneParamSelfHermMat):
    r"""
    Hadamard gate with matrix as:

    .. math::

        {\rm H}=\frac{1}{\sqrt{2}}\begin{pmatrix}1&1\\1&-1\end{pmatrix}

    More usage, please see :class:`mindquantum.core.gates.XGate`.
    """
    def __init__(self):
        super().__init__(
            name='H',
            n_qubits=1,
            matrix_value=_GLOBAL_MAT_VALUE['H'],
        )


class XGate(PauliGate):
    r"""
    Pauli X gate with matrix as:

    .. math::

        {\rm X}=\begin{pmatrix}0&1\\1&0\end{pmatrix}

    For simplicity, we define ```X``` as a instance of ```XGate()```. For more
    redefine, please refer the functional table below.

    Note:
        For simplicity, you can do power operator on pauli gate (only works
        for pauli gate at this time). The rules is set below as:

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
        super().__init__(
            name='X',
            n_qubits=1,
            matrix_value=_GLOBAL_MAT_VALUE['X'],
        )


class YGate(PauliGate):
    r"""
    Pauli Y gate with matrix as:

    .. math::

        {\rm Y}=\begin{pmatrix}0&-i\\i&0\end{pmatrix}

    More usage, please see :class:`mindquantum.core.gates.XGate`.
    """
    def __init__(self):
        super().__init__(
            name='Y',
            n_qubits=1,
            matrix_value=_GLOBAL_MAT_VALUE['Y'],
        )


class ZGate(PauliGate):
    r"""
    Pauli Z gate with matrix as:

    .. math::

        {\rm Z}=\begin{pmatrix}1&0\\0&-1\end{pmatrix}

    More usage, please see :class:`mindquantum.core.gates.XGate`.
    """
    def __init__(self):
        super().__init__(
            name='Z',
            n_qubits=1,
            matrix_value=_GLOBAL_MAT_VALUE['Z'],
        )


class IGate(PauliGate):
    r"""
    Identity gate with matrix as:

    .. math::

        {\rm I}=\begin{pmatrix}1&0\\0&1\end{pmatrix}

    More usage, please see :class:`mindquantum.core.gates.XGate`.
    """
    def __init__(self):
        super().__init__(
            name='I',
            n_qubits=1,
            matrix_value=_GLOBAL_MAT_VALUE['I'],
        )


class CNOTGate(NoneParamSelfHermMat):
    r"""
    Control-X gate.

    More usage, please see :class:`mindquantum.core.gates.XGate`.
    """
    def __init__(self):
        super().__init__(
            name='CNOT',
            n_qubits=2,
            matrix_value=_GLOBAL_MAT_VALUE['CNOT'],
        )

    def on(self, obj_qubits, ctrl_qubits=None):
        if ctrl_qubits is None:
            raise ValueError("A control qubit is needed for CNOT gate!")
        if isinstance(ctrl_qubits, (int, np.int64)):
            ctrl_qubits = [ctrl_qubits]
        elif not isinstance(ctrl_qubits, list) or not ctrl_qubits:
            raise ValueError(f"ctrl_qubits requires a list, but get {type(ctrl_qubits)}")
        out = super().on([obj_qubits, ctrl_qubits[0]], ctrl_qubits[1:])
        return out


class SWAPGate(NoneParamSelfHermMat):
    """
    SWAP gate that swap two different qubits.

    More usage, please see :class:`mindquantum.core.gates.XGate`.
    """
    def __init__(self):
        super().__init__(
            name='SWAP',
            n_qubits=2,
            matrix_value=_GLOBAL_MAT_VALUE['SWAP'],
        )


class ISWAPGate(NoneParamNonHermMat):
    r"""
    ISWAP gate that swap two different qubits and phase the
    :math:`\left|01\right>` and :math:`\left|10\right>` amplitudes by
    :math:`i`.

    More usage, please see :class:`mindquantum.core.gates.XGate`.
    """
    def __init__(self):
        super().__init__(
            name='ISWAP',
            n_qubits=2,
            matrix_value=_GLOBAL_MAT_VALUE['ISWAP'],
        )


class TGate(NoneParamNonHermMat):
    r"""
    T gate with matrix as :

    .. math::
        {\rm T}=\begin{pmatrix}1&0\\0&(1+i)/\sqrt(2)\end{pmatrix}

    More usage, please see :class:`mindquantum.core.gates.XGate`.
    """
    def __init__(self):
        super().__init__(
            name='T',
            n_qubits=1,
            matrix_value=_GLOBAL_MAT_VALUE['T'],
        )


class SGate(NoneParamNonHermMat):
    r"""
    S gate with matrix as :

    .. math::
        {\rm S}=\begin{pmatrix}1&0\\0&i\end{pmatrix}

    More usage, please see :class:`mindquantum.core.gates.XGate`.
    """
    def __init__(self):
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
    coefficience is one.

    3. If you initialize it with a dict, e.g. `{'a':1,'b':2}`, this gate
    can have multiple parameters with certain coefficiences. In this case,
    it can be expressed as:


    .. math::

        RX(a+2b)

    Args:
        coeff (Union[int, float, str, dict, ParameterResolver]): the parameters of
            parameterized gate, see above for detail explanation. Default: None.

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
        super().__init__(
            pr=PR(pr),
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
        coeff (Union[int, float, str, dict, ParameterResolver]): the parameters of
            parameterized gate, see above for detail explanation. Default: None.
    """
    def __init__(self, pr):
        super().__init__(
            pr=PR(pr),
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
        coeff (Union[int, float, str, dict, ParameterResolver]): the parameters of
            parameterized gate, see above for detail explanation. Default: None.
    """
    def __init__(self, pr):
        super().__init__(
            pr=PR(pr),
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
        coeff (Union[int, float, str, dict, ParameterResolver]): the parameters of
            parameterized gate, see above for detail explanation. Default: None.
    """
    def __init__(self, pr):
        super().__init__(
            pr=PR(pr),
            name='ZZ',
            n_qubits=2,
            core=PauliStringGate([Z, Z]),
        )

    def matrix(self, pr=None, frac=1):
        return super().matrix(pr, frac)

    def diff_matrix(self, pr=None, about_what=None, frac=1):
        return super().diff_matrix(pr, about_what, frac)


class XX(RotSelfHermMat):
    r"""
    Ising XX gate. More usage, please see :class:`mindquantum.core.gates.RX`.

    .. math::

        {\rm XX_\theta}=\cos(\theta)I\otimes I-i\sin(\theta)\sigma_x\otimes\sigma_x

    Args:
        coeff (Union[int, float, str, dict, ParameterResolver]): the parameters of
            parameterized gate, see above for detail explanation. Default: None.
    """
    def __init__(self, pr):
        super().__init__(
            pr=PR(pr),
            name='XX',
            n_qubits=2,
            core=PauliStringGate([X, X]),
        )

    def matrix(self, pr=None, frac=1):
        return super().matrix(pr, frac)

    def diff_matrix(self, pr=None, about_what=None, frac=1):
        return super().diff_matrix(pr, about_what, frac)


class YY(RotSelfHermMat):
    r"""
    Ising YY  gate. More usage, please see :class:`mindquantum.core.gates.RX`.

    .. math::

        {\rm YY_\theta}=\cos(\theta)I\otimes I-i\sin(\theta)\sigma_y\otimes\sigma_y

    Args:
        coeff (Union[int, float, str, dict, ParameterResolver]): the parameters of
            parameterized gate, see above for detail explanation. Default: None.
    """
    def __init__(self, pr):
        super().__init__(
            pr=PR(pr),
            name='YY',
            n_qubits=2,
            core=PauliStringGate([Y, Y]),
        )

    def matrix(self, pr=None, frac=1):
        return super().matrix(pr, frac)

    def diff_matrix(self, pr=None, about_what=None, frac=1):
        return super().diff_matrix(pr, about_what, frac)


class BarrierGate(FunctionalGate):
    def __init__(self, show=True):
        super().__init__(name='BARRIER', n_qubits=0)
        self.show = show

    def on(self, obj_qubits, ctrl_qubits=None):
        raise RuntimeError("Cannot call on for BarrierGate.")


class GlobalPhase(RotSelfHermMat):
    r"""
    Global phase gate. More usage, please see :class:`mindquantum.core.gates.RX`.

    .. math::

        {\rm GlobalPhase}=\begin{pmatrix}\exp(-i\theta)&0\\
                        0&\exp(-i\theta)\end{pmatrix}

    Args:
        coeff (Union[int, float, str, dict, ParameterResolver]): the parameters of
            parameterized gate, see above for detail explanation. Default: None.
    """
    def __init__(self, pr):
        super().__init__(
            pr=PR(pr),
            name='GP',
            n_qubits=1,
            core=IGate(),
        )

    def matrix(self, pr=None, **kwargs):
        return RotSelfHermMat.matrix(self, pr, 1)

    def diff_matrix(self, pr=None, about_what=None, **kwargs):
        return RotSelfHermMat.diff_matrix(self, pr, about_what, 1)


BARRIER = BarrierGate(show=False)


class PhaseShift(ParameterOppsGate):
    r"""
    Phase shift gate. More usage, please see :class:`mindquantum.core.gates.RX`.

    .. math::

        {\rm PhaseShift}=\begin{pmatrix}1&0\\
                         0&\exp(i\theta)\end{pmatrix}

    Args:
        coeff (Union[int, float, str, dict, ParameterResolver]): the parameters of
            parameterized gate, see above for detail explanation. Default: None.
    """
    def __init__(self, pr):
        super().__init__(
            pr=PR(pr),
            name='PS',
            n_qubits=1,
        )

    def matrix(self, pr=None):
        """
        Get the matrix of this none parameterized gate.
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
        Get the  matrix of this none parameterized gate.
        """
        if self.coeff.is_const():
            return np.zeros((2, 2))
        new_pr = self.coeff.combination(pr)
        if not new_pr.is_const():
            raise ValueError("The parameter is not set completed.")
        val = new_pr.const
        if about_what is None:
            if len(self.coeff) != 1:
                raise ValueError(f"Should specific which parameter are going to do derivation.")
            for i in self.coeff:
                about_what = i
        return np.array([[0, 0], [0, 1j * self.coeff[about_what] * np.exp(1j * val)]])


class Power(NoneParamNonHermMat):
    r"""
    Power operator on a non parameterized gate.

    Args:
        gates (:class:`mindquantum.core.gates.NoneParameterGate`): The basic gate you need to apply power operator.
        t (int, float): The exponenet. Default: 0.5.

    Examples:
        >>> from mindquantum import Power
        >>> import numpy as np
        >>> rx1 = RX(0.5)
        >>> rx2 = RX(1)
        >>> assert np.all(np.isclose(Power(rx2,0.5).matrix(), rx1.matrix()))
    """
    def __init__(self, gate, t=0.5):
        name = f'{gate}^{t}'
        n_qubits = gate.n_qubits
        matrix_value = fractional_matrix_power(gate.matrix(), t)
        super().__init__(
            name=name,
            n_qubits=n_qubits,
            matrix_value=matrix_value,
        )
        self.gate = gate
        self.t = t

    def get_cpp_obj(self):
        mat = mb.dim2matrix(self.matrix())
        cpp_gate = mb.basic_gate(False, self.name, 1, mat)
        cpp_gate.daggered = self.hermitianed
        cpp_gate.obj_qubits = self.obj_qubits
        cpp_gate.ctrl_qubits = self.ctrl_qubits
        return cpp_gate


def gene_univ_parameterized_gate(name, matrix_generator, diff_matrix_generator):
    """
    Generate a customer parameterized gate based on the single parameter defined
    unitary matrix.

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
        >>> from mindquantum import gene_univ_parameterized_gate
        >>> from mindquantum import Simulator, Circuit
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
    m = matrix_generator(0)
    n_qubits = int(np.log2(m.shape[0]))

    class _ParamNonHerm(ParamNonHerm):
        """The customer parameterized gate."""
        def __init__(self, pr):
            super().__init__(pr=PR(pr),
                             name=name,
                             n_qubits=n_qubits,
                             matrix_generator=matrix_generator,
                             diff_matrix_generator=diff_matrix_generator)

        def get_cpp_obj(self):
            if not self.hermitianed:
                cpp_gate = mb.basic_gate(self.name, 1, self.matrix_generator, self.diff_matrix_generator)
            else:
                cpp_gate = mb.basic_gate(
                    self.name,
                    1,
                    lambda x: np.conj(self.matrix_generator(x).T),
                    lambda x: np.conj(self.diff_matrix_generator(x).T),
                )
            cpp_gate.daggered = self.hermitianed
            cpp_gate.obj_qubits = self.obj_qubits
            cpp_gate.ctrl_qubits = self.ctrl_qubits
            if not self.parameterized:
                cpp_gate.apply_value(self.coeff.const)
            else:
                cpp_gate.params = self.coeff.get_cpp_obj()
            return cpp_gate

    return _ParamNonHerm


X = XGate()
Y = YGate()
Z = ZGate()
I = IGate()
H = HGate()
T = TGate()
S = SGate()
CNOT = CNOTGate()
ISWAP = ISWAPGate()
SWAP = SWAPGate()
