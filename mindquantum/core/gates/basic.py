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

# pylint: disable=abstract-method,duplicate-code
"""Basic module for quantum gate."""

import copy
from abc import ABC, abstractmethod
from typing import Iterable, List, Tuple

import numpy as np
import scipy

from mindquantum.utils.f import pauli_string_matrix
from mindquantum.utils.quantifiers import s_quantifier
from mindquantum.utils.string_utils import join_without_empty
from mindquantum.utils.type_value_check import (
    _check_gate_type,
    _check_input_type,
    _check_qubit_id,
)

from ..parameterresolver import ParameterResolver

HERMITIAN_PROPERTIES = {
    'self_hermitian': 0,  # the hermitian of this gate is its self
    'do_hermitian': 1,  # just do hermitian when you need hermitian
    'params_opposite': 2,  # use the negative parameters for hermitian
}


class BasicGate(ABC):
    """
    BasicGate is the base class of all gates.

    Args:
        name (str): the name of this gate.
        n_qubits (int): how many qubits is this gate.
        obj_qubits (int, list[int]): Specific which qubits the gate act on.
        ctrl_qubits (int, list[int]): Specific the control qubits. Default, ``None``.
    """

    def __init__(self, name, n_qubits, obj_qubits=None, ctrl_qubits=None):
        """Initialize a BasicGate."""
        super().__init__()
        if obj_qubits is None:
            obj_qubits = []
        if ctrl_qubits is None:
            ctrl_qubits = []
        if not isinstance(name, str):
            raise TypeError(f"Excepted string for gate name, get {type(name)}")
        _check_qubit_id(n_qubits)
        self.name = name
        self.n_qubits = n_qubits
        self.obj_qubits = obj_qubits
        self.ctrl_qubits = ctrl_qubits

    @property
    def acted(self):
        """Check whether this gate is acted on qubits."""
        if not self.obj_qubits:
            return False
        if self.n_qubits != len(self.obj_qubits):
            raise RuntimeError(
                f"{self.name} gate requires {s_quantifier(self.n_qubits, 'qubit')}, but get {len(self.obj_qubits)}"
            )
        return self.n_qubits == len(self.obj_qubits)

    def __decompose__(self):
        """
        Decompose this gate into more basic gate.

        Returns:
            None, if the decomposition is not defined.
            List[Circuit], all possible decompositions.
        """
        return

    def __commutate__(self, _):
        """Indicate whether a gate commutes."""
        return False

    def __qubits_expression__(self):
        """Qubit expression generator."""
        obj_s = ' '.join([str(i) for i in self.obj_qubits])
        ctrl_s = ' '.join([str(i) for i in self.ctrl_qubits])
        return join_without_empty(' <-: ', [obj_s, ctrl_s])

    def __str_in_svg__(self):
        """Return a string representation of the object."""
        return self.name

    def __str_in_circ__(self):
        """Return a string representation of the object."""
        return self.name

    def __str_in_terminal__(self):
        """Return a string representation of the object."""
        expression = self.__qubits_expression__()
        return self.name + (f"({expression})" if expression else '')

    def __type_specific_str__(self):
        """Return a string representation of the object."""
        return ''

    def __merge__(self, other: "BasicGate") -> Tuple[bool, List["BasicGate"], "GlobalPhase"]:
        """Merge with other gate."""
        return (False, [self, other], None)

    def matrix(self, *args):
        """Matrix of the gate."""
        raise NotImplementedError(f"Matrix for gate {self.name} not implement.")

    def hermitian(self):
        """Return the hermitian gate of this gate."""
        raise NotImplementedError(f"Hermitian for gate {self.name} not implement.")

    @property
    def parameterized(self):
        """Check whether this gate is a parameterized gate."""
        return isinstance(self, ParameterGate) and not self.coeff.is_const()  # pylint: disable=no-member

    def define_projectq_gate(self):
        """Define the corresponded projectq gate."""
        raise NotImplementedError(f"projectq version of gate {self.name} is not implement.")

    def no_grad(self):
        """Set this gate to not calculate gradient."""
        return self

    def requires_grad(self):
        """
        Set this gate to calculate gradient.

        In default, a parameterized gate will requires grad when you init it.
        """
        return self

    def on(self, obj_qubits, ctrl_qubits=None):  # pylint: disable=invalid-name
        """
        Define which qubit the gate act on and the control qubit.

        Note:
            In this framework, the qubit that the gate act on is specified
            first, even for control gate, e.g. CNOT, the second arg is control
            qubits.

        Args:
            obj_qubits (int, list[int]): Specific which qubits the gate act on.
            ctrl_qubits (int, list[int]): Specific the control qubits. Default, ``None``.

        Returns:
            Gate, Return a new gate.

        Examples:
            >>> from mindquantum.core.gates import X
            >>> x = X.on(1)
            >>> x.obj_qubits
            [1]
            >>> x.ctrl_qubits
            []
            >>> x = X.on(2, [0, 1])
            >>> x.ctrl_qubits
            [0, 1]
        """
        if isinstance(obj_qubits, int):
            obj_qubits = [obj_qubits]
        if isinstance(ctrl_qubits, int):
            ctrl_qubits = [ctrl_qubits]
        if ctrl_qubits is None:
            ctrl_qubits = []
        _check_input_type("obj_qubits", (int, Iterable), obj_qubits)
        _check_input_type("ctrl_qubits", (int, Iterable), ctrl_qubits)
        if set(obj_qubits) & set(ctrl_qubits):
            raise ValueError("Obj_qubit and ctrl_qubit cannot have same qubits.")
        if len(set(obj_qubits)) != len(obj_qubits):
            raise ValueError("obj_qubits cannot be repeated.")
        if len(set(ctrl_qubits)) != len(ctrl_qubits):
            raise ValueError("ctrl_qubits cannot be repeated.")
        if self.n_qubits:
            if len(obj_qubits) != self.n_qubits:
                raise ValueError(
                    f"{self.name} gate requires {s_quantifier(self.n_qubits, 'qubit')}, but get {len(obj_qubits)}"
                )
        new = copy.deepcopy(self)
        new.obj_qubits = []
        new.ctrl_qubits = []
        for i in obj_qubits:
            _check_qubit_id(i)
            new.obj_qubits.append(i)
        for i in ctrl_qubits:
            _check_qubit_id(i)
            new.ctrl_qubits.append(i)
        return new

    def __or__(self, qubits):
        """Support for ProjectQ-like syntax."""
        if not isinstance(qubits, tuple):
            qubits = (qubits,)

        qubits = list(qubits)

        for i, qubit in enumerate(qubits):
            if hasattr(qubit, "qubit_id"):
                qubits[i] = [qubit]
        ctrls = []
        objs = []
        if len(qubits) == 1:
            objs = [qubit.qubit_id for qubit in qubits[0]]
        else:
            ctrls = [qubit.qubit_id for qubit in qubits[0]]
            objs = [qubit.qubit_id for qubit in qubits[1]]
        qubits[0][0].circuit_.append(self.on(objs, ctrls))

    def __str__(self):
        """Return a string representation of the object."""
        return self.__str_in_terminal__()

    def __repr__(self):
        """Return a string representation of the object."""
        return self.__str__()

    def __eq__(self, other):
        """Equality comparison operator."""
        _check_gate_type(other)
        if self.__class__ is not other.__class__:
            return False
        if (
            self.name != other.name
            or self.n_qubits != other.n_qubits
            or self.obj_qubits != other.obj_qubits
            or set(self.ctrl_qubits) != set(other.ctrl_qubits)
        ):
            return False
        return True

    @abstractmethod
    def get_cpp_obj(self):
        """Get the underlying C++ object."""


class FunctionalGate(BasicGate):
    """Base class for functional gates."""

    def hermitian(self):
        """Return the hermitian of this parameter gate."""
        return copy.deepcopy(self)


class QuantumGate(BasicGate):
    """
    Base class for quantum gates.

    Args:
        name (str): the name of this gate.
        n_qubits (int): how many qubits is this gate.
        obj_qubits (int, list[int]): Specific which qubits the gate act on.
        ctrl_qubits (int, list[int]): Specific the control qubits. Default, ``None``.
    """

    def __commutate__(self, other: BasicGate):
        """Indicate whether a gate commutes."""
        # pylint: disable=import-outside-toplevel,cyclic-import
        from mindquantum.core.circuit import UN, Circuit

        from .basicgate import I

        if not isinstance(other, BasicGate):
            return False
        total_circ = Circuit([self, other]).compress()
        circ_1 = total_circ[:1]
        circ_2 = total_circ[1:]
        qubit_set_2 = set(circ_2.all_qubits.keys())
        qubit_set_1 = set(circ_1.all_qubits.keys())
        circ_1 += UN(I, qubit_set_2 - qubit_set_1)
        circ_2 += UN(I, qubit_set_1 - qubit_set_2)
        matrix_1 = circ_1.matrix()
        matrix_2 = circ_2.matrix()
        return np.allclose(matrix_1 @ matrix_2, matrix_2 @ matrix_1)


class SelfHermitianGate(QuantumGate):
    """Base class for self-hermitian gates."""

    def hermitian(self):
        """Return the hermitian of this parameter gate."""
        return copy.deepcopy(self)


class AntiHermitianGate(QuantumGate):
    """Base class for anti-hermitian gates."""


class NonHermitianGate(QuantumGate):
    """
    The basic class of gate that is non hermitian.

    Args:
        name (str): the name of this gate.
        n_qubits (int): how many qubits is this gate.
        obj_qubits (int, list[int]): Specific which qubits the gate act on.
        ctrl_qubits (int, list[int]): Specific the control qubits. Default, ``None``.
    """

    def __init__(self, name, n_qubits, *args, obj_qubits=None, ctrl_qubits=None, hermitianed=False, **kwargs):
        """Initialize a NonHermitianGate object."""
        super().__init__(name, n_qubits, *args, obj_qubits=obj_qubits, ctrl_qubits=ctrl_qubits, **kwargs)
        self.hermitianed = hermitianed

    def hermitian(self):
        """Return the hermitian of this parameter gate."""
        new = copy.deepcopy(self)
        new.hermitianed = not new.hermitianed
        return new

    def __type_specific_str__(self):
        """Return a string representation of the object."""
        # pylint: disable=import-outside-toplevel,cyclic-import
        from mindquantum.io.display._config import _DAGGER_MASK

        return _DAGGER_MASK if self.hermitianed else ''

    def __str_in_terminal__(self):
        """Return a string representation of the object."""
        string = super().__str_in_terminal__()
        return f"{self.name}{self.__type_specific_str__()}{string[len(self.name):]}"

    def __str_in_circ__(self):
        """Return a string representation of the object."""
        string = super().__str_in_circ__()
        return f"{self.name}{self.__type_specific_str__()}{string[len(self.name):]}"

    def __str_in_svg__(self):
        """Return a string representation of the object."""
        # pylint: disable=import-outside-toplevel,cyclic-import
        from mindquantum.io.display._config import _DAGGER_MASK

        string = super().__str_in_svg__()
        if self.hermitianed:
            string += _DAGGER_MASK
        return string

    def __eq__(self, other):
        """Equality comparison operator."""
        return super().__eq__(other) and self.hermitianed == other.hermitianed


class MatrixGate(QuantumGate):
    """Gate that has matrix defined."""

    def __init__(self, matrix_value, name, n_qubits, *args, obj_qubits=None, ctrl_qubits=None, **kwargs):
        """Initialize a MatrixGate object."""
        super().__init__(name, n_qubits, *args, obj_qubits=obj_qubits, ctrl_qubits=ctrl_qubits, **kwargs)
        self.matrix_value = matrix_value

    def matrix(self, full=False):  # pylint: disable=arguments-differ
        """
        Matrix of parameterized gate.

        Args:
            full (bool): Whether to get the full matrix of this gate (the gate
                should be acted on some qubits). Default: ``False``.
        """
        _check_input_type('full', bool, full)
        if full:
            # pylint: disable=import-outside-toplevel
            from mindquantum.core.circuit import Circuit

            return Circuit([self]).matrix()
        return self.matrix_value

    def __eq__(self, other):
        """Equality comparison operator."""
        return super().__eq__(other) and np.allclose(self.matrix(), other.matrix())


class NoneParameterGate(QuantumGate):
    """
    Base class for non-parametric gates.

    Args:
        name (str): the name of this gate.
        n_qubits (int): how many qubits is this gate.
        obj_qubits (int, list[int]): Specific which qubits the gate act on.
        ctrl_qubits (int, list[int]): Specific the control qubits. Default, ``None``.
    """

    def __call__(self, obj_qubits, ctrl_qubits=None):
        """Definition of a function call operator."""
        return self.on(obj_qubits, ctrl_qubits)


class ParameterGate(QuantumGate):
    """
    Gate that is parameterized.

    Args:
        pr (ParameterResolver): the parameter for parameterized gate.
        name (str): the name of this parameterized gate.
        n_qubits (int): the qubit number of this parameterized gate.
        args (list): other arguments for quantum gate.
        obj_qubits (Union[int, List[int]]): the qubit that this gate act on. Default: ``None``.
        ctrl_qubits (Union[int, List[int]]): the control qubit of this gate. Default: ``None``.
        kwargs (dict): other arguments for quantum gate.
    """

    def __init__(self, pr: ParameterResolver, name, n_qubits, *args, obj_qubits=None, ctrl_qubits=None, **kwargs):
        """Initialize a ParameterGate object."""
        super().__init__(name, n_qubits, *args, obj_qubits=obj_qubits, ctrl_qubits=ctrl_qubits, **kwargs)
        self.coeff = pr

    def __type_specific_str__(self):
        """Return a string representation of the coefficients."""
        return self.coeff.expression()

    def __str_in_terminal__(self):
        """Return a string representation of the object."""
        qubit_s = QuantumGate.__qubits_expression__(self)
        pr_s = self.__type_specific_str__()
        string = join_without_empty('|', [pr_s, qubit_s])
        return self.name + (f'({string})' if string else '')

    def __str_in_circ__(self):
        """Return a string representation of the object."""
        pr_s = self.__type_specific_str__()
        string = join_without_empty('|', [pr_s])
        return self.name + (f'({string})' if string else '')

    def __call__(self, pr):
        """Definition of a function call operator."""
        new = copy.deepcopy(self)
        new.coeff = ParameterResolver(pr)
        return new

    def __eq__(self, other):
        """Equality comparison operator."""
        return super().__eq__(other) and self.coeff == other.coeff

    def __params_prop__(self):
        """Get properties of all parameters."""
        return list(self.coeff.keys()), list(self.coeff.ansatz_parameters), list(self.coeff.encoder_parameters)

    def get_parameters(self) -> List[ParameterResolver]:
        """Return a list of parameters of parameterized gate."""
        return [self.coeff]

    def no_grad(self):
        """Mark all parameters as *not* requiring gradient calculations."""
        self.coeff.no_grad()
        return self

    def requires_grad(self):
        """Mark all parameters as requiring gradient calculations."""
        self.coeff.requires_grad()
        return self

    def requires_grad_part(self, *names):
        """
        Set certain parameters that need grad. Inplace operation.

        Args:
            names (tuple[str]): Parameters that requires grad.

        Returns:
            BasicGate, with some part of parameters need to update gradient.
        """
        self.coeff.requires_grad_part(*names)
        return self

    def no_grad_part(self, *names):
        """
        Set certain parameters that do not need grad. Inplace operation.

        Args:
            names (tuple[str]): Parameters that not requires grad.

        Returns:
            BasicGate, with some part of parameters not need to update gradient.
        """
        self.coeff.no_grad_part(*names)
        return self


class ParameterOppsGate(ParameterGate):
    """ParameterOppsGate class."""

    def __merge__(self, other: BasicGate) -> Tuple[bool, List[BasicGate], "GlobalPhase"]:
        """Merge two gate."""
        if not isinstance(other, self.__class__):
            return (False, [], None)
        if self.obj_qubits != other.obj_qubits or set(self.ctrl_qubits) != set(other.ctrl_qubits):
            return (False, [], None)
        new_coeff = self.coeff + other.coeff
        if new_coeff == 0.0:
            return (True, [], None)
        return (True, [self(new_coeff)], None)

    def hermitian(self):
        """Return the hermitian of this parameter gate."""
        new = copy.deepcopy(self)
        new.coeff = -new.coeff
        return new


class NoneParamNonHermMat(NoneParameterGate, MatrixGate, NonHermitianGate):
    """Gate that is both none parameterized and non hermitian."""

    # pylint: disable=too-many-arguments
    def __init__(self, matrix_value, name, n_qubits, obj_qubits=None, ctrl_qubits=None, hermitianed=False):
        """Initialize a NoneParamNonHermMat object."""
        super().__init__(
            matrix_value, name, n_qubits, obj_qubits=obj_qubits, ctrl_qubits=ctrl_qubits, hermitianed=hermitianed
        )

    def __eq__(self, other):
        """Equality comparison operator."""
        return NonHermitianGate.__eq__(self, other)

    def __merge__(self, other: BasicGate) -> Tuple[bool, List[BasicGate], "GlobalPhase"]:
        """Merge two gate."""
        if not isinstance(other, self.__class__):
            return (False, [self, other], None)
        if self.obj_qubits != other.obj_qubits or set(self.ctrl_qubits) != set(other.ctrl_qubits):
            return (False, [self, other], None)
        if self.hermitianed ^ other.hermitianed:
            return (True, [], None)
        return (False, [self, other], None)

    def matrix(self, full=False):
        """
        Matrix of parameterized gate.

        Args:
            full (bool): Whether to get the full matrix of this gate (the gate
                should be acted on some qubits). Default: ``False``.
        """
        _check_input_type('full', bool, full)
        if full:
            # pylint: disable=import-outside-toplevel
            from mindquantum.core.circuit import Circuit

            return Circuit([self]).matrix()

        if self.hermitianed:
            return np.conj(self.matrix_value.T)
        return self.matrix_value


class NoneParamSelfHermMat(NoneParameterGate, SelfHermitianGate, MatrixGate):
    """Non-parametric self hermitian matrix gate."""

    def __eq__(self, other):
        """Equality comparison operator."""
        return MatrixGate.__eq__(self, other)

    def __merge__(self, other: BasicGate) -> Tuple[bool, List[BasicGate], "GlobalPhase"]:
        """Merge with other gate."""
        if isinstance(other, self.__class__):
            if self.obj_qubits == other.obj_qubits and set(self.ctrl_qubits) == set(other.ctrl_qubits):
                return (True, [], None)
        return (False, [self, other], None)


class PauliGate(NoneParamSelfHermMat):
    """Pauli Gate."""

    def __pow__(self, coeff):
        """Calculate the power of a Pauli gate."""
        from .basicgate import (  # pylint: disable=import-outside-toplevel,cyclic-import
            RX,
            RY,
            RZ,
        )

        gate_map = {'X': RX, 'Y': RY, 'Z': RZ}
        if self.name not in gate_map:
            raise NotImplementedError(f"Power of gate {self.name} not implement yet.")
        r_gate = gate_map[self.name]
        pr = ParameterResolver(coeff) * np.pi
        return r_gate(pr)

    def __eq__(self, other):
        """Equality comparison operator."""
        return QuantumGate.__eq__(self, other)


class PauliStringGate(NoneParamSelfHermMat):
    """Gate construct by pauli string."""

    def __init__(self, pauli_string):
        """Initialize a PauliStringGate object."""
        name = ''.join([i.name for i in pauli_string])
        n_qubits = len(pauli_string)
        matrix_value = pauli_string_matrix(name)
        super().__init__(name=name, n_qubits=n_qubits, matrix_value=matrix_value)
        self.pauli_string = pauli_string

    def __eq__(self, other):
        """Equality comparison operator."""
        return QuantumGate.__eq__(self, other)

    def get_cpp_obj(self):
        """Get cpp obj."""
        raise NotImplementedError


class RotSelfHermMat(ParameterOppsGate):
    """Exponential of a self hermitian operator gate."""

    def __init__(
        self, core, name, n_qubits, obj_qubits=None, ctrl_qubits=None, pr=ParameterResolver()
    ):  # pylint: disable=too-many-arguments
        """Initialize a RotSelfHermMat object."""
        super().__init__(pr, name, n_qubits, obj_qubits=obj_qubits, ctrl_qubits=ctrl_qubits)
        self.core = core

    def __merge__(self, other: BasicGate) -> Tuple[bool, List[BasicGate], "GlobalPhase"]:
        """Merge with other gate."""
        if not isinstance(other, self.__class__):
            return (False, [self, other], None)
        if self.obj_qubits != other.obj_qubits or set(self.ctrl_qubits) != set(other.ctrl_qubits):
            return (False, [self, other], None)
        new_coeff = self.coeff + other.coeff
        if new_coeff.is_const() and np.allclose(new_coeff.const % (4 * np.pi), 0.0):
            return (True, [], None)
        return (True, [self(new_coeff)], None)

    def matrix(self, pr=None, frac=0.5, full=False):  # pylint: disable=arguments-differ
        """
        Matrix of parameterized gate.

        Args:
            pr (Union[ParameterResolver, dict]): The parameter of this gate. Default: None.
            frac (numbers.Number): The multiple of the coefficient. Default: ``0.5``.
            full (bool): Whether to get the full matrix of this gate (the gate
                should be acted on some qubits). Default: ``False``.
        """
        _check_input_type('full', bool, full)
        if full:
            # pylint: disable=import-outside-toplevel
            from mindquantum.core.circuit import Circuit

            return Circuit([self]).matrix(pr=pr)

        val = 0
        if not self.parameterized:
            val = self.coeff.const
        else:
            if pr is None:
                raise ValueError("Parameterized gate need a parameter resolver to get matrix.")
            new = self.coeff.combination(pr)
            if not new.is_const():
                raise ValueError("Parameter not set completed.")
            val = new.const
        return scipy.linalg.expm(-1j * val * frac * self.core.matrix())

    def diff_matrix(self, pr=None, about_what=None, frac=0.5):
        """Differential form of this parameterized gate."""
        if not self.parameterized:
            return np.zeros_like(self.core.matrix())
        if about_what is None:
            if len(self.coeff) != 1:
                raise ValueError("Should specific which parameter are going to do derivation.")
            for i in self.coeff:
                about_what = i
        return -1j * frac * self.core.matrix() @ self.matrix(pr=pr, frac=frac) * self.coeff[about_what]


class ParamNonHerm(ParameterGate, NonHermitianGate):
    """Gate that is parameterized and non hermitian."""

    def __init__(
        self, pr, matrix_generator, diff_matrix_generator, name, n_qubits, obj_qubits=None, ctrl_qubits=None
    ):  # pylint: disable=too-many-arguments
        """Initialize a ParamNonHerm object."""
        super().__init__(pr, name, n_qubits, obj_qubits=obj_qubits, ctrl_qubits=ctrl_qubits)
        self.matrix_generator = matrix_generator
        self.diff_matrix_generator = diff_matrix_generator

    def __str_in_circ__(self):
        """Return a string representation of the object."""
        string = ParameterGate.__str_in_circ__(self)
        return self.name + NonHermitianGate.__type_specific_str__(self) + string[len(self.name) :]  # noqa: E203

    def __str_in_svg__(self):
        """Return a string representation of the object."""
        string = ParameterGate.__str_in_svg__(self)
        return self.name + NonHermitianGate.__type_specific_str__(self) + string[len(self.name) :]  # noqa: E203

    def __str_in_terminal__(self):
        """Return a string representation of the object."""
        string = ParameterGate.__str_in_terminal__(self)
        return self.name + NonHermitianGate.__type_specific_str__(self) + string[len(self.name) :]  # noqa: E203

    def matrix(self, pr=None, full=False):  # pylint: disable=arguments-differ
        """
        Matrix of parameterized gate.

        Note:
            If the parameterized gate convert to non parameterized gate, then
            you don't need any parameters to get this matrix.

        Args:
            pr (Union[dict, ParameterResolver]): Parameters of this gate.
            full (bool): Whether to get the full matrix of this gate (the gate
                should be acted on some qubits). Default: ``False``.

        Returns:
            numpy.ndarray, Return the numpy array of the matrix.

        Examples:
            >>> from mindquantum.core.gates import RX
            >>> rx1 = RX(0)
            >>> rx1.matrix()
            array([[1.+0.j, 0.-0.j],
                   [0.-0.j, 1.+0.j]])
            >>> rx2 = RX({'a' : 1.2})
            >>> np.round(rx2.matrix({'a': 2}), 2)
            array([[0.36+0.j  , 0.  -0.93j],
                   [0.  -0.93j, 0.36+0.j  ]])
        """
        _check_input_type('full', bool, full)
        if full:
            # pylint: disable=import-outside-toplevel
            from mindquantum.core.circuit import Circuit

            return Circuit([self]).matrix(pr=pr)

        val = 0
        if not self.parameterized:
            val = self.coeff.const
        else:
            new = self.coeff.combination(pr)
            if not new.is_const():
                raise ValueError("Parameter not set completed.")
            val = new.const
        return self.matrix_generator(val)

    def diff_matrix(self, pr=None, about_what=None):
        """
        Differential form of this parameterized gate.

        Args:
            paras_out (Union[dict, ParameterResolver]): Parameters of this gate.
            about_what (str): Specific the differential is about
                which parameter. Default: ``None``.

        Returns:
            numpy.ndarray, Return the numpy array of the differential matrix.

        Examples:
            >>> from mindquantum import RX
            >>> rx = RX('a')
            >>> np.round(rx.diff_matrix({'a' : 2}), 2)
            array([[-0.42+0.j  ,  0.  -0.27j],
                   [ 0.  -0.27j, -0.42+0.j  ]])
        """
        if not self.parameterized:
            return np.zeros_like(self.matrix_generator(0))
        new = self.coeff.combination(pr)
        if not new.is_const():
            raise ValueError("Parameter not set completed.")
        val = new.const
        if about_what is None:
            if len(self.coeff) != 1:
                raise ValueError("Should specific which parameter are going to do derivation.")
            for i in self.coeff:
                about_what = i
        return self.coeff[about_what] * self.diff_matrix_generator(val)


class NoiseGate(NoneParameterGate):
    """
    Noise gate class.

    Args:
        name (str): the name of this gate.
        n_qubits (int): how many qubits is this gate.
        obj_qubits (int, list[int]): Specific which qubits the gate act on.
        ctrl_qubits (int, list[int]): Specific the control qubits. Default, ``None``.
    """

    def __str_in_terminal__(self):
        """Return a string representation of the object."""
        qubit_s = QuantumGate.__qubits_expression__(self)
        pr_s = self.__type_specific_str__()
        string = join_without_empty('|', [pr_s, qubit_s])
        return self.name + (f'({string})' if string else '')

    def __str_in_circ__(self):
        """Return a string representation of the object."""
        pr_s = self.__type_specific_str__()
        string = join_without_empty('|', [pr_s])
        return self.name + (f'({string})' if string else '')

    def on(self, obj_qubits, ctrl_qubits=None):
        """
        Define which qubit the gate act on.

        Args:
            obj_qubits (int, list[int]): Specific which qubits the gate act on.
            ctrl_qubits (int, list[int]): Control qubit for noise gate should always be ``None``.
        """
        out = super().on(obj_qubits, ctrl_qubits)
        if out.ctrl_qubits:
            raise ValueError("Noise gate cannot have control qubits.")
        return out
