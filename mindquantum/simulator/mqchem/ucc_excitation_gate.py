# Copyright 2025 Huawei Technologies Co., Ltd
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
"""UCC Excitation Gate for MQ Chemistry simulator."""
# pylint: disable=abstract-method

from ...utils.type_value_check import (
    _check_input_type,
    _check_int_type,
    _check_value_should_not_less,
    _check_value_should_between_close_set,
)
from ...core.operators import FermionOperator
from ...core.gates.basic import ParameterGate


class UCCExcitationGate(ParameterGate):
    r"""
    Unitary Coupled-Cluster (UCC) excitation gate, for use with the
    :class:`~.simulator.mqchem.MQChemSimulator`.

    This gate represents the unitary operator :math:`e^{G}` where :math:`G` is an
    anti-Hermitian generator of the form :math:`G = \theta (T - T^\dagger)`.
    Here, :math:`T` is a fermionic excitation operator that conserves both spin
    and electron number. The operator is commonly used in the UCC ansatz for
    variational quantum algorithms in quantum chemistry.

    Note:
        This gate is specifically designed for the `MQChemSimulator` and relies
        on its internal CI-space representation. It is not compatible with the
        standard state-vector `Simulator`.

    The gate is defined as:

    .. math::

        U(\theta) = \exp(\theta(T - T^\dagger))

    where :math:`T` must be a single-term :class:`~.core.operators.FermionOperator`,
    for example, :math:`a_p^\dagger a_q`.

    Args:
        fermion_operator (FermionOperator): The fermionic excitation operator :math:`T`.
            It must contain exactly one term. The coefficient of this term is used as the
            rotation angle :math:`\theta`. If the coefficient is a variable, the gate
            will be parametric.

    Examples:
    >>> from mindquantum.simulator import mqchem
    >>> from mindquantum.core.operators import FermionOperator
        >>> # Non-parametric gate
        >>> t = FermionOperator('2^ 0', 0.5)
        >>> non_parametric_gate = mqchem.UCCExcitationGate(t)
        >>> non_parametric_gate
        exp{(1/2)([2^ 0] + [2 0^])}
        >>>
        >>> # Parametric gate
        >>> t_para = FermionOperator('3^ 1', 'theta')
        >>> parametric_gate = mqchem.UCCExcitationGate(t_para)
        >>> parametric_gate
        exp{(theta)([3^ 1] + [3 1^])}
    """

    def __init__(self, fermion_operator: FermionOperator):
        """Initialize a UCCExcitationGate."""
        _check_input_type("fermion_operator", FermionOperator, fermion_operator)
        if len(fermion_operator.terms) != 1:
            raise ValueError("The fermionic excitation operator T must have exactly one term.")
        raw_operator = fermion_operator.normal_ordered()
        raw_term, raw_coef = next(iter(raw_operator.terms.items()))
        qubits = [idx for idx, _ in raw_term]
        if len(qubits) != len(set(qubits)):
            raise ValueError("The fermionic excitation operator T must not contain duplicate indices in a term.")
        herm_operator = raw_operator.hermitian()
        self.fermion_operator = raw_operator - herm_operator

        herm_term, herm_coef = next(iter(herm_operator.terms.items()))

        obj_qubits = [qubit for qubit, _ in raw_term]

        # Separate coefficient and terms: pr * (T - T^dagger)
        pr = raw_coef
        sign = -1 if herm_coef == raw_coef else 1
        term_data = [(list(raw_term), 1), (list(herm_term), sign)]
        self.term_data = term_data

        term_str = "[" + ' '.join(f"{i}{'^' if j else ''}" for i, j in raw_term) + "]"
        herm_term_str = "[" + ' '.join(f"{i}{'^' if j else ''}" for i, j in herm_term) + "]"
        op_str = "exp{" + "(" + str(pr) + ")(" + term_str + f" {'-' if sign == -1 else '+'} " + herm_term_str + ")}"

        super().__init__(pr, op_str, len(obj_qubits), obj_qubits=obj_qubits)

    def __str_in_terminal__(self):
        return self.name

    def get_cpp_obj(self, n_qubits: int, n_electrons: int, backend):  # pylint: disable=arguments-differ
        """
        Get the underlying C++ CppExcitationOperator binding object.

        Note:
            This method is for internal use by the :class:`~.simulator.mqchem.MQChemSimulator`.

        Args:
            n_qubits (int): The total number of qubits (spin-orbitals) in the system.
            n_electrons (int): The total number of electrons in the system.
            backend: The C++ backend module (_mq_chem.float or _mq_chem.double).

        Returns:
            The C++ object used for simulation.
        """
        _check_int_type("n_qubits", n_qubits)
        _check_value_should_not_less("n_qubits", 1, n_qubits)
        _check_int_type("n_electrons", n_electrons)
        _check_value_should_not_less("n_electrons", 0, n_electrons)
        _check_value_should_between_close_set("n_electrons", 0, n_qubits, n_electrons)
        return backend.CppExcitationOperator(self.term_data, n_qubits, n_electrons, self.coeff)
