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
"""Sequential UCC ansatz for MQ Chemistry simulator."""

from ...core.circuit import Circuit
from ...core.operators import FermionOperator
from ...utils.type_value_check import _check_int_type, _check_value_should_not_less
from .ucc_excitation_gate import UCCExcitationGate


class SequentialUCCAnsatz:
    """
    Sequential Unitary Coupled-Cluster (UCC) Ansatz.

    This class allows constructing a UCC ansatz by adding excitation operators sequentially.
    Unlike ``FermionOperator`` which merges terms with the same index, this class preserves
    the order and distinctness of each added term, enabling the construction of
    multi-layer or Trotterized ansatzes where the same excitation operator may appear
    multiple times with different parameters.

    Note:
        The generated circuit uses :class:`~.simulator.mqchem.UCCExcitationGate`,
        which is intended for :class:`~.simulator.mqchem.MQChemSimulator`.

    Args:
        n_qubits (int): The number of qubits (spin-orbitals) in the simulation. Default: ``None``.
        n_electrons (int): The number of electrons of the given molecule. Default: ``None``.

    Examples:
        >>> from mindquantum.simulator import mqchem
        >>> from mindquantum.core.operators import FermionOperator
        >>> ansatz = mqchem.SequentialUCCAnsatz()
        >>> ansatz.append(FermionOperator("3^ 1", "a"))
        >>> ansatz.append(FermionOperator("4^ 2", "b"))
        >>> ansatz.append(FermionOperator("3^ 1", "c"))
        >>> print(len(ansatz.circuit))
        3
        >>> print(ansatz.circuit.params_name)
        ['a', 'b', 'c']
    """

    def __init__(self, n_qubits=None, n_electrons=None):
        """Initialize a SequentialUCCAnsatz object."""
        if n_qubits is not None:
            _check_int_type('n_qubits', n_qubits)
            _check_value_should_not_less('n_qubits', 1, n_qubits)
        if n_electrons is not None:
            _check_int_type('n_electrons', n_electrons)
            _check_value_should_not_less('n_electrons', 0, n_electrons)
        if n_qubits is not None and n_electrons is not None and n_electrons > n_qubits:
            raise ValueError(
                "The number of electrons must be smaller than the number of qubits (spin-orbitals) in the ansatz!"
            )
        self.name = "Sequential UCC"
        self.n_qubits = n_qubits
        self._operators = []
        self._circuit = Circuit()

    def _sanitize_operator(self, operator: FermionOperator) -> FermionOperator:
        """Validate operator invariants and return a detached canonical copy."""
        if not isinstance(operator, FermionOperator):
            raise TypeError(f"operator must be a FermionOperator, but got {type(operator)}")

        operator_copied = FermionOperator(operator)
        if len(operator_copied.terms) != 1:
            raise ValueError("The added FermionOperator must have exactly one term.")

        operator_copied = operator_copied.normal_ordered()
        if len(operator_copied.terms) != 1:
            raise ValueError("The added FermionOperator must have exactly one non-zero term after normal ordering.")

        term = next(iter(operator_copied.terms.keys()))
        if not term:
            raise ValueError("The added FermionOperator cannot be an identity term.")

        qubits = [idx for idx, _ in term]
        if len(qubits) != len(set(qubits)):
            raise ValueError("The added FermionOperator must not contain duplicate indices in a term.")

        if self.n_qubits is not None and max(qubits) >= self.n_qubits:
            raise ValueError(f"Operator acts on qubit {max(qubits)}, which exceeds n_qubits ({self.n_qubits}).")

        return operator_copied

    def append(self, operator: FermionOperator):
        """
        Append a FermionOperator to the ansatz.

        Args:
            operator (FermionOperator): The excitation operator to add. Must have exactly one term.
        """
        operator_copied = self._sanitize_operator(operator)
        self._operators.append(operator_copied)
        self._circuit += UCCExcitationGate(operator_copied)

    def __iadd__(self, other):
        """
        Support the += operator to append an operator.

        Args:
            other (FermionOperator): The operator to append.
        """
        self.append(other)
        return self

    def remove(self, index: int) -> FermionOperator:
        """
        Remove and return the operator at the specified index.

        Args:
            index (int): The index of the operator to remove.

        Returns:
            FermionOperator, the removed operator.
        """
        _check_int_type('index', index)
        if index < 0 or index >= len(self._operators):
            raise IndexError("Index out of range.")

        removed = self._operators.pop(index)
        self._rebuild_circuit()
        return FermionOperator(removed)

    def _rebuild_circuit(self):
        """Reconstruct the circuit from the current list of operators."""
        self._circuit = Circuit()
        for op in self._operators:
            self._circuit += UCCExcitationGate(op)

    def __len__(self):
        """Return the number of operators in the ansatz."""
        return len(self._operators)

    def __getitem__(self, index):
        """Return the operator at the specified index."""
        if isinstance(index, slice):
            return [FermionOperator(op) for op in self._operators[index]]
        _check_int_type('index', index)
        if index < -len(self._operators) or index >= len(self._operators):
            raise IndexError("Index out of range.")
        return FermionOperator(self._operators[index])

    @property
    def circuit(self) -> Circuit:
        """Get the generated ansatz circuit."""
        return self._circuit

    @property
    def operators(self):
        """Return the list of operators in the ansatz."""
        return [FermionOperator(op) for op in self._operators]
