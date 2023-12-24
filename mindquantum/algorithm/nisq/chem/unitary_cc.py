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

# pylint: disable=duplicate-code

"""Unitary coupled-cluster ansatz."""

from mindquantum.core.circuit import Circuit, add_prefix
from mindquantum.core.operators import TimeEvolution

from .._ansatz import Ansatz
from .transform import Transform
from .uccsd0 import uccsd0_singlet_generator


def _check_int_list(input_list, name):
    """Check if input_list is a list of integers."""
    if not isinstance(input_list, list):
        raise ValueError(f"The input {str(name)} should be a list, but get {type(input_list)}.")
    for i in input_list:
        if not isinstance(i, int):
            raise ValueError(f"The indices of {str(name)} should be integer, but get {type(i)}.")


# pylint: disable=too-few-public-methods
class UCCAnsatz(Ansatz):
    r"""
    The unitary coupled-cluster ansatz for molecular simulations.

    .. math::

        U(\vec{\theta}) = \prod_{j=1}^{N(N\ge1)}{\prod_{i=0}^{N_{j}}{\exp{(\theta_{i}\hat{\tau}_{i})}}}

    where :math:`\hat{\tau}` are anti-Hermitian operators.

    Note:
        Currently, the circuit is constructed using JW transformation.
        In addition, the reference state wave function (Hartree-Fock) will NOT be
        included.

    Args:
        n_qubits(int): Number of qubits (spin-orbitals). Default: ``None``.
        n_electrons(int): Number of electrons (occupied spin-orbitals). Default: ``None``.
        occ_orb(list): Indices of manually assigned occupied spatial
            orbitals, for ansatz construction only. Default: ``None``.
        vir_orb(list): Indices of manually assigned virtual spatial
            orbitals, for ansatz construction only. Default: ``None``.
        generalized(bool): Whether to use generalized excitations which
            do not distinguish occupied or virtual orbitals (UCCGSD). Default: ``False``.
        trotter_step(int): The order of Trotterization step. Default: ``1``.

    Examples:
        >>> from mindquantum.algorithm.nisq import UCCAnsatz
        >>> ucc = UCCAnsatz(12, 4, occ_orb=[1],
        ...                 vir_orb=[2, 3],
        ...                 generalized=True,
        ...                 trotter_step=2)
        >>> circuit = ucc.circuit.remove_barrier()
        >>> len(circuit)
        3624
        >>> params_list = ucc.circuit.params_name
        >>> len(params_list)
        48
        >>> circuit[-10:]
                    ┏━━━━━━━━━━┓ ┏━━━┓                                      ┏━━━┓
        q7: ────■───┨ RX(7π/2) ┠─┨ H ┠───■──────────────────────────────■───┨ H ┠──────────
                ┃   ┗━━━━━━━━━━┛ ┗━━━┛   ┃                              ┃   ┗━━━┛
              ┏━┻━┓ ┏━━━┓ ┏━━━━━━━━━┓  ┏━┻━┓ ┏━━━━━━━━━━━━━━━━━━━━━━┓ ┏━┻━┓ ┏━━━━━━━━━━┓
        q5: ──┨╺╋╸┠─┨ H ┠─┨ RX(π/2) ┠──┨╺╋╸┠─┨ RZ(-1/2*t_1_d0_d_17) ┠─┨╺╋╸┠─┨ RX(7π/2) ┠───
              ┗━━━┛ ┗━━━┛ ┗━━━━━━━━━┛  ┗━━━┛ ┗━━━━━━━━━━━━━━━━━━━━━━┛ ┗━━━┛ ┗━━━━━━━━━━┛
    """

    # pylint: disable=too-many-arguments
    def __init__(self, n_qubits=None, n_electrons=None, occ_orb=None, vir_orb=None, generalized=False, trotter_step=1):
        """Initialize a UCCAnsatz object."""
        if n_qubits is not None and not isinstance(n_qubits, int):
            raise ValueError(f"The number of qubits should be integer, but get {type(n_qubits)}.")
        if n_electrons is not None and not isinstance(n_electrons, int):
            raise ValueError(f"The number of electrons should be integer, but get {type(n_electrons)}.")
        if isinstance(n_electrons, int) and n_electrons > n_qubits:
            raise ValueError(
                "The number of electrons must be smaller than the number of qubits (spin-orbitals) in the ansatz!"
            )
        if occ_orb is not None:
            _check_int_list(occ_orb, "occupied orbitals")
        if vir_orb is not None:
            _check_int_list(vir_orb, "virtual orbitals")
        if not isinstance(generalized, bool):
            raise ValueError(f"The parameter generalized should be bool, but get {type(generalized)}.")
        if not isinstance(trotter_step, int) or trotter_step < 1:
            raise ValueError("Trotter step must be a positive integer!")

        super().__init__("Unitary CC", n_qubits, n_qubits, n_electrons, occ_orb, vir_orb, generalized, trotter_step)

    # pylint: disable=arguments-differ,too-many-arguments
    def _implement(self, n_qubits, n_electrons, occ_orb=None, vir_orb=None, generalized=False, trotter_step=1):
        """Implement the UCC ansatz using uccsd0."""
        ansatz_circuit = Circuit()
        for trotter_idx in range(trotter_step):
            uccsd0_fermion_op = uccsd0_singlet_generator(n_qubits, n_electrons, True, occ_orb, vir_orb, generalized)
            uccsd0_circuit = TimeEvolution(Transform(uccsd0_fermion_op).jordan_wigner().imag, 1).circuit
            # Modify parameter names
            uccsd0_circuit_modified = add_prefix(uccsd0_circuit, "t_" + str(trotter_idx))
            ansatz_circuit += uccsd0_circuit_modified
        n_qubits_circuit = 0
        if list(ansatz_circuit):
            n_qubits_circuit = ansatz_circuit.n_qubits
        # If the ansatz's n_qubits is not set by user, use n_qubits_circuit.
        if self.n_qubits is None:
            self.n_qubits = n_qubits_circuit
        if self.n_qubits < n_qubits_circuit:
            raise ValueError(
                f"The number of qubits in the ansatz circuit {n_qubits_circuit} is larger than the input"
                f" n_qubits {n_qubits}! Please check input parameters such as occ_orb, etc."
            )
        self._circuit = ansatz_circuit
