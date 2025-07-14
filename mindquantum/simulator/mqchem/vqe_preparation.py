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
"""High-level VQE preparation factory for MQChemSimulator."""

import numpy as np

from mindquantum.simulator import mqchem
from ...core.circuit import Circuit
from ...core.operators import FermionOperator, InteractionOperator
from ...algorithm.nisq.chem import uccsd_singlet_generator, uccsd_singlet_get_packed_amplitudes


def prepare_uccsd_vqe(molecular, threshold: float = 1e-6):
    """
    Prepare all components for a UCCSD-VQE simulation with the MQChemSimulator.

    This factory function streamlines the setup for a VQE simulation by:
    1. Generating all singlet UCCSD excitation operators using
       :func:`~.algorithm.nisq.chem.uccsd_singlet_generator`.
    2. Extracting the corresponding amplitudes from a pre-computed CCSD
       calculation included in the `molecular` data.
    3. Filtering excitations based on their CCSD amplitudes via the `threshold`.
    4. Constructing a parameterized UCCSD ansatz circuit using
       :class:`~.simulator.mqchem.UCCExcitationGate`.
    5. Creating a :class:`~.simulator.mqchem.CIHamiltonian` for expectation evaluation.
    6. Returning all necessary components to run a VQE experiment.

    Args:
        molecular (openfermion.MolecularData): Molecular data object that must
            contain CCSD calculation results.
        threshold (float): The threshold for CCSD amplitudes. Excitations with
            amplitudes below this value will be discarded. Default: ``1e-6``.

    Returns:
        - **hamiltonian** (mqchem.CIHamiltonian), The CI-space Hamiltonian.
        - **ansatz_circuit** (Circuit), The parameterized UCCSD ansatz circuit.
        - **initial_amplitudes** (numpy.ndarray), The CCSD amplitudes corresponding
          to the parameters in `ansatz_circuit`, suitable as an initial
          guess for an optimizer.

    Examples:
        >>> from openfermionpyscf import run_pyscf
        >>> from openfermion import MolecularData
        >>> from scipy.optimize import minimize
        >>> from mindquantum.simulator import mqchem
        >>>
        >>> molecule = MolecularData([("H", (i, 0, 0)) for i in range(6)], 'sto-3g', 1, 0)
        >>> mol = run_pyscf(molecule, run_ccsd=True)
        >>> hamiltonian, ansatz_circuit, initial_amplitudes = mqchem.prepare_uccsd_vqe(mol, 1e-3)
        >>>
        >>> simulator = mqchem.MQChemSimulator(mol.n_qubits, mol.n_electrons)
        >>> grad_ops = simulator.get_expectation_with_grad(hamiltonian, ansatz_circuit)
        >>>
        >>> result = minimize(grad_ops, initial_amplitudes, method='L-BFGS-B', jac=True)
        >>> print(f"VQE energy: {result.fun}")
        VQE energy: -3.2354494505390528
    """
    ham_of = molecular.get_molecular_hamiltonian()
    inter_ops = InteractionOperator(*ham_of.n_body_tensors.values())
    fermion_ham = FermionOperator(inter_ops).normal_ordered()
    hamiltonian = mqchem.CIHamiltonian(fermion_ham)

    excitation_generator = uccsd_singlet_generator(molecular.n_qubits, molecular.n_electrons, anti_hermitian=False)

    if molecular.ccsd_single_amps is None or molecular.ccsd_double_amps is None:
        raise ValueError("Molecular data must contain CCSD amplitudes. Please run pyscf with 'run_ccsd=True'.")

    packed_amplitudes = uccsd_singlet_get_packed_amplitudes(
        molecular.ccsd_single_amps,
        molecular.ccsd_double_amps,
        molecular.n_qubits,
        molecular.n_electrons,
    )

    ansatz_circuit = Circuit()
    for term_op in excitation_generator:
        param_names = term_op.singlet_coeff().params_name
        include_gate = False
        for p_name in param_names:
            if abs(packed_amplitudes[p_name]) > threshold:
                include_gate = True
                break

        if include_gate:
            gate = mqchem.UCCExcitationGate(term_op)
            ansatz_circuit += gate

    ordered_params = ansatz_circuit.params_name
    initial_amplitudes = [packed_amplitudes[p] for p in ordered_params]
    return hamiltonian, ansatz_circuit, np.array(initial_amplitudes)
