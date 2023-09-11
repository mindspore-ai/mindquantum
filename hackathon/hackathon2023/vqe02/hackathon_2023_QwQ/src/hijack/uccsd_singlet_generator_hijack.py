from itertools import product, combinations
from collections import defaultdict
from typing import List, Dict

from mindquantum.core.circuit import Circuit
from mindquantum.core.operators import FermionOperator
from mindquantum.core.operators.utils import down_index, up_index
from qupack.vqe.gates import ExpmPQRSFermionGate

from mindquantum.third_party.unitary_cc import *
assert uccsd_singlet_generator


def uccsd_singlet_get_packed_amplitudes_hijack(single_amplitudes, double_amplitudes, n_qubits, n_electrons, n_trotter=1) -> ParameterResolver:
    single_amplitudes *= single_amplitudes > 1e-8
    double_amplitudes *= double_amplitudes > 1e-8

    n_spatial_orbitals = n_qubits // 2
    n_occupied = int(numpy.ceil(n_electrons / 2))
    n_virtual = n_spatial_orbitals - n_occupied

    singles = ParameterResolver()
    doubles_1 = ParameterResolver()
    doubles_2 = ParameterResolver()

    # pylint: disable=invalid-name
    # Get singles and doubles amplitudes associated with one spatial occupied-virtual pair
    for p, q in product(range(n_virtual), range(n_occupied)):
        # Get indices of spatial orbitals
        virtual_spatial = n_occupied + p
        occupied_spatial = q
        # Get indices of spin orbitals
        virtual_up    = up_index(virtual_spatial)
        virtual_down  = down_index(virtual_spatial)
        occupied_up   = up_index(occupied_spatial)
        occupied_down = down_index(occupied_spatial)

        # Get singles amplitude
        # Just get up amplitude, since down should be the same'
        singles[f's_{len(singles)}'] = single_amplitudes[virtual_up, occupied_up]

        # Get doubles amplitude
        doubles_1[f'd1_{len(doubles_1)}'] = double_amplitudes[virtual_up, occupied_up, virtual_down, occupied_down]

    # Get doubles amplitudes associated with two spatial occupied-virtual pairs
    for (p, q), (r, s) in combinations(product(range(n_virtual), range(n_occupied)), 2):
        # Get indices of spatial orbitals
        virtual_spatial_1  = n_occupied + p
        occupied_spatial_1 = q
        virtual_spatial_2  = n_occupied + r
        occupied_spatial_2 = s

        # Get indices of spin orbitals
        # Just get up amplitudes, since down and cross terms should be the same
        virtual_1_up  = up_index(virtual_spatial_1)
        occupied_1_up = up_index(occupied_spatial_1)
        virtual_2_up  = up_index(virtual_spatial_2)
        occupied_2_up = up_index(occupied_spatial_2)

        # Get amplitude
        doubles_2[f'd2_{len(doubles_2)}'] = double_amplitudes[virtual_1_up, occupied_1_up, virtual_2_up, occupied_2_up]

    ret = ParameterResolver()
    pr: ParameterResolver = singles + doubles_1 + doubles_2
    for k, v in pr.items():
        for t in range(n_trotter):
            ret[f't_{t}_{k}'] = v
    return ret


def uccsd_singlet_generator_hijack(n_qubits, n_electrons, anti_hermitian=True, n_trotter=1) -> Circuit:
    n_spatial_orbitals = n_qubits // 2
    n_occupied = int(numpy.ceil(n_electrons / 2))
    n_virtual = n_spatial_orbitals - n_occupied

    ops: List[FermionOperator] = []
    spin_index_functions = [up_index, down_index]

    for t in range(n_trotter):
        # Generate all spin-conserving single and double excitations derived from one spatial occupied-virtual pair
        for i, (p, q) in enumerate(product(range(n_virtual), range(n_occupied))):
            # Get indices of spatial orbitals
            virtual_spatial = n_occupied + p
            occupied_spatial = q

            for spin in range(2):
                # Get the functions which map a spatial orbital index to a spin orbital index
                this_index  = spin_index_functions[spin]
                other_index = spin_index_functions[1 - spin]

                # Get indices of spin orbitals
                virtual_this   = this_index (virtual_spatial)
                virtual_other  = other_index(virtual_spatial)
                occupied_this  = this_index (occupied_spatial)
                occupied_other = other_index(occupied_spatial)

                # Generate single excitations
                coeff = ParameterResolver({f't_{t}_s_{i}': 1})
                ops += [FermionOperator(((virtual_this, 1), (occupied_this, 0)), coeff)]
                if anti_hermitian:
                    ops += [FermionOperator(((occupied_this, 1), (virtual_this, 0)), -1 * coeff)]

                # Generate double excitation
                coeff = ParameterResolver({f't_{t}_d1_{i}': 1})
                ops += [FermionOperator(((virtual_this, 1), (occupied_this, 0), (virtual_other, 1), (occupied_other, 0)), coeff)]
                if anti_hermitian:
                    ops += [FermionOperator(((occupied_other, 1), (virtual_other, 0), (occupied_this, 1), (virtual_this, 0)), -1 * coeff)]

        # Generate all spin-conserving double excitations derived from two spatial occupied-virtual pairs
        for i, ((p, q), (r, s)) in enumerate(combinations(product(range(n_virtual), range(n_occupied)), 2)):
            # Get indices of spatial orbitals
            virtual_spatial_1  = n_occupied + p
            occupied_spatial_1 = q
            virtual_spatial_2  = n_occupied + r
            occupied_spatial_2 = s

            # Generate double excitations
            coeff = ParameterResolver({f't_{t}_d2_{i}': 1})
            for (spin_a, spin_b) in product(range(2), repeat=2):
                # Get the functions which map a spatial orbital index to a
                # spin orbital index
                index_a = spin_index_functions[spin_a]
                index_b = spin_index_functions[spin_b]

                # Get indices of spin orbitals
                virtual_1_a  = index_a(virtual_spatial_1)
                occupied_1_a = index_a(occupied_spatial_1)
                virtual_2_b  = index_b(virtual_spatial_2)
                occupied_2_b = index_b(occupied_spatial_2)

                if virtual_1_a == virtual_2_b or occupied_1_a == occupied_2_b: continue

                ops += [FermionOperator(((virtual_1_a, 1), (occupied_1_a, 0), (virtual_2_b, 1), (occupied_2_b, 0)), coeff)]
                if anti_hermitian:
                    ops += [FermionOperator(((occupied_2_b, 1), (virtual_2_b, 0), (occupied_1_a, 1), (virtual_1_a, 0)), -1 * coeff)]

    circ = Circuit()
    for op in ops:
        for term in op:
            circ += ExpmPQRSFermionGate(term)
    return circ
