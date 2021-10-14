#   Copyright 2017 The OpenFermion Developers.
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
"""Module to create and manipulate unitary coupled cluster operators."""

import itertools

import numpy
from mindquantum.ops import FermionOperator
from openfermion.utils.indexing import down_index, up_index
from mindquantum.parameterresolver import ParameterResolver as PR


def uccsd_singlet_get_packed_amplitudes(single_amplitudes, double_amplitudes,
                                        n_qubits, n_electrons):
    r"""Convert amplitudes for use with singlet UCCSD

    The output list contains only those amplitudes that are relevant to
    singlet UCCSD, in an order suitable for use with the function
    `uccsd_singlet_generator`.

    Args:
        single_amplitudes(numpy.ndarray): :math:`N\times N` array storing single excitation
            amplitudes corresponding to :math:`t_{i,j} * (a_i^\dagger a_j - \text{H.C.})`
        double_amplitudes(numpy.ndarray): :math:`N\times N\times N\times N` array storing double
            excitation amplitudes corresponding to
            :math:`t_{i,j,k,l} * (a_i^\dagger a_j a_k^\dagger a_l - \text{H.C.})`
        n_qubits(int): Number of spin-orbitals used to represent the system,
            which also corresponds to number of qubits in a non-compact map.
        n_electrons(int): Number of electrons in the physical system.

    Returns:
        ParameterResolver, List storing the unique single
        and double excitation amplitudes for a singlet UCCSD operator.
        The ordering lists unique single excitations before double
        excitations.

    Examples:
        >>> import numpy as np
        >>> from mindquantum.hiqfermion.ucc import uccsd_singlet_get_packed_amplitudes
        >>> n_qubits, n_electrons = 4, 2
        >>> np.random.seed(42)
        >>> ccsd_single_amps = np.random.random((4, 4))
        >>> ccsd_double_amps = np.random.random((4, 4, 4, 4))
        >>> uccsd_singlet_get_packed_amplitudes(ccsd_single_amps, ccsd_double_amps,
        ...                                     n_qubits, n_electrons)
        {'s_0': 0.6011150117432088, 'd1_0': 0.7616196153287176}
    """
    n_spatial_orbitals = n_qubits // 2
    n_occupied = int(numpy.ceil(n_electrons / 2))
    n_virtual = n_spatial_orbitals - n_occupied

    singles = PR()
    doubles_1 = PR()
    doubles_2 = PR()

    # Get singles and doubles amplitudes associated with one
    # spatial occupied-virtual pair
    for p, q in itertools.product(range(n_virtual), range(n_occupied)):
        # Get indices of spatial orbitals
        virtual_spatial = n_occupied + p
        occupied_spatial = q
        # Get indices of spin orbitals
        virtual_up = up_index(virtual_spatial)
        virtual_down = down_index(virtual_spatial)
        occupied_up = up_index(occupied_spatial)
        occupied_down = down_index(occupied_spatial)

        # Get singles amplitude
        # Just get up amplitude, since down should be the same
        singles[f's_{len(singles)}'] = single_amplitudes[virtual_up,
                                                         occupied_up]

        # Get doubles amplitude
        doubles_1[f'd1_{len(doubles_1)}'] = double_amplitudes[virtual_up,
                                                              occupied_up,
                                                              virtual_down,
                                                              occupied_down]

    # Get doubles amplitudes associated with two spatial occupied-virtual pairs
    for (p, q), (r, s) in itertools.combinations(
            itertools.product(range(n_virtual), range(n_occupied)), 2):
        # Get indices of spatial orbitals
        virtual_spatial_1 = n_occupied + p
        occupied_spatial_1 = q
        virtual_spatial_2 = n_occupied + r
        occupied_spatial_2 = s

        # Get indices of spin orbitals
        # Just get up amplitudes, since down and cross terms should be the same
        virtual_1_up = up_index(virtual_spatial_1)
        occupied_1_up = up_index(occupied_spatial_1)
        virtual_2_up = up_index(virtual_spatial_2)
        occupied_2_up = up_index(occupied_spatial_2)

        # Get amplitude
        doubles_2[f'd2_{len(doubles_2)}'] = double_amplitudes[virtual_1_up,
                                                              occupied_1_up,
                                                              virtual_2_up,
                                                              occupied_2_up]

    return singles + doubles_1 + doubles_2


def uccsd_singlet_generator(n_qubits, n_electrons, anti_hermitian=True):
    """Create a singlet UCCSD generator for a system with n_electrons

    This function generates a FermionOperator for a UCCSD generator designed
    to act on a single reference state consisting of n_qubits spin orbitals
    and n_electrons electrons, that is a spin singlet operator, meaning it
    conserves spin.

    Args:
        n_qubits(int): Number of spin-orbitals used to represent the system,
            which also corresponds to number of qubits in a non-compact map.
        n_electrons(int): Number of electrons in the physical system.
        anti_hermitian(bool): Flag to generate only normal CCSD operator
            rather than unitary variant, primarily for testing

    Returns:
        FermionOperator, Generator of the UCCSD operator that
        builds the UCCSD wavefunction.

    Examples:
        >>> from mindquantum.hiqfermion.ucc import uccsd_singlet_generator
        >>> uccsd_singlet_generator(4, 2)
        -s_0 [0^ 2] +
        -d1_0 [0^ 2 1^ 3] +
        -s_0 [1^ 3] +
        -d1_0 [1^ 3 0^ 2] +
        s_0 [2^ 0] +
        d1_0 [2^ 0 3^ 1] +
        s_0 [3^ 1] +
        d1_0 [3^ 1 2^ 0]
    """
    if n_qubits % 2 != 0:
        raise ValueError('The total number of spin-orbitals should be even.')

    n_spatial_orbitals = n_qubits // 2
    n_occupied = int(numpy.ceil(n_electrons / 2))
    n_virtual = n_spatial_orbitals - n_occupied

    # Initialize operator
    generator = FermionOperator()

    # Generate excitations
    spin_index_functions = [up_index, down_index]
    # Generate all spin-conserving single and double excitations derived
    # from one spatial occupied-virtual pair
    for i, (p, q) in enumerate(
            itertools.product(range(n_virtual), range(n_occupied))):

        # Get indices of spatial orbitals
        virtual_spatial = n_occupied + p
        occupied_spatial = q

        for spin in range(2):
            # Get the functions which map a spatial orbital index to a
            # spin orbital index
            this_index = spin_index_functions[spin]
            other_index = spin_index_functions[1 - spin]

            # Get indices of spin orbitals
            virtual_this = this_index(virtual_spatial)
            virtual_other = other_index(virtual_spatial)
            occupied_this = this_index(occupied_spatial)
            occupied_other = other_index(occupied_spatial)

            # Generate single excitations
            coeff = PR({f's_{i}': 1})
            generator += FermionOperator(
                ((virtual_this, 1), (occupied_this, 0)), coeff)
            if anti_hermitian:
                generator += FermionOperator(
                    ((occupied_this, 1), (virtual_this, 0)), -1 * coeff)

            # Generate double excitation
            coeff = PR({f'd1_{i}': 1})
            generator += FermionOperator(
                ((virtual_this, 1), (occupied_this, 0), (virtual_other, 1),
                 (occupied_other, 0)), coeff)
            if anti_hermitian:
                generator += FermionOperator(
                    ((occupied_other, 1), (virtual_other, 0),
                     (occupied_this, 1), (virtual_this, 0)), -1 * coeff)

    # Generate all spin-conserving double excitations derived
    # from two spatial occupied-virtual pairs
    for i, ((p, q), (r, s)) in enumerate(
            itertools.combinations(
                itertools.product(range(n_virtual), range(n_occupied)), 2)):

        # Get indices of spatial orbitals
        virtual_spatial_1 = n_occupied + p
        occupied_spatial_1 = q
        virtual_spatial_2 = n_occupied + r
        occupied_spatial_2 = s

        # Generate double excitations
        coeff = PR({f'd2_{i}': 1})
        for (spin_a, spin_b) in itertools.product(range(2), repeat=2):
            # Get the functions which map a spatial orbital index to a
            # spin orbital index
            index_a = spin_index_functions[spin_a]
            index_b = spin_index_functions[spin_b]

            # Get indices of spin orbitals
            virtual_1_a = index_a(virtual_spatial_1)
            occupied_1_a = index_a(occupied_spatial_1)
            virtual_2_b = index_b(virtual_spatial_2)
            occupied_2_b = index_b(occupied_spatial_2)

            if virtual_1_a == virtual_2_b:
                continue
            if occupied_1_a == occupied_2_b:
                continue
            else:

                generator += FermionOperator(
                    ((virtual_1_a, 1), (occupied_1_a, 0), (virtual_2_b, 1),
                     (occupied_2_b, 0)), coeff)
                if anti_hermitian:
                    generator += FermionOperator(
                        ((occupied_2_b, 1), (virtual_2_b, 0),
                         (occupied_1_a, 1), (virtual_1_a, 0)), -1 * coeff)

    return generator
