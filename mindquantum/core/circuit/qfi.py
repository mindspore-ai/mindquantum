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
"""Calculate quantum fisher information."""

import numpy as np

from mindquantum.core.gates.basicgate import MultiParamsGate

from ...utils.type_value_check import _check_and_generate_pr_type, _check_input_type
from ..parameterresolver import ParameterResolver
from .circuit import Circuit


def pr_converter(pr_map, origin: ParameterResolver):
    """Convert original parameters to redefined parameters."""
    part_a = {}
    for k, v in pr_map.items():
        part_a[k] = v.combination(origin).const
    return ParameterResolver(part_a)


def apply_gate(sim, gate, g_cpp, pr_cpp, diff):
    """Apply a gate."""
    if gate.parameterized:
        sim.apply_gate(g_cpp, pr_cpp, diff)
    else:
        sim.apply_gate(g_cpp)


# pylint: disable=too-many-statements,too-many-locals
def _qfi_matrix_base(circuit: Circuit, which_part='both', backend='mqvector'):
    """Calculate Quantum Fisher Information (QFI)."""
    from ...simulator import (  # pylint: disable=import-outside-toplevel,cyclic-import
        Simulator,
        inner_product,
    )

    _check_input_type('circuit', Circuit, circuit)
    if which_part not in ['A', 'B', 'both']:
        raise ValueError(f"which part should be 'A', 'B' or 'both', but get {which_part}.")
    circuit = circuit.remove_barrier()
    if circuit.has_measure_gate:
        raise ValueError("circuit can not has measure gate for calculate qfi similar value.")
    if circuit.is_noise_circuit:
        raise ValueError("circuit can not be noise circuit for calculate qfi similar value.")
    if not circuit.params_name:
        raise ValueError("circuit need a parameterized quantum circuit, but get non-parameterized one.")

    pure_circ = Circuit()
    n_params = 0
    jac = {}
    pr_map = {}
    for gate in circuit:
        if isinstance(gate, MultiParamsGate):
            raise ValueError(f"qfi doesn't support multi parameters gate: {gate}")
        if gate.parameterized:
            n_params += 1
            new_p = f'p{n_params}'
            pure_circ += gate(ParameterResolver(new_p)).on(gate.obj_qubits, gate.ctrl_qubits)
            jac[new_p] = dict(gate.coeff.items())
            pr_map[new_p] = gate.coeff
        else:
            pure_circ += gate
    old_idx_map = {p: idx for idx, p in enumerate(circuit.params_name)}
    new_idx_map = {p: idx for idx, p in enumerate(pure_circ.params_name)}
    tmp = np.zeros((len(new_idx_map), len(old_idx_map)), np.complex128)
    for new_p, matrix in jac.items():
        for old_p, v in matrix.items():
            tmp[new_idx_map[new_p], old_idx_map[old_p]] = v
    jac = tmp
    cpp_obj = pure_circ.get_cpp_obj()
    c_len = len(pure_circ)
    ket = Simulator(backend, pure_circ.n_qubits)

    # pylint: disable=too-many-branches
    def qfi_ops(pr: ParameterResolver):
        pr = _check_and_generate_pr_type(pr, circuit.params_name)
        ket.reset()
        pr_cpp = pr_converter(pr_map, pr).to_real_obj()
        if which_part != 'B':
            part_a = np.zeros((len(new_idx_map), len(new_idx_map)), np.complex128)
        if which_part != 'A':
            part_b = np.zeros(len(new_idx_map), np.complex128)
        for i in range(c_len):
            gate = pure_circ[i]
            g_cpp = cpp_obj[i]
            if gate.parameterized:
                idx_i = new_idx_map[gate.coeff.params_name[0]]
                bra = Simulator(backend, pure_circ.n_qubits)
                ket_tmp = ket.copy()
                ket_tmp.backend.sim.apply_gate(g_cpp, pr_cpp, True)
                if which_part != 'B':
                    grad_current = inner_product(ket_tmp, ket_tmp)
                    part_a[idx_i, idx_i] = grad_current
                for j in range(i):
                    g_j = pure_circ[j]
                    g_cpp_j = cpp_obj[j]
                    if which_part != 'B':
                        bra_tmp = bra.copy()
                        for k in range(j, i + 1):
                            g_k = pure_circ[k]
                            g_cpp_k = cpp_obj[k]
                            apply_gate(bra_tmp.backend.sim, g_k, g_cpp_k, pr_cpp, (k == j))

                    if g_j.parameterized:
                        bra.backend.sim.apply_gate(g_cpp_j, pr_cpp, False)
                        if which_part != 'B':
                            idx_j = new_idx_map[g_j.coeff.params_name[0]]
                            part_a[idx_j, idx_i] = inner_product(bra_tmp, ket_tmp)
                            part_a[idx_i, idx_j] = np.conj(part_a[idx_j, idx_i])
                    else:
                        bra.backend.sim.apply_gate(g_cpp_j)
                bra.backend.sim.apply_gate(g_cpp, pr_cpp, False)
                if which_part != 'A':
                    part_b[idx_i] = np.conj(inner_product(bra, ket_tmp))
                ket.backend.sim.apply_gate(g_cpp, pr_cpp, False)
            else:
                ket.backend.sim.apply_gate(g_cpp)
        if which_part != 'B':
            first_part = jac.T @ part_a @ jac
        if which_part != 'A':
            second_part = jac.T @ part_b
        if which_part == 'A':
            return first_part
        if which_part == 'B':
            return second_part
        return first_part, second_part

    return qfi_ops


def qfi(circuit: Circuit, backend='mqvector'):
    r"""
    Calculate the quantum fisher information of the given parameterized circuit with given parameters.

    The quantum fisher information of a parameterized circuit is defined as:

    .. math::

        \text{QFI}_{i,j} = 4\text{Re}(A_{i,j} - B_{i,j})

    where

    .. math::

        A_{i,j} = \frac{\partial \left<\psi\right| }{\partial x_{i}}
        \frac{\partial \left|\psi\right> }{\partial x_{j}}

    and

    .. math::

        B_{i,j} = \frac{\partial \left<\psi\right| }{\partial x_i}\left|\psi\right>
        \left<\psi\right|\frac{\partial \left|\psi\right> }{\partial x_{j}}

    Args:
        circuit (Circuit): A parameterized quantum circuit.
        backend (str): A supported simulator backend. Please refer description
            of :class:`~.simulator.Simulator`. Default: ``'mqvector'``.

    Returns:
        Function, a function that can calculate quantum fisher information.

    Examples:
        >>> import numpy as np
        >>> from mindquantum.core.circuit import qfi, Circuit
        >>> circ = Circuit().rx('a', 0).ry('b', 0).rz('c', 0)
        >>> qfi_ops = qfi(circ)
        >>> qfi_ops(np.array([1, 2, 3]))
        array([[ 1.        ,  0.        , -0.90929743],
               [ 0.        ,  0.29192658, -0.18920062],
               [-0.90929743, -0.18920062,  0.94944468]])
    """
    qfi_ops_tmp = _qfi_matrix_base(circuit, backend=backend)

    def qfi_ops(pr):
        # pylint: disable=invalid-name
        a, b = qfi_ops_tmp(pr)
        b = np.outer(b, np.conj(b))
        return np.real(a - b) * 4

    return qfi_ops


def partial_psi_partial_psi(circuit: Circuit, backend='mqvector'):
    r"""
    Calculate the following value of the given parameterized quantum circuit.

    .. math::

        A_{i,j} = \frac{\partial \left<\psi\right| }{\partial x_{i}}
        \frac{\partial \left|\psi\right> }{\partial x_{j}}

    Args:
        circuit (Circuit): A parameterized quantum circuit.
        backend (str): A supported simulator backend. Please refer description
            of :class:`~.simulator.Simulator`. Default: ``'mqvector'``.

    Returns:
        Function, a function that can calculate inner product of partial psi and partial psi.

    Examples:
        >>> import numpy as np
        >>> from mindquantum.core.circuit import partial_psi_partial_psi, Circuit
        >>> circ = Circuit().rx('a', 0).ry('b', 0).rz('c', 0)
        >>> pppp_ops = partial_psi_partial_psi(circ)
        >>> pppp_ops(np.array([1, 2, 3]))
        array([[ 0.25      +0.j        ,  0.        +0.13507558j,
                -0.22732436-0.08754387j],
               [ 0.        -0.13507558j,  0.25      +0.j        ,
                 0.        +0.12282387j],
               [-0.22732436+0.08754387j,  0.        -0.12282387j,
                 0.25      +0.j        ]])
    """
    return _qfi_matrix_base(circuit, 'A', backend=backend)


def partial_psi_psi(circuit: Circuit, backend='mqvector'):
    r"""
    Calculate the following value of the given parameterized quantum circuit.

    .. math::

        B_i = \frac{\partial \left<\psi\right| }{\partial x_i}\left|\psi\right>

    Args:
        circuit (Circuit): A parameterized quantum circuit.
        backend (str): A supported simulator backend. Please refer description
            of :class:`~.simulator.Simulator`. Default: ``'mqvector'``.

    Returns:
        Function, a function that can calculate inner product of partial psi and psi.

    Examples:
        >>> import numpy as np
        >>> from mindquantum.core.circuit import partial_psi_psi, Circuit
        >>> circ = Circuit().rx('a', 0).ry('b', 0).rz('c', 0)
        >>> ppp = partial_psi_psi(circ)
        >>> ppp(np.array([1, 2, 3]))
        array([0.+0.j        , 0.-0.42073549j, 0.-0.11242255j])
    """
    return _qfi_matrix_base(circuit, 'B', backend=backend)
