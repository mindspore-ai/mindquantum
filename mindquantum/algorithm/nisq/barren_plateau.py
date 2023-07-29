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
"""Barren plateau related module."""
import typing

import numpy as np

from mindquantum.core.circuit import Circuit
from mindquantum.core.operators import Hamiltonian
from mindquantum.simulator import Simulator


# pylint: disable=too-many-arguments,too-many-locals
def ansatz_variance(
    ansatz: Circuit,
    ham: Hamiltonian,
    focus: str,
    var_range: typing.Tuple[float, float] = (0, np.pi * 2),
    other_var: np.array = None,
    atol: float = 0.1,
    init_batch: int = 20,
    sim: typing.Union[Simulator, str] = 'mqvector',
):
    r"""
    Calculate the variance of the gradient of certain parameters of parameterized quantum circuit.

    Args:
        ansatz (:class:`~.core.circuit.Circuit`): The given parameterized quantum circuit.
        ham (:class:`~.core.operators.Hamiltonian`): The objective observable.
        focus (str): Which parameters you want to check.
        var_range (Tuple[float, float]): The random range for focusing parameter.
            Default: ``(0, 2*np.pi)``.
        other_var (numpy.array): The value of other parameters. If ``None``,
            it will be random number at every sampling. Default: ``None``.
        atol (float): Tolerance for variance when samping. Default: ``0.1``.
        init_batch (int): Initial sampling size. Default: ``20``.
        sim (Union[:class:`~.simulator.Simulator`, str]): Simulator you want to use. Default: ``mqvector``.

    Examples:
        >>> import numpy as np
        >>> from mindquantum.algorithm.nisq import ansatz_variance
        >>> from mindquantum.algorithm.nisq.chem import HardwareEfficientAnsatz
        >>> from mindquantum.core.gates import RY, RZ, Z
        >>> from mindquantum.core.operators import Hamiltonian, QubitOperator
        >>> np.random.seed(42)
        >>> ham = Hamiltonian(QubitOperator('Z0 Z1'))
        >>> q_list = [4, 6, 8, 10]
        >>> vars = []
        >>> for q in q_list:
        ...     circ = HardwareEfficientAnsatz(q, [RY, RZ], Z, depth=50).circuit
        ...     vars.append(ansatz_variance(circ, ham, circ.params_name[0]))
        ...     print(f'qubit: {q},\t var: {vars[-1]}')
        qubit: 4,        var: 0.03366677155540075
        qubit: 6,        var: 0.007958129595835611
        qubit: 8,        var: 0.0014247908876269244
        qubit: 10,       var: 0.0006696567877430079
    """
    params_name = ansatz.params_name
    if focus not in params_name:
        raise ValueError(f"Parameter {focus} is not in given ansatz.")
    if len(var_range) != 2:
        raise ValueError("var_range should have two float.")
    if other_var is not None and len(other_var) != len(params_name) - 1:
        raise ValueError(f"other_var should have length of {len(params_name)-1}, but get {len(other_var)}")
    if other_var is None:
        encoder = ansatz.as_encoder(False)
    else:
        params_name.remove(focus)
        encoder = ansatz.apply_value(dict(zip(params_name, other_var))).as_encoder(False)
    params_name = encoder.params_name
    focus_idx = params_name.index(focus)

    if not isinstance(sim, Simulator):
        sim = Simulator(sim, encoder.n_qubits)
    grad_ops = sim.get_expectation_with_grad(ham, encoder)

    def run(grad_ops, n_sampling):
        params_0 = np.random.uniform(var_range[0], var_range[1], size=(n_sampling, len(params_name)))
        _, g = grad_ops(params_0)
        return g.real[:, 0, focus_idx]

    gradients = run(grad_ops, init_batch)
    var_i = 1
    step = 0
    while var_i > atol:
        gradients = np.append(gradients, run(grad_ops, init_batch * 2**step))
        half_l = len(gradients) // 2
        ori_var = np.var(gradients[:half_l])
        this_var = np.var(gradients[half_l:])
        try:
            var_i = np.abs(ori_var - this_var) / ori_var
        except ZeroDivisionError as exc:
            raise ZeroDivisionError("ori_val cannot be zero.") from exc
        step += 1

    return np.var(gradients)
