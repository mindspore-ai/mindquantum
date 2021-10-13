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
"""Circuit for time evolution."""

from mindquantum.core.operators import QubitOperator
from mindquantum.core.parameterresolver import ParameterResolver
from mindquantum.core.circuit.utils import decompose_single_term_time_evolution


#TODO: 🔥🔥🔥🔥🔥《含时演化算符教程》↪️编写含时演化算符的使用教程，介绍Trotter分解，介绍该模块的应用
class TimeEvolution:
    r"""
    The time evolution operator that can generate a crosponded circuit.

    The time evolution operator will do the following evolution:

    .. math::

        \left|\varphi(t)\right>=e^{-itH}\left|\varphi(0)\right>

    Note:
        The hamiltonian should be a parameterized or non parameterized
        QubitOperator. If the QubitOperator has multiple terms, the
        first order trotter decomposition will be used.

    Args:
        ops (QubitOperator): The qubit operator hamiltonian, could be parameterized or
            non parameterized.
        time (Union[numbers.Number, dict, ParameterResolver]): The evolution time,
            could be a number or a parameter resolver. If None, the time will be set to 1.
            Default: None.

    Examples:
        >>> from mindquantum.core.operators import TimeEvolution, QubitOperator
        >>> h = QubitOperator('Z0 Z1', 'p')
        >>> TimeEvolution(h).circuit
        q0: ──●───────────────●──
              │               │
        q1: ──X────RZ(2*p)────X──
    """
    def __init__(self, ops: QubitOperator, time=None):
        if time is None:
            time = 1
        self.time = time
        if isinstance(time, dict):
            self.time = ParameterResolver(time)
        self.ops = ops

    @property
    def circuit(self):
        """Get the first order trotter decomposition circuit of this time evolution operator."""
        from ..circuit import Circuit
        circ = Circuit()
        for k, v in self.ops.terms.items():
            pr_tmp = self.time * v
            tmp_circ = decompose_single_term_time_evolution(k, pr_tmp)
            circ += tmp_circ
        return circ
