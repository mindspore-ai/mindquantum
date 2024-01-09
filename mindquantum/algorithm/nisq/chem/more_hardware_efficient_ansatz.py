# Copyright 2023 Huawei Technologies Co., Ltd
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
# Author : Xiaoxiao Xiao <xxxwwzi@163.com>
#          Zhendong Li <zhendongli@bnu.edu.cn>
# Affiliation: College of Chemistry, Beijing Normal University
# Publication: Xiao, X., Zhao, H., Ren, J., Fang, W. H., & Li, Z. (2023). arXiv:2307.03563.
"""More hadaware efficient ansatz."""
# pylint: disable=arguments-differ,too-few-public-methods
import numpy as np

from mindquantum.algorithm.nisq._ansatz import (
    Ansatz,
    Initializer,
    single_qubit_gate_layer,
)
from mindquantum.core.circuit import Circuit
from mindquantum.core.gates import RX, RY, RZ, FSim, X


class RYLinear(Ansatz, Initializer):
    r"""
    HEA with :class:`~.core.gates.RY` as single qubit gate and linearly mapped CNOT gate as entanglement gate.

    .. image:: ./ansatz_images/RYLinear.png
        :height: 180px

    For more information about this ansatz, please refer to `Hardware-efficient variational
    quantum eigensolver for small molecules and quantum magnets <https://www.nature.com/articles/nature23879>`_.

    Args:
        n_qubits (int): total qubits number of this ansatz.
        depth (int): depth of ansatz.
        prefix (str): prefix of parameters. Default: ``''``.
        suffix (str): suffix of parameters. Default: ``''``.

    Examples:
        >>> from mindquantum.algorithm.nisq import RYLinear
        >>> RYLinear(3, 2, prefix='a').circuit
              ┏━━━━━━━━━━┓             ┏━━━━━━━━━━┓             ┏━━━━━━━━━━┓
        q0: ──┨ RY(a_p0) ┠───■─────────┨ RY(a_p3) ┠───■─────────┨ RY(a_p6) ┠───
              ┗━━━━━━━━━━┛   ┃         ┗━━━━━━━━━━┛   ┃         ┗━━━━━━━━━━┛
              ┏━━━━━━━━━━┓ ┏━┻━┓       ┏━━━━━━━━━━┓ ┏━┻━┓       ┏━━━━━━━━━━┓
        q1: ──┨ RY(a_p1) ┠─┨╺╋╸┠───■───┨ RY(a_p4) ┠─┨╺╋╸┠───■───┨ RY(a_p7) ┠───
              ┗━━━━━━━━━━┛ ┗━━━┛   ┃   ┗━━━━━━━━━━┛ ┗━━━┛   ┃   ┗━━━━━━━━━━┛
              ┏━━━━━━━━━━┓       ┏━┻━┓ ┏━━━━━━━━━━┓       ┏━┻━┓ ┏━━━━━━━━━━┓
        q2: ──┨ RY(a_p2) ┠───────┨╺╋╸┠─┨ RY(a_p5) ┠───────┨╺╋╸┠─┨ RY(a_p8) ┠───
              ┗━━━━━━━━━━┛       ┗━━━┛ ┗━━━━━━━━━━┛       ┗━━━┛ ┗━━━━━━━━━━┛
    """

    def __init__(self, n_qubits: int, depth: int, prefix: str = '', suffix: str = ''):
        """Construct ansatz."""
        Initializer.__init__(self, n_qubits, depth, prefix, suffix)
        Ansatz.__init__(self, 'RYLinear', n_qubits, depth)

    def _implement(self, depth):
        """Implement of RYLinear."""
        self._circuit += single_qubit_gate_layer(RY, self.n_qubits, pr_gen=self.pr_gen)
        ent = Circuit()
        for i in range(1, self.n_qubits):
            ent.x(i, i - 1)
        ent.barrier(show=False)
        for _ in range(depth):
            self._circuit += ent
            self._circuit += single_qubit_gate_layer(RY, self.n_qubits, pr_gen=self.pr_gen)


class RYFull(Ansatz, Initializer):
    r"""
    HEA with :class:`~.core.gates.RY` as single qubit gate and fully mapped CNOT gate as entanglement gate.

    .. image:: ./ansatz_images/RYFull.png
        :height: 180px

    For more information about this ansatz, please refer to `Challenges in the Use of
    Quantum Computing Hardware-Efficient Ansätze in Electronic Structure
    Theory <https://pubs.acs.org/doi/10.1021/acs.jpca.2c08430>`_.

    Args:
        n_qubits (int): total qubits number of this ansatz.
        depth (int): depth of ansatz.
        prefix (str): prefix of parameters. Default: ``''``.
        suffix (str): suffix of parameters. Default: ``''``.

    Examples:
        >>> from mindquantum.algorithm.nisq import RYFull
        >>> RYFull(3, 2, prefix='a').circuit
              ┏━━━━━━━━━━┓                   ┏━━━━━━━━━━┓                   ┏━━━━━━━━━━┓
        q0: ──┨ RY(a_p0) ┠───■─────■─────────┨ RY(a_p3) ┠───■─────■─────────┨ RY(a_p6) ┠───
              ┗━━━━━━━━━━┛   ┃     ┃         ┗━━━━━━━━━━┛   ┃     ┃         ┗━━━━━━━━━━┛
              ┏━━━━━━━━━━┓ ┏━┻━┓   ┃         ┏━━━━━━━━━━┓ ┏━┻━┓   ┃         ┏━━━━━━━━━━┓
        q1: ──┨ RY(a_p1) ┠─┨╺╋╸┠───╂─────■───┨ RY(a_p4) ┠─┨╺╋╸┠───╂─────■───┨ RY(a_p7) ┠───
              ┗━━━━━━━━━━┛ ┗━━━┛   ┃     ┃   ┗━━━━━━━━━━┛ ┗━━━┛   ┃     ┃   ┗━━━━━━━━━━┛
              ┏━━━━━━━━━━┓       ┏━┻━┓ ┏━┻━┓ ┏━━━━━━━━━━┓       ┏━┻━┓ ┏━┻━┓ ┏━━━━━━━━━━┓
        q2: ──┨ RY(a_p2) ┠───────┨╺╋╸┠─┨╺╋╸┠─┨ RY(a_p5) ┠───────┨╺╋╸┠─┨╺╋╸┠─┨ RY(a_p8) ┠───
              ┗━━━━━━━━━━┛       ┗━━━┛ ┗━━━┛ ┗━━━━━━━━━━┛       ┗━━━┛ ┗━━━┛ ┗━━━━━━━━━━┛
    """

    def __init__(self, n_qubits: int, depth: int, prefix: str = '', suffix: str = ''):
        """Construct ansatz."""
        Initializer.__init__(self, n_qubits, depth, prefix, suffix)
        Ansatz.__init__(self, 'RYFull', n_qubits, depth)

    def _implement(self, depth):
        """Implement of RYFull."""
        self._circuit += single_qubit_gate_layer(RY, self.n_qubits, pr_gen=self.pr_gen)
        ent = Circuit()
        for j in range(self.n_qubits - 1):
            for k in range(j + 1, self.n_qubits):
                ent.x(k, j)
        ent.barrier(show=False)
        for _ in range(depth):
            self._circuit += ent
            self._circuit += single_qubit_gate_layer(RY, self.n_qubits, pr_gen=self.pr_gen)


class RYRZFull(Ansatz, Initializer):
    r"""
    HEA with RY and RZ as single qubit gate and fully mapped CNOT gate as entangle gate.

    .. image:: ./ansatz_images/RYRZFull.png
        :height: 180px

    Args:
        n_qubits (int): total qubits number of this ansatz.
        depth (int): depth of ansatz.
        prefix (str): prefix of parameters. Default: ``''``.
        suffix (str): suffix of parameters. Default: ``''``.

    Examples:
        >>> from mindquantum.algorithm.nisq import RYRZFull
        >>> RYRZFull(3, 1, prefix='a').circuit
              ┏━━━━━━━━━━┓ ┏━━━━━━━━━━┓                   ┏━━━━━━━━━━┓ ┏━━━━━━━━━━┓
        q0: ──┨ RY(a_p0) ┠─┨ RZ(a_p3) ┠───■─────■─────────┨ RY(a_p6) ┠─┨ RZ(a_p9) ┠────
              ┗━━━━━━━━━━┛ ┗━━━━━━━━━━┛   ┃     ┃         ┗━━━━━━━━━━┛ ┗━━━━━━━━━━┛
              ┏━━━━━━━━━━┓ ┏━━━━━━━━━━┓ ┏━┻━┓   ┃         ┏━━━━━━━━━━┓ ┏━━━━━━━━━━━┓
        q1: ──┨ RY(a_p1) ┠─┨ RZ(a_p4) ┠─┨╺╋╸┠───╂─────■───┨ RY(a_p7) ┠─┨ RZ(a_p10) ┠───
              ┗━━━━━━━━━━┛ ┗━━━━━━━━━━┛ ┗━━━┛   ┃     ┃   ┗━━━━━━━━━━┛ ┗━━━━━━━━━━━┛
              ┏━━━━━━━━━━┓ ┏━━━━━━━━━━┓       ┏━┻━┓ ┏━┻━┓ ┏━━━━━━━━━━┓ ┏━━━━━━━━━━━┓
        q2: ──┨ RY(a_p2) ┠─┨ RZ(a_p5) ┠───────┨╺╋╸┠─┨╺╋╸┠─┨ RY(a_p8) ┠─┨ RZ(a_p11) ┠───
              ┗━━━━━━━━━━┛ ┗━━━━━━━━━━┛       ┗━━━┛ ┗━━━┛ ┗━━━━━━━━━━┛ ┗━━━━━━━━━━━┛
    """

    def __init__(self, n_qubits: int, depth: int, prefix: str = '', suffix: str = ''):
        """Construct ansatz."""
        Initializer.__init__(self, n_qubits, depth, prefix, suffix)
        Ansatz.__init__(self, 'RYRZFull', n_qubits, depth)

    def _implement(self, depth):
        """Implement of RYRZFull."""
        self._circuit += single_qubit_gate_layer(RY, self.n_qubits, pr_gen=self.pr_gen)
        self._circuit += single_qubit_gate_layer(RZ, self.n_qubits, pr_gen=self.pr_gen)
        ent = Circuit()
        for j in range(self.n_qubits - 1):
            for k in range(j + 1, self.n_qubits):
                ent.x(k, j)
        ent.barrier(show=False)
        for _ in range(depth):
            self._circuit += ent
            self._circuit += single_qubit_gate_layer(RY, self.n_qubits, pr_gen=self.pr_gen)
            self._circuit += single_qubit_gate_layer(RZ, self.n_qubits, pr_gen=self.pr_gen)


class RYCascade(Ansatz, Initializer):
    r"""
    HEA with RY as single qubit gate and two layers of linearly mapped CNOT gate as entangle gate.

    .. image:: ./ansatz_images/RYCascade.png
        :height: 180px

    For more information about this ansatz, please refer to `Challenges in the Use of
    Quantum Computing Hardware-Efficient Ansätze in Electronic Structure
    Theory <https://pubs.acs.org/doi/10.1021/acs.jpca.2c08430>`_.

    Args:
        n_qubits (int): total qubits number of this ansatz.
        depth (int): depth of ansatz.
        prefix (str): prefix of parameters. Default: ``''``.
        suffix (str): suffix of parameters. Default: ``''``.

    Examples:
        >>> from mindquantum.algorithm.nisq import RYCascade
        >>> RYCascade(3, 1, prefix='a').circuit
              ┏━━━━━━━━━━┓             ┏━━━━━━━━━━┓             ┏━━━━━━━━━━┓
        q0: ──┨ RY(a_p0) ┠───■─────────┨ RY(a_p3) ┠─────────■───┨ RY(a_p6) ┠───
              ┗━━━━━━━━━━┛   ┃         ┗━━━━━━━━━━┛         ┃   ┗━━━━━━━━━━┛
              ┏━━━━━━━━━━┓ ┏━┻━┓       ┏━━━━━━━━━━┓       ┏━┻━┓ ┏━━━━━━━━━━┓
        q1: ──┨ RY(a_p1) ┠─┨╺╋╸┠───■───┨ RY(a_p4) ┠───■───┨╺╋╸┠─┨ RY(a_p7) ┠───
              ┗━━━━━━━━━━┛ ┗━━━┛   ┃   ┗━━━━━━━━━━┛   ┃   ┗━━━┛ ┗━━━━━━━━━━┛
              ┏━━━━━━━━━━┓       ┏━┻━┓ ┏━━━━━━━━━━┓ ┏━┻━┓       ┏━━━━━━━━━━┓
        q2: ──┨ RY(a_p2) ┠───────┨╺╋╸┠─┨ RY(a_p5) ┠─┨╺╋╸┠───────┨ RY(a_p8) ┠───
              ┗━━━━━━━━━━┛       ┗━━━┛ ┗━━━━━━━━━━┛ ┗━━━┛       ┗━━━━━━━━━━┛
    """

    def __init__(self, n_qubits: int, depth: int, prefix: str = '', suffix: str = ''):
        """Construct ansatz."""
        Initializer.__init__(self, n_qubits, depth, prefix, suffix)
        Ansatz.__init__(self, 'RYCascade', n_qubits, depth)

    def _implement(self, depth):
        """Implement of RYCascade."""
        self._circuit += single_qubit_gate_layer(RY, self.n_qubits, pr_gen=self.pr_gen)
        ent1 = Circuit()
        for i in range(1, self.n_qubits):
            ent1.x(i, i - 1)
        ent2 = ent1.hermitian()
        ent1.barrier(show=False)
        ent2.barrier(show=False)
        for _ in range(depth):
            self._circuit += ent1
            self._circuit += single_qubit_gate_layer(RY, self.n_qubits, pr_gen=self.pr_gen)
            self._circuit += ent2
            self._circuit += single_qubit_gate_layer(RY, self.n_qubits, pr_gen=self.pr_gen)


class ASWAP(Ansatz, Initializer):
    r"""
    A swap like hardware efficient ansatz.

    .. image:: ./ansatz_images/ASWAP.png
        :height: 180px

    For more information about this ansatz, please refer to `Efficient symmetry-preserving
    state preparation circuits for the variational quantum eigensolver
    algorithm <https://www.nature.com/articles/s41534-019-0240-1>`_.

    Args:
        n_qubits (int): total qubits number of this ansatz.
        depth (int): depth of ansatz.
        prefix (str): prefix of parameters. Default: ``''``.
        suffix (str): suffix of parameters. Default: ``''``.

    Examples:
        >>> from mindquantum.algorithm.nisq import ASWAP
        >>> ASWAP(3, 1, prefix='a').circuit
              ┏━━━┓
        q0: ──┨╺╋╸┠───────────────────────────────────────────────────■──────────────────────────────↯─
              ┗━┳━┛                                                   ┃
                ┃   ┏━━━━━━━━━━┓ ┏━━━━━━━┓ ┏━━━━━━━━━━┓ ┏━━━━━━━━━┓ ┏━┻━┓ ┏━━━━━━━━━━┓ ┏━━━━━━━━━━━┓
        q1: ────■───┨ RZ(a_p0) ┠─┨ RZ(π) ┠─┨ RY(a_p1) ┠─┨ RY(π/2) ┠─┨╺╋╸┠─┨ RY(-π/2) ┠─┨ RY(-a_p1) ┠─↯─
                    ┗━━━━━━━━━━┛ ┗━━━━━━━┛ ┗━━━━━━━━━━┛ ┗━━━━━━━━━┛ ┗━━━┛ ┗━━━━━━━━━━┛ ┗━━━━━━━━━━━┛
        q2: ─────────────────────────────────────────────────────────────────────────────────────────↯─
                                       ┏━━━┓
        q0: ───────────────────────────┨╺╋╸┠───────────────────────────────────────────────────────↯─
                                       ┗━┳━┛
              ┏━━━━━━━━┓ ┏━━━━━━━━━━━┓   ┃   ┏━━━┓
        q1: ──┨ RZ(-π) ┠─┨ RZ(-a_p0) ┠───■───┨╺╋╸┠─────────────────────────────────────────────────↯─
              ┗━━━━━━━━┛ ┗━━━━━━━━━━━┛       ┗━┳━┛
                                               ┃   ┏━━━━━━━━━━┓ ┏━━━━━━━┓ ┏━━━━━━━━━━┓ ┏━━━━━━━━━┓
        q2: ───────────────────────────────────■───┨ RZ(a_p2) ┠─┨ RZ(π) ┠─┨ RY(a_p3) ┠─┨ RY(π/2) ┠─↯─
                                                   ┗━━━━━━━━━━┛ ┗━━━━━━━┛ ┗━━━━━━━━━━┛ ┗━━━━━━━━━┛
        q0: ────────────────────────────────────────────────────────────────────
                                                                        ┏━━━┓
        q1: ────■───────────────────────────────────────────────────────┨╺╋╸┠───
                ┃                                                       ┗━┳━┛
              ┏━┻━┓ ┏━━━━━━━━━━┓ ┏━━━━━━━━━━━┓ ┏━━━━━━━━┓ ┏━━━━━━━━━━━┓   ┃
        q2: ──┨╺╋╸┠─┨ RY(-π/2) ┠─┨ RY(-a_p3) ┠─┨ RZ(-π) ┠─┨ RZ(-a_p2) ┠───■─────
              ┗━━━┛ ┗━━━━━━━━━━┛ ┗━━━━━━━━━━━┛ ┗━━━━━━━━┛ ┗━━━━━━━━━━━┛
    """

    def __init__(self, n_qubits: int, depth: int, prefix: str = '', suffix: str = ''):
        """Construct ansatz."""
        Initializer.__init__(self, n_qubits, depth, prefix, suffix)
        Ansatz.__init__(self, 'ASWAP', n_qubits, depth)

    def _implement(self, depth):
        """Implement of ASWAP."""
        for _ in range(depth):
            self._circuit += self._aswap_unit()

    def _a_gate(self, q1, q2, phi, theta) -> Circuit:
        """Construct A Gate."""
        rot = Circuit().rz(phi, q2).rz(np.pi, q2).ry(theta, q2).ry(np.pi / 2, q2)
        rot_herm = rot.hermitian()
        circ = Circuit().x(q1, q2)
        circ += rot
        circ += X.on(q2, q1)
        circ += rot_herm
        circ += X.on(q1, q2)
        return circ

    def _aswap_unit(self) -> Circuit:
        """Construct a swap unit."""
        circ = Circuit()
        for n in [0, 1]:
            for i in range(n, self.n_qubits - 1, 2):
                circ += self._a_gate(i, i + 1, self.pr_gen.new(), self.pr_gen.new())
        return circ


class PCHeaXYZ1F(Ansatz, Initializer):
    r"""
    PCHeaXYZ1F ansatz.

    .. image:: ./ansatz_images/PCHeaXYZ1F.png
        :height: 180px

    For more information about this ansatz, please refer to `Physics-Constrained Hardware-Efficient
    Ansatz on Quantum Computers that is Universal, Systematically Improvable, and
    Size-consistent <https://arxiv.org/abs/2307.03563>`_.

    Args:
        n_qubits (int): total qubits number of this ansatz.
        depth (int): depth of ansatz.
        prefix (str): prefix of parameters. Default: ``''``.
        suffix (str): suffix of parameters. Default: ``''``.

    Examples:
        >>> from mindquantum.algorithm.nisq import PCHeaXYZ1F
        >>> PCHeaXYZ1F(3, 1).circuit
              ┏━━━━━━━━┓ ┏━━━━━━━━┓                 ┏━━━━━━━━━━━━━━━━━━┓
        q0: ──┨ RX(p0) ┠─┨ RY(p3) ┠─────────────────┨                  ┠────────────────────────────────↯─
              ┗━━━━━━━━┛ ┗━━━━━━━━┛                 ┃                  ┃
              ┏━━━━━━━━┓ ┏━━━━━━━━┓ ┏━━━━━━━━━━━━━┓ ┃ FSim(θ=p7, φ=p6) ┃ ┏━━━━━━━━━━━━┓
        q1: ──┨ RX(p1) ┠─┨ RY(p4) ┠─┨ RY(-1/2*p6) ┠─┨                  ┠─┨ RY(1/2*p6) ┠─────────────────↯─
              ┗━━━━━━━━┛ ┗━━━━━━━━┛ ┗━━━━━━━━━━━━━┛ ┗━━━━━━━━━━━━━━━━━━┛ ┗━━━━━━━━━━━━┛
              ┏━━━━━━━━┓ ┏━━━━━━━━┓                                                     ┏━━━━━━━━━━━━━┓
        q2: ──┨ RX(p2) ┠─┨ RY(p5) ┠─────────────────────────────────────────────────────┨ RY(-1/2*p8) ┠─↯─
              ┗━━━━━━━━┛ ┗━━━━━━━━┛                                                     ┗━━━━━━━━━━━━━┛
        q0: ─────────────────────────────────────────────────────────────────────────────────────────↯─
              ┏━━━━━━━━━━━━━━━━━━┓                                            ┏━━━━━━━━━━━━━━━━━━━━┓
        q1: ──┨                  ┠────────────────────────────────────────────┨                    ┠─↯─
              ┃                  ┃                                            ┃                    ┃
              ┃ FSim(θ=p9, φ=p8) ┃ ┏━━━━━━━━━━━━┓ ┏━━━━━━━━━┓ ┏━━━━━━━━━━━━━┓ ┃ FSim(θ=-p9, φ=-p8) ┃
        q2: ──┨                  ┠─┨ RY(1/2*p8) ┠─┨ RZ(p10) ┠─┨ RY(-1/2*p8) ┠─┨                    ┠─↯─
              ┗━━━━━━━━━━━━━━━━━━┛ ┗━━━━━━━━━━━━┛ ┗━━━━━━━━━┛ ┗━━━━━━━━━━━━━┛ ┗━━━━━━━━━━━━━━━━━━━━┛
                                             ┏━━━━━━━━━━━━━━━━━━━━┓                ┏━━━━━━━━━┓ ┏━━━━━━━━━┓
        q0: ─────────────────────────────────┨                    ┠────────────────┨ RY(-p3) ┠─┨ RX(-p0) ┠───
                                             ┃                    ┃                ┗━━━━━━━━━┛ ┗━━━━━━━━━┛
                             ┏━━━━━━━━━━━━━┓ ┃ FSim(θ=-p7, φ=-p6) ┃ ┏━━━━━━━━━━━━┓ ┏━━━━━━━━━┓ ┏━━━━━━━━━┓
        q1: ─────────────────┨ RY(-1/2*p6) ┠─┨                    ┠─┨ RY(1/2*p6) ┠─┨ RY(-p4) ┠─┨ RX(-p1) ┠───
                             ┗━━━━━━━━━━━━━┛ ┗━━━━━━━━━━━━━━━━━━━━┛ ┗━━━━━━━━━━━━┛ ┗━━━━━━━━━┛ ┗━━━━━━━━━┛
              ┏━━━━━━━━━━━━┓                                                       ┏━━━━━━━━━┓ ┏━━━━━━━━━┓
        q2: ──┨ RY(1/2*p8) ┠───────────────────────────────────────────────────────┨ RY(-p5) ┠─┨ RX(-p2) ┠───
              ┗━━━━━━━━━━━━┛                                                       ┗━━━━━━━━━┛ ┗━━━━━━━━━┛
    """

    def __init__(self, n_qubits: int, depth: int, prefix: str = '', suffix: str = ''):
        """Construct ansatz."""
        Initializer.__init__(self, n_qubits, depth, prefix, suffix)
        Ansatz.__init__(self, 'PCHeaXYZ1F', n_qubits, depth)

    def _implement(self, depth):
        """Implement of PCHeaXYZ1F."""
        for _ in range(depth):
            self._circuit += self._gate_unit()

    def _half(self) -> Circuit:
        """Construct first half."""
        circ = Circuit()
        circ += single_qubit_gate_layer(RX, self.n_qubits, pr_gen=self.pr_gen)
        circ += single_qubit_gate_layer(RY, self.n_qubits, pr_gen=self.pr_gen)
        circ.barrier(False)
        for i in range(1, self.n_qubits):
            phi = self.pr_gen.new()
            theta = self.pr_gen.new()
            circ.ry(-0.5 * phi, i)
            circ += FSim(theta, phi).on([i - 1, i])
            circ.ry(0.5 * phi, i)
            circ.barrier(False)
        return circ

    def _gate_unit(self) -> Circuit:
        """Construct a swap unit."""
        circ = Circuit()
        half = self._half()
        circ += half
        circ.rz(self.pr_gen.new(), self.n_qubits - 1)
        circ += half.hermitian()
        return circ


class PCHeaXYZ2F(Ansatz, Initializer):
    r"""
    PCHeaXYZ2F ansatz.

    .. image:: ./ansatz_images/PCHeaXYZ2F.png
        :height: 180px

    For more information about this ansatz, please refer to `Physics-Constrained Hardware-Efficient
    Ansatz on Quantum Computers that is Universal, Systematically Improvable, and
    Size-consistent <https://arxiv.org/abs/2307.03563>`_.

    Args:
        n_qubits (int): total qubits number of this ansatz.
        depth (int): depth of ansatz.
        prefix (str): prefix of parameters. Default: ``''``.
        suffix (str): suffix of parameters. Default: ``''``.

    Examples:
        >>> from mindquantum.algorithm.nisq import PCHeaXYZ2F
        >>> PCHeaXYZ2F(3, 1).circuit
              ┏━━━━━━━━┓ ┏━━━━━━━━┓                 ┏━━━━━━━━━━━━━━━━━━┓
        q0: ──┨ RX(p0) ┠─┨ RY(p3) ┠─────────────────┨                  ┠────────────────────────────────↯─
              ┗━━━━━━━━┛ ┗━━━━━━━━┛                 ┃                  ┃
              ┏━━━━━━━━┓ ┏━━━━━━━━┓ ┏━━━━━━━━━━━━━┓ ┃ FSim(θ=p7, φ=p6) ┃ ┏━━━━━━━━━━━━┓
        q1: ──┨ RX(p1) ┠─┨ RY(p4) ┠─┨ RY(-1/2*p6) ┠─┨                  ┠─┨ RY(1/2*p6) ┠─────────────────↯─
              ┗━━━━━━━━┛ ┗━━━━━━━━┛ ┗━━━━━━━━━━━━━┛ ┗━━━━━━━━━━━━━━━━━━┛ ┗━━━━━━━━━━━━┛
              ┏━━━━━━━━┓ ┏━━━━━━━━┓                                                     ┏━━━━━━━━━━━━━┓
        q2: ──┨ RX(p2) ┠─┨ RY(p5) ┠─────────────────────────────────────────────────────┨ RY(-1/2*p8) ┠─↯─
              ┗━━━━━━━━┛ ┗━━━━━━━━┛                                                     ┗━━━━━━━━━━━━━┛
                                                  ┏━━━━━━━━━┓
        q0: ──────────────────────────────────────┨ RZ(p10) ┠────────────────────────────────────────↯─
                                                  ┗━━━━━━━━━┛
              ┏━━━━━━━━━━━━━━━━━━┓                ┏━━━━━━━━━┓                 ┏━━━━━━━━━━━━━━━━━━━━┓
        q1: ──┨                  ┠────────────────┨ RZ(p11) ┠─────────────────┨                    ┠─↯─
              ┃                  ┃                ┗━━━━━━━━━┛                 ┃                    ┃
              ┃ FSim(θ=p9, φ=p8) ┃ ┏━━━━━━━━━━━━┓ ┏━━━━━━━━━┓ ┏━━━━━━━━━━━━━┓ ┃ FSim(θ=-p9, φ=-p8) ┃
        q2: ──┨                  ┠─┨ RY(1/2*p8) ┠─┨ RZ(p12) ┠─┨ RY(-1/2*p8) ┠─┨                    ┠─↯─
              ┗━━━━━━━━━━━━━━━━━━┛ ┗━━━━━━━━━━━━┛ ┗━━━━━━━━━┛ ┗━━━━━━━━━━━━━┛ ┗━━━━━━━━━━━━━━━━━━━━┛
                                             ┏━━━━━━━━━━━━━━━━━━━━┓                ┏━━━━━━━━━┓ ┏━━━━━━━━━┓
        q0: ─────────────────────────────────┨                    ┠────────────────┨ RY(-p3) ┠─┨ RX(-p0) ┠───
                                             ┃                    ┃                ┗━━━━━━━━━┛ ┗━━━━━━━━━┛
                             ┏━━━━━━━━━━━━━┓ ┃ FSim(θ=-p7, φ=-p6) ┃ ┏━━━━━━━━━━━━┓ ┏━━━━━━━━━┓ ┏━━━━━━━━━┓
        q1: ─────────────────┨ RY(-1/2*p6) ┠─┨                    ┠─┨ RY(1/2*p6) ┠─┨ RY(-p4) ┠─┨ RX(-p1) ┠───
                             ┗━━━━━━━━━━━━━┛ ┗━━━━━━━━━━━━━━━━━━━━┛ ┗━━━━━━━━━━━━┛ ┗━━━━━━━━━┛ ┗━━━━━━━━━┛
              ┏━━━━━━━━━━━━┓                                                       ┏━━━━━━━━━┓ ┏━━━━━━━━━┓
        q2: ──┨ RY(1/2*p8) ┠───────────────────────────────────────────────────────┨ RY(-p5) ┠─┨ RX(-p2) ┠───
              ┗━━━━━━━━━━━━┛                                                       ┗━━━━━━━━━┛ ┗━━━━━━━━━┛
    """

    def __init__(self, n_qubits: int, depth: int, prefix: str = '', suffix: str = ''):
        """Construct ansatz."""
        Initializer.__init__(self, n_qubits, depth, prefix, suffix)
        Ansatz.__init__(self, 'PCHeaXYZ2F', n_qubits, depth)

    def _implement(self, depth):
        """Implement of PCHeaXYZ2F."""
        for _ in range(depth):
            self._circuit += self._gate_unit()

    def _half(self) -> Circuit:
        """Construct first half."""
        circ = Circuit()
        circ += single_qubit_gate_layer(RX, self.n_qubits, pr_gen=self.pr_gen)
        circ += single_qubit_gate_layer(RY, self.n_qubits, pr_gen=self.pr_gen)
        circ.barrier(False)
        for i in range(1, self.n_qubits):
            phi = self.pr_gen.new()
            theta = self.pr_gen.new()
            circ.ry(-0.5 * phi, i)
            circ += FSim(theta, phi).on([i - 1, i])
            circ.ry(0.5 * phi, i)
            circ.barrier(False)
        return circ

    def _gate_unit(self) -> Circuit:
        """Construct a swap unit."""
        circ = Circuit()
        half = self._half()
        circ += half
        circ += single_qubit_gate_layer(RZ, self.n_qubits, pr_gen=self.pr_gen)
        circ += half.hermitian()
        return circ
