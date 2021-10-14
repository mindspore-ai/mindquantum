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
"""Projector module."""

import re


def _check_projector_str(proj):
    if not isinstance(proj, str):
        raise TypeError(f"Projector requires a string, but get {type(proj)}!")
    for i in proj:
        if i not in ['0', '1', 'I']:
            raise ValueError(
                f"Unkonw character '{i}' for a projector. Projector must onstructed by '0', '1' and 'I'."
            )


class Projector:
    r"""
    Projector operator.

    For a projector shown as below:

    .. math::
        \left|01\right>\left<01\right|\otimes I^2

    The string format would be '01II'.

    Note:
        The lower index qubit is at the right end of string format of bra and ket.

    Args:
        proj (str): The string format of the projector.

    Examples:
        >>> from mindquantum.gate import Projector
        >>> p = Projector('II010')
        >>> p
        I2 ⊗ ¦010⟩⟨010¦
    """
    def __init__(self, proj):
        _check_projector_str(proj)
        self.proj = proj
        self.n_qubits = len(proj)

    def __str__(self):
        res = re.split(r'(\d+)', self.proj)
        out = []
        for i in res:
            if i.isalpha():
                out.append(f'I{len(i)}' if len(i) > 1 else 'I')
            if i.isdigit():
                out.append(f'¦{i}⟩⟨{i}¦')
        return ' ⊗ '.join(out)

    def __repr__(self):
        return self.__str__()

    def mindspore_data(self):
        """
        Generate projector information for PQC operator.
        """
        m_data = {'projectors': self.proj}
        return m_data
