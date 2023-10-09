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
"""MindSpore Quantum dtype module."""
# pylint: disable=import-error,invalid-name
from mindquantum._math.ops import f_term_value, p_term_value

term_map = {
    0: f_term_value.a,
    1: f_term_value.adg,
    'a': f_term_value.a,
    'adg': f_term_value.adg,
    '': f_term_value.a,
    '^': f_term_value.adg,
    'X': p_term_value.X,
    'Y': p_term_value.Y,
    'Z': p_term_value.Z,
    'x': p_term_value.X,
    'y': p_term_value.Y,
    'z': p_term_value.Z,
    f_term_value.a: 0,
    f_term_value.adg: 1,
    f_term_value.I: 'I',
    p_term_value.X: 'X',
    p_term_value.Y: 'Y',
    p_term_value.Z: 'Z',
    p_term_value.I: 'I',
}


class TermValue__:
    """Bind TermValue to python."""

    @property
    def fermion_I(self):  # noqa: E743,E741,N802
        """Bind pauli I operator."""
        return f_term_value.I

    @property
    def pauli_I(self):  # noqa: E743,E741,N802
        """Bind pauli I operator."""
        return p_term_value.I

    @property
    def X(self):  # noqa: N802
        """Bind pauli X operator."""
        return p_term_value.X

    @property
    def Y(self):  # noqa: N802
        """Bind pauli Y operator."""
        return p_term_value.Y

    @property
    def Z(self):  # noqa: N802
        """Bind pauli Z operator."""
        return p_term_value.Z

    @property
    def a(self):
        """Bind annihilation operator."""
        return f_term_value.a

    @property
    def adg(self):
        """Bind creation operator."""
        return f_term_value.adg

    def __getitem__(self, val):
        """Get operator."""
        return term_map[val]


TermValue = TermValue__()
