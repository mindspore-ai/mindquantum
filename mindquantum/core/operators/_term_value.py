#   Copyright 2022 <Huawei Technologies Co., Ltd>
#
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

"""Module used for (temporary) compatibility with Python terms operators."""

from ...mqbackend import TermValue as TermValue_

term_map = {
    0: TermValue_.a,
    1: TermValue_.adg,
    'a': TermValue_.a,
    'adg': TermValue_.adg,
    '': TermValue_.a,
    '^': TermValue_.adg,
    'X': TermValue_.X,
    'Y': TermValue_.Y,
    'Z': TermValue_.Z,
    'I': TermValue_.I,
    'x': TermValue_.X,
    'y': TermValue_.Y,
    'z': TermValue_.Z,
    'i': TermValue_.I,
    TermValue_.a: 0,
    TermValue_.adg: 1,
    TermValue_.X: 'X',
    TermValue_.Y: 'Y',
    TermValue_.Z: 'Z',
    TermValue_.I: 'I',
}


class TermValue__:
    """Bind TermValue to python."""

    # pylint: disable=invalid-name
    @property
    def I(self):  # noqa: E743,E741,N802
        """Bind pauli I operator."""
        return TermValue_.I

    @property
    def X(self):  # noqa: N802
        """Bind pauli X operator."""
        return TermValue_.X

    @property
    def Y(self):  # noqa: N802
        """Bind pauli Y operator."""
        return TermValue_.Y

    @property
    def Z(self):  # noqa: N802
        """Bind pauli Z operator."""
        return TermValue_.Z

    @property
    def a(self):
        """Bind annihilation operator."""
        return TermValue_.a

    @property
    def adg(self):
        """Bind creation operator."""
        return TermValue_.adg

    def __getitem__(self, val):
        """Get operator."""
        return term_map[val]


TermValue = TermValue__()
