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
"""Different qubit topology lattice."""

import numpy as np

from mindquantum.device.topo_editor import TopoEditor
from mindquantum.device.topology import QubitNode, QubitsTopology
from mindquantum.utils.type_value_check import _check_int_type


class LinearQubits(QubitsTopology):
    """
    Linear qubit topology.

    Args:
        n_qubits (int): total qubit size.

    Examples:
        >>> from mindquantum.device import LinearQubits
        >>> topology = LinearQubits(5)
        >>> topology.is_coupled_with(0, 1)
        True
        >>> topology.is_coupled_with(0, 2)
        False
    """

    def __init__(self, n_qubits: int):
        """Initialize a linear topology."""
        _check_int_type("n_qubits", n_qubits)
        super().__init__([QubitNode(i) for i in range(n_qubits)])
        TopoEditor(self).select_all_node().h_expand().connect_selected_nearest_neighbor_node()


class GridQubits(QubitsTopology):
    """
    Grid qubit topology.

    Args:
        n_row (int): how many rows of your grid qubits.
        n_col (int): how many columns of your grid quits.

    Examples:
        >>> from mindquantum.device import GridQubits
        >>> topology = GridQubits(2, 3)
        >>> topology.n_row()
        2
    """

    def __init__(self, n_row: int, n_col: int) -> None:
        """Initialize a grid topology with row and col number."""
        _check_int_type("n_row", n_row)
        _check_int_type("n_col", n_col)
        self.n_row_ = n_row
        self.n_col_ = n_col
        super().__init__([QubitNode(i) for i in range(self.n_row() * self.n_col())])
        te = TopoEditor(self)
        for i in range(self.n_row()):
            te.select_node_range(i * self.n_col(), (i + 1) * self.n_col())
            te.h_expand().shift_node_poi(0, i)
        te.select_all_node().connect_selected_nearest_neighbor_node()

    def n_col(self) -> int:
        """
        Get column number.

        Returns:
            int, the column number.

        Examples:
            >>> from mindquantum.device import GridQubits
            >>> topology = GridQubits(2, 3)
            >>> topology.n_col()
            3
        """
        return self.n_col_

    def n_row(self) -> int:
        """
        Get row number.

        Returns:
            int, the row number.

        Examples:
            >>> from mindquantum.device import GridQubits
            >>> topology = GridQubits(2, 3)
            >>> topology.n_row()
            2
        """
        return self.n_row_


class Kagome(QubitsTopology):
    def __init__(self, n_layer: int = 1):
        self.n_layer = n_layer
        self.n_long = list(range(self.n_layer, self.n_layer * 2 + 1))
        self.n_long = self.n_long + self.n_long[-2::-1]
        self.n_short = list(range(2 * (self.n_layer + 1), 4 * self.n_layer + 1, 2))
        self.n_short = self.n_short + self.n_short[::-1]
        self.n_qubits = sum(self.n_long + self.n_short)
        super().__init__([QubitNode(i) for i in range(self.n_qubits)])
        self.place_qubits()
        TopoEditor(self).select_all_node().connect_selected_nearest_neighbor_node()

    def place_qubits(self):
        current_id = 0
        end_id = 0
        te = TopoEditor(self)
        poi_x, poi_y = 0, 0
        n_line = len(self.n_long) + len(self.n_short)
        for i in range(n_line):
            current_id = end_id
            end_id, dis = current_id + (self.n_short if i & 1 else self.n_long)[i // 2], (1 if i & 1 else 2)
            te.select_node_range(current_id, end_id).h_expand(dis).shift_node_poi(poi_x, poi_y)
            if i < n_line // 2:
                poi_x += 0.5 * (i & 1) - 1.5 * (i & 1 == 0)
            else:
                poi_x += 1.5 * (i & 1) - 0.5 * (i & 1 == 0)
            poi_y += np.sqrt(3) / 2


class HeavyHexgon(QubitsTopology):
    pass


class TriangularLattice(QubitsTopology):
    pass
