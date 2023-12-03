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
"""Topology editor."""
import time
from typing import Callable, List, NewType, Tuple, Union

import numpy as np

from mindquantum.device.topology import QubitNode, QubitsTopology
from mindquantum.utils.f import random_rng

Edge = NewType('Edge', Tuple[QubitNode, QubitNode])
Edgeable = NewType('Edgeable', Tuple[Union[int, QubitNode], Union[int, QubitNode]])


class TopoEditor:
    def __init__(self, topo: QubitsTopology) -> None:
        self.selected_node: List[QubitNode] = []
        self.selected_edge: List[Edge] = []
        self.topo = topo

    def top_of_node(self):
        if not self.selected_node:
            return None
        return min(i.poi_y for i in self.selected_node)

    def bottom_of_node(self):
        if not self.selected_node:
            return None
        return max(i.poi_y for i in self.selected_node)

    def left_of_node(self):
        if not self.selected_node:
            return None
        return min(i.poi_x for i in self.selected_node)

    def right_of_node(self):
        if not self.selected_node:
            return None
        return max(i.poi_x for i in self.selected_node)

    def clean_selected_node(self):
        self.selected_node = []
        return self

    def clean_selected_edge(self):
        self.selected_edge = []
        return self

    def select_node(self, qids: Union[int, QubitNode, List[Union[int, QubitNode]]], append: bool = False):
        if not append:
            self.clean_selected_node()
        if isinstance(qids, (int, QubitNode)):
            qids = [qids]
        self.selected_node.extend([i if isinstance(i, QubitNode) else self.topo[i] for i in qids])
        return self

    def select_edge(self, edges: Union[Edgeable, List[Edgeable]], append: bool = False):
        if not append:
            self.clean_selected_edge()
        if isinstance(edges, tuple):
            edges = [edges]
        for e1, e2 in edges:
            if isinstance(e1, int):
                e1 = self.topo[e1]
            if isinstance(e2, int):
                e2 = self.topo[e2]
            self.selected_edge.append((e1, e2))
        return self

    def select_node_with(self, callback: Callable[[QubitNode], bool], append: bool = False):
        if not append:
            self.clean_selected_node()
        self.selected_node.extend([i for i in self.topo.qubits.values() if callback(i)])
        return self

    def select_edge_with(self, callback: Callable[[QubitNode, QubitNode], bool], append: bool = False):
        if not append:
            self.clean_selected_edge()
        for e1, e2 in self.topo.edges_with_id():
            e1, e2 = self.topo[e1], self.topo[e2]
            if callback(e1, e2):
                self.selected_edge.append((e1, e2))
        return self

    def select_edge_in_direction(self, ang: float, atol: float = 1e-6, append: bool = False):
        ang = ang % (2 * np.pi)

        def ang_close(e1: QubitNode, e2: QubitNode):
            e1_x, e1_y = e1.poi_x, e1.poi_y
            e2_x, e2_y = e2.poi_x, e2.poi_y
            ang0 = np.angle(e2_x - e1_x + 1j * (e2_y - e1_y))
            return np.abs(ang0 - ang) / ang < atol

        return self.select_edge_with(ang_close, append)

    def select_edge_in_current_node(self, append: bool = False):
        if not append:
            self.clean_selected_edge()
        for e1, e2 in self.topo.edges_with_id():
            e1, e2 = self.topo[e1], self.topo[e2]
            if e1 in self.selected_node and e2 in self.selected_node:
                self.selected_edge.append((e1, e2))
        return self

    def select_node_range(self, start: int, stop: int, append: bool = False):
        return self.select_node_with(lambda i: start <= i.qubit_id < stop, append)

    def select_all_node(self):
        return self.select_node_with(lambda _: True, False)

    def shift_node_poi(self, x, y):
        for i in self.selected_node:
            i.set_poi(i.poi_x + x, i.poi_y + y)
        return self

    def top_align(self):
        if len(self.selected_node) < 2:
            return self
        top = self.top_of_node()
        for i in self.selected_node:
            i.set_poi(i.poi_x, top)
        return self

    def bottom_align(self):
        if len(self.selected_node) < 2:
            return self
        bottom = self.bottom_of_node()
        for i in self.selected_node:
            i.set_poi(i.poi_x, bottom)
        return self

    def left_align(self):
        if len(self.selected_node) < 2:
            return self
        left = self.left_of_node()
        for i in self.selected_node:
            i.set_poi(left, i.poi_y)
        return self

    def right_align(self):
        if len(self.selected_node) < 2:
            return self
        right = self.right_of_node()
        for i in self.selected_node:
            i.set_poi(right, i.poi_y)
        return self

    def h_expand(self, dist: int = 1):
        if len(self.selected_node) < 2:
            return self
        sorted_qubit_node = sorted(self.selected_node, key=lambda x: x.poi_x)
        curr_x = sorted_qubit_node[0].poi_x
        for i in sorted_qubit_node[1:]:
            curr_x += dist
            i.set_poi(curr_x, i.poi_y)
        return self

    def random_distribution(self, width: int = 10, high: int = 10, seed=None):
        rng = random_rng(seed)
        poi = rng.random(size=(self.topo.size(), 2))
        poi[:, 0] *= width
        poi[:, 1] *= high
        for position, node in zip(poi, self.topo.qubits.values()):
            node.set_poi(*position)
        return self

    def __calc_distance__(self, i: QubitNode, j: QubitNode, round=6):
        v1 = np.array([i.poi_x, i.poi_y])
        v2 = np.array([j.poi_x, j.poi_y])
        return np.round(((v1 - v2) ** 2).sum(), round)

    def disconnect_selected_node(self):
        if len(self.selected_node) < 2:
            return self
        for i in self.selected_node:
            for j in self.selected_node:
                if i.qubit_id <= j.qubit_id:
                    continue
                i < j
        return self

    def connect_selected_nearest_neighbor_node(self):
        if len(self.selected_node) < 2:
            return self
        for i in self.selected_node:
            dist = []
            min_d = np.inf
            for j in self.selected_node:
                if i.qubit_id == j.qubit_id:
                    continue
                d = self.__calc_distance__(i, j)
                dist.append((d, j))
                min_d = min(min_d, d)
            for j in dist:
                if j[0] == min_d:
                    i >> j[1]
        return self

    def set_node_color(self, color: str):
        for i in self.selected_node:
            i.set_color(color)
        return self

    def set_edge_color(self, color: str):
        for e1, e2 in self.selected_edge:
            self.topo.set_edge_color(e1.qubit_id, e2.qubit_id, color)
        return self


if __name__ == "__main__":
    from mindquantum.device import GridQubits

    # topo = GridQubits(3, 3)
    # te = TopoEditor(topo)
    # te.random_distribution(5, 5, seed=42)
    # te.select_node([4, 8, 7]).top_align().h_expand()
    # te.select_node([6, 3, 1]).h_expand().bottom_align()
    # te.select_node([2, 5, 0]).left_align()
    # te.select_node([0, 1]).right_align()
    # te.select_node([0, 5]).top_align()
    # te.select_node([1, 3, 4, 5, 7]).disconnect_selected_node()
    # topo.set_edge_color(7, 8, '#12ab45')
    # topo.show()

    class Kagome:
        def __init__(self, n_layer: int = 1):
            self.n_layer = n_layer
            self.n_long = list(range(self.n_layer, self.n_layer * 2 + 1))
            self.n_long = self.n_long + self.n_long[-2::-1]
            self.n_short = list(range(2 * (self.n_layer + 1), 4 * self.n_layer + 1, 2))
            self.n_short = self.n_short + self.n_short[::-1]
            self.n_qubits = sum(self.n_long + self.n_short)
            self.topo = QubitsTopology([QubitNode(i) for i in range(self.n_qubits)])
            self.place_qubits()
            TopoEditor(self.topo).select_all_node().connect_selected_nearest_neighbor_node()

        def place_qubits(self):
            current_id = 0
            end_id = 0
            te = TopoEditor(self.topo)
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

    kagome = Kagome(3)
    te = TopoEditor(kagome.topo)
    te.select_node([35, 36, 44, 45, 53, 54]).set_node_color('#aaa300').select_edge_in_current_node().set_edge_color(
        '#00ffff'
    )
    te.select_edge_in_direction(np.pi / 3).set_edge_color('#ff00ff')
    te.select_edge_in_direction(2 * np.pi / 3).set_edge_color('#ff0000')
    kagome.topo.show()
