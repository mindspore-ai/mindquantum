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
"""Method to draw a qubit topology."""
from typing import Dict, Tuple, Union

from ...core.circuit import Circuit
from ...device import QubitNode, QubitsTopology
from ._config import _topology_default_style
from .circuit_svg_drawer import Circle, Line, Rect, SVGContainer, Text, box, super_align


def _ext_edges(circ: Circuit):
    out = []
    for g in circ:
        all_qubits = g.obj_qubits + g.ctrl_qubits
        all_qubits.sort()
        if len(all_qubits) == 2:
            out.append((all_qubits[0], all_qubits[1]))
        elif len(all_qubits) > 2:
            raise ValueError(f"gate {g} too big, need to do decomposition first.")
    return set(out)


class SVGQNode(SVGContainer):
    """Qubit node on topology svg graph."""

    def __init__(self, qnode: QubitNode, svg_config):
        """Initialize a svg qubit node."""
        super().__init__()
        q_circle = Circle(0, 0, svg_config['r']).fill(qnode.color)
        q_id = Text(0, 0, str(qnode.qubit_id)).fill(svg_config['id_color']).font_size(svg_config['id_fs'])
        self.add(q_circle)
        self.add(q_id)
        self.shift(qnode.poi_x * svg_config['f'], qnode.poi_y * svg_config['f'])


# pylint: disable=too-many-locals
def draw_topology(
    topo: QubitsTopology,
    circuit: Circuit = None,
    style: Dict = None,
    edge_color: Union[str, Dict[Tuple[int, int], str]] = None,
):
    """
    Draw a qubit topology as a svg picture.

    Args:
        topo (:class:`.device.QubitsTopology`): The qubit topology.
        circuit (:class:`~.core.circuit.Circuit`): The given quantum circuit you want to execute on
            given qubit topology. Default: ``None``.
        style (Dict): The picture style configuration. Default: ``None``.
        edge_color (Union[str, Dict[Tuple[int, int], str]]): The color of edge. If it is a color string,
            every edge will display as this color. It can also be a dict with edge as key and color string as
            value.
    """
    if style is None:
        style = _topology_default_style
    f = style['f']
    if circuit is None:
        selected = []
    else:
        selected = _ext_edges(circ=circuit)
    for q1, q2 in selected:
        if not topo.is_coupled_with(q1, q2):
            raise RuntimeError(f"Failed to execute circuit: qubit {q1} is not coupled with {q2}.")
    qubits_view = SVGContainer()
    for q_id in topo.all_qubit_id():
        qubits_view.add(SVGQNode(topo[q_id], style))

    edges_view = SVGContainer()
    if edge_color is None:
        edge_color = style['couple_color']
    if isinstance(edge_color, str):
        edge_color = {i: edge_color for i in topo.edges_with_id()}
    for id1, id2 in topo.edges_with_id():
        q1, q2 = topo[id1], topo[id2]
        x1, y1 = q1.poi_x, q1.poi_y
        x2, y2 = q2.poi_x, q2.poi_y
        line = Line(x1 * f, x2 * f, y1 * f, y2 * f)
        color = edge_color.get((id1, id2), style['couple_color'])
        if (id1, id2) in selected or (id2, id1) in selected:
            color = style['selected_edge_c']
        line.stroke(color).stroke_width(style['couple_w'])
        edges_view.add(line)

    qubits_box = box(qubits_view)
    background = Rect(0, 0, qubits_box['width'], qubits_box['height'])
    background.fill("#ffffff")

    super_align(background, qubits_view, 'middle', 'middle', 'h')
    super_align(background, qubits_view, 'middle', 'middle', 'v')
    new_q_box = box(qubits_view)

    edges_view.shift(new_q_box['left'] - qubits_box['left'], new_q_box['top'] - qubits_box['top'])
    view = SVGContainer()
    view.add(background)
    view.add(edges_view)
    view.add(qubits_view)
    return view
