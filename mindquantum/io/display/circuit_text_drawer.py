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
"""Text draw a circuit."""
# pylint: disable=invalid-name,too-many-arguments,non-parent-init-called,super-init-not-called,too-many-locals
# pylint: disable=too-many-branches,too-many-statements
from typing import List

import numpy as np

from mindquantum.core import (
    BarrierGate,
    BasicGate,
    Circuit,
    CNOTGate,
    GroupedPauli,
    GroupedPauliChannel,
    Measure,
    NoiseGate,
    PauliChannel,
    SWAPGate,
    XGate,
)
from mindquantum.io.display._config import TextCircConfig
from mindquantum.io.display.circuit_text_drawer_helper import (
    BasicObj,
    BoxStyle,
    Container,
    Frame,
    HLine,
    ObjectEditor,
    Rect,
    Text,
    VLine,
    fix_line_rec_cross,
    fix_lines_cross,
)


class QubitLine(Container):
    """Qubit Line object."""

    def __init__(self, qids: List[int], style: TextCircConfig, qubit_map=None):
        """Construct a qubit line."""
        self.qids = qids
        self.qids_map = {j: i for i, j in enumerate(self.qids)}
        if qubit_map is None:
            self.text_qubits = [Text(f"q{i}: ") for i in self.qids]
        else:
            self.text_qubits = [Text(f"q{qubit_map[i]}: ") for i in self.qids]
        ObjectEditor.batch_ops(self.text_qubits, ObjectEditor.right_align)
        ObjectEditor.v_expand(self.text_qubits, style.qubit_line_v_dist)
        self.text_container = Container(self.text_qubits)
        self.lines = {
            i: HLine(thickness=style.qubit_line_thickness, line_style=TextCircConfig.qubit_line_style)
            for i in self.qids
        }
        ObjectEditor.v_expand(list(self.lines.values()), style.qubit_line_v_dist)
        self.lines_container = Container(list(self.lines.values()))
        ObjectEditor.to_right(self.text_container, self.lines_container)
        super().__init__([self.text_container, self.lines_container])
        self.left_top_origin()

    def get_qline(self, idx: int) -> HLine:
        """Get qubit line with given qubit index."""
        return self.lines[idx]

    def extend_line_to(self, x_poi: float):
        """Extend the qubit line to given x position."""
        length = x_poi - self.lines[self.qids[0]].left() + 1
        for i in self.lines.values():
            i.set_length(length)


class SimpleGate(Container):
    """Basic simple gate style."""

    def __init__(self, gate_name, obj_qubits, ctrl_qubits, ql: QubitLine, style: TextCircConfig):
        """Construct a simple gate."""
        self.style = style
        self.obj_qubits = obj_qubits
        self.ctrl_qubits = ctrl_qubits
        self.min_obj = min(obj_qubits)
        self.max_obj = max(obj_qubits)
        self.text = Text(gate_name, str_style=style.simple_gate_str_style)
        self.box = Rect(
            width=self.text.get_width() + 2 * style.simple_gate_width_padding,
            high=self.__calc_high__(ql.qids_map),
            box_style=getattr(BoxStyle, style.simple_gate_box_style),
        )
        for i in ql.lines.values():
            fix_line_rec_cross(self.box, i)
        ObjectEditor.center_align(self.box, self.text)
        all_elems = []
        all_elems.extend(self.handle_ctrl_bar(ql))
        all_elems.extend([self.box, self.text])
        all_elems.extend(self.handle_box_dots(ql.qids_map))
        super().__init__(all_elems)
        self.shift(0, ql.get_qline(self.min_obj).top() - self.box.top() - 1)

    def __calc_high__(self, qids_map) -> int:
        """Calculate the height of gate."""
        return (qids_map[self.max_obj] - qids_map[self.min_obj] + 1) * (self.style.qubit_line_v_dist + 1)

    def handle_ctrl_bar(self, ql: QubitLine) -> List[BasicObj]:
        """Handle the control relate component."""
        qids_map = ql.qids_map
        ctrl_dots = []
        min_ctrl = self.obj_qubits[0]
        max_ctrl = min_ctrl
        for i in self.ctrl_qubits:
            if not self.min_obj < i < self.max_obj:
                min_ctrl = min(min_ctrl, i)
                max_ctrl = max(max_ctrl, i)
                text = Text(self.style.control_dot)
                ObjectEditor.v_align(self.box, text)
                text.shift(0, (qids_map[i] - qids_map[self.min_obj]) * (self.style.qubit_line_v_dist + 1) + 1)
                ctrl_dots.append(text)
        lines = []
        if not ((self.min_obj < min_ctrl < self.max_obj) and (self.min_obj < max_ctrl < self.max_obj)):
            line = VLine(
                length=(qids_map[max_ctrl] - qids_map[min_ctrl]) * (self.style.qubit_line_v_dist + 1),
                thickness=self.style.control_line_thickness,
            )
            ObjectEditor.v_align(self.box, line)
            line.shift(0, (qids_map[min_ctrl] - qids_map[self.min_obj]) * (self.style.qubit_line_v_dist + 1) + 1)
            lines.append(line)
            fix_line_rec_cross(self.box, line)
            for i in qids_map:
                if min_ctrl < i < max_ctrl:
                    fix_lines_cross(ql.get_qline(i), line)
        return lines + ctrl_dots

    def handle_box_dots(self, qids_map) -> List[BasicObj]:
        """Handle the component around the box."""
        ctrl_texts = []
        for i in self.ctrl_qubits:
            if self.min_obj < i < self.max_obj:
                text = Text(self.style.control_dot)
                text.shift(0, (qids_map[i] - qids_map[self.min_obj]) * (self.style.qubit_line_v_dist + 1) + 1)
                ctrl_texts.append(text)
        obj_texts = []
        if ctrl_texts or qids_map[self.max_obj] - qids_map[self.min_obj] + 1 > len(self.obj_qubits):
            for i in self.obj_qubits:
                textl = Text(self.style.obj_dot, str_style=self.style.obj_dot_style)
                textl.shift(0, (qids_map[i] - qids_map[self.min_obj]) * (self.style.qubit_line_v_dist + 1) + 1)

                obj_texts.append(textl)

        return ctrl_texts + obj_texts


class CNOTTGate(SimpleGate):
    """A CNOT gate."""

    def __init__(self, obj_qubits, ctrl_qubits, ql: QubitLine, style: TextCircConfig):
        """Construct a CNOT gate."""
        self.style = style
        self.obj_qubits = obj_qubits
        self.ctrl_qubits = ctrl_qubits
        self.min_obj = min(obj_qubits)
        self.max_obj = max(obj_qubits)
        self.text = Text(style.cnot_symbol)
        self.box = Rect(
            width=self.text.get_width() + 2,
            high=self.__calc_high__(ql.qids_map),
            box_style=BoxStyle.regular_heavy,
        )
        for i in ql.lines.values():
            fix_line_rec_cross(self.box, i)
        ObjectEditor.center_align(self.box, self.text)
        all_elems = []
        all_elems.extend(self.handle_ctrl_bar(ql))
        all_elems.extend([self.box, self.text])
        all_elems.extend(self.handle_box_dots(ql.qids_map))
        Container.__init__(self, all_elems)
        self.shift(0, ql.get_qline(self.min_obj).top() - self.box.top() - 1)


class SWAPTGate(SimpleGate):
    """A SWAP gate."""

    def __init__(self, obj_qubits, ctrl_qubits, ql: QubitLine, style: TextCircConfig):
        """Construct a SWAP gate."""
        self.style = style
        self.obj_qubits = obj_qubits
        self.ctrl_qubits = ctrl_qubits
        self.min_obj = min(obj_qubits + ctrl_qubits)
        self.max_obj = max(obj_qubits + ctrl_qubits)
        self.line = VLine(
            length=(ql.qids_map[self.max_obj] - ql.qids_map[self.min_obj]) * (style.qubit_line_v_dist + 1),
            thickness=style.control_line_thickness,
        )
        self.line.shift(0, ql.get_qline(self.min_obj).top() - self.line.top())
        self.fix_cross(ql)
        all_elems = [self.line]
        all_elems.extend(self.handle_dots(ql))
        Container.__init__(self, all_elems)

    def handle_dots(self, ql: QubitLine) -> List[BasicObj]:
        """Handle swap dot and control dot."""
        ctrl_dots = []
        for i in self.ctrl_qubits:
            dot = Text(self.style.control_dot)
            ObjectEditor.h_align(ql.get_qline(i), dot)
            ctrl_dots.append(dot)
        obj_dots = []
        for i in self.obj_qubits:
            dot = Text(self.style.swap_symbol, str_style=self.style.swap_symbol_style)
            ObjectEditor.h_align(ql.get_qline(i), dot)
            obj_dots.append(dot)
        return ctrl_dots + obj_dots

    def fix_cross(self, ql: QubitLine):
        """Fix cross problem."""
        max_obj = max(self.obj_qubits)
        min_obj = min(self.obj_qubits)
        for i in ql.qids:
            if self.min_obj < i < self.max_obj and not min_obj < i < max_obj:
                fix_lines_cross(ql.get_qline(i), self.line)


class BarrierTGate(SimpleGate):
    """A barrier gate."""

    def __init__(self, obj_qubits, ql: QubitLine, style: TextCircConfig):
        """Construct a barrier gate."""
        self.obj_qubits = obj_qubits
        self.min_obj = min(obj_qubits)
        self.max_obj = max(obj_qubits)
        self.line = VLine(
            length=(ql.qids_map[self.max_obj] - ql.qids_map[self.min_obj]) * (style.qubit_line_v_dist + 1) + 1,
            line_style=style.barrier_style,
        )
        self.line.char = style.barrier_symbol
        self.line.shift(0, ql.get_qline(self.min_obj).top() - self.line.top())
        Container.__init__(self, [self.line])


class MeasureTGate(SimpleGate):
    """A measurement gate."""

    def __init__(self, m: Measure, ql: QubitLine, style: TextCircConfig):
        """Construct a measurement gate."""
        obj_qubit = m.obj_qubits[0]
        if style.simple_mode:
            name = f"{style.measurement_symbol}"
        else:
            name = f"{style.measurement_symbol} {m.key}"
            if m.reset_to is not None:
                name = f"{name}, reset to {m.reset_to}"
        name = f" {name} "
        name_text = Text(name, str_style="black")
        name_text.shift(0, ql.get_qline(obj_qubit).top() - name_text.top())
        box = Rect(
            width=name_text.get_width() + style.simple_gate_width_padding,
            box_style=getattr(BoxStyle, style.measurement_box),
        )
        fix_line_rec_cross(box, ql.get_qline(obj_qubit))
        ObjectEditor.center_align(name_text, box)
        elems = [box, name_text]
        Container.__init__(self, elems)


class SubCirc(SimpleGate):
    """A sub circuit."""

    def __init__(self, circ: Circuit, ql: QubitLine, name: str = "", father_qubits=None, style: TextCircConfig = None):
        """Construct a sub circuit."""
        self.obj_qubits = sorted(circ.all_qubits.keys())
        self.min_obj = min(self.obj_qubits)
        self.max_obj = max(self.obj_qubits)

        sub_circ = rich_circuit(circ, np.inf, father_qubits, style=style)
        sub_circ_h = sub_circ.get_high()
        sub_ql: QubitLine = sub_circ.eles[0]
        sub_ql.eles = sub_ql.eles[1:]
        sub_circ.shift(0, ql.get_qline(self.min_obj).top() - sub_circ.top() - 1)

        name_text = Text(name, str_style="red bold")
        tmp_rect = Rect()
        dist = style.qubit_line_v_dist + 1
        tmp_rect.set_high(int(sub_circ_h // dist * dist + np.ceil(sub_circ_h % dist / dist) * dist))
        tmp_rect.set_width(max(name_text.get_width() + style.simple_gate_width_padding, sub_circ.get_width()))
        ObjectEditor.center_align(sub_circ, tmp_rect)
        ObjectEditor.top_align(sub_circ, tmp_rect)
        left = VLine(length=tmp_rect.get_high())
        right = VLine(length=tmp_rect.get_high())

        def left_post(frame: Frame):
            frame.data[0] = BoxStyle.rounded[0]
            frame.data[-1] = BoxStyle.rounded[6]
            return frame

        def right_post(frame: Frame):
            frame.data[0] = BoxStyle.rounded[2]
            frame.data[-1] = BoxStyle.rounded[4]
            return frame

        left.append_post_process(left_post)
        right.append_post_process(right_post)
        ObjectEditor.left_align(tmp_rect, left)
        ObjectEditor.right_align(tmp_rect, right)
        right.shift(1, 0)
        ObjectEditor.batch_ops([sub_circ, left, right], ObjectEditor.top_align)
        for i in ql.qids:
            if self.min_obj <= i <= self.max_obj:
                fix_lines_cross(ql.get_qline(i), left, must_zero=[2])
                fix_lines_cross(ql.get_qline(i), right, must_zero=[0])

        elems = [sub_circ, left, right]
        if name:
            ObjectEditor.top_align(tmp_rect, name_text)
            ObjectEditor.v_align(tmp_rect, name_text)
            elems.append(name_text)
        Container.__init__(self, elems)


def rich_circuit(
    circ: Circuit, screen_width, father_qubits: List[int] = None, style: TextCircConfig = None, qubit_map=None
) -> Container:
    """Display a circuit with rich."""
    if father_qubits is None:
        all_qubits = sorted(circ.all_qubits.keys())
    else:
        child_qubits = sorted(circ.all_qubits.keys())
        min_qid = min(child_qubits)
        max_qid = max(child_qubits)
        all_qubits = [i for i in range(min_qid, max_qid + 1) if i in father_qubits]
    rev_all_qubits = {j: i for i, j in enumerate(all_qubits)}

    ql = QubitLine(all_qubits, style, qubit_map)
    gate_container = Container()
    all_ql = [ql]
    all_gc = [gate_container]

    rights = {i: j.right() for i, j in ql.lines.items()}
    g_total = len(circ)
    g_cur = 0
    while True:
        if g_cur >= g_total:
            break
        g: BasicGate = circ[g_cur]
        if isinstance(g, CNOTGate):
            text_g = CNOTTGate(g.obj_qubits[:1], [g.obj_qubits[1]] + g.ctrl_qubits, ql, style)
        elif isinstance(g, XGate):
            text_g = CNOTTGate(g.obj_qubits, g.ctrl_qubits, ql, style)
        elif isinstance(g, SWAPGate):
            text_g = SWAPTGate(g.obj_qubits, g.ctrl_qubits, ql, style)
        elif isinstance(g, BarrierGate):
            text_g = BarrierTGate(g.obj_qubits if g.obj_qubits else all_qubits, ql, style)
        elif isinstance(g, GroupedPauliChannel):
            if style.simple_mode:
                text_g = SimpleGate('GPC', g.obj_qubits, g.ctrl_qubits, ql, style)
            else:
                text_g = SubCirc(
                    Circuit([PauliChannel(*j).on(i) for i, j in zip(g.obj_qubits, g.probs)]),
                    ql,
                    "Grouped Pauli Channel",
                    all_qubits,
                    style,
                )
        elif isinstance(g, GroupedPauli):
            if style.simple_mode:
                text_g = SimpleGate('GP', g.obj_qubits, g.ctrl_qubits, ql, style)
            else:
                text_g = SubCirc(g.__decompose__(), ql, "Grouped Pauli", all_qubits, style)
        elif isinstance(g, Measure):
            text_g = MeasureTGate(g, ql, style)
        else:
            name = g.name if style.simple_mode else g.__str_in_circ__()
            text_g = SimpleGate(name, g.obj_qubits, g.ctrl_qubits, ql, style)
        if isinstance(g, NoiseGate) and not isinstance(g, GroupedPauliChannel):
            text_g.box.box_style = getattr(BoxStyle, style.noise_box)
        if g.parameterized:
            text_g.box.box_color = style.parameterized_box_style
            text_g.box.box_style = getattr(BoxStyle, style.parameterized_box)
        if isinstance(g, BarrierGate) and not g.obj_qubits:
            min_id = all_qubits[0]
            max_id = all_qubits[-1]
        else:
            min_id = min(g.obj_qubits + g.ctrl_qubits)
            max_id = max(g.obj_qubits + g.ctrl_qubits)
        max_right = max(rights[all_qubits[i]] for i in range(rev_all_qubits[min_id], rev_all_qubits[max_id] + 1))
        if not isinstance(g, BarrierGate) or (isinstance(g, BarrierGate) and g.show):
            text_g.shift(max_right - text_g.left() + style.gate_v_dist + (1 if style.simple_mode else 2), 0)
            max_right = text_g.right()
        if not isinstance(g, BarrierGate) or (isinstance(g, BarrierGate) and g.show):
            if text_g.right() > screen_width - style.window_padding and gate_container.eles:
                ql.extend_line_to(max(rights.values()) + style.qubit_line_final_width)
                ql = QubitLine(all_qubits, style, qubit_map)
                gate_container = Container()
                all_ql.append(ql)
                all_gc.append(gate_container)
                rights = {i: j.right() for i, j in ql.lines.items()}
                continue
            gate_container.add(text_g)
        for i in range(rev_all_qubits[min_id], rev_all_qubits[max_id] + 1):
            rights[all_qubits[i]] = max_right
        g_cur += 1
    ql.extend_line_to(max(rights.values()) + style.qubit_line_final_width)

    pics = [Container([i, j]) for i, j in zip(all_ql, all_gc)]
    if len(pics) == 1:
        pics[0].left_top_origin()
        return pics[0]
    ObjectEditor.v_expand(pics, 1)
    out = Container()
    for i in pics[:-1]:
        pl: QubitLine = i.eles[0]

        def add_continue(frame: Frame):
            frame.data[-2] = f"[{style.continue_symbol_style}]{style.continue_symbol}[/]"
            return frame

        for line in pl.lines.values():
            line.append_post_process(add_continue)
        out.add(i)
    out.add(pics[-1])
    return out.left_top_origin()
