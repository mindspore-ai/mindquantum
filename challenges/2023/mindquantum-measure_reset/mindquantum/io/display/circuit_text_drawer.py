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

import numpy as np

from ._config import _text_drawer_config


def _get_qubit_range(gate):
    """Get qubit range."""
    out = []
    out.extend(gate.obj_qubits)
    out.extend(gate.ctrl_qubits)
    if not out:
        raise ValueError(f"{gate.name} gate should apply to certain qubit first.")
    return out


# pylint: disable=too-many-statements,too-many-branches
def brick_model(circ, qubits_name=None, width=np.inf):  # pylint: disable=too-many-locals
    """Split a circuit into layers."""
    from mindquantum import (  # pylint: disable=import-outside-toplevel,cyclic-import
        gates,
    )

    n = circ.n_qubits
    if qubits_name is None:
        qubits_name = list(range(n))
    v_n = _text_drawer_config['v_n']
    blocks = []
    qubit_hight = np.zeros(n, dtype=int)
    for gate in circ:
        if isinstance(gate, gates.BarrierGate):
            if gate.obj_qubits:
                qrange = _get_qubit_range(gate)
            else:
                qrange = range(n)
        else:
            qrange = _get_qubit_range(gate)
        max_hight = np.max(qubit_hight[range(min(qrange), max(qrange) + 1)])
        if len(blocks) <= max_hight:
            blocks.append([])
        blocks[max_hight].append(gate)
        qubit_hight[range(min(qrange), max(qrange) + 1)] = max_hight + 1
    blocks = [_single_block_drawer(i, n) for i in blocks]
    res = {}
    max_q = 0
    for i in range(n):
        string = f'q{qubits_name[i]}: '
        res[i * (v_n + 1)] = string
        max_q = max(max_q, len(string))
    for i in range(n):
        res[i * (v_n + 1)] = res[i * (v_n + 1)].ljust(max_q, ' ')
        if i != n - 1:
            for j in range(v_n):
                res[i * (v_n + 1) + j + 1] = ' ' * max_q
    text_circ = {i: [j] for i, j in res.items()}
    current_width = max_q
    continue_left = _text_drawer_config.get('continue_left')
    continue_right = _text_drawer_config.get('continue_right')
    continue_buff = len(continue_left) + len(continue_right)

    if max_q >= width - continue_buff:
        raise ValueError("Window width too small to display circuit.")
    n_split = 1
    block_widths = []
    for block in blocks:
        block_widths.append(max(len(j) for j in block.values()))
        if block_widths[-1] + max_q >= width - 4:
            raise ValueError("Window width too small to display a single gate.")
    for block_id, block in enumerate(blocks):
        if current_width + block_widths[block_id] >= width - continue_buff:
            for i, j in res.items():
                text_circ[i][-1] += continue_right
                text_circ[i].append(j + continue_left)
            n_split += 1
            current_width = max_q
        for k, v in block.items():
            text_circ[k][-1] += v
        current_width += block_widths[block_id]
    output = []
    for j in range(n_split):
        for i in range((n - 1) * (v_n + 1) + 1):
            output.append(text_circ[i][j].rstrip())
        if j != n_split - 1:
            output.append(_text_drawer_config['horizontal_split'] * width)
    return output


def _single_gate_drawer(gate):
    """Single gate drawer."""
    from mindquantum import (  # pylint: disable=import-outside-toplevel,cyclic-import
        gates,
    )

    if isinstance(gate, gates.CNOTGate):
        gate = gates.X.on(*gate.obj_qubits)
    main_text = gate.__str_in_circ__()
    if isinstance(gate, gates.SWAPGate):
        main_text = _text_drawer_config['swap_mask'][0]
    if isinstance(gate, gates.BarrierGate):
        main_text = _text_drawer_config['barrier']
    main_text = _text_drawer_config['edge'] + main_text + _text_drawer_config['edge']
    res = {}
    for i in gate.obj_qubits:
        res[i] = main_text
    if isinstance(gate, gates.SWAPGate):
        max_idx = max(gate.obj_qubits)
        lower_txt = _text_drawer_config['swap_mask'][1]
        res[max_idx] = _text_drawer_config['edge'] + lower_txt + _text_drawer_config['edge']
    for i in gate.ctrl_qubits:
        res[i] = _text_drawer_config['ctrl_mask']
        res[i] = res[i].center(len(main_text), _text_drawer_config['circ_line'])
    res['len'] = len(main_text)
    return res


def _single_block_drawer(block, n_qubits):  # pylint: disable=too-many-branches,too-many-locals
    """Single block drawer."""
    from mindquantum import (  # pylint: disable=import-outside-toplevel,cyclic-import
        gates,
    )

    v_n = _text_drawer_config['v_n']
    text_gates = {}
    if isinstance(block[0], gates.BarrierGate) and not block[0].obj_qubits:
        if not block[0].show:
            tmp = ''
        else:
            tmp = _text_drawer_config['barrier']
        for i in range((n_qubits - 1) * v_n + n_qubits):
            text_gates[i] = tmp
        return text_gates
    for gate in block:
        text_gate = _single_gate_drawer(gate)
        qrange = _get_qubit_range(gate)
        for qubit in range(min(qrange), max(qrange) + 1):
            ind = qubit * (v_n + 1)
            if qubit in qrange:
                text_gates[ind] = text_gate[qubit]
            else:
                text_gates[ind] = _text_drawer_config['cross_mask']
                text_gates[ind] = text_gates[ind].center(text_gate['len'], _text_drawer_config['circ_line'])
        ctrl_line = _text_drawer_config['ctrl_line']
        if isinstance(gate, gates.BarrierGate):
            ctrl_line = _text_drawer_config['barrier']
        for qubit in range(min(qrange), max(qrange)):
            for i in range(v_n):
                ind = qubit * (v_n + 1) + i + 1
                text = ctrl_line
                text_gates[ind] = text
                text_gates[ind] = text.center(text_gate.get('len', 0), ' ')
    max_l = max(len(j) for j in text_gates.values())
    for k, v in text_gates.items():
        if len(v) != max_l:
            if k % (v_n + 1) == 0:
                text_gates[k] = text_gates[k].center(max_l, _text_drawer_config['circ_line'])
            else:
                text_gates[k] = text_gates[k].center(max_l, ' ')
    for i in range((n_qubits - 1) * v_n + n_qubits):
        if i not in text_gates:
            if i % (v_n + 1) == 0:
                text_gates[i] = _text_drawer_config['circ_line'] * max_l
            else:
                text_gates[i] = ' ' * max_l
    return text_gates
