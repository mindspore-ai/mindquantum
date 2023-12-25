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
# pylint: disable=cyclic-import

"""Configuration."""
from dataclasses import dataclass

CIRCUIT_HTML_FORMAT = """\
<div style="{stylesheet}color: {foreground}; background-color: {background}"></div>
<pre style="white-space: pre;">{code}</pre>
"""

MEA_HTML_FORMAT = """\
<div style="{stylesheet}color: {foreground}; background-color: {background}"></div>
<pre style="white-space: pre;">{code}</pre>
"""

_res_text_drawer_config = {
    'vline': '│',
    'max_size': 60,
    'spilit': 5,
    'hline': '─',
    'cross_mask': '┼',
    'axis_mask': '┴',
    'box_high': '▓',
    'box_low': '▒',
    'deci': 3,
}

_MEA_RES_STYLE = 'yellow'
_DAGGER_MASK = '†'

_svg_measure_config_official = {
    "table_box_line_width": 1,
    "table_box_line_stroke": "#adb0b8",
    "table_box_w": 400,
    "vline_width": 1,
    "vline_stroke": "#dfe1e6",
    "v_dis": 60,
    "bar_dis": 30,
    "stick_len": 7,
    "n_stick": 6,
    "first_color": "#16acff",
    "second_color": "#5e7ce0",
    "label_fontsize": 12,
    "label_fontcolor": "#575d6c",
    "rec_h": 24,
    "anim_time": 0.3,
    "max_color": "#fac209",
    "background": "#ffffff",
}

_svg_config_official = {
    'gate_start_distance': 24,
    'padding_x': 0,
    'padding_y': 0,
    'swap_icon_ratio': 0.6,
    'background': '#ffffff',
    # for qubit
    'qubit_name_font_family': 'Arial',
    'qubit_name_font_weight': 'normal',
    'qubit_font_size': 16,
    'qubit_name_color': '#252b3a',
    # for circuit line
    'circuit_line_stroke': '#adb0b8',
    'circuit_line_stroke_width': 1,
    'circuit_line_distance': 60,
    # for gate
    'gate_name_color': '#ffffff',
    'gate_name_font_size': 20,
    'gate_name_font_family': 'Arial',
    'gate_name_font_weight': 'normal',
    'gate_chamfer': 4,
    'gate_size': 40,
    'gate_v_distance': 20,
    'gate_stroke_width': 0,
    # for quantum noise
    'noise_fill': '#c77532',
    # for non parameterized gate
    'npg_stroke': '#ffffff',
    'npg_fill': '#5e7ce0',
    'npg_fill_opacity': 1,
    # for parameterized gate
    'pg_stroke': '#ffffff',
    'pg_fill': '#fac209',
    'pg_fill_opacity': 1,
    # for cnot gate
    'cnot_cross_size': 28,
    'cnot_cross_stroke': '#ffffff',
    'cnot_cross_stroke_width': 4,
    'cnot_fill': '#16acff',
    'cnot_fill_opacity': 1,
    'cnot_stroke': '#ffffff',
    'cnot_stroke_width': 0,
    # for swap gate
    'swap_fill': '#16acff',
    'swap_fill_opacity': 1,
    'swap_stroke': '#ffffff',
    'swap_icon_color': '#ffffff',
    # for measure gate
    'measure_fill': '#ff7272',
    'measure_fill_opacity': 1,
    'measure_stroke': '#ffffff',
    'measure_icon_color': '#ffffff',
    'measure_arc_stroke_width': 3,
    'measure_icon_scale': 0.8,
    # gate dots
    'obj_dot_r': 4,
    'ctrl_dot_r': 4,
    'ctrl_line_stroke_width': 3,
    # for barrier
    'barrier_width': 20,
    'barrier_fill': 'gray',
    'barrier_opacity': 0.8,
}

_svg_config_dark = {
    'gate_start_distance': 24,
    'padding_x': 0,
    'padding_y': 0,
    'swap_icon_ratio': 0.6,
    'background': '#180835',
    # for qubit
    'qubit_name_font_family': 'Arial-BoldMT, Arial',
    'qubit_name_font_weight': 'bold',
    'qubit_font_size': 16,
    'qubit_name_color': '#d8d8d8',
    # for circuit line
    'circuit_line_stroke': '#d8d8d8',
    'circuit_line_stroke_width': 4,
    'circuit_line_distance': 60,
    # for gate
    'gate_name_color': '#d8d8d8',
    'gate_name_font_size': 18,
    'gate_name_font_family': 'Arial-BoldMT, Arial',
    'gate_name_font_weight': 'bold',
    'gate_chamfer': 9,
    'gate_size': 40,
    'gate_v_distance': 40,
    'gate_stroke_width': 4,
    # for quantum noise
    'noise_fill': '#c77532',
    # for non parameterized gate
    'npg_stroke': '#e9b645',
    'npg_fill': '#180835',
    'npg_fill_opacity': 1,
    # for parameterized gate
    'pg_stroke': '#ad20e9',
    'pg_fill': '#180835',
    'pg_fill_opacity': 1,
    # for cnot gate
    'cnot_cross_size': 30,
    'cnot_cross_stroke': '#180835',
    'cnot_cross_stroke_width': 6,
    'cnot_fill': '#66c31a',
    'cnot_fill_opacity': 1,
    'cnot_stroke': '#ffffff',
    'cnot_stroke_width': 0,
    # for swap gate
    'swap_fill': '#180835',
    'swap_fill_opacity': 1,
    'swap_stroke': '#30ac1a',
    'swap_icon_color': '#595bea',
    # for measure gate
    'measure_fill': '#180835',
    'measure_fill_opacity': 1,
    'measure_stroke': '#ffffff',
    'measure_icon_color': '#e92734',
    'measure_arc_stroke_width': 3,
    'measure_icon_scale': 0.8,
    # gate dots
    'obj_dot_r': 5.5,
    'ctrl_dot_r': 5.5,
    'ctrl_line_stroke_width': 4,
    # for barrier
    'barrier_width': 20,
    'barrier_fill': '#8a744d',
    'barrier_opacity': 0.95,
}

_svg_config_light = {
    'gate_start_distance': 24,
    'padding_x': 0,
    'padding_y': 0,
    'swap_icon_ratio': 0.6,
    'background': '#ffffff',
    # for qubit
    'qubit_name_font_family': 'Arial-BoldMT, Arial',
    'qubit_name_font_weight': 'bold',
    'qubit_font_size': 16,
    'qubit_name_color': '#180835',
    # for circuit line
    'circuit_line_stroke': '#180835',
    'circuit_line_stroke_width': 4,
    'circuit_line_distance': 60,
    # for gate
    'gate_name_color': '#180835',
    'gate_name_font_size': 18,
    'gate_name_font_family': 'Arial-BoldMT, Arial',
    'gate_name_font_weight': 'bold',
    'gate_chamfer': 9,
    'gate_size': 40,
    'gate_v_distance': 40,
    'gate_stroke_width': 4,
    # for quantum noise
    'noise_fill': '#c77532',
    # for non parameterized gate
    'npg_stroke': '#e9b645',
    'npg_fill': '#ffffff',
    'npg_fill_opacity': 1,
    # for parameterized gate
    'pg_stroke': '#ad20e9',
    'pg_fill': '#ffffff',
    'pg_fill_opacity': 1,
    # for cnot gate
    'cnot_cross_size': 30,
    'cnot_cross_stroke': '#ffffff',
    'cnot_cross_stroke_width': 6,
    'cnot_fill': '#66c31a',
    'cnot_fill_opacity': 1,
    'cnot_stroke': '#ffffff',
    'cnot_stroke_width': 0,
    # for swap gate
    'swap_fill': '#ffffff',
    'swap_fill_opacity': 1,
    'swap_stroke': '#30ac1a',
    'swap_icon_color': '#595bea',
    # for measure gate
    'measure_fill': '#ffffff',
    'measure_fill_opacity': 1,
    'measure_stroke': '#f38064',
    'measure_icon_color': '#e92734',
    'measure_arc_stroke_width': 3,
    'measure_icon_scale': 0.8,
    # gate dots
    'obj_dot_r': 5.5,
    'ctrl_dot_r': 5.5,
    'ctrl_line_stroke_width': 4,
    # for barrier
    'barrier_width': 20,
    'barrier_fill': '#8a744d',
    'barrier_opacity': 0.95,
}

_bloch_drawer_config = {
    'radius': 1,
    'ca_width': 1,
    'ca_color': 'gray',
    'sp_alpha': 0.2,
    'sp_color': 'silver',
    'sp_width': 1,
    'arr_color': 'red',
    'arr_width': 3,
    'arr_size': 0.1,
    'arr_alpha': False,
    'label': ['x', 'y', 'z'],
    'label_size': 20,
    'label_color': 'black',
    'ax_offset': (0, 0),
}

_bloch_default_style_ = {
    'ket_label_fs': 18,
    'stick_c': 'k',
    'stick_w': 3,
    'point_c': 'k',
    'point_s': 50,
    'point_m': 'o',
    'proj_c': 'g',
    'state_mode': 'both',
    'with_proj': True,
    'trace_c': '#345678',
    'trace_m': '*',
    'trace_lw': 1.5,
    'trace_ms': 3,
    'trace_ls': '-',
    'arrowstyle': '-|>',
    'mutation_scale': 20,
    'arrow_ls': 'dashed',
    'arrow_c': 'k',
    'plane_alpha': 0.04,
    'xy_plane_color': 'b',
    'yz_plane_color': 'r',
    'zx_plane_color': 'g',
    'frame_alpha': 0.15,
    'fig_color': "#ffffff",
    'fig_w': 8,
    'fig_h': 8,
    'axis_delta': 0.1,
    'axis_label_c': 'k',
}

_bloch_default_style_dark_ = {
    'ket_label_fs': 18,
    'stick_c': 'w',
    'stick_w': 3,
    'point_c': 'w',
    'point_s': 50,
    'point_m': 'o',
    'proj_c': 'g',
    'state_mode': 'both',
    'with_proj': True,
    'trace_c': '#f5e532',
    'trace_m': '*',
    'trace_lw': 1.5,
    'trace_ms': 3,
    'trace_ls': '-',
    'arrowstyle': '-|>',
    'mutation_scale': 20,
    'arrow_ls': 'dashed',
    'arrow_c': 'w',
    'plane_alpha': 0.2,
    'xy_plane_color': '#70f580',
    'yz_plane_color': '#72d2f5',
    'zx_plane_color': '#db6ff5',
    'frame_alpha': 0.2,
    'fig_color': "#24283b",
    'fig_w': 8,
    'fig_h': 8,
    'axis_delta': 0.1,
    'axis_label_c': 'w',
}

_topology_default_style = {
    'f': 60,
    'r': 12,
    'id_color': "#ffffff",
    'id_fs': 18,
    'couple_color': '#ababab',
    'couple_w': 8,
    'selected_edge_c': '#ff0000',
}


# pylint: disable=too-many-instance-attributes
@dataclass
class TextCircConfig:
    """The configuration for display circuit in text."""

    qubit_line_v_dist: int = 2
    qubit_line_thickness: str = 'normal'
    qubit_line_style: str = '#D2D4D9'
    gate_v_dist: int = 0
    simple_gate_str_style: str = 'bold'
    simple_gate_width_padding: int = 2
    simple_gate_box_style: str = 'regular_heavy'
    control_dot: str = "■"
    control_line_thickness: str = 'thick'
    obj_dot: str = "●"
    obj_dot_style: str = "#1877FC"
    cnot_symbol: str = "╺╋╸"
    swap_symbol: str = "╳"
    swap_symbol_style: str = "#0e36d6"
    barrier_style: str = '#ACACAC'
    barrier_symbol: str = "▓"
    measurement_symbol: str = "M"
    measurement_box: str = "ud_heavy"
    noise_box: str = "double"
    parameterized_box: str = "regular_heavy"
    parameterized_box_style: str = "#FEB439"
    window_padding: int = 10
    continue_symbol: str = "↯"
    continue_symbol_style: str = "red bold"
    simple_mode: bool = False
    compress_unuse_qubit: bool = True
    qubit_line_final_width: int = 3


_text_circ_config = TextCircConfig()
