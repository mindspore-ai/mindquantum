# -*- coding: utf-8 -*-
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
"""Configuration"""

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

_text_drawer_config = {
    'ctrl_mask': '●',  #⨉
    'circ_line': '─',
    'ctrl_line': '│',
    'cross_mask': '┼',
    'v_n': 1,
    'swap_mask': ['@', '@'],  # ✖, ⨯⨯
    'edge_num': 2,
    'barrier': '‖'
}

_text_drawer_config['edge'] = _text_drawer_config[
    'circ_line'] * _text_drawer_config['edge_num']

_CIRCUIT_STYLE = {'style': 'blue bold'}
_MEA_RES_STYLE = {'style': 'yellow'}
_DAGGER_MASK = '†'
