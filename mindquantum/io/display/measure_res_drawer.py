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
"""Text measure result"""
# from mindquantum.core.gates import MeasureResult
import math
from ._config import _res_text_drawer_config


def _trans(v, l, m):
    return math.ceil(v / m * l)


def measure_text_drawer(res):
    """Draw measure result."""
    max_size = _res_text_drawer_config['max_size']
    vline = _res_text_drawer_config['vline']
    hline = _res_text_drawer_config['hline']
    cross_mask = _res_text_drawer_config['cross_mask']
    box_high = _res_text_drawer_config['box_high']
    box_low = _res_text_drawer_config['box_low']
    axis_mask = _res_text_drawer_config['axis_mask']
    split = _res_text_drawer_config['spilit']
    deci = _res_text_drawer_config['deci']
    keys = res.keys
    max_shot = max(res.data.values())
    max_prop = max_shot / res.shots
    if max_prop == 0:
        max_prop = 1
    if max_prop / 0.8 > 1:
        max_prop = 1
    else:
        max_prop /= 0.8
    ket_exp = 'Keys: '
    ket_exp += ' '.join(keys[::-1])
    s = [f'shots: {res.shots}']
    s.append(ket_exp + vline)
    axis_num = ''
    axis = ''
    for i in range(split):
        axis_num = str(round(max_prop / split * (split - i), deci)).rjust(
            int(max_size / split)) + axis_num
        axis = axis_mask.rjust(int(max_size / split), hline) + axis
    axis_num = '0.00' + axis_num[4:]
    s[-1] += axis_num
    s.append(hline * len(ket_exp) + cross_mask + axis)
    for k, v in res.data.items():
        state = k
        state = state.rjust(len(ket_exp))
        state += vline
        state += (box_high if v == max_shot else box_low) * _trans(
            v / res.shots, max_size, max_prop)
        s.append(state)
        s.append(' ' * len(ket_exp) + vline)
    return s
