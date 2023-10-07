"""utility functions."""

import numpy as np
import mindspore.numpy as msnp


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
"""Print information in block shape."""


def _fill_fist_sep(string, sep, length, fill_char=' '):
    poi = string.find(sep)
    if length < poi:
        raise Exception(f"Original length is {poi}, can not fill it to length {length}.")
    return string[:poi] + fill_char * (length - poi) + string[poi:]


def _check_str(string, name):
    if not isinstance(string, str):
        raise TypeError(f"{name} requires str, but get {type(string)}!")


# pylint: disable=too-many-arguments
def bprint(strings: list, align=":", title='', v_around='=', h_around='|', fill_char=' '):
    """
    Print the information in block shape.

    Args:
        strings (list[str]): Information you want to output.
        align (str): The align char alone vertal direction. Default: ``":"``.
        title (str): The title of this information block. Default: ``""``.
        v_around (str): Vertical boundary char. Default: ``"="``.
        h_around (str): horizontal boundary char. Default: ``"|"``.
        fill_char (str): Empty space fill with this char. Default: ``" "``.

    Returns:
        list, a list of formatted string.

    Examples:
        >>> from mindquantum.io import bprint
        >>> title='Info of Bob'
        >>> o = bprint(['Name:Bob', 'Age:17', 'Nationality:China'],
        ...     title=title)
        >>> for i in o:
        ...     print(i)
        ====Info of Bob====
        |Name       :Bob  |
        |Age        :17   |
        |Nationality:China|
        ===================
    """
    if not isinstance(strings, list):
        raise TypeError(f"strings requires a list, but get {type(strings)}")
    for string in strings:
        _check_str(string, "string")
    _check_str(align, 'align')
    _check_str(title, 'title')
    _check_str(v_around, 'v_around')
    _check_str(h_around, 'h_around')
    _check_str(fill_char, 'fill_char')
    maxmim_len = strings[0].find(align)
    for sub_str in strings:
        m_poi = sub_str.find(align)
        if m_poi > maxmim_len:
            maxmim_len = m_poi
    strings = [_fill_fist_sep(i, align, maxmim_len, fill_char) for i in strings]
    n_around = 3
    title = v_around * n_around + title + v_around * n_around
    maxmim = max(len(i) for i in strings)
    if len(title) > maxmim:
        len_total = (len(title) - maxmim) // 2 + (len(title) - maxmim) % 2
        strings = [h_around + ' ' * len_total + i + ' ' * (len(title) - len(i) - len_total) + h_around for i in strings]
        title = h_around + title + h_around
    else:
        len_total = (maxmim - len(title)) // 2 + (maxmim - len(title)) % 2
        title = v_around + v_around * len_total + title + v_around * len_total + v_around
        strings = [h_around + i + ' ' * (len(title) - 2 - len(i)) + h_around for i in strings]
    bot = v_around + v_around * (len(title) - 2) + v_around
    output = []
    output.append(title)
    output.extend(strings)
    output.append(bot)
    return output


__all__ = ['bprint']
