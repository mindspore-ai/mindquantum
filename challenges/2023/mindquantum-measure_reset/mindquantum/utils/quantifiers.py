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

"""Quantifier helper functions."""

from .type_value_check import _check_int_type, _check_value_should_not_less


def quantifier_selector(num, single, plural):
    """Quantifier selector."""
    _check_int_type('num', num)
    _check_value_should_not_less('num', 0, num)
    if num > 1:
        return f'{num} {plural}'
    return f'{num} {single}'


def s_quantifier(num, quantifier):
    """S quantifier."""
    return quantifier_selector(num, quantifier, f'{quantifier}s')


def es_quantifier(num, quantifier):
    """ES quantifier."""
    return quantifier_selector(num, quantifier, f'{quantifier}es')
