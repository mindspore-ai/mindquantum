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
"""type and value check helper"""
import numpy as np

_num_type = (int, float, complex, np.int32, np.int64, np.float32, np.float64)


def _check_seed(seed):
    """check seed"""
    _check_int_type("seed", seed)
    _check_value_should_between_close_set("seed", 0, 2**23, seed)


def _check_input_type(arg_msg, require_type, arg):
    """check input type"""
    if not isinstance(arg, require_type):
        raise TypeError(f"{arg_msg} requires a {require_type}, but get {type(arg)}")


def _check_int_type(args_msg, arg):
    """check int type"""
    if not isinstance(arg, (int, np.int64)) or isinstance(arg, bool):
        raise TypeError(f"{args_msg} requires an int, but get {type(arg)}")


def _check_value_should_not_less(arg_msg, require_value, arg):
    """check value should not less"""
    if arg < require_value:
        raise ValueError(f'{arg_msg} should be not less than {require_value}, but get {arg}')


def _check_value_should_between_close_set(arg_ms, min_value, max_value, arg):
    """Check value should between"""
    if arg < min_value or arg > max_value:
        raise ValueError(f"{arg_ms} should between {min_value} and {max_value}, but get {arg}")


def _check_and_generate_pr_type(pr, names=None):
    """_check_and_generate_pr_type"""
    from mindquantum.core import ParameterResolver
    if isinstance(pr, _num_type):
        if len(names) != 1:
            raise ValueError(f"number of given parameters value is less than parameters ({len(names)})")
        pr = np.array([pr])
    _check_input_type('parameter', (ParameterResolver, np.ndarray, list, dict), pr)
    if isinstance(pr, dict):
        pr = ParameterResolver(pr)
        if len(pr) != len(names):
            raise ValueError(f"given parameter value size ({len(pr)}) not match with parameter size ({len(names)})")
    elif isinstance(pr, (np.ndarray, list)):
        pr = np.array(pr)
        if len(pr) != len(names) or len(pr.shape) != 1:
            raise ValueError(f"given parameter value size ({pr.shape}) not match with parameter size ({len(names)})")
        pr = ParameterResolver(dict(zip(names, pr)))

    return pr
