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
"""Check input for quantum neural network."""

from collections.abc import Iterable
from mindquantum.circuit import Circuit


def _check_circuit(circuit, msg):
    if not isinstance(circuit, Circuit):
        raise TypeError("{} requires a quantum circuit, but get {}!".format(
            msg, type(circuit)))


def _check_non_parameterized_circuit(circuit: Circuit):
    if not isinstance(circuit, Circuit):
        raise TypeError(
            "Requires a non parameterized quantum circuit, but get {}!".format(
                type(circuit)))
    for g in circuit:
        if g.isparameter:
            raise ValueError(
                "Requires a non parameterized quantum circuit, but {} is parameterized gate!"
                .format(g))


def _check_type_or_iterable_type(inputs, require, msg):
    if not isinstance(inputs, Iterable):
        if not isinstance(inputs, require):
            raise TypeError(
                "{msg} requires {req} or several {req}s, but get {inp}!".
                format(msg=msg, req=require, inp=type(inputs)))
    else:
        for inp in inputs:
            if not isinstance(inp, require):
                raise TypeError(
                    "{msg} requires {req} or several {req}s, but {inp} is not {req}!"
                    .format(msg=msg, req=require, inp=inp))


def _check_list_of_string(inputs, msg):
    if not isinstance(inputs, list):
        raise TypeError("{} requires a list of string, but get {}!".format(
            msg, type(inputs)))
    for inp in inputs:
        if not isinstance(inp, str):
            raise TypeError(
                "{} requires a list of string, but {} is not string.".format(
                    msg, inp))


def _check_parameters_of_circuit(encoder_params_names, ansatz_params_names,
                                 circuit: Circuit):
    _check_list_of_string(encoder_params_names, 'Encoder parameter names')
    _check_list_of_string(ansatz_params_names, 'Ansatz parameter names')
    all_names = []
    all_names.extend(encoder_params_names)
    all_names.extend(ansatz_params_names)
    circ_names = circuit.para_name
    if not set(all_names) == set(circ_names):
        raise ValueError(
            "Parameter names you input not match with parameters in circuit.")
