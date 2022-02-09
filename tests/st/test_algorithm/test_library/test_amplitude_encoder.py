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
'''test for amplitude encoder'''
from mindquantum.algorithm import amplitude_encoder
from mindquantum.simulator import Simulator

def test_amplitude_encoder():
    '''
    Feature: amplitude_encoder
    Description:
    Expectation:
    '''
    sim = Simulator('projectq', 3)
    circuit, params = amplitude_encoder([0.5, 0.5, 0.5, 0.5], 3)
    sim.apply_circuit(circuit, params)
    st = sim.get_qs(False)
    assert abs(st[0].real - 0.5) < 1e-10
    assert abs(st[2].real - 0.5) < 1e-10
    assert abs(st[4].real - 0.5) < 1e-10
    assert abs(st[6].real - 0.5) < 1e-10
    circuit, params = amplitude_encoder([0, 0, 0.5, 0.5, 0.5, 0.5], 3)
    sim.reset()
    sim.apply_circuit(circuit, params)
    st = sim.get_qs(False)
    assert abs(st[1].real - 0.5) < 1e-10
    assert abs(st[2].real - 0.5) < 1e-10
    assert abs(st[5].real - 0.5) < 1e-10
    assert abs(st[6].real - 0.5) < 1e-10
    circuit, params = amplitude_encoder([0.5, -0.5, 0.5, 0.5], 3)
    sim.reset()
    sim.apply_circuit(circuit, params)
    st = sim.get_qs(False)
    assert abs(st[0].real - 0.5) < 1e-10
    assert abs(st[2].real - 0.5) < 1e-10
    assert abs(st[4].real + 0.5) < 1e-10
    assert abs(st[6].real - 0.5) < 1e-10
