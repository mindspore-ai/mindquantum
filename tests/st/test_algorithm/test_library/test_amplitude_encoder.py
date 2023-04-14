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

import warnings

import pytest

from mindquantum.simulator.available_simulator import SUPPORTED_SIMULATOR

with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=UserWarning, message='MindSpore not installed.*')
    warnings.filterwarnings(
        'ignore', category=DeprecationWarning, message=r'Please use `OptimizeResult` from the `scipy\.optimize`'
    )

    from mindquantum.algorithm.library import amplitude_encoder
    from mindquantum.simulator import Simulator

AVAILABLE_BACKEND = list(SUPPORTED_SIMULATOR)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('config', AVAILABLE_BACKEND)
def test_amplitude_encoder(config):
    '''
    Feature: amplitude_encoder
    Description: test amplitude encoder.
    Expectation: success.
    '''
    backend, dtype = config
    sim = Simulator(backend, 3, dtype=dtype)
    circuit, params = amplitude_encoder([0.5, 0.5, 0.5, 0.5], 3)
    sim.apply_circuit(circuit, params)
    state = sim.get_qs(False)
    if backend == "mqmatrix":
        assert abs(state[0][0].real - 0.25) < 1e-6
        assert abs(state[1][1].real - 0.25) < 1e-6
        assert abs(state[2][2].real - 0.25) < 1e-6
        assert abs(state[3][3].real - 0.25) < 1e-6
    else:
        assert abs(state[0].real - 0.5) < 1e-6
        assert abs(state[1].real - 0.5) < 1e-6
        assert abs(state[2].real - 0.5) < 1e-6
        assert abs(state[3].real - 0.5) < 1e-6
    circuit, params = amplitude_encoder([0, 0, 0.5, 0.5, 0.5, 0.5], 3)
    sim.reset()
    sim.apply_circuit(circuit, params)
    state = sim.get_qs(False)
    if backend == "mqmatrix":
        assert abs(state[2][2].real - 0.25) < 1e-6
        assert abs(state[3][3].real - 0.25) < 1e-6
        assert abs(state[4][4].real - 0.25) < 1e-6
        assert abs(state[5][5].real - 0.25) < 1e-6
    else:
        assert abs(state[2].real - 0.5) < 1e-6
        assert abs(state[3].real - 0.5) < 1e-6
        assert abs(state[4].real - 0.5) < 1e-6
        assert abs(state[5].real - 0.5) < 1e-6
    circuit, params = amplitude_encoder([0.5, -0.5, 0.5, 0.5], 3)
    sim.reset()
    sim.apply_circuit(circuit, params)
    state = sim.get_qs(False)
    if backend == "mqmatrix":
        assert abs(state[0][0].real - 0.25) < 1e-6
        assert abs(state[1][1].real - 0.25) < 1e-6
        assert abs(state[2][2].real - 0.25) < 1e-6
        assert abs(state[3][3].real - 0.25) < 1e-6
    else:
        assert abs(state[0].real - 0.5) < 1e-6
        assert abs(state[1].real + 0.5) < 1e-6
        assert abs(state[2].real - 0.5) < 1e-6
        assert abs(state[3].real - 0.5) < 1e-6
