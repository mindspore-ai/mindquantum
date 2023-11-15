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
"""List available simulators."""
import typing
import warnings

from mindquantum import _mq_matrix, _mq_vector, mqbackend
from mindquantum.dtype import complex64, complex128
from mindquantum.simulator.backend_base import BackendBase
from mindquantum.utils.error import SimNotAvailableError

try:
    from mindquantum import _mq_vector_gpu

    # pylint: disable=no-member
    _mq_vector_gpu.double.mqvector_gpu(1).apply_gate(mqbackend.gate.HGate([0]))
    MQVECTOR_GPU_SUPPORTED = True
except ImportError as err:
    warnings.warn(f"Unable import mqvector gpu backend due to: {err}", stacklevel=2)
    MQVECTOR_GPU_SUPPORTED = False
except RuntimeError as err:
    warnings.warn(f"Disable mqvector gpu backend due to: {err}", stacklevel=2)
    MQVECTOR_GPU_SUPPORTED = False


class _AvailableSimulator:
    """Set available simulator."""

    def __init__(self):
        """Init available simulator obj."""
        self.base_module = {
            'mqvector': _mq_vector,
            'mqmatrix': _mq_matrix,
        }
        self.sims = {
            'mqvector': {
                complex64: _mq_vector.float,
                complex128: _mq_vector.double,
            },
            'mqmatrix': {
                complex64: _mq_matrix.float,
                complex128: _mq_matrix.double,
            },
        }
        if MQVECTOR_GPU_SUPPORTED:
            self.base_module['mqvector_gpu'] = _mq_vector_gpu
            self.sims['mqvector_gpu'] = {
                complex64: _mq_vector_gpu.float,
                complex128: _mq_vector_gpu.double,
            }

    def is_available(self, sim: typing.Union[str, BackendBase], dtype) -> bool:
        """Check a simulator with given data type is available or not."""
        if isinstance(sim, BackendBase):
            return True
        if sim in self.sims and dtype in self.sims[sim]:
            return True
        return False

    def c_module(self, sim: str, dtype=None):
        """Get available simulator c module."""
        if dtype is None:
            if sim not in self.base_module:
                raise SimNotAvailableError(sim)
            return self.base_module[sim]
        if not self.is_available(sim, dtype):
            raise SimNotAvailableError(sim, dtype)
        return self.sims[sim][dtype]

    def py_class(self, sim: str):
        """Get python base class of simulator."""
        if sim in self.sims:
            if sim in ['mqvector', 'mqvector_gpu', 'mqmatrix']:
                # pylint: disable=import-outside-toplevel
                from mindquantum.simulator.mqsim import MQSim

                return MQSim
            raise SimNotAvailableError(sim)
        raise SimNotAvailableError(sim)

    def __iter__(self):
        """List available simulator with data type."""
        for k, v in self.sims.items():
            for dtype in v:
                yield [k, dtype]


SUPPORTED_SIMULATOR = _AvailableSimulator()
