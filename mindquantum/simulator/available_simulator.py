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

import os
import typing
import warnings

from mindquantum import _mq_matrix, _mq_mps, _mq_stabilizer, _mq_vector, mqbackend
from mindquantum.dtype import complex64, complex128
from mindquantum.simulator.backend_base import BackendBase
from mindquantum.simulator.npu_backend import probe_npu_backend
from mindquantum.utils.error import SimNotAvailableError

GPU_DISABLED_REASON = None
CUQUANTUM_DISABLED_REASON = None
NPU_DISABLED_REASON = None
MQVECTOR_GPU_SUPPORTED = False
MQVECTOR_CUQUANTUM_SUPPORTED = False
MQVECTOR_NPU_SUPPORTED = False

try:
    from mindquantum import _mq_vector_gpu

    # pylint: disable=no-member
    _mq_vector_gpu.double.mqvector_gpu(1).apply_gate(mqbackend.gate.HGate([0]))
    MQVECTOR_GPU_SUPPORTED = True
except ImportError:
    GPU_DISABLED_REASON = "GPU backend is not available. This backend requires CUDA 11 or higher."
except Exception as e:  # pylint: disable=broad-except
    GPU_DISABLED_REASON = f"GPU backend is not available due to: {e}"

if MQVECTOR_GPU_SUPPORTED:
    if hasattr(_mq_vector_gpu, 'cuquantum_float'):
        MQVECTOR_CUQUANTUM_SUPPORTED = True
    else:
        CUQUANTUM_DISABLED_REASON = (
            "The 'mqvector_cq' backend is not available. "
            "Please build with ENABLE_CUQUANTUM=ON and ensure that NVIDIA's cuQuantum SDK is installed correctly."
        )
else:
    CUQUANTUM_DISABLED_REASON = GPU_DISABLED_REASON

_mq_vector_npu, _npu_disabled_reason = probe_npu_backend()
if _mq_vector_npu is not None:
    MQVECTOR_NPU_SUPPORTED = True
else:
    NPU_DISABLED_REASON = _npu_disabled_reason


class _AvailableSimulator:
    """Set available simulator."""

    def __init__(self):
        """Init available simulator obj."""
        self.base_module = {
            'mqvector': _mq_vector,
            'mqmatrix': _mq_matrix,
            'stabilizer': _mq_stabilizer,
            'mqmps': _mq_mps,
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
            'mqmps': {
                complex64: _mq_mps.float,
                complex128: _mq_mps.double,
            },
            'stabilizer': _mq_stabilizer,
        }
        if MQVECTOR_GPU_SUPPORTED:
            self.base_module['mqvector_gpu'] = _mq_vector_gpu
            self.sims['mqvector_gpu'] = {
                complex64: _mq_vector_gpu.float,
                complex128: _mq_vector_gpu.double,
            }
        if MQVECTOR_CUQUANTUM_SUPPORTED:
            self.base_module['mqvector_cq'] = _mq_vector_gpu
            self.sims['mqvector_cq'] = {
                complex64: _mq_vector_gpu.cuquantum_float,
                complex128: _mq_vector_gpu.cuquantum_double,
            }
        if MQVECTOR_NPU_SUPPORTED:
            self.base_module['mqvector_npu'] = _mq_vector_npu
            self.sims['mqvector_npu'] = {
                complex64: _mq_vector_npu.float,
            }

    def is_available(self, sim: typing.Union[str, BackendBase], dtype) -> bool:
        """Check a simulator with given data type is available or not."""
        if isinstance(sim, BackendBase):
            return True
        if sim == 'stabilizer':
            return True
        if sim in self.sims and dtype in self.sims.get(sim, {}):
            return True
        return False

    def c_module(self, sim: str, dtype=None):
        """Get available simulator c module."""
        if sim == 'mqvector_gpu' and not MQVECTOR_GPU_SUPPORTED:
            warnings.warn(f"{GPU_DISABLED_REASON}", stacklevel=3)
        if sim == 'mqvector_cq' and not MQVECTOR_CUQUANTUM_SUPPORTED:
            warnings.warn(f"{CUQUANTUM_DISABLED_REASON}", stacklevel=3)
        if sim == 'mqvector_npu' and not MQVECTOR_NPU_SUPPORTED:
            warnings.warn(f"{NPU_DISABLED_REASON}", stacklevel=3)

        if dtype is None:
            if sim not in self.base_module:
                raise SimNotAvailableError(sim)
            return self.base_module[sim]
        if not self.is_available(sim, dtype):
            raise SimNotAvailableError(sim, dtype)
        return self.sims[sim][dtype]

    def py_class(self, sim: str):
        """Get python base class of simulator."""
        if sim == 'mqvector_gpu' and not MQVECTOR_GPU_SUPPORTED:
            raise ValueError(f"{GPU_DISABLED_REASON}")
        if sim == 'mqvector_cq' and not MQVECTOR_CUQUANTUM_SUPPORTED:
            raise ValueError(f"{CUQUANTUM_DISABLED_REASON}")
        if sim == 'mqvector_npu' and not MQVECTOR_NPU_SUPPORTED:
            raise ValueError(f"{NPU_DISABLED_REASON}")

        if sim in self.sims:
            if sim == 'mqvector_npu':
                # pylint: disable=import-outside-toplevel
                from mindquantum.simulator.mqvector_npu import MQVectorNPU

                return MQVectorNPU
            if sim in ['mqvector', 'mqvector_gpu', 'mqmatrix', 'mqvector_cq', 'mqmps']:
                # pylint: disable=import-outside-toplevel
                from mindquantum.simulator.mqsim import MQSim

                return MQSim
            if sim == 'stabilizer':
                # pylint: disable=import-outside-toplevel
                from mindquantum.simulator.stabilizer import Stabilizer

                return Stabilizer
        raise SimNotAvailableError(sim)

    def __iter__(self):
        """List available simulator with data type."""
        skip_mqmps = os.getenv("MQ_SKIP_MQMPS", "").lower() in {"1", "true", "yes", "on"}
        for k, v in self.sims.items():
            if skip_mqmps and k == "mqmps":
                continue
            if not isinstance(v, dict):
                yield k
            else:
                for dtype in v:
                    yield [k, dtype]


SUPPORTED_SIMULATOR = _AvailableSimulator()
