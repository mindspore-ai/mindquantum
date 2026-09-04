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
"""实验性 Ascend/NPU 向量后端辅助工具。"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from mindquantum import mqbackend

NPU_BACKEND_NAME = "mqvector_npu"
NPU_EXTENSION_NAME = "mindquantum._mq_vector_npu"
NPU_DISABLED_REASON = (
    "NPU backend is not available. This backend requires a CANN/Ascend build with the experimental "
    "'mqvector_npu' module and the AscendCL runtime."
)


def probe_npu_backend() -> Tuple[Optional[object], Optional[str]]:
    """尝试导入并验证最小态演化，确认实验性 NPU 向量后端是否可用。"""
    try:
        from mindquantum import _mq_vector_npu
    except ImportError:
        return None, NPU_DISABLED_REASON

    try:
        if not hasattr(_mq_vector_npu, "float"):
            raise ImportError("The mqvector_npu backend module does not expose its complex64 submodule.")
        sim = _mq_vector_npu.float.mqvector_npu(1)
        if sim.sim_name() != NPU_BACKEND_NAME:
            raise ImportError("The mqvector_npu backend reports an unexpected simulator name.")
        sim.apply_gate(mqbackend.gate.HGate([0]))
        state = np.asarray(sim.get_qs())
        expected = np.array([1 / np.sqrt(2), 1 / np.sqrt(2)], dtype=np.complex64)
        if state.shape != (2,):
            raise ImportError("The mqvector_npu backend returned an invalid one-qubit state vector.")
        if not np.all(np.isfinite(state)):
            raise ImportError("The mqvector_npu backend returned a non-finite one-qubit state vector.")
        if not np.allclose(state, expected, atol=1e-5, rtol=1e-5):
            raise ImportError("The mqvector_npu backend failed the H|0> state-vector smoke check.")
        return _mq_vector_npu, None
    except Exception as exc:  # pylint: disable=broad-except
        return None, f"NPU backend is not available due to: {exc}"


__all__ = [
    "NPU_BACKEND_NAME",
    "NPU_DISABLED_REASON",
    "NPU_EXTENSION_NAME",
    "probe_npu_backend",
]
