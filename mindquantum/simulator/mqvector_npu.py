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
"""Experimental Ascend/NPU state vector simulator backend."""

from __future__ import annotations

from mindquantum.dtype import complex64
from mindquantum.simulator.mqsim import MQSim
from mindquantum.utils.type_value_check import _check_mq_type


class MQVectorNPU(MQSim):
    """Experimental Ascend/NPU backend for the state vector simulator."""

    def __init__(self, name, n_qubits, seed=None, dtype=complex64, internal=False, **kwargs):
        """Initialize a complex64-only NPU simulator."""
        if not internal:
            if dtype is None:
                dtype = complex64
            _check_mq_type(dtype)
            if dtype != complex64:
                raise TypeError("mqvector_npu only supports mindquantum.complex64.")
        super().__init__(name, n_qubits, seed, dtype, internal, **kwargs)

    def get_qs_of_qubits(self, qubits, ket=False):
        """Reject reduced-state extraction that is not implemented by the NPU backend."""
        if isinstance(qubits, int):
            qubits = [qubits]
        if not isinstance(qubits, (list, tuple)):
            raise TypeError(f"qubits requires an int or a list/tuple of ints, but get {type(qubits)}")
        if not all(isinstance(qubit, int) for qubit in qubits):
            raise TypeError("all elements in qubits must be integers")
        if not all(0 <= qubit < self.n_qubits for qubit in qubits):
            raise ValueError(f"qubit indices must be between 0 and {self.n_qubits-1}")
        if len(set(qubits)) == self.n_qubits:
            return self.get_qs(ket=ket)
        raise NotImplementedError(
            "mqvector_npu does not implement reduced state extraction for a subset of qubits."
        )

    def set_threads_number(self, number):
        """Reject CPU-only thread tuning for the NPU backend."""
        raise NotImplementedError("set_threads_number is not supported for mqvector_npu.")

    def copy(self):
        """Copy the NPU simulator."""
        sim = self.__class__(self.sim.copy(), self.n_qubits, self.seed, dtype=complex64, internal=True)
        sim.name = self.name
        sim._init_kwargs = self._init_kwargs.copy()
        return sim

    def astype(self, dtype, seed):
        """Convert the NPU simulator to another complex precision."""
        _check_mq_type(dtype)
        if dtype != complex64:
            raise TypeError("mqvector_npu only supports mindquantum.complex64.")
        sim = self.__class__(self.sim.copy(), self.n_qubits, seed, dtype=complex64, internal=True)
        sim.name = self.name
        sim._init_kwargs = self._init_kwargs.copy()
        return sim


__all__ = ["MQVectorNPU"]
