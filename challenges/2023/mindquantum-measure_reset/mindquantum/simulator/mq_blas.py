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
"""BLAS operator of Mindquantum simulator."""

from typing import Union

from mindquantum.utils.type_value_check import _check_input_type

# This import is required to register some of the C++ types (e.g. ParameterResolver)
from .. import mqbackend  # noqa: F401  # pylint: disable=unused-import

# isort: split

from mindquantum.simulator.available_simulator import SUPPORTED_SIMULATOR

from .mqsim import MQSim


class MQBlas:  # pylint: disable=too-few-public-methods
    """MindQuantum blas module."""

    @staticmethod
    def inner_product(bra: MQSim, ket: MQSim) -> Union[float, complex]:
        """Get the inner product of two quantum state."""
        _check_input_type("bra", MQSim, bra)
        _check_input_type("ket", MQSim, ket)
        if bra.name != ket.name:
            raise ValueError("inner product can only do between two same simulator backend.")
        if bra.n_qubits != ket.n_qubits:
            raise ValueError("Qubit of these two vector state should be same.")
        blas = MQBlas._get_blas(bra)
        return blas.inner_product(bra.sim, ket.sim)

    @staticmethod
    def _get_blas(simulator: MQSim):
        """Get blas module w.r.t given backend."""
        return SUPPORTED_SIMULATOR.c_module(simulator.name, simulator.dtype).blas
