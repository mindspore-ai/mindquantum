# Copyright 2023 Huawei Technologies Co., Ltd
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
"""Generate random circuit."""

import json
import os

from mindquantum.utils import random_circuit
from circuit_convert import convert_circ
from mindquantum.algorithm.compiler import BasicDecompose, compile_circuit
from mindquantum.core import U3, Circuit

current_path = os.path.dirname(os.path.abspath(__file__))


def prepare_random_circ(
    d_path: str, qubit_range: list, size: int = 100, seed: int = None
):
    """
    Prepare random circuit data.
    """
    for i in qubit_range:
        data_name = f"simple_circuit_qubit_{str(i).zfill(2)}_size_{size}.json"
        simple_circ = compile_circuit(
            BasicDecompose(), random_circuit(i, size, seed=seed)
        )
        circ = convert_circ(Circuit([i for i in simple_circ if not isinstance(i, U3)]))
        path = os.path.join(os.path.abspath(d_path), data_name)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(circ, f, indent=2)


if __name__ == "__main__":
    data_path = os.path.join(current_path, "../data")
    prepare_random_circ(data_path, range(4, 28), seed=42)
