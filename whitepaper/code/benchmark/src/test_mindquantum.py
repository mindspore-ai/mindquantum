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
"""Benchmark MindQuantum."""

import os
import re
import pytest
import json

current_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_path, os.path.abspath("../data"))

circs = []

for file_name in os.listdir(data_path):
    if file_name.startswith("random_circuit"):
        n_qubits = int(re.search(r"\d+", file_name).group())
        full_path = os.path.join(data_path, file_name)
        with open(full_path, "r", encoding="utf-8") as f:
            str_circ = json.load(f)
