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
"""Generate graph."""
import networkx as nx
import json
import os

current_path = os.path.dirname(os.path.abspath(__file__))


def prepare_4_regular(d_path: str, qubit_range: list, seed: int = None):
    """
    Prepare a 4 regular graph.
    """
    for i in qubit_range:
        data_name = f"regular_4_qubit_{str(i).zfill(2)}.json"
        graph = nx.generators.random_regular_graph(4, i, seed=seed)
        out = []
        for i, j in graph.edges:
            out.append((i, j))
        path = os.path.join(os.path.abspath(d_path), data_name)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    data_path = os.path.join(current_path, "../data")
    prepare_4_regular(data_path, range(5, 28), seed=42)
