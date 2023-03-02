//   Copyright 2023 <Huawei Technologies Co., Ltd>
//
//   Licensed under the Apache License, Version 2.0 (the "License");
//   you may not use this file except in compliance with the License.
//   You may obtain a copy of the License at
//
//       http://www.apache.org/licenses/LICENSE-2.0
//
//   Unless required by applicable law or agreed to in writing, software
//   distributed under the License is distributed on an "AS IS" BASIS,
//   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//   See the License for the specific language governing permissions and
//   limitations under the License.

#include <iostream>

#include "device/mapping.hpp"
int main() {
    auto grid_qubits = mindquantum::mapping::GridQubits(3, 3);
    grid_qubits.RemoveQubitNode(4);
    std::cout << grid_qubits.size() << std::endl;
    auto edges = grid_qubits.EdgesWithID();
    for (auto& [id_1, id_2] : edges) {
        std::cout << id_1 << ", " << id_2 << std::endl;
    }
}
