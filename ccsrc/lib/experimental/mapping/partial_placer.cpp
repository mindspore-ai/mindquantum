//   Copyright 2021 <Huawei Technologies Co., Ltd>
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

#include "experimental/mapping/partial_placer.hpp"

#include <algorithm>
#include <cassert>
#include <type_traits>
#include <vector>

namespace mindquantum::mapping {
PartialPlacer::PartialPlacer(const device_t& device, placement_t& placement) : device_(device), placement_(placement) {
}

void PartialPlacer::run(const std::vector<qubit_t>& new_qubits) {
    const auto num_v_qubits = std::count_if(begin(placement_.v_to_phy()), end(placement_.v_to_phy()),
                                            [](const auto& v) { return v != qubit_t::invalid(); });
    assert(device_.num_qubits() >= num_v_qubits + size(new_qubits));

    const auto available_v_qubits = size(placement_.v_to_phy()) - num_v_qubits;

    // Dumb algorithm... simply do first come first serve
    std::vector<uint32_t> available_phys;
    for (auto phy(0UL); phy < device_.num_qubits(); ++phy) {
        if (placement_.phy_to_v(phy) == qubit_t::invalid()) {
            available_phys.emplace_back(phy);
        }
    }

    assert(std::size(new_qubits) <= std::size(available_phys));

    if (available_v_qubits >= size(new_qubits)) {
        for (const auto& new_qubit : new_qubits) {
            assert(placement_.v_to_phy(new_qubit) == qubit_t::invalid());

            auto phy = available_phys.back();
            placement_.map_v_phy(new_qubit, phy);
            available_phys.pop_back();
        }
    } else {
        placement_t new_placement(device_.num_qubits(), device_.num_qubits());
        for (const auto& phy : placement_.v_to_phy()) {
            if (phy != qubit_t::invalid()) {
                new_placement.map_v_phy(placement_.phy_to_v(phy), phy);
            }
        }

        for (const auto& new_qubit : new_qubits) {
            assert(new_placement.v_to_phy(new_qubit) == qubit_t::invalid());

            auto phy = available_phys.back();
            new_placement.map_v_phy(new_qubit, phy);
            available_phys.pop_back();
        }

        placement_ = std::move(new_placement);
    }
}
}  // namespace mindquantum::mapping
