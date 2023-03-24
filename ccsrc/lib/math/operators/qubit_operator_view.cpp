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

#include "math/operators/qubit_operator_view.hpp"

#include <cstdint>
#include <iostream>

namespace operators::qubit {
SinglePauliStr::SinglePauliStr(const std::vector<std::tuple<TermValue, size_t>>& terms, const tn::Tensor& t) {
    for (auto& [term, idx] : terms) {
        this->InplaceMulPauli(term, idx);
    }
}

void SinglePauliStr::InplaceMulPauli(TermValue term, size_t idx) {
    if (term == TermValue::I) {
        return;
    }
    size_t group_id = idx >> 5;
    size_t local_id = ((idx & 31) << 1);
    size_t local_mask = (1UL << local_id) | (1UL << (local_id + 1));
    if (this->pauli_string.size() < group_id + 1) {
        for (size_t i = pauli_string.size(); i < group_id + 1; i++) {
            this->pauli_string.push_back(0);
        }
        pauli_string[group_id] = pauli_string[group_id] & (~local_mask) | (static_cast<uint8_t>(term)) << local_id;
    } else {
        TermValue lhs = static_cast<TermValue>((pauli_string[group_id] & local_mask) >> local_id);
        auto [t, res] = pauli_product_map.at(lhs).at(term);
        this->coeff = this->coeff * t;
        pauli_string[group_id] = pauli_string[group_id] & (~local_mask) | (static_cast<uint8_t>(res)) << local_id;
    }
}
}  // namespace operators::qubit
