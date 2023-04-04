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

#include "math/operators/transform.hpp"

namespace operators::transform {
qubit_op_t bravyi_kitaev(const fermion_op_t& ops, int n_qubits) {
    auto transf_op = qubit_op_t();
    for (const auto& [term, coeff] : ops.get_terms()) {
        auto transformed_term = qubit_op_t("", coeff);
        for (const auto& [idx, value] : term) {
            auto update_set_ = update_set(idx, n_qubits);
            auto occupation_set_ = occupation_set(idx);
            auto parity_set_ = parity_set(idx - 1);
            qlist_t x1(update_set_.begin(), update_set_.end());
            qlist_t y1 = {};
            qlist_t z1(parity_set_.begin(), parity_set_.end());
            std::unordered_set<qubit_op_t::term_t::first_type> x2_set(update_set_);
            x2_set.erase(idx);
            qlist_t x2(x2_set.begin(), x2_set.end());
            qlist_t y2 = {idx};
            std::unordered_set<qubit_op_t::term_t::first_type> z2_set(parity_set_);
            for (auto it = occupation_set_.begin(); it != occupation_set_.end(); it++) {
                if (z2_set.count(*it)) {
                    z2_set.erase(*it);
                } else {
                    z2_set.insert(*it);
                }
            }
            z2_set.erase(idx);
            qlist_t z2(z2_set.begin(), z2_set.end());
            transformed_term *= transform_ladder_operator(value, x1, y1, z1, x2, y2, z2);
        }
        transf_op += transformed_term;
    }
    return transf_op;
}

std::unordered_set<qubit_op_t::term_t::first_type> parity_set(qubit_op_t::term_t::first_type idx) {
    /*
    Bits whose parity store the parity of the bits 0 .. `index`.
    Used in Bravyi-Kitaev transform.
    */
    std::unordered_set<qubit_op_t::term_t::first_type> indices;
    qubit_op_t::term_t::first_type index = idx + 1;
    while (index > 0) {
        indices.insert(index - 1);
        index &= index - 1;
    }
    return indices;
}
std::unordered_set<qubit_op_t::term_t::first_type> occupation_set(qubit_op_t::term_t::first_type idx) {
    /*
    Bits whose parity stores the occupation of mode `index`.
    Used in Bravyi-Kitaev transform.
    */
    std::unordered_set<qubit_op_t::term_t::first_type> indices;
    qubit_op_t::term_t::first_type index = idx + 1;
    indices.insert(index - 1);
    qubit_op_t::term_t::first_type parent = index & (index - 1);
    index -= 1;
    while (index != parent) {
        indices.insert(index - 1);
        index &= index - 1;
    }
    return indices;
}
std::unordered_set<qubit_op_t::term_t::first_type> update_set(qubit_op_t::term_t::first_type idx, int n_qubits) {
    /*
    Bits that need to be updated upon flipping the occupancy of a mode.
    Used in Bravyi-Kitaev transform.
    */
    std::unordered_set<qubit_op_t::term_t::first_type> indices;
    qubit_op_t::term_t::first_type index = idx + 1;
    while (index <= n_qubits) {
        indices.insert(index - 1);
        index += index & (-index);
    }
    return indices;
}
}  // namespace operators::transform
