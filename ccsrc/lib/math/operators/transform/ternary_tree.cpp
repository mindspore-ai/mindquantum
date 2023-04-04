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
qubit_op_t ternary_tree(const fermion_op_t& ops, int n_qubits) {
    int h = static_cast<int>(std::floor(std::log1p(2 * n_qubits) / std::log(3)));
    int d = n_qubits - (static_cast<int>(std::round(std::pow(3, h))) - 1) / 2;
    auto transf_op = qubit_op_t();
    for (const auto& [term, coeff] : ops.get_terms()) {
        auto transformed_term = qubit_op_t("", coeff);
        for (const auto& [idx, value] : term) {
            qlist_t p1 = {};
            if (2 * idx < 3 * d) {
                for (int k = h; k > -1; k--) {
                    p1.push_back((2 * idx / static_cast<int>(std::round(std::pow(3, k))) % 3));
                }
            } else {
                for (int k = h - 1; k > -1; k--) {
                    p1.push_back(((2 * idx - 2 * d) / static_cast<int>(std::round(std::pow(3, k))) % 3));
                }
            }
            qlist_t x1 = {};
            qlist_t y1 = {};
            qlist_t z1 = {};
            for (int k = 0; k < p1.size(); k++) {
                auto tmp = p1[k];
                if (tmp == 0) {
                    x1.push_back((get_qubit_index(p1, k)));
                } else if (tmp == 1) {
                    y1.push_back((get_qubit_index(p1, k)));
                } else {
                    z1.push_back((get_qubit_index(p1, k)));
                }
            }
            qlist_t p2 = {};
            if (2 * idx < 3 * d) {
                for (int k = h; k > -1; k--) {
                    p2.push_back(((2 * idx + 1) / static_cast<int>(std::round(std::pow(3, k))) % 3));
                }
            } else {
                for (int k = h - 1; k > -1; k--) {
                    p2.push_back(((2 * idx + 1 - 2 * d) / static_cast<int>(std::round(std::pow(3, k))) % 3));
                }
            }
            qlist_t x2 = {};
            qlist_t y2 = {};
            qlist_t z2 = {};
            for (int k = 0; k < p2.size(); k++) {
                auto tmp = p2[k];
                if (tmp == 0) {
                    x2.push_back((get_qubit_index(p2, k)));
                } else if (tmp == 1) {
                    y2.push_back((get_qubit_index(p2, k)));
                } else {
                    z2.push_back((get_qubit_index(p2, k)));
                }
            }
            transformed_term *= transform_ladder_operator(value, x1, y1, z1, x2, y2, z2);
        }
        transf_op += transformed_term;
    }
    return transf_op;
}

int get_qubit_index(const qlist_t& p, int i) {
    /*
    Get the qubit index.
    */
    int qubit_idx = static_cast<int>(std::round(std::pow(3, i) - 1)) / 2;
    for (int j = 0; j < i; j++) {
        qubit_idx += static_cast<int>(std::round(std::pow(3, i - 1 - j) * p[j]));
    }
    return qubit_idx;
}
}  // namespace operators::transform
