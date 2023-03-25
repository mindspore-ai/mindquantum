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

#include "math/tensor/traits.hpp"

namespace operators::qubit {
std::tuple<operators::qubit::TermValue, size_t> parse_token(const std::string& token) {
    if (token.size() <= 1) {
        throw std::runtime_error("Wrong token: '" + token + "'. Need a pauli word following a int, for example 'X0'.");
    }
    std::string pauli_word = token.substr(0, 1);
    operators::qubit::TermValue pauli;
    if (pauli_word == "X") {
        pauli = operators::qubit::TermValue::X;
    } else if (pauli_word == "Y") {
        pauli = operators::qubit::TermValue::Y;
    } else if (pauli_word == "Z") {
        pauli = operators::qubit::TermValue::Z;
    } else {
        throw std::runtime_error("Wrong token: '" + token + "'; Can not convert '" + pauli_word
                                 + "' to pauli operator.");
    }
    std::string idx_str = token.substr(1);
    int idx;
    try {
        size_t pos;
        idx = std::stoi(idx_str, &pos);
        if (pos != idx_str.length()) {
            throw std::runtime_error("");
        }
    } catch (const std::exception& e) {
        throw std::runtime_error("Wrong token: '" + token + "'; Can not convert '" + idx_str + "' to int.");
    }
    if (idx < 0) {
        throw std::runtime_error("Wrong token: '" + token + "'; Qubit index should not less than zero, but get "
                                 + idx_str + ".");
    }
    return {pauli, idx};
}

SinglePauliStr::SinglePauliStr(const std::string& pauli_string, const tn::Tensor& coeff) {
    this->coeff = coeff;
    std::istringstream iss(pauli_string);
    for (std::string s; iss >> s;) {
        auto [term, idx] = parse_token(s);
        this->InplaceMulPauli(term, idx);
    }
}

SinglePauliStr::SinglePauliStr(const std::vector<std::tuple<TermValue, size_t>>& terms, const tn::Tensor& coeff) {
    this->coeff = coeff;
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

SinglePauliStr SinglePauliStr::astype(tn::TDtype dtype) {
    auto out = *this;
    out.coeff = out.coeff.astype(dtype);
    return out;
}

bool SinglePauliStr::IsSameString(const SinglePauliStr& other) {
    if (this->pauli_string[0] != other.pauli_string[0]) {
        return false;
    }
    for (int i = 1; i < std::max(this->pauli_string.size(), other.pauli_string.size()); i++) {
        uint64_t this_pauli, other_pauli;
        if (i >= this->pauli_string.size()) {
            this_pauli = 0;
            other_pauli = other.pauli_string[i];
        } else if (i >= other.pauli_string.size()) {
            this_pauli = this->pauli_string[i];
            other_pauli = 0;
        } else {
            this_pauli = this->pauli_string[i];
            other_pauli = other.pauli_string[i];
        }
        if (this_pauli != other_pauli) {
            return false;
        }
    }
    return true;
}

std::string SinglePauliStr::GetString() const {
    std::string out = "";
    int group_id = 0;
    for (auto& i : this->pauli_string) {
        if (i != 0) {
            for (int j = 0; j < 32; j++) {
                
            }
        }
        group_id += 1;
    }
}
}  // namespace operators::qubit
