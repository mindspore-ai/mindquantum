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
#include <stdexcept>
#include <string>

#include "math/tensor/ops/memory_operator.hpp"
#include "math/tensor/tensor.hpp"
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

auto SinglePauliStr::init(const std::string& pauli_string, const tn::Tensor& coeff) -> pauli_t {
    pauli_t out = {key_t{0}, coeff};
    std::istringstream iss(pauli_string);
    for (std::string s; iss >> s;) {
        auto [term, idx] = parse_token(s);
        InplaceMulPauli(term, idx, out);
    }
    return out;
}

auto SinglePauliStr::init(const std::vector<std::tuple<TermValue, size_t>>& terms, const tn::Tensor& coeff) -> pauli_t {
    pauli_t out = {key_t{0}, coeff};
    for (auto& [term, idx] : terms) {
        InplaceMulPauli(term, idx, out);
    }
    return out;
}

void SinglePauliStr::InplaceMulPauli(TermValue term, size_t idx, pauli_t& pauli) {
    if (term == TermValue::I) {
        return;
    }
    size_t group_id = idx >> 5;
    size_t local_id = ((idx & 31) << 1);
    size_t local_mask = (1UL << local_id) | (1UL << (local_id + 1));
    auto& [pauli_string, coeff] = pauli;
    if (pauli_string.size() < group_id + 1) {
        for (size_t i = pauli_string.size(); i < group_id + 1; i++) {
            pauli_string.push_back(0);
        }
        pauli_string[group_id] = pauli_string[group_id] & (~local_mask) | (static_cast<uint8_t>(term)) << local_id;
    } else {
        TermValue lhs = static_cast<TermValue>((pauli_string[group_id] & local_mask) >> local_id);
        auto [t, res] = pauli_product_map.at(lhs).at(term);
        coeff = coeff * t;
        pauli_string[group_id] = pauli_string[group_id] & (~local_mask) | (static_cast<uint8_t>(res)) << local_id;
    }
}

bool SinglePauliStr::IsSameString(const key_t& k1, const key_t& k2) {
    if (k1[0] != k2[0]) {
        return false;
    }
    for (int i = 1; i < std::max(k1.size(), k2.size()); i++) {
        uint64_t this_pauli, other_pauli;
        if (i >= k1.size()) {
            this_pauli = 0;
            other_pauli = k2[i];
        } else if (i >= k2.size()) {
            this_pauli = k1[i];
            other_pauli = 0;
        } else {
            this_pauli = k1[i];
            other_pauli = k2[i];
        }
        if (this_pauli != other_pauli) {
            return false;
        }
    }
    return true;
}

std::string SinglePauliStr::GetString(const pauli_t& pauli) {
    std::string out = "";
    int group_id = 0;
    auto& [pauli_string, coeff] = pauli;
    for (auto& i : pauli_string) {
        auto k = i;
        int local_id = 0;
        while (k != 0) {
            auto j = static_cast<TermValue>(k & 3);
            switch (j) {
                case TermValue::X:
                    out += "X";
                    out += std::to_string(local_id + group_id * 32);
                    out += " ";
                    break;
                case TermValue::Y:
                    out += "Y";
                    out += std::to_string(local_id + group_id * 32);
                    out += " ";
                    break;
                case TermValue::Z:
                    out += "Z";
                    out += std::to_string(local_id + group_id * 32);
                    out += " ";
                    break;
                default:
                    break;
            }
            local_id += 1;
            k = k >> 2;
        }
        group_id += 1;
    }
    return tn::ops::to_string(coeff, true) + " [" + out + "]";
}

auto SinglePauliStr::Mul(const pauli_t& lhs, const pauli_t& rhs) -> pauli_t {
    auto& [l_k, l_v] = lhs;
    auto& [r_k, r_v] = rhs;
    key_t pauli_string = {};
    value_t coeff = l_v * r_v;
    if (l_k.size() == 1 && r_k.size() == 1) {
        auto [t, s] = mul_pauli_str(l_k[0], r_k[0]);
        coeff = coeff * t;
        pauli_string.push_back(s);
        return {pauli_string, coeff};
    }
    int min_size = std::min(l_k.size(), r_k.size());
    int max_size = std::max(l_k.size(), r_k.size());
    for (int i = 0; i < max_size; i++) {
        if (i < min_size) {
            auto [t, s] = mul_pauli_str(l_k[i], r_k[i]);
            coeff = coeff * t;
            pauli_string.push_back(s);
        } else if (i >= l_k.size()) {
            pauli_string.push_back(r_k[i]);
        } else {
            pauli_string.push_back(l_k[i]);
        }
    }
    return {pauli_string, coeff};
}

// bool SinglePauliStr::operator<(const SinglePauliStr& other) const {
//     if (this->pauli_string.size() == 1 && other.pauli_string.size() == 1) {
//         return this->pauli_string[0] < other.pauli_string[0];
//     }
//     for (int i = std::max(this->pauli_string.size(), other.pauli_string.size()) - 1; i >= 0; i--) {
//         uint64_t lhs, rhs;
//         if (i >= this->pauli_string.size()) {
//             lhs = 0;
//         } else {
//             lhs = this->pauli_string[i];
//         }
//         if (i >= other.pauli_string.size()) {
//             rhs = 0;
//         } else {
//             rhs = other.pauli_string[i];
//         }
//         if (lhs < rhs) {
//             return true;
//         }
//         if (lhs > rhs) {
//             return false;
//         }
//     }
//     return false;
// }

std::tuple<tn::Tensor, uint64_t> mul_pauli_str(uint64_t a, uint64_t b) {
    auto res = (~a & b) | (a & ~b);
    auto idx_0 = (~(a >> 1) & a & (b >> 1)) | (a & (b >> 1) & ~b) | ((a >> 1) & ~a & b) | ((a >> 1) & ~(b >> 1) & b);
    auto idx_1 = (~(a >> 1) & a & (b >> 1) & b) | ((a >> 1) & ~a & ~(b >> 1) & b) | ((a >> 1) & a & (b >> 1) & ~b);
    idx_0 = idx_0 & M_B;
    idx_1 = idx_1 & M_B;
    auto num_I = __builtin_popcount(~idx_1 & idx_0);
    auto num_M_ONE = __builtin_popcount(idx_1 & ~idx_0);
    auto num_M_I = __builtin_popcount(idx_1 & idx_0);
    auto out = (num_I + 2 * num_M_ONE + 3 * num_M_I) & 3;
    switch (out) {
        case (0):
            return {tn::ops::init_with_value(static_cast<double>(1.0)), res};
        case (1):
            return {tn::ops::init_with_value(std::complex<double>(0.0, 1.0)), res};
        case (2):
            return {tn::ops::init_with_value(static_cast<double>(-1.0)), res};
        case (3):
            return {tn::ops::init_with_value(std::complex<double>(0.0, -1.0)), res};
        default:
            throw std::runtime_error("Error in multiply of pauli string.");
    }
}

// -----------------------------------------------------------------------------

QubitOperator::QubitOperator(const std::string& pauli_string, const tn::Tensor& coeff) {
    auto term = SinglePauliStr::init(pauli_string, coeff);
    this->terms.insert(term.first, term.second);
}

bool QubitOperator::Contains(const key_t& term) const {
    return this->terms.m_map.find(term) != this->terms.m_map.end();
}

void QubitOperator::Update(const pauli_t& pauli) {
    if (this->Contains(pauli.first)) {
        this->terms[pauli.first] = pauli.second;
    } else {
        this->terms.insert(pauli);
    }
}
size_t QubitOperator::size() const {
    return this->terms.size();
}

std::string QubitOperator::ToString() const {
    if (this->size() == 0) {
        return "0";
    }
    std::string out = "";
    for (const auto& term : this->terms) {
        out += SinglePauliStr::GetString(term) + "\n";
    }
    return out;
}

QubitOperator& QubitOperator::operator+=(const tn::Tensor& c) {
    if (this->Contains(key_t{0})) {
        this->terms[key_t{0}] = this->terms[key_t{0}] + c;
    } else {
        this->terms.insert({key_t{0}, c});
    }
    return *this;
}

QubitOperator QubitOperator::operator+(const tn::Tensor& c) {
    auto out = *this;
    if (out.Contains(key_t{0})) {
        out.terms[key_t{0}] = out.terms[key_t{0}] + c;
    } else {
        out.terms.insert({key_t{0}, c});
    }
    return out;
}

QubitOperator& QubitOperator::operator+=(const QubitOperator& other) {
    for (const auto& term : other.terms) {
        if (this->Contains(term.first)) {
            // std::cout << "Before:" << this->terms[term.first] << " " << term.second << std::endl;
            this->terms[term.first] = this->terms[term.first] + term.second;
            // std::cout << "After: " << this->terms[term.first] << std::endl;
        } else {
            this->terms.insert(term);
        }
    }
    std::cout << "--- " << this->ToString() << std::endl;
    return *this;
}
QubitOperator QubitOperator::operator+(const QubitOperator& other) {
    auto out = *this;
    std::cout << out.ToString() << std::endl;
    out += other;
    std::cout << out.ToString() << std::endl;
    std::cout << other.ToString() << std::endl;

    return out;
}

QubitOperator QubitOperator::operator*=(const QubitOperator& other) {
    auto out = QubitOperator();
    for (auto& this_term : this->terms) {
        for (const auto& other_term : other.terms) {
            auto new_term = SinglePauliStr::Mul(this_term, other_term);
            if (out.Contains(new_term.first)) {
                out.terms[new_term.first] = this->terms[new_term.first] + new_term.second;
            } else {
                out.terms.insert(new_term);
            }
        }
    }
    return out;
}

QubitOperator QubitOperator::operator*(const QubitOperator& other) {
    auto out = QubitOperator();
    for (auto& this_term : this->terms) {
        for (const auto& other_term : other.terms) {
            auto new_term = SinglePauliStr::Mul(this_term, other_term);
            if (out.Contains(new_term.first)) {
                out.terms[new_term.first] = this->terms[new_term.first] + new_term.second;
            } else {
                out.terms.insert(new_term);
            }
        }
    }
    return out;
}
}  // namespace operators::qubit
