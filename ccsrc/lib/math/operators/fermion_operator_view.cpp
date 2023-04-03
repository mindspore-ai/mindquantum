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

#include "math/operators/fermion_operator_view.hpp"

#include <algorithm>
#include <cstdint>
#include <iostream>

#include <sys/types.h>

#include "math/operators/utils.hpp"
#include "math/pr/parameter_resolver.hpp"
#include "math/tensor/ops/concrete_tensor.hpp"
#include "math/tensor/tensor.hpp"

namespace operators::fermion {
auto SingleFermionStr::ParseToken(const std::string& token) -> term_t {
    if (token.size() == 0) {
        throw std::runtime_error(
            "Wrong token: '" + token
            + "'. Need a fermion word index following a dag or only fermion word index, for example '2^'.");
    }
    bool is_dag = (token.back() == '^');
    std::string idx_str = token;
    if (is_dag) {
        idx_str = token.substr(0, token.size() - 1);
    }
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
    if (is_dag) {
        return {idx, TermValue::Ad};
    }
    return {idx, TermValue::A};
}

auto SingleFermionStr::init(const std::string& fermion_string, const parameter::ParameterResolver& var)
    -> compress_term_t {
    compress_term_t out = {key_t{0}, var};
    std::istringstream iss(fermion_string);
    for (std::string s; iss >> s;) {
        InplaceMulCompressTerm(ParseToken(s), out);
    }
    return out;
}

auto SingleFermionStr::init(const terms_t& terms, const parameter::ParameterResolver& var) -> compress_term_t {
    compress_term_t out = {key_t{0}, var};
    for (auto& term : terms) {
        InplaceMulCompressTerm(term, out);
    }
    return out;
}

void SingleFermionStr::InplaceMulCompressTerm(const term_t& term, compress_term_t& fermion) {
    auto [idx, word] = term;
    if (word == TermValue::I) {
        return;
    }
    auto& [ori_term, coeff] = fermion;
    if ((word == TermValue::nll) || std::any_of(ori_term.begin(), ori_term.end(), [](auto j) {
            return j == static_cast<uint64_t>(TermValue::nll);
        })) {
        for (size_t i = 0; i < ori_term.size(); i++) {
            ori_term[i] = static_cast<uint64_t>(TermValue::nll);
        }
        return;
    }
    size_t group_id = idx / 21;
    size_t local_id = ((idx % 21) * 3);
    size_t low_mask = (1UL << local_id) - 1;
    size_t local_mask = (1UL << local_id) | (1UL << (local_id + 1)) | (1UL << (local_id + 2));
    if (ori_term.size() < group_id + 1) {
        for (size_t i = ori_term.size(); i < group_id + 1; i++) {
            ori_term.push_back(0);
        }
        ori_term[group_id] = ori_term[group_id] & (~local_mask) | (static_cast<uint64_t>(word)) << local_id;
    } else {
        TermValue lhs = static_cast<TermValue>((ori_term[group_id] & local_mask) >> local_id);
        auto res = fermion_product_map.at(lhs).at(word);
        if (res == TermValue::nll) {
            for (size_t i = 0; i < ori_term.size(); i++) {
                ori_term[i] = static_cast<uint64_t>(TermValue::nll);
            }
            return;
        }
        ori_term[group_id] = ori_term[group_id] & (~local_mask) | (static_cast<uint64_t>(res)) << local_id;
    }
    int count_one = 0;
    for (size_t i = 0; i < ori_term.size(); i++) {
        if (i == group_id) {
            count_one += __builtin_popcount(ori_term[i] & low_mask);
            break;
        } else {
            count_one += __builtin_popcount(ori_term[i]);
        }
    }
    if (count_one & 1) {
        coeff *= -1.0;
    }
}

bool SingleFermionStr::IsSameString(const key_t& k1, const key_t& k2) {
    if (k1[0] != k2[0]) {
        return false;
    }
    for (int i = 1; i < std::max(k1.size(), k2.size()); i++) {
        uint64_t this_fermion, other_fermion;
        if (i >= k1.size()) {
            this_fermion = 0;
            other_fermion = k2[i];
        } else if (i >= k2.size()) {
            this_fermion = k1[i];
            other_fermion = 0;
        } else {
            this_fermion = k1[i];
            other_fermion = k2[i];
        }
        if (this_fermion != other_fermion) {
            return false;
        }
    }
    return true;
}

std::string SingleFermionStr::GetString(const compress_term_t& fermion) {
    std::string out = "";
    int group_id = 0;
    auto& [fermion_string, coeff] = fermion;
    if (std::any_of(fermion_string.begin(), fermion_string.end(),
                    [](auto i) { return i == static_cast<uint64_t>(TermValue::nll); })) {
        return "0";
    }
    for (auto& i : fermion_string) {
        auto k = i;
        int local_id = 0;
        while (k != 0) {
            auto j = static_cast<TermValue>(k & 7);
            switch (j) {
                case TermValue::A:
                    out = " " + out;
                    out = std::to_string(local_id + group_id * 21) + out;
                    break;
                case TermValue::Ad:
                    out = "^ " + out;
                    out = std::to_string(local_id + group_id * 21) + out;
                    break;
                case TermValue::AAd:
                    out = "^ " + out;
                    out = std::to_string(local_id + group_id * 21) + out;
                    out = " " + out;
                    out = std::to_string(local_id + group_id * 21) + out;
                    break;
                case TermValue::AdA:
                    out = " " + out;
                    out = std::to_string(local_id + group_id * 21) + out;
                    out = "^ " + out;
                    out = std::to_string(local_id + group_id * 21) + out;
                    break;
                default:
                    break;
            }
            local_id += 1;
            k = k >> 3;
        }
        group_id += 1;
    }
    out.resize(out.find_last_not_of(" ") + 1);
    if (coeff.IsConst()) {
        return tn::ops::to_string(coeff.const_value, true) + " [" + out + "]";
    }
    return coeff.ToString() + " [" + out + "]";
}

auto SingleFermionStr::Mul(const compress_term_t& lhs, const compress_term_t& rhs) -> compress_term_t {
    auto& [l_k, l_v] = lhs;
    auto& [r_k, r_v] = rhs;
    key_t fermion_string = {};
    value_t coeff = l_v * r_v;
    if (l_k.size() == 1 && r_k.size() == 1) {
        auto [t, s] = MulSingleCompressTerm(l_k[0], r_k[0]);
        coeff = coeff * t;
        fermion_string.push_back(s);
        return {fermion_string, coeff};
    }
    int min_size = std::min(l_k.size(), r_k.size());
    int max_size = std::max(l_k.size(), r_k.size());
    int one_in_low = 0;
    int total_one = 0;
    for (int i = 0; i < max_size; i++) {
        if (i < min_size) {
            total_one += one_in_low;
            one_in_low += __builtin_popcount(l_k[i]);
            auto [t, s] = MulSingleCompressTerm(l_k[i], r_k[i]);
            coeff = coeff * t;
            fermion_string.push_back(s);
        } else if (i >= l_k.size()) {
            total_one += one_in_low;
            fermion_string.push_back(r_k[i]);
        } else {
            fermion_string.push_back(l_k[i]);
        }
    }
    if (total_one & 1) {
        coeff *= -1.0;
    }
    return {fermion_string, coeff};
}

std::tuple<tn::Tensor, uint64_t> SingleFermionStr::MulSingleCompressTerm(uint64_t a, uint64_t b) {
    if (a == static_cast<uint64_t>(TermValue::nll) || b == static_cast<uint64_t>(TermValue::nll)
        || a == static_cast<uint64_t>(TermValue::I) || b == static_cast<uint64_t>(TermValue::I)) {
        return {tn::ops::ones(1), a | b};
    }
    int total_one = 0;
    int one_in_low = 0;
    uint64_t out = 0;
    int poi = 0;
    for (size_t b_i = b; b_i != 0; b_i >>= 3) {
        auto lhs = static_cast<TermValue>(a & 7);
        auto rhs = static_cast<TermValue>(b & 7);
        if (rhs != TermValue::I) {
            total_one += one_in_low;
        }
        one_in_low += __builtin_popcount(a & 7);
        auto res = fermion_product_map.at(lhs).at(rhs);
        if (res == TermValue::nll) {
            return {tn::ops::ones(1), static_cast<uint64_t>(TermValue::nll)};
        }
        out += (static_cast<uint64_t>(res) << poi);
        poi += 3;
        a >>= 3;
        b >>= 3;
    }
    out += (a << poi);
    auto coeff = tn::ops::ones(1);
    if (total_one & 1) {
        coeff *= -1.0;
    }
    return {coeff, out};
}

// -----------------------------------------------------------------------------

FermionOperator::FermionOperator(const std::string& fermion_string, const parameter::ParameterResolver& var) {
    auto term = SingleFermionStr::init(fermion_string, var);
    this->terms.insert(term.first, term.second);
}

FermionOperator::FermionOperator(const terms_t& t, const parameter::ParameterResolver& var) {
    auto term = SingleFermionStr::init(t, var);
    this->terms.insert(term.first, term.second);
}

bool FermionOperator::Contains(const key_t& term) const {
    return this->terms.m_map.find(term) != this->terms.m_map.end();
}

void FermionOperator::Update(const compress_term_t& fermion) {
    if (this->Contains(fermion.first)) {
        this->terms[fermion.first] = fermion.second;
    } else {
        this->terms.insert(fermion);
    }
}
size_t FermionOperator::size() const {
    return this->terms.size();
}

std::string FermionOperator::ToString() const {
    if (this->size() == 0) {
        return "0";
    }
    std::string out = "";
    for (const auto& term : this->terms) {
        out += SingleFermionStr::GetString(term) + "\n";
    }
    return out;
}

auto FermionOperator::get_terms() const -> dict_t {
    dict_t out{};
    for (auto& [k, v] : this->terms.m_list) {
        if (std::any_of(k.begin(), k.end(), [](auto i) { return i == static_cast<uint64_t>(TermValue::nll); })) {
            continue;
        }
        terms_t terms;
        int group_id = 0;
        for (auto fermion_word : k) {
            int local_id = 0;
            while (fermion_word != 0) {
                auto word = static_cast<TermValue>(fermion_word & 7);
                if (word == TermValue::AAd) {
                    terms.push_back({group_id * 21 + local_id, TermValue::A});
                    terms.push_back({group_id * 21 + local_id, TermValue::Ad});
                } else if (word == TermValue::AdA) {
                    terms.push_back({group_id * 21 + local_id, TermValue::Ad});
                    terms.push_back({group_id * 21 + local_id, TermValue::A});
                } else if (word != TermValue::I) {
                    terms.push_back({group_id * 21 + local_id, word});
                }
                fermion_word >>= 3;
                local_id += 1;
            }
            group_id += 1;
        }
        out.push_back({terms, v});
    }
    return out;
}

bool FermionOperator::is_singlet() const {
    return this->size() == 1;
}

parameter::ParameterResolver FermionOperator::singlet_coeff() const {
    if (!this->is_singlet()) {
        throw std::runtime_error("Operator is not singlet.");
    }
    return this->terms.m_list.begin()->second;
}

size_t FermionOperator::count_qubits() const {
    int n_qubits = 0;
    for (auto& [k, v] : this->terms.m_list) {
        int group_id = k.size() - 1, local_id = 0;
        for (auto word = k.rbegin(); word != k.rend(); ++word) {
            if ((*word) != 0) {
                n_qubits = std::max(n_qubits, (63 - __builtin_clzll(*word)) / 3 + group_id * 21);
                break;
            }
            group_id -= 1;
        }
    }
    return n_qubits;
}
// -----------------------------------------------------------------------------

FermionOperator& FermionOperator::operator+=(const tn::Tensor& c) {
    if (this->Contains(key_t{0})) {
        this->terms[key_t{0}] = this->terms[key_t{0}] + c;
    } else {
        this->terms.insert({key_t{0}, parameter::ParameterResolver(c)});
    }
    return *this;
}

FermionOperator operator+(FermionOperator lhs, const tensor::Tensor& rhs) {
    if (lhs.Contains(key_t{0})) {
        lhs.terms[key_t{0}] = lhs.terms[key_t{0}] + rhs;
    } else {
        lhs.terms.insert({key_t{0}, parameter::ParameterResolver(rhs)});
    }
    return lhs;
}

FermionOperator& FermionOperator::operator+=(const FermionOperator& other) {
    for (const auto& term : other.terms) {
        if (this->Contains(term.first)) {
            this->terms[term.first] = this->terms[term.first] + term.second;
        } else {
            this->terms.insert(term);
        }
    }
    return *this;
}

FermionOperator operator+(FermionOperator lhs, const FermionOperator& rhs) {
    lhs += rhs;
    return lhs;
}

FermionOperator FermionOperator::operator*=(const FermionOperator& other) {
    auto out = FermionOperator();
    for (auto& this_term : this->terms) {
        for (const auto& other_term : other.terms) {
            auto new_term = SingleFermionStr::Mul(this_term, other_term);
            if (out.Contains(new_term.first)) {
                out.terms[new_term.first] = out.terms[new_term.first] + new_term.second;
            } else {
                out.terms.insert(new_term);
            }
        }
    }
    std::swap(out.terms, this->terms);
    return *this;
}

FermionOperator FermionOperator::operator*(const FermionOperator& other) {
    auto out = FermionOperator();
    for (auto& this_term : this->terms) {
        for (const auto& other_term : other.terms) {
            auto new_term = SingleFermionStr::Mul(this_term, other_term);
            if (out.Contains(new_term.first)) {
                out.terms[new_term.first] = out.terms[new_term.first] + new_term.second;
            } else {
                out.terms.insert(new_term);
            }
        }
    }
    return out;
}

FermionOperator FermionOperator::operator*=(const parameter::ParameterResolver& other) {
    for (auto& [k, v] : this->terms.m_list) {
        v *= other;
    }
    return *this;
}
}  // namespace operators::fermion

std::ostream& operator<<(std::ostream& os, const operators::fermion::FermionOperator& t) {
    os << t.ToString();
    return os;
}
