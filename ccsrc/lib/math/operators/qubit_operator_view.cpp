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
#include <iterator>
#include <stdexcept>
#include <string>

#include <sys/types.h>

#include "math/operators/utils.hpp"
#include "math/pr/parameter_resolver.hpp"
#include "math/tensor/ops/memory_operator.hpp"
#include "math/tensor/tensor.hpp"
#include "math/tensor/traits.hpp"

namespace operators::qubit {
std::string to_string(const TermValue& term) {
    switch (term) {
        case TermValue::I:
            return "I";
        case TermValue::X:
            return "X";
        case TermValue::Y:
            return "Y";
        case TermValue::Z:
            return "Z";
    }
}

auto SinglePauliStr::ParseToken(const std::string& token) -> term_t {
    if (token.size() <= 1) {
        throw std::runtime_error("Wrong token: '" + token + "'. Need a pauli word following a int, for example 'X0'.");
    }
    std::string pauli_word = token.substr(0, 1);
    operators::qubit::TermValue pauli;
    if (pauli_word == "X" || pauli_word == "x") {
        pauli = operators::qubit::TermValue::X;
    } else if (pauli_word == "Y" || pauli_word == "y") {
        pauli = operators::qubit::TermValue::Y;
    } else if (pauli_word == "Z" || pauli_word == "z") {
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
    return {idx, pauli};
}

auto SinglePauliStr::init(const std::string& pauli_string, const parameter::ParameterResolver& var) -> compress_term_t {
    compress_term_t out = {key_t{0}, var};
    std::istringstream iss(pauli_string);
    for (std::string s; iss >> s;) {
        InplaceMulCompressTerm(ParseToken(s), out);
    }
    return out;
}

auto SinglePauliStr::init(const terms_t& terms, const parameter::ParameterResolver& var) -> compress_term_t {
    compress_term_t out = {key_t{0}, var};
    for (auto& term : terms) {
        InplaceMulCompressTerm(term, out);
    }
    return out;
}

void SinglePauliStr::InplaceMulCompressTerm(const term_t& term, compress_term_t& pauli) {
    auto [idx, word] = term;
    if (word == TermValue::I) {
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
        pauli_string[group_id] = pauli_string[group_id] & (~local_mask) | (static_cast<uint64_t>(word)) << local_id;
    } else {
        TermValue lhs = static_cast<TermValue>((pauli_string[group_id] & local_mask) >> local_id);
        auto [t, res] = pauli_product_map.at(lhs).at(word);
        coeff = coeff * t;
        pauli_string[group_id] = pauli_string[group_id] & (~local_mask) | (static_cast<uint64_t>(res)) << local_id;
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

std::string SinglePauliStr::GetString(const compress_term_t& pauli) {
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
                case TermValue::Y:
                case TermValue::Z:
                    out += to_string(j);
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
    out.resize(out.find_last_not_of(" ") + 1);
    if (coeff.IsConst()) {
        return tn::ops::to_string(coeff.const_value, true) + " [" + out + "]";
    }
    return coeff.ToString() + " [" + out + "]";
}

auto SinglePauliStr::Mul(const compress_term_t& lhs, const compress_term_t& rhs) -> compress_term_t {
    auto& [l_k, l_v] = lhs;
    auto& [r_k, r_v] = rhs;
    key_t pauli_string = {};
    value_t coeff = l_v * r_v;
    if (l_k.size() == 1 && r_k.size() == 1) {
        auto [t, s] = MulSingleCompressTerm(l_k[0], r_k[0]);
        coeff = coeff * t;
        pauli_string.push_back(s);
        return {pauli_string, coeff};
    }
    int min_size = std::min(l_k.size(), r_k.size());
    int max_size = std::max(l_k.size(), r_k.size());
    for (int i = 0; i < max_size; i++) {
        if (i < min_size) {
            auto [t, s] = MulSingleCompressTerm(l_k[i], r_k[i]);
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

std::tuple<tn::Tensor, uint64_t> SinglePauliStr::MulSingleCompressTerm(uint64_t a, uint64_t b) {
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

auto SinglePauliStr::py_term_to_term(const py_term_t& term) -> term_t {
    auto& [idx, word] = term;
    if (word == "X" || word == "x") {
        return {idx, TermValue::X};
    }
    if (word == "Y" || word == "y") {
        return {idx, TermValue::Y};
    }
    if (word == "Z" || word == "z") {
        return {idx, TermValue::Z};
    }
    throw std::runtime_error("Unknown term: (" + std::to_string(idx) + ", " + word + ").");
}

auto SinglePauliStr::py_terms_to_terms(const py_terms_t& terms) -> terms_t {
    terms_t out(terms.size());
    std::transform(terms.begin(), terms.end(), out.begin(), py_term_to_term);
    return out;
}

// -----------------------------------------------------------------------------

QubitOperator::QubitOperator(const std::string& pauli_string, const parameter::ParameterResolver& var) {
    auto term = SinglePauliStr::init(pauli_string, var);
    this->terms.insert(term.first, term.second);
    this->dtype = term.second.GetDtype();
}

QubitOperator::QubitOperator(const terms_t& t, const parameter::ParameterResolver& var) {
    auto term = SinglePauliStr::init(t, var);
    this->terms.insert(term.first, term.second);
    this->dtype = term.second.GetDtype();
}

QubitOperator::QubitOperator(const term_t& t, const parameter::ParameterResolver& var) {
    auto term = SinglePauliStr::init(terms_t{t}, var);
    this->terms.insert(term.first, term.second);
    this->dtype = term.second.GetDtype();
}

QubitOperator::QubitOperator(const py_term_t& t, const parameter::ParameterResolver& var) {
    auto term = SinglePauliStr::init(terms_t{SinglePauliStr::py_term_to_term(t)}, var);
    this->terms.insert(term.first, term.second);
    this->dtype = term.second.GetDtype();
}

QubitOperator::QubitOperator(const py_terms_t& t, const parameter::ParameterResolver& var) {
    auto term = SinglePauliStr::init(SinglePauliStr::py_terms_to_terms(t), var);
    this->terms.insert(term.first, term.second);
    this->dtype = term.second.GetDtype();
}

bool QubitOperator::Contains(const key_t& term) const {
    return this->terms.m_map.find(term) != this->terms.m_map.end();
}

void QubitOperator::Update(const compress_term_t& pauli) {
    if (this->Contains(pauli.first)) {
        this->terms[pauli.first] = pauli.second;
    } else {
        this->terms.insert(pauli);
    }
}

size_t QubitOperator::size() const {
    return this->terms.size();
}

void QubitOperator::CastTo(tn::TDtype dtype) {
    if (dtype != this->dtype) {
        for (auto& [k, v] : this->terms) {
            v.CastTo(dtype);
        }
        this->dtype = dtype;
    }
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

size_t QubitOperator::count_qubits() const {
    int n_qubits = 0;
    for (auto& [k, v] : this->terms.m_list) {
        int group_id = k.size() - 1, local_id = 0;
        for (auto word = k.rbegin(); word != k.rend(); ++word) {
            if ((*word) != 0) {
                n_qubits = std::max(n_qubits, (63 - __builtin_clzll(*word)) / 2 + group_id * 32);
                break;
            }
            group_id -= 1;
        }
    }
    return n_qubits + 1;
}

auto QubitOperator::get_terms() const -> dict_t {
    dict_t out{};
    for (auto& [k, v] : this->terms.m_list) {
        terms_t terms{};
        int group_id = 0;
        for (auto qubit_word : k) {
            int local_id = 0;
            while (qubit_word != 0) {
                auto word = static_cast<TermValue>(qubit_word & 3);
                if (word != TermValue::I) {
                    terms.push_back({group_id * 32 + local_id, word});
                }
                local_id += 1;
                qubit_word >>= 2;
            }
        }
        // std::reverse(terms.begin(), terms.end());
        out.push_back({terms, v});
    }
    return out;
}

value_t QubitOperator::get_coeff(const terms_t& term) {
    auto terms = SinglePauliStr::init(term, parameter::ParameterResolver(tn::ops::ones(1)));
    if (this->Contains(terms.first)) {
        return this->terms[terms.first] * terms.second;
    }
    throw std::out_of_range("term not in fermion operator");
}

QubitOperator QubitOperator::hermitian_conjugated() const {
    if (this->dtype == tn::TDtype::Complex128 || this->dtype == tn::TDtype::Complex64) {
        auto out = *this;
        for (auto& [k, v] : out.terms.m_list) {
            v = v.Conjugate();
        }
        return out;
    }
    return *this;
}

bool QubitOperator::is_singlet() const {
    return this->size() == 1;
}
bool QubitOperator::parameterized() const {
    for (auto& [k, v] : this->terms.m_list) {
        if (!v.IsConst()) {
            return true;
        }
    }
    return false;
}
void QubitOperator::set_coeff(const terms_t& term, const parameter::ParameterResolver& value) {
    auto terms = SinglePauliStr::init(term, parameter::ParameterResolver(tn::ops::ones(1)));
    if (this->Contains(terms.first)) {
        this->terms[terms.first] = terms.second * value;
    } else {
        terms.second = terms.second * value;
        this->terms.insert(terms);
    }
    auto upper_t = tensor::upper_type_v(this->dtype, this->terms[terms.first].GetDtype());
    if (this->dtype != upper_t) {
        this->CastTo(upper_t);
    }
    if (this->terms[terms.first].GetDtype() != upper_t) {
        this->terms[terms.first].CastTo(upper_t);
    }
}

QubitOperator::QubitOperator(const key_t& k, const value_t& v) {
    this->terms.insert(k, v);
    this->dtype = v.GetDtype();
}
std::vector<std::pair<parameter::ParameterResolver, QubitOperator>> QubitOperator::split() const {
    auto out = std::vector<std::pair<parameter::ParameterResolver, QubitOperator>>();
    for (auto& [k, v] : this->terms.m_list) {
        out.push_back({v, QubitOperator(k, parameter::ParameterResolver(tn::ops::ones(1, v.GetDtype())))});
    }
    return out;
}

parameter::ParameterResolver QubitOperator::singlet_coeff() const {
    if (!this->is_singlet()) {
        throw std::runtime_error("Operator is not singlet.");
    }
    return this->terms.m_list.begin()->second;
}

std::vector<QubitOperator> QubitOperator::singlet() const {
    if (!this->is_singlet()) {
        throw std::runtime_error("Given QubitOperator is not singlet.");
    }
    std::vector<QubitOperator> out;
    for (auto& [term, value] : this->get_terms()) {
        std::transform(term.begin(), term.end(), std::back_inserter(out), [](auto& word) {
            return QubitOperator({word}, parameter::ParameterResolver(tn::ops::ones(1)));
        });
        break;
    }
    return out;
}

void QubitOperator::subs(const parameter::ParameterResolver& other) {
    auto new_type = this->dtype;
    std::vector<key_t> will_pop;
    for (auto& [k, v] : this->terms.m_list) {
        if (v.subs(other).size() != 0) {
            new_type = tensor::upper_type_v(this->dtype, v.GetDtype());
        }
        if (!v.IsNotZero()) {
            will_pop.push_back(k);
        }
    }
    for (auto& k : will_pop) {
        this->terms.erase(k);
    }
    if (new_type != this->dtype) {
        this->CastTo(new_type);
    }
}

QubitOperator QubitOperator::imag() const {
    auto out = *this;
    std::vector<key_t> will_pop;
    for (auto& [k, v] : out.terms.m_list) {
        v = v.Imag();
        if (!v.IsNotZero()) {
            will_pop.push_back(k);
        }
    }
    out.dtype = tn::ToRealType(this->dtype);
    std::for_each(will_pop.begin(), will_pop.end(), [&](auto i) { out.terms.erase(i); });
    return out;
}

QubitOperator QubitOperator::real() const {
    auto out = *this;
    std::vector<key_t> will_pop;
    for (auto& [k, v] : out.terms.m_list) {
        v = v.Real();
        if (!v.IsNotZero()) {
            will_pop.push_back(k);
        }
    }
    out.dtype = tn::ToRealType(this->dtype);
    std::for_each(will_pop.begin(), will_pop.end(), [&](auto i) { out.terms.erase(i); });
    return out;
}

tn::TDtype QubitOperator::GetDtype() const {
    return this->dtype;
}
// -----------------------------------------------------------------------------

QubitOperator& QubitOperator::operator+=(const tn::Tensor& c) {
    if (this->Contains(key_t{0})) {
        this->terms[key_t{0}] = this->terms[key_t{0}] + c;
    } else {
        this->terms.insert({key_t{0}, parameter::ParameterResolver(c)});
    }
    auto upper_t = tn::upper_type_v(c.dtype, this->dtype);
    if (upper_t != this->GetDtype()) {
        this->CastTo(upper_t);
        return *this;
    }
    if (c.dtype != upper_t) {
        this->terms[key_t{0}].CastTo(upper_t);
    }
    if (!this->terms[key_t{0}].IsNotZero()) {
        this->terms.erase(key_t{0});
    }
    return *this;
}
QubitOperator& QubitOperator::operator-=(const tn::Tensor& c) {
    *this += (0.0 - c);
    return *this;
}

QubitOperator& QubitOperator::operator+=(const QubitOperator& other) {
    for (const auto& term : other.terms) {
        if (this->Contains(term.first)) {
            this->terms[term.first] = this->terms[term.first] + term.second;
            if (this->dtype != this->terms[term.first].GetDtype()) {
                this->CastTo(this->terms[term.first].GetDtype());
            }
            if (!this->terms[term.first].IsNotZero()) {
                this->terms.erase(term.first);
            }
        } else {
            this->terms.insert(term);
            auto upper_t = tn::upper_type_v(this->dtype, term.second.GetDtype());
            if (this->GetDtype() != upper_t) {
                this->CastTo(upper_t);
            }
            if (term.second.GetDtype() != upper_t) {
                this->terms[term.first].CastTo(upper_t);
            }
            if (!this->terms[term.first].IsNotZero()) {
                this->terms.erase(term.first);
            }
        }
    }
    return *this;
}

QubitOperator& QubitOperator::operator-=(const QubitOperator& other) {
    for (const auto& term : other.terms) {
        if (this->Contains(term.first)) {
            this->terms[term.first] = this->terms[term.first] - term.second;
            if (this->dtype != this->terms[term.first].GetDtype()) {
                this->CastTo(this->terms[term.first].GetDtype());
            }
            if (!this->terms[term.first].IsNotZero()) {
                this->terms.erase(term.first);
            }
        } else {
            this->terms.insert({term.first, parameter::ParameterResolver(tensor::ops::zeros(1, term.second.GetDtype()))
                                                - term.second});
            auto upper_t = tn::upper_type_v(this->dtype, term.second.GetDtype());
            if (this->GetDtype() != upper_t) {
                this->CastTo(upper_t);
            }
            if (term.second.GetDtype() != upper_t) {
                this->terms[term.first].CastTo(upper_t);
            }
            if (!this->terms[term.first].IsNotZero()) {
                this->terms.erase(term.first);
            }
        }
    }
    return *this;
}

QubitOperator operator+(QubitOperator lhs, const tensor::Tensor& rhs) {
    if (lhs.Contains(key_t{0})) {
        lhs.terms[key_t{0}] = lhs.terms[key_t{0}] + rhs;
    } else {
        lhs.terms.insert({key_t{0}, parameter::ParameterResolver(rhs)});
    }
    auto upper_t = tn::upper_type_v(lhs.GetDtype(), rhs.dtype);
    if (upper_t != lhs.GetDtype()) {
        lhs.CastTo(upper_t);
        return lhs;
    }
    if (upper_t != rhs.dtype) {
        lhs.terms[key_t{0}].CastTo(upper_t);
    }
    if (!lhs.terms[key_t{0}].IsNotZero()) {
        lhs.terms.erase(key_t{0});
    }
    return lhs;
}

QubitOperator operator+(QubitOperator lhs, const QubitOperator& rhs) {
    lhs += rhs;
    return lhs;
}

QubitOperator operator-(QubitOperator lhs, const tensor::Tensor& rhs) {
    lhs -= rhs;
    return lhs;
}

QubitOperator operator-(QubitOperator lhs, const QubitOperator& rhs) {
    lhs -= rhs;
    return lhs;
}

QubitOperator QubitOperator::operator*=(const QubitOperator& other) {
    auto out = QubitOperator();
    for (auto& this_term : this->terms) {
        for (const auto& other_term : other.terms) {
            auto new_term = SinglePauliStr::Mul(this_term, other_term);
            if (out.Contains(new_term.first)) {
                out.terms[new_term.first] = out.terms[new_term.first] + new_term.second;
                if (this->dtype != out.terms[new_term.first].GetDtype()) {
                    this->CastTo(out.terms[new_term.first].GetDtype());
                }
            } else {
                out.terms.insert(new_term);
                auto upper_t = tn::upper_type_v(this->dtype, out.terms[new_term.first].GetDtype());
                if (this->GetDtype() != upper_t) {
                    this->CastTo(upper_t);
                }
                if (new_term.second.GetDtype() != upper_t) {
                    out.terms[new_term.first].CastTo(upper_t);
                }
            }
            if (!out.terms[new_term.first].IsNotZero()) {
                out.terms.erase(new_term.first);
            }
        }
    }
    std::swap(out.terms, this->terms);
    return *this;
}

QubitOperator operator*(QubitOperator lhs, const QubitOperator& rhs) {
    lhs *= rhs;
    return lhs;
}

QubitOperator operator*(QubitOperator lhs, const tensor::Tensor& other) {
    lhs *= QubitOperator("", parameter::ParameterResolver(other));
    return lhs;
}

QubitOperator QubitOperator::operator*=(const parameter::ParameterResolver& other) {
    if (!other.IsNotZero()) {
        this->terms = {};
        return *this;
    }
    for (auto& [k, v] : this->terms.m_list) {
        v *= other;
        this->dtype = v.GetDtype();
    }
    return *this;
}
}  // namespace operators::qubit

std::ostream& operator<<(std::ostream& os, const operators::qubit::QubitOperator& t) {
    os << t.ToString();
    return os;
}
