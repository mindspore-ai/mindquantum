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

#ifndef MATH_OPERATORS_QUBIT_OPERATOR_VIEW_HPP_
#define MATH_OPERATORS_QUBIT_OPERATOR_VIEW_HPP_
#include <cstdint>
#include <list>
#include <map>
#include <set>
#include <vector>

#include "math/tensor/ops.hpp"
#include "math/tensor/ops/memory_operator.hpp"
#include "math/tensor/tensor.hpp"
#include "math/tensor/traits.hpp"
namespace operators::qubit {
// ABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABAB
constexpr static uint64_t M_B = 6148914691236517205;
constexpr static uint64_t M_A = M_B << 1;

namespace tn = tensor;
enum class TermValue : uint8_t {
    //! DO NOT CHANGE VALUE.
    I = 0,
    X = 1,
    Y = 2,
    Z = 3,
};

enum class CValue : uint8_t {
    //! DO NOT CHANGE VALUE.
    ONE = 0,    // 00 -> (1, 0)
    I = 1,      // 01 -> (0, 1)
    M_ONE = 2,  // 10 -> (-1, 0)
    M_I = 3,    // 11 -> (0, -1)
};

using pauli_product_map_t = std::map<TermValue, std::map<TermValue, std::tuple<tn::Tensor, TermValue>>>;

// clang-format off
const pauli_product_map_t pauli_product_map = {
    {
            TermValue::I, {
                            {TermValue::I, {tn::ops::ones(1), TermValue::I}},
                            {TermValue::X, {tn::ops::ones(1), TermValue::X}},
                            {TermValue::Y, {tn::ops::ones(1), TermValue::Y}},
                            {TermValue::Z, {tn::ops::ones(1), TermValue::Z}},
                          }
        },
    {
            TermValue::X, {
                            {TermValue::I, {tn::ops::ones(1), TermValue::X}},
                            {TermValue::X, {tn::ops::ones(1), TermValue::I}},
                            {TermValue::Y, {tn::ops::init_with_value(std::complex<double>(0.0, 1.0)), TermValue::Z}},
                            {TermValue::Z, {tn::ops::init_with_value(std::complex<double>(0.0, -1.0)), TermValue::Y}},
                          }
        },
    {
            TermValue::Y, {
                            {TermValue::I, {tn::ops::ones(1), TermValue::Y}},
                            {TermValue::X, {tn::ops::init_with_value(std::complex<double>(0.0, -1.0)), TermValue::Z}},
                            {TermValue::Y, {tn::ops::ones(1), TermValue::I}},
                            {TermValue::Z, {tn::ops::init_with_value(std::complex<double>(0.0, 1.0)), TermValue::X}},
                          }
        },
    {
            TermValue::Z, {
                            {TermValue::I, {tn::ops::ones(1), TermValue::Z}},
                            {TermValue::X, {tn::ops::init_with_value(std::complex<double>(0.0, 1.0)), TermValue::Y}},
                            {TermValue::Y, {tn::ops::init_with_value(std::complex<double>(0.0, -1.0)), TermValue::X}},
                            {TermValue::Z, {tn::ops::ones(1), TermValue::I}},
                          }
        },
};
// clang-format on

using cvalue_product_map_t = std::map<CValue, std::map<CValue, CValue>>;

const cvalue_product_map_t cvalue_product_map = {
    {CValue::ONE,
     {
         {CValue::ONE, CValue::ONE},
         {CValue::I, CValue::I},
         {CValue::M_ONE, CValue::M_ONE},
         {CValue::M_I, CValue::M_I},
     }},
    {CValue::I,
     {
         {CValue::ONE, CValue::I},
         {CValue::I, CValue::M_ONE},
         {CValue::M_ONE, CValue::M_I},
         {CValue::M_I, CValue::ONE},
     }},
    {CValue::M_ONE,
     {
         {CValue::ONE, CValue::M_ONE},
         {CValue::I, CValue::M_I},
         {CValue::M_ONE, CValue::ONE},
         {CValue::M_I, CValue::I},
     }},
    {CValue::M_I,
     {
         {CValue::ONE, CValue::M_I},
         {CValue::I, CValue::ONE},
         {CValue::M_ONE, CValue::I},
         {CValue::M_I, CValue::M_ONE},
     }},
};

// -----------------------------------------------------------------------------

std::tuple<TermValue, size_t> parse_token(const std::string& token);
struct SinglePauliStr {
    // tn::Tensor coeff = tn::ops::ones(1);
    // std::vector<uint64_t> pauli_string = {0};
    using key_t = std::vector<uint64_t>;
    using value_t = tn::Tensor;
    using pauli_t = std::pair<key_t, value_t>;
    // -----------------------------------------------------------------------------

    static pauli_t init(const std::string& pauli_string, const tn::Tensor& coeff = tn::ops::ones(1));
    static pauli_t init(const std::vector<std::tuple<TermValue, size_t>>& terms, const tn::Tensor& coeff);

    // -----------------------------------------------------------------------------

    static void InplaceMulPauli(TermValue term, size_t idx, pauli_t& pauli);
    static bool IsSameString(const key_t& k1, const key_t& k2);
    static std::string GetString(const pauli_t& pauli);
    static pauli_t Mul(const pauli_t& lhs, const pauli_t& rhs);
};

struct KeyCompare {
    bool operator()(const SinglePauliStr::key_t& a, const SinglePauliStr::key_t& b) const {
        if (a.size() == b.size() && a.size() == 1) {
            return a[0] < b[0];
        }
        if (a.size() < b.size()) {
            return true;
        } else if (a.size() == b.size()) {
            return std::lexicographical_compare(a.begin(), a.end(), b.begin(), b.end());
        } else {
            return false;
        }
    }
};

class QTerm_t {
    using K = SinglePauliStr::key_t;
    using V = SinglePauliStr::value_t;

 public:
    void insert(const K& key, const V& value) {
        if (m_map.count(key)) {
            m_list.erase(m_map[key]);
            m_map.erase(key);
        }
        m_list.push_back({key, value});
        m_map[key] = --m_list.end();
    }
    void insert(const SinglePauliStr::pauli_t& t) {
        this->insert(t.first, t.second);
    }
    V& operator[](const K& key) {
        return m_map[key]->second;
    }

    typename std::list<std::pair<K, V>>::iterator begin() {
        return m_list.begin();
    }

    typename std::list<std::pair<K, V>>::iterator end() {
        return m_list.end();
    }

    typename std::list<std::pair<K, V>>::const_iterator begin() const {
        return m_list.begin();
    }

    typename std::list<std::pair<K, V>>::const_iterator end() const {
        return m_list.end();
    }
    size_t size() const {
        return this->m_map.size();
    }

 public:
    std::list<std::pair<K, V>> m_list;
    std::map<K, std::list<std::pair<K, V>>::iterator, KeyCompare> m_map;
};

std::tuple<tn::Tensor, uint64_t> mul_pauli_str(uint64_t a, uint64_t b);

// -----------------------------------------------------------------------------

struct QubitOperator {
    using key_t = SinglePauliStr::key_t;
    using value_t = SinglePauliStr::value_t;
    using pauli_t = SinglePauliStr::pauli_t;
    QTerm_t terms{};

    // -----------------------------------------------------------------------------

    QubitOperator() = default;
    QubitOperator(const std::string& pauli_string, const tn::Tensor& coeff = tn::ops::ones(1));

    bool Contains(const key_t& term) const;
    void Update(const pauli_t& pauli);
    size_t size() const;
    std::string ToString() const;
    QubitOperator& operator+=(const tn::Tensor& c);
    QubitOperator operator+(const tn::Tensor& c);
    QubitOperator& operator+=(const QubitOperator& other);
    QubitOperator operator+(const QubitOperator& other);
    QubitOperator operator*=(const QubitOperator& other);
    QubitOperator operator*(const QubitOperator& other);
};
}  // namespace operators::qubit
#endif /* MATH_OPERATORS_QUBIT_OPERATOR_VIEW_HPP_ */
