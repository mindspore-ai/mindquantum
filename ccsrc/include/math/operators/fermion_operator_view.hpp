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

#ifndef MATH_OPERATORS_FERMION_OPERATOR_VIEW_HPP_
#define MATH_OPERATORS_FERMION_OPERATOR_VIEW_HPP_

#include <cstdint>
#include <limits>
#include <list>
#include <map>
#include <tuple>
#include <utility>
#include <vector>

#include "math/pr/parameter_resolver.hpp"
#include "math/tensor/ops/concrete_tensor.hpp"
#include "math/tensor/tensor.hpp"

namespace operators::fermion {
namespace tn = tensor;
enum class TermValue : uint64_t {
    //! DO NOT CHANGE VALUE.
    I = 0,                                       // 000
    A = 1,                                       // 001
    Ad = 2,                                      // 010
    AdA = 3,                                     // 011
    AAd = 6,                                     // 110
    nll = std::numeric_limits<uint64_t>::max(),  // 11111...
};

using fermion_product_t = std::map<TermValue, std::map<TermValue, TermValue>>;
const fermion_product_t fermion_product_map = {
    {
        TermValue::I,
        {
            {TermValue::I, TermValue::I},
            {TermValue::A, TermValue::A},
            {TermValue::Ad, TermValue::Ad},
            {TermValue::AdA, TermValue::AdA},
            {TermValue::AAd, TermValue::AAd},
            {TermValue::nll, TermValue::nll},
        },
    },
    {
        TermValue::A,
        {
            {TermValue::I, TermValue::A},
            {TermValue::A, TermValue::nll},
            {TermValue::Ad, TermValue::AAd},
            {TermValue::AdA, TermValue::A},
            {TermValue::AAd, TermValue::nll},
            {TermValue::nll, TermValue::nll},
        },
    },
    {
        TermValue::Ad,
        {
            {TermValue::I, TermValue::Ad},
            {TermValue::A, TermValue::AdA},
            {TermValue::Ad, TermValue::nll},
            {TermValue::AdA, TermValue::nll},
            {TermValue::AAd, TermValue::Ad},
            {TermValue::nll, TermValue::nll},
        },
    },
    {
        TermValue::AdA,
        {
            {TermValue::I, TermValue::AdA},
            {TermValue::A, TermValue::nll},
            {TermValue::Ad, TermValue::Ad},
            {TermValue::AdA, TermValue::AdA},
            {TermValue::AAd, TermValue::nll},
            {TermValue::nll, TermValue::nll},
        },
    },
    {
        TermValue::AAd,
        {
            {TermValue::I, TermValue::AAd},
            {TermValue::A, TermValue::A},
            {TermValue::Ad, TermValue::nll},
            {TermValue::AdA, TermValue::nll},
            {TermValue::AAd, TermValue::AAd},
            {TermValue::nll, TermValue::nll},
        },
    },
    {
        TermValue::nll,
        {
            {TermValue::I, TermValue::nll},
            {TermValue::A, TermValue::nll},
            {TermValue::Ad, TermValue::nll},
            {TermValue::AdA, TermValue::nll},
            {TermValue::AAd, TermValue::nll},
            {TermValue::nll, TermValue::nll},
        },
    },
};

// -----------------------------------------------------------------------------
std::tuple<TermValue, size_t> parse_token(const std::string& token);

struct SingleFermionStr {
    using key_t = std::vector<uint64_t>;
    using value_t = parameter::ParameterResolver;
    using fermion_t = std::pair<key_t, value_t>;

    // -----------------------------------------------------------------------------

    static fermion_t init(const std::string& fermion_string,
                          const parameter::ParameterResolver& var = tn::ops::ones(1));
    static fermion_t init(const std::vector<std::tuple<TermValue, size_t>>& terms,
                          const parameter::ParameterResolver& var = tn::ops::ones(1));

    // -----------------------------------------------------------------------------

    static void InplaceMulPauli(TermValue term, size_t idx, fermion_t& fermion);
    static bool IsSameString(const key_t& k1, const key_t& k2);
    static std::string GetString(const fermion_t& fermion);
    static fermion_t Mul(const fermion_t& lhs, const fermion_t& rhs);
};

struct KeyCompare {
    bool operator()(const SingleFermionStr::key_t& a, const SingleFermionStr::key_t& b) const {
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
    using K = SingleFermionStr::key_t;
    using V = SingleFermionStr::value_t;

 public:
    void insert(const K& key, const V& value) {
        if (m_map.find(key) != m_map.end()) {
            m_list.erase(m_map[key]);
            m_map.erase(key);
        }
        m_list.push_back({key, value});
        m_map[key] = --m_list.end();
    }
    void insert(const SingleFermionStr::fermion_t& t) {
        this->insert(t.first, t.second);
    }
    V& operator[](const K& key) {
        return (*m_map[key]).second;
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

    // -----------------------------------------------------------------------------

    QTerm_t() = default;
    QTerm_t(const QTerm_t& other) {
        for (auto& p : other.m_list) {
            insert(p);
        }
    }

    QTerm_t(QTerm_t&& other) {
        m_list = std::move(other.m_list);
        m_map = std::move(other.m_map);
    }
    QTerm_t& operator=(const QTerm_t& other) {
        if (this != &other) {
            m_list.clear();
            m_map.clear();
            for (auto& p : other.m_list) {
                insert(p);
            }
        }
        return *this;
    }
    QTerm_t& operator=(QTerm_t&& other) {
        if (this != &other) {
            m_list = std::move(other.m_list);
            m_map = std::move(other.m_map);
        }
        return *this;
    }

 public:
    std::list<std::pair<K, V>> m_list;
    std::map<K, std::list<std::pair<K, V>>::iterator, KeyCompare> m_map;
};

std::tuple<tn::Tensor, uint64_t> mul_fermion_str(uint64_t a, uint64_t b);

// -----------------------------------------------------------------------------

struct FermionOperator {
    using key_t = SingleFermionStr::key_t;
    using value_t = SingleFermionStr::value_t;
    using pauli_t = SingleFermionStr::fermion_t;
    using term_t = std::pair<uint64_t, TermValue>;
    using terms_t = std::vector<term_t>;
    using dict_t = std::vector<std::pair<terms_t, parameter::ParameterResolver>>;
    QTerm_t terms{};

    // -----------------------------------------------------------------------------

    FermionOperator() = default;
    FermionOperator(const std::string& fermion_string, const parameter::ParameterResolver& var = tn::ops::ones(1));
    FermionOperator(const terms_t& t, const parameter::ParameterResolver& var = tn::ops::ones(1));
    FermionOperator(const term_t& t);
    // -----------------------------------------------------------------------------

    bool Contains(const key_t& term) const;
    void Update(const pauli_t& pauli);
    size_t size() const;
    std::string ToString() const;
    dict_t get_terms() const;

    // -----------------------------------------------------------------------------

    FermionOperator& operator+=(const tn::Tensor& c);
    FermionOperator operator+(const tn::Tensor& c);
    FermionOperator& operator+=(const FermionOperator& other);
    FermionOperator operator+(const FermionOperator& other);
    FermionOperator operator*=(const FermionOperator& other);
    FermionOperator operator*(const FermionOperator& other);
};
}  // namespace operators::fermion
std::ostream& operator<<(std::ostream& os, const operators::fermion::FermionOperator& t);
#endif /* MATH_OPERATORS_FERMION_OPERATOR_VIEW_HPP_ */
