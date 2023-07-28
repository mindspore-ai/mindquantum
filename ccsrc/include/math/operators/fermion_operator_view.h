/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef MATH_OPERATORS_FERMION_OPERATOR_VIEW_HPP_
#define MATH_OPERATORS_FERMION_OPERATOR_VIEW_HPP_

#include <cstdint>
#include <limits>
#include <list>
#include <map>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "math/operators/utils.h"
#include "math/pr/parameter_resolver.h"
#include "math/tensor/ops/concrete_tensor.h"
#include "math/tensor/tensor.h"
#include "math/tensor/traits.h"
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

TermValue hermitian_conjugated(const TermValue& t);
std::string to_string(const TermValue& term);

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

struct SingleFermionStr {
    using term_t = std::pair<uint64_t, TermValue>;
    using terms_t = std::vector<term_t>;
    using py_term_t = std::pair<uint64_t, uint64_t>;
    using py_terms_t = std::vector<py_term_t>;

    // -----------------------------------------------------------------------------

    static std::pair<compress_term_t, bool> init(const std::string& fermion_string,
                                                 const parameter::ParameterResolver& var
                                                 = parameter::ParameterResolver(tn::ops::ones(1)));
    static std::pair<compress_term_t, bool> init(const terms_t& terms,
                                                 const parameter::ParameterResolver& var
                                                 = parameter::ParameterResolver(tn::ops::ones(1)));

    // -----------------------------------------------------------------------------

    static std::tuple<tn::Tensor, uint64_t> MulSingleCompressTerm(uint64_t a, uint64_t b);
    static bool InplaceMulCompressTerm(const term_t& term, compress_term_t& fermion);
    static compress_term_t Mul(const compress_term_t& lhs, const compress_term_t& rhs);
    static bool IsSameString(const key_t& k1, const key_t& k2);
    static std::string GetString(const compress_term_t& fermion);
    static term_t ParseToken(const std::string& token);
    static std::vector<uint64_t> NumOneMask(const compress_term_t& fermion);
    static uint64_t PrevOneMask(const std::vector<uint64_t>& one_mask, size_t idx);
    static bool has_a_ad(uint64_t t);
    static term_t py_term_to_term(const py_term_t& term);
    static terms_t py_terms_to_terms(const py_terms_t& terms);
};

// -----------------------------------------------------------------------------

class FermionOperator {
 public:
    using term_t = SingleFermionStr::term_t;
    using terms_t = SingleFermionStr::terms_t;
    using py_term_t = SingleFermionStr::py_term_t;
    using py_terms_t = SingleFermionStr::py_terms_t;

    using dict_t = std::vector<std::pair<terms_t, parameter::ParameterResolver>>;
    using py_dict_t = std::vector<std::pair<py_terms_t, parameter::ParameterResolver>>;

 private:
    bool Contains(const key_t& term) const;
    void Update(const compress_term_t& fermion);

    // -----------------------------------------------------------------------------
 public:
    FermionOperator() = default;
    explicit FermionOperator(const std::string& fermion_string,
                             const parameter::ParameterResolver& var = parameter::ParameterResolver(tn::ops::ones(1)));
    explicit FermionOperator(const terms_t& t,
                             const parameter::ParameterResolver& var = parameter::ParameterResolver(tn::ops::ones(1)));
    explicit FermionOperator(const term_t& t,
                             const parameter::ParameterResolver& var = parameter::ParameterResolver(tn::ops::ones(1)));
    explicit FermionOperator(const py_terms_t& t,
                             const parameter::ParameterResolver& var = parameter::ParameterResolver(tn::ops::ones(1)));
    explicit FermionOperator(const py_term_t& t,
                             const parameter::ParameterResolver& var = parameter::ParameterResolver(tn::ops::ones(1)));
    explicit FermionOperator(const py_dict_t& t);
    FermionOperator(const key_t& k, const value_t& v);
    // -----------------------------------------------------------------------------

    size_t size() const;
    std::string ToString() const;
    dict_t get_terms() const;
    value_t get_coeff(const terms_t& term);
    void set_coeff(const terms_t& term, const parameter::ParameterResolver& value);
    bool is_singlet() const;
    parameter::ParameterResolver singlet_coeff() const;
    std::vector<FermionOperator> singlet() const;
    size_t count_qubits() const;
    FermionOperator imag() const;
    FermionOperator real() const;
    tn::TDtype GetDtype() const;
    void CastTo(tn::TDtype dtype);
    std::vector<std::pair<parameter::ParameterResolver, FermionOperator>> split() const;
    bool parameterized() const;
    void subs(const parameter::ParameterResolver& other);
    FermionOperator hermitian_conjugated() const;
    FermionOperator normal_ordered() const;
    // -----------------------------------------------------------------------------

    FermionOperator& operator+=(const tn::Tensor& c);
    FermionOperator& operator+=(const FermionOperator& other);
    friend FermionOperator operator+(FermionOperator lhs, const tensor::Tensor& rhs);
    friend FermionOperator operator+(FermionOperator lhs, const FermionOperator& rhs);

    // -----------------------------------------------------------------------------

    FermionOperator operator*=(const FermionOperator& other);
    FermionOperator operator*=(const parameter::ParameterResolver& other);
    friend FermionOperator operator*(FermionOperator lhs, const FermionOperator& rhs);

 public:
    QTerm_t terms{};
    tn::TDtype dtype = tn::TDtype::Float64;
};
}  // namespace operators::fermion
std::ostream& operator<<(std::ostream& os, const operators::fermion::FermionOperator& t);
#endif /* MATH_OPERATORS_FERMION_OPERATOR_VIEW_HPP_ */
