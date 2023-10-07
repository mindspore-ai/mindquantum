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

#ifndef MATH_OPERATORS_QUBIT_OPERATOR_VIEW_HPP_
#define MATH_OPERATORS_QUBIT_OPERATOR_VIEW_HPP_
#include <cstdint>
#include <list>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "math/operators/utils.h"
#include "math/pr/parameter_resolver.h"
#include "math/tensor/ops.h"
#include "math/tensor/ops/memory_operator.h"
#include "math/tensor/tensor.h"
#include "math/tensor/traits.h"
namespace operators::qubit {

namespace tn = tensor;
enum class TermValue : uint8_t {
    //! DO NOT CHANGE VALUE.
    I = 0,
    X = 1,
    Y = 2,
    Z = 3,
};

std::string to_string(const TermValue& term);

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

// -----------------------------------------------------------------------------

struct SinglePauliStr {
    using term_t = std::pair<size_t, TermValue>;
    using terms_t = std::vector<term_t>;
    using py_term_t = std::pair<size_t, std::string>;
    using py_terms_t = std::vector<py_term_t>;
    // -----------------------------------------------------------------------------

    static compress_term_t init(const std::string& pauli_string, const parameter::ParameterResolver& var
                                                                 = parameter::ParameterResolver(tn::ops::ones(1)));
    static compress_term_t init(const terms_t& terms, const parameter::ParameterResolver& var
                                                      = parameter::ParameterResolver(tn::ops::ones(1)));

    // -----------------------------------------------------------------------------
    // ABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABAB
    static constexpr uint64_t M_B = 6148914691236517205;
    static constexpr uint64_t M_A = M_B << 1;
    static std::tuple<tn::Tensor, uint64_t> MulSingleCompressTerm(uint64_t a, uint64_t b);
    static void InplaceMulCompressTerm(const term_t& term, compress_term_t& pauli);
    static compress_term_t Mul(const compress_term_t& lhs, const compress_term_t& rhs);
    static bool IsSameString(const key_t& k1, const key_t& k2);
    static std::string GetString(const compress_term_t& pauli);
    static term_t ParseToken(const std::string& token);
    static term_t py_term_to_term(const py_term_t& term);
    static terms_t py_terms_to_terms(const py_terms_t& terms);
};

// -----------------------------------------------------------------------------

class QubitOperator {
 public:
    using term_t = SinglePauliStr::term_t;
    using terms_t = SinglePauliStr::terms_t;
    using py_term_t = SinglePauliStr::py_term_t;
    using py_terms_t = SinglePauliStr::py_terms_t;
    using dict_t = std::vector<std::pair<terms_t, parameter::ParameterResolver>>;
    using py_dict_t = std::vector<std::pair<py_terms_t, parameter::ParameterResolver>>;

 private:
    bool Contains(const key_t& term) const;
    void Update(const compress_term_t& pauli);

    // -----------------------------------------------------------------------------

 public:
    QubitOperator() = default;
    explicit QubitOperator(const std::string& pauli_string,
                           const parameter::ParameterResolver& var = parameter::ParameterResolver(tn::ops::ones(1)));
    explicit QubitOperator(const terms_t& t,
                           const parameter::ParameterResolver& var = parameter::ParameterResolver(tn::ops::ones(1)));
    explicit QubitOperator(const term_t& t,
                           const parameter::ParameterResolver& var = parameter::ParameterResolver(tn::ops::ones(1)));
    explicit QubitOperator(const py_terms_t& t,
                           const parameter::ParameterResolver& var = parameter::ParameterResolver(tn::ops::ones(1)));
    explicit QubitOperator(const py_term_t& t,
                           const parameter::ParameterResolver& var = parameter::ParameterResolver(tn::ops::ones(1)));
    explicit QubitOperator(const py_dict_t& t);
    QubitOperator(const key_t& k, const value_t& v);

    // -----------------------------------------------------------------------------

    size_t size() const;
    std::string ToString() const;
    size_t count_qubits() const;
    dict_t get_terms() const;
    value_t get_coeff(const terms_t& term);
    bool is_singlet() const;
    parameter::ParameterResolver singlet_coeff() const;
    QubitOperator imag() const;
    QubitOperator real() const;
    tn::TDtype GetDtype() const;
    void CastTo(tn::TDtype dtype);
    QubitOperator hermitian_conjugated() const;
    bool parameterized() const;
    void set_coeff(const terms_t& term, const parameter::ParameterResolver& value);
    std::vector<std::pair<parameter::ParameterResolver, QubitOperator>> split() const;
    std::vector<QubitOperator> singlet() const;
    void subs(const parameter::ParameterResolver& other);

    // -----------------------------------------------------------------------------

    QubitOperator& operator+=(const tn::Tensor& c);
    QubitOperator& operator+=(const QubitOperator& other);
    friend QubitOperator operator+(QubitOperator lhs, const tensor::Tensor& rhs);
    friend QubitOperator operator+(QubitOperator lhs, const QubitOperator& rhs);

    QubitOperator& operator-=(const tn::Tensor& c);
    QubitOperator& operator-=(const QubitOperator& other);
    friend QubitOperator operator-(QubitOperator lhs, const tensor::Tensor& rhs);
    friend QubitOperator operator-(QubitOperator lhs, const QubitOperator& rhs);

    QubitOperator operator*=(const QubitOperator& other);
    QubitOperator operator*=(const parameter::ParameterResolver& other);
    friend QubitOperator operator*(QubitOperator lhs, const tensor::Tensor& other);
    friend QubitOperator operator*(QubitOperator lhs, const QubitOperator& rhs);

 public:
    QTerm_t terms{};
    tn::TDtype dtype = tn::TDtype::Float64;
};
}  // namespace operators::qubit
std::ostream& operator<<(std::ostream& os, const operators::qubit::QubitOperator& t);
#endif /* MATH_OPERATORS_QUBIT_OPERATOR_VIEW_HPP_ */
