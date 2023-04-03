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
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "math/operators/utils.hpp"
#include "math/pr/parameter_resolver.hpp"
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

    // -----------------------------------------------------------------------------

    static compress_term_t init(const std::string& pauli_string,
                                const parameter::ParameterResolver& var = tn::ops::ones(1));
    static compress_term_t init(const terms_t& terms, const parameter::ParameterResolver& var = tn::ops::ones(1));

    // -----------------------------------------------------------------------------

    static std::tuple<tn::Tensor, uint64_t> MulSingleCompressTerm(uint64_t a, uint64_t b);
    static void InplaceMulCompressTerm(const term_t& term, compress_term_t& pauli);
    static compress_term_t Mul(const compress_term_t& lhs, const compress_term_t& rhs);
    static bool IsSameString(const key_t& k1, const key_t& k2);
    static std::string GetString(const compress_term_t& pauli);
    static term_t ParseToken(const std::string& token);
};

// -----------------------------------------------------------------------------

class QubitOperator {
 public:
    using term_t = SinglePauliStr::term_t;
    using terms_t = SinglePauliStr::terms_t;
    using dict_t = std::vector<std::pair<terms_t, parameter::ParameterResolver>>;

 private:
    bool Contains(const key_t& term) const;
    void Update(const compress_term_t& pauli);

    // -----------------------------------------------------------------------------

 public:
    QubitOperator() = default;
    explicit QubitOperator(const std::string& pauli_string, const parameter::ParameterResolver& var = tn::ops::ones(1));
    explicit QubitOperator(const terms_t& t, const parameter::ParameterResolver& var = tn::ops::ones(1));

    // -----------------------------------------------------------------------------

    size_t size() const;
    std::string ToString() const;
    size_t count_qubits() const;
    dict_t get_terms() const;
    bool is_singlet() const;
    parameter::ParameterResolver singlet_coeff() const;

    // -----------------------------------------------------------------------------

    QubitOperator& operator+=(const tn::Tensor& c);
    QubitOperator& operator+=(const QubitOperator& other);
    friend QubitOperator operator+(QubitOperator lhs, const tensor::Tensor& rhs);
    friend QubitOperator operator+(QubitOperator lhs, const QubitOperator& rhs);

    QubitOperator& operator-=(const QubitOperator& other);
    friend QubitOperator operator-(QubitOperator lhs, const QubitOperator& rhs);

    QubitOperator operator*=(const QubitOperator& other);
    QubitOperator operator*=(const parameter::ParameterResolver& other);
    QubitOperator operator*(const tensor::Tensor& other);
    QubitOperator operator*(const QubitOperator& other);

 public:
    QTerm_t terms{};
};
}  // namespace operators::qubit
std::ostream& operator<<(std::ostream& os, const operators::qubit::QubitOperator& t);
#endif /* MATH_OPERATORS_QUBIT_OPERATOR_VIEW_HPP_ */
