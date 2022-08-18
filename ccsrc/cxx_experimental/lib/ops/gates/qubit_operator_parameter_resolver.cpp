//   Copyright 2022 <Huawei Technologies Co., Ltd>
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

#include "ops/gates/qubit_operator_parameter_resolver.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <numeric>
#include <optional>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <boost/range/adaptor/transformed.hpp>
#include <boost/range/algorithm/copy.hpp>
#include <boost/range/algorithm_ext/push_back.hpp>

#include <unsupported/Eigen/KroneckerProduct>

#include <tweedledum/Operators/Standard/X.h>
#include <tweedledum/Operators/Standard/Y.h>
#include <tweedledum/Operators/Standard/Z.h>

#include <fmt/format.h>

#include "core/format/format_complex.hpp"
#include "core/logging.hpp"
#include "ops/gates.hpp"
#include "ops/gates/terms_operator.hpp"

// -----------------------------------------------------------------------------

using namespace std::literals::string_literals;  // NOLINT(build/namespaces_literals)

namespace mindquantum::ops {
// =============================================================================

uint32_t QubitOperatorPR::count_gates() const noexcept {
    return std::accumulate(begin(terms_), end(terms_), 0UL, [](const auto& val, const auto& term) {
        const auto& [ops, coeff] = term;
        return val + std::size(ops);
    });
}
// =============================================================================

auto QubitOperatorPR::matrix(std::optional<uint32_t> n_qubits) const -> std::optional<csr_matrix_t> {
    namespace Op = tweedledum::Op;
    using dense_2x2_t = Eigen::Matrix2cd;

    // NB: required since we are indexing into the array below
    constexpr auto offset = static_cast<std::underlying_type_t<TermValue>>(TermValue::I);
    static_assert(static_cast<std::underlying_type_t<TermValue>>(TermValue::I) - offset == 0);
    static_assert(static_cast<std::underlying_type_t<TermValue>>(TermValue::X) - offset == 1);
    static_assert(static_cast<std::underlying_type_t<TermValue>>(TermValue::Y) - offset == 2);
    static_assert(static_cast<std::underlying_type_t<TermValue>>(TermValue::Z) - offset == 3);

    static const std::array<csr_matrix_t, 4> pauli_matrices = {
        dense_2x2_t::Identity().sparseView(),
        Op::X::matrix().sparseView(),
        Op::Y::matrix().sparseView(),
        Op::Z::matrix().sparseView(),
    };

    if (std::empty(terms_)) {
        return std::nullopt;
    }

    // NB: this is assuming that num_targets_ is up-to-date
    const auto& n_qubits_local = num_targets_;

    if (n_qubits_local == 0UL && !n_qubits) {
        std::cerr << "You should specify n_qubits for converting an identity qubit operator.";
        return std::nullopt;
    }

    if (n_qubits && n_qubits.value() < n_qubits_local) {
        std::cerr << fmt::format(
            "Given n_qubits {} is smaller than the number of qubits of this qubit operator, which is {}.\n",
            n_qubits.value(), n_qubits_local);
        return std::nullopt;
    }

    const auto n_qubits_value = n_qubits.value_or(n_qubits_local);

    const auto process_term = [n_qubits_value](const auto& local_ops, const auto& coeff) -> csr_matrix_t {
        if (std::empty(local_ops)) {
            return (Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>::Identity(
                        1U << n_qubits_value, 1U << n_qubits_value))
                       .sparseView()
                   * coeff.const_value;
        }

        // TODO(dnguyen): The `total` variable is probably not required and could be removed altogether...
        std::vector<csr_matrix_t> total(
            n_qubits_value, pauli_matrices[static_cast<std::underlying_type_t<TermValue>>(TermValue::I) - offset]);
        for (const auto& [qubit_id, local_op] : local_ops) {
            total[qubit_id] = pauli_matrices[static_cast<std::underlying_type_t<TermValue>>(local_op) - offset];
        }

        csr_matrix_t init(1, 1);
        init.insert(0, 0) = coeff.const_value;
        return std::accumulate(begin(total), end(total), init, [](const csr_matrix_t& init, const auto& matrix) {
            return Eigen::kroneckerProduct(matrix, init).eval();
        });
    };

    auto it = begin(terms_);
    if (!it->second.IsConst()) {
        MQ_ERROR("Coeff is not const! ({})", it->second);
        return {};
    }
    auto result = process_term(it->first, it->second);
    ++it;
    for (; it != end(terms_); ++it) {
        if (!it->second.IsConst()) {
            MQ_ERROR("Coeff is not const! ({})", it->second);
            return {};
        }
        result += process_term(it->first, it->second);
    }

    return result;
}

// =============================================================================

auto QubitOperatorPR::simplify_(terms_t terms, coefficient_t coeff) -> std::tuple<terms_t, coefficient_t> {
    if (std::empty(terms)) {
        return {terms_t{}, coeff};
    }
    std::sort(
        begin(terms), end(terms), [](const auto& lhs, const auto& rhs) constexpr { return lhs.first < rhs.first; });

    terms_t reduced_terms;
    auto left_term = terms.front();
    for (auto it(begin(terms) + 1); it != end(terms); ++it) {
        const auto& [left_qubit_id, left_operator] = left_term;
        const auto& [right_qubit_id, right_operator] = *it;

        if (left_qubit_id == right_qubit_id) {
            const auto [new_coeff, new_op] = pauli_products(left_operator, right_operator);
            left_term = term_t{left_qubit_id, new_op};
            coeff *= new_coeff;
        } else {
            if (left_term.second != TermValue::I) {
                reduced_terms.emplace_back(left_term);
            }
            left_term = *it;
        }
    }

    return {std::move(terms), coeff};
}

// -----------------------------------------------------------------------------

auto QubitOperatorPR::simplify_(py_terms_t py_terms, coefficient_t coeff) -> std::tuple<terms_t, coefficient_t> {
    terms_t terms;
    terms.reserve(std::size(py_terms));
    boost::range::push_back(
        terms, py_terms | boost::adaptors::transformed([](const auto& value) -> term_t {
                   return {std::get<0>(value), static_cast<mindquantum::ops::TermValue>(std::get<1>(value))};
               }));

    return simplify_(terms, coeff);
}

// =============================================================================

auto QubitOperatorPR::sort_terms_(terms_t local_ops, coefficient_t coeff) -> std::pair<terms_t, coefficient_t> {
    std::sort(begin(local_ops), end(local_ops));
    return {std::move(local_ops), coeff};  // TODO(dnguyen): Should we move? or can (N)RVO take care of that?
}

// =============================================================================

}  // namespace mindquantum::ops
