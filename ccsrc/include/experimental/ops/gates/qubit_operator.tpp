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

#ifndef QUBIT_OPERATOR_TPP
#define QUBIT_OPERATOR_TPP

#include <cstdint>
#include <numeric>
#include <optional>
#include <utility>
#include <vector>

#include <unsupported/Eigen/KroneckerProduct>

#include <tweedledum/Operators/Standard/X.h>
#include <tweedledum/Operators/Standard/Y.h>
#include <tweedledum/Operators/Standard/Z.h>

#include "experimental/core/logging.hpp"
#include "experimental/ops/gates/details/eigen_sparse_identity.hpp"
#include "experimental/ops/gates/qubit_operator.hpp"

// =============================================================================

namespace mindquantum::ops {

template <typename coeff_t>
uint32_t QubitOperator<coeff_t>::count_gates() const noexcept {
    return std::accumulate(begin(base_t::terms_), end(base_t::terms_), 0UL, [](const auto& val, const auto& term) {
        const auto& [ops, coeff] = term;
        return val + std::size(ops);
    });
}

// =============================================================================

template <typename coeff_t>
auto QubitOperator<coeff_t>::matrix() const -> std::optional<matrix_t> {
    return std::nullopt;
}

// =============================================================================

template <typename coeff_t>
auto QubitOperator<coeff_t>::sparse_matrix(std::optional<uint32_t> n_qubits) const -> std::optional<sparse_matrix_t> {
    namespace Op = tweedledum::Op;
    using scalar_t = typename sparse_matrix_t::Scalar;
    using dense_2x2_t = Eigen::Matrix<scalar_t, 2, 2>;

    // NB: required since we are indexing into the array below
    constexpr auto offset = static_cast<std::underlying_type_t<TermValue>>(TermValue::I);
    static_assert(static_cast<std::underlying_type_t<TermValue>>(TermValue::I) - offset == 0);
    static_assert(static_cast<std::underlying_type_t<TermValue>>(TermValue::X) - offset == 1);
    static_assert(static_cast<std::underlying_type_t<TermValue>>(TermValue::Y) - offset == 2);
    static_assert(static_cast<std::underlying_type_t<TermValue>>(TermValue::Z) - offset == 3);

    static const std::array<sparse_matrix_t, 4> pauli_matrices = {
        sparse_matrix_t{::generate_eigen_diagonal<scalar_t, 2>()},
        Op::X::matrix().sparseView(),
        Op::Y::matrix().sparseView(),
        Op::Z::matrix().sparseView(),
    };

    if (std::empty(base_t::terms_)) {
        return std::nullopt;
    }

    // NB: this is assuming that num_targets_ is up-to-date
    const auto& n_qubits_local = base_t::num_targets_;

    if (n_qubits_local == 0UL && !n_qubits) {
        MQ_ERROR("You should specify n_qubits for converting an identity qubit operator.");
        return std::nullopt;
    }

    if (n_qubits && n_qubits.value() < n_qubits_local) {
        MQ_ERROR("Given n_qubits {} is smaller than the number of qubits of this qubit operator, which is {}.\n",
                 n_qubits.value(), n_qubits_local);
        return std::nullopt;
    }

    const auto n_qubits_value = n_qubits.value_or(n_qubits_local);
    MQ_INFO("n_qubits_value {}", n_qubits_value);

    const auto process_term = [n_qubits_value](const auto& local_ops, const auto& coeff) -> sparse_matrix_t {
        if (std::empty(local_ops)) {
            MQ_INFO("Empty local ops: {}", 1U << n_qubits_value);
            return details::n_identity<scalar_t>(n_qubits_value) * coeff;
        }

        // NB: IMPORTANT! This below assumes that simplify() has been called correctly to eliminate terms like X1 Y1
        std::vector<int64_t> order(n_qubits_value, -1);
        int64_t idx = 0L;
        for (const auto& term : local_ops) {
            order[term.first] = idx++;
        }
        sparse_matrix_t init(1, 1);
        init.insert(0, 0) = coeff;
        return std::accumulate(
            begin(order), end(order), init, [&local_ops](const sparse_matrix_t& init, const auto& idx) {
                MQ_INFO("init {}x{}, idx = {}", init.rows(), init.cols(), idx);
#ifdef _MSC_VER
                constexpr auto offset = static_cast<std::underlying_type_t<TermValue>>(TermValue::I);
#endif  // _MSC_VER
                if (idx < 0) {
                    static_assert(static_cast<std::underlying_type_t<TermValue>>(TermValue::I) - offset == 0);
                    const auto tmp = Eigen::kroneckerProduct(pauli_matrices[0], init).eval();
                    MQ_INFO("return {}x{}", tmp.rows(), tmp.cols());
                    return Eigen::kroneckerProduct(pauli_matrices[0], init).eval();
                }
                const auto tmp
                    = Eigen::kroneckerProduct(
                          pauli_matrices[static_cast<std::underlying_type_t<TermValue>>(local_ops[idx].second)
                                         - offset],
                          init)
                          .eval();
                MQ_INFO("return {}x{}", tmp.rows(), tmp.cols());
                return Eigen::kroneckerProduct(
                           pauli_matrices[static_cast<std::underlying_type_t<TermValue>>(local_ops[idx].second)
                                          - offset],
                           init)
                    .eval();
            });

        // NB: IMPORTANT! This below assumes that simplify() has been called correctly to eliminate terms like X1 Y1
        // std::vector<sparse_matrix_t> total(
        //     n_qubits_value, pauli_matrices[static_cast<std::underlying_type_t<TermValue>>(TermValue::I) - offset]);

        //         // Workaround internal compiler error with GCC 8.1
        // #if (defined __GNUC__) && (__GNUC__ == 8 && __GNUC_MINOR__ == 1)
        //         for (const auto& term : local_ops) {
        //             const auto [qubit_id, local_op] = term;
        //             total[qubit_id] = pauli_matrices[static_cast<std::underlying_type_t<TermValue>>(local_op) -
        //             offset];
        //         }
        // #else
        //         for (const auto& [qubit_id, local_op] : local_ops) {
        //             total[qubit_id] = pauli_matrices[static_cast<std::underlying_type_t<TermValue>>(local_op) -
        //             offset];
        //         }
        // #endif  // MINGW

        //         sparse_matrix_t init(1, 1);
        //         init.insert(0, 0) = coeff;
        //         return std::accumulate(begin(total), end(total), init, [](const sparse_matrix_t& init, const auto&
        //         matrix) {
        //             return Eigen::kroneckerProduct(matrix, init).eval();
        //         });
    };

    // NB: if the coefficient type is always constant (e.g. float, double), then the compiler should be able to
    // remove both if() below.

    auto it = begin(base_t::terms_);
    if (!coeff_policy_t::is_const(it->second)) {
        MQ_ERROR("Coeff is not const! ({})", it->second);
        return {};
    }
    auto result = process_term(it->first, coeff_policy_t::get_num(it->second));
    MQ_INFO("result: {}x{}", result.rows(), result.cols());
    ++it;
    for (; it != end(base_t::terms_); ++it) {
        if (!coeff_policy_t::is_const(it->second)) {
            MQ_ERROR("Coeff is not const! ({})", it->second);
            return {};
        }
        result += process_term(it->first, coeff_policy_t::get_num(it->second));
        MQ_INFO("result: {}x{}", result.rows(), result.cols());
    }
    MQ_INFO("result: {}x{}", result.rows(), result.cols());

    return result;
}

}  // namespace mindquantum::ops

#endif /* QUBIT_OPERATOR_TPP */
