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

#include "config/conversion.hpp"
#include "config/format/eigen_matrices.hpp"
#include "config/format/std_optional.hpp"
#include "config/logging.hpp"
#include "config/type_traits.hpp"

#include "ops/gates/details/eigen_sparse_identity.hpp"
#include "ops/gates/qubit_operator.hpp"

// =============================================================================

namespace mindquantum::ops {
template <typename coeff_t>
auto QubitOperator<coeff_t>::get_op_matrix(Op op_type) -> op_matrix_t {
    constexpr static std::array<typename sparse_matrix_t::Scalar, 4> x_mat = {0., 1., 1., 0.};
    constexpr static std::array<typename sparse_matrix_t::Scalar, 4> y_mat = {0., {0., 1.}, {0., -1.}, 0.};
    constexpr static std::array<typename sparse_matrix_t::Scalar, 4> z_mat = {1., 0., 0., -1.};
    constexpr static std::array<typename sparse_matrix_t::Scalar, 4> invalid_mat = {0., 0., 0., 0.};

    if (op_type == Op::X) {
        return op_matrix_t{x_mat.data()};
    } else if (op_type == Op::Y) {
        return op_matrix_t{y_mat.data()};
    } else if (op_type == Op::Z) {
        return op_matrix_t{z_mat.data()};
    }
    return op_matrix_t{invalid_mat.data()};
}

template <typename coeff_t>
auto QubitOperator<coeff_t>::simplify(const self_t& qubit_op) -> derived_cmplx_t {
    if constexpr (!is_real_valued) {
        return qubit_op;
    } else {
        using conv_helper_t = traits::conversion_helper<typename derived_cmplx_t::coefficient_t>;

        typename derived_cmplx_t::coeff_term_dict_t terms;
        for (const auto& [local_ops, coeff] : qubit_op.get_terms()) {
            const auto [new_terms, new_coeff] = derived_cmplx_t::term_policy_t::simplify(local_ops,
                                                                                         conv_helper_t::apply(coeff));
            terms.emplace_back(new_terms, new_coeff);
        }
        return derived_cmplx_t{std::move(terms)};
    }
}

// =============================================================================

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
    MQ_TRACE("Calling QubitOperator<{}>::matrix()", get_type_name<coeff_t>());
    return std::nullopt;
}

// =============================================================================

template <typename coeff_t>
auto QubitOperator<coeff_t>::sparse_matrix(std::optional<uint32_t> n_qubits) const -> std::optional<sparse_matrix_t> {
    MQ_TRACE("Calling QubitOperator<{}>::sparse_matrix({})", get_type_name<coeff_t>(), n_qubits);
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
        get_op_matrix(Op::X).sparseView(),
        get_op_matrix(Op::Y).sparseView(),
        get_op_matrix(Op::Z).sparseView(),
    };

    if (std::empty(base_t::terms_)) {
        return std::nullopt;
    }

    const auto& n_qubits_local = base_t::num_targets();

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

    const auto process_term = [n_qubits_value](const auto& local_ops, const auto& coeff) -> sparse_matrix_t {
        if (std::empty(local_ops)) {
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
#ifdef _MSC_VER
                constexpr auto offset = static_cast<std::underlying_type_t<TermValue>>(TermValue::I);
#endif  // _MSC_VER
                if (idx < 0) {
                    static_assert(static_cast<std::underlying_type_t<TermValue>>(TermValue::I) - offset == 0);
                    const auto tmp = Eigen::kroneckerProduct(pauli_matrices[0], init).eval();
                    return Eigen::kroneckerProduct(pauli_matrices[0], init).eval();
                }
                const auto tmp
                    = Eigen::kroneckerProduct(
                          pauli_matrices[static_cast<std::underlying_type_t<TermValue>>(local_ops[idx].second)
                                         - offset],
                          init)
                          .eval();
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
    ++it;
    for (; it != end(base_t::terms_); ++it) {
        if (!coeff_policy_t::is_const(it->second)) {
            MQ_ERROR("Coeff is not const! ({})", it->second);
            return {};
        }
        result += process_term(it->first, coeff_policy_t::get_num(it->second));
    }

    return result;
}

}  // namespace mindquantum::ops

#endif /* QUBIT_OPERATOR_TPP */
