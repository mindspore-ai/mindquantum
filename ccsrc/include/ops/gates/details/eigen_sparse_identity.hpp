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

#ifndef MQ_OPS_DETAILS_EIGEN_SPARSE_IDENTITY_HPP
#define MQ_OPS_DETAILS_EIGEN_SPARSE_IDENTITY_HPP

#include "ops/gates/details/eigen_diagonal_identity.hpp"
#include "ops/gates/types.hpp"

namespace mindquantum::ops::details {
template <typename coeff_t>
auto n_identity(std::size_t n_qubits) -> types::csr_matrix_t<coeff_t> {
    using matrix_t = types::csr_matrix_t<coeff_t>;
    using scalar_t = typename matrix_t::Scalar;
    // NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define MQ_CASE_FOR_NQUBITS(n)                                                                                         \
    case n:                                                                                                            \
        return matrix_t {                                                                                              \
            ::generate_eigen_diagonal<scalar_t, 1U << (n)>()                                                           \
        }

    switch (n_qubits) {
        // clang-format off
        MQ_CASE_FOR_NQUBITS(0U); break;  // NOLINT
        MQ_CASE_FOR_NQUBITS(1U); break;  // NOLINT
        MQ_CASE_FOR_NQUBITS(2U); break;  // NOLINT
        MQ_CASE_FOR_NQUBITS(3U); break;  // NOLINT
        MQ_CASE_FOR_NQUBITS(4U); break;  // NOLINT
        MQ_CASE_FOR_NQUBITS(5U); break;  // NOLINT
        MQ_CASE_FOR_NQUBITS(6U); break;  // NOLINT
        // clang-format on
        default:
            auto tmp = Eigen::DiagonalMatrix<scalar_t, Eigen::Dynamic>(1U << n_qubits);
            tmp.setIdentity();
            return matrix_t{tmp};
            break;
    }
#undef MQ_CASE_FOR_NQUBITS
}
}  // namespace mindquantum::ops::details

#endif /* MQ_OPS_DETAILS_EIGEN_SPARSE_IDENTITY_HPP */
