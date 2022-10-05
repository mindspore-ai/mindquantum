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

#ifndef MQ_CATCH2_EIGEN_HPP
#define MQ_CATCH2_EIGEN_HPP

#include <string>

#include <Eigen/SparseCore>

#include <catch2/matchers/catch_matchers_templated.hpp>

// =============================================================================

namespace mindquantum::catch2 {
template <typename scalar_t, int options, typename index_t>
struct EigenSparseMatrixMatcher : Catch::Matchers::MatcherGenericBase {
    using matrix_t = Eigen::SparseMatrix<scalar_t, options, index_t>;
    explicit EigenSparseMatrixMatcher(const matrix_t& matrix_ref) : ref_matrix{matrix_ref} {
    }

    template <typename other_scalar_t, int other_options, typename other_index_t>
    bool match(const Eigen::SparseMatrix<other_scalar_t, other_options, other_index_t>& matrix) const {
        return matrix.isApprox(ref_matrix);
    }

    std::string describe() const override {
        return "Equals: " + ::Catch::Detail::stringify(ref_matrix);
    }

 private:
    const matrix_t& ref_matrix;
};

// -----------------------------------------------------------------------------

template <typename scalar_t, int options, typename index_t>
auto Equals(const Eigen::SparseMatrix<scalar_t, options, index_t>& ref_matrix) {
    return EigenSparseMatrixMatcher<scalar_t, options, index_t>(ref_matrix);
}
}  // namespace mindquantum::catch2

// =============================================================================
#endif /* MQ_CATCH2_EIGEN_HPP */
