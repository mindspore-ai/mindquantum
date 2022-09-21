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

#ifndef TESTS_EQUALITY_OPERATORS_HPP
#define TESTS_EQUALITY_OPERATORS_HPP

#include <Eigen/SparseCore>

template <typename scalar_t, int options, typename index_t>
auto operator==(const Eigen::SparseMatrix<scalar_t, options, index_t>& lhs,
                const Eigen::SparseMatrix<scalar_t, options, index_t>& rhs) {
    return lhs.isApprox(rhs);
}

#endif /* TESTS_EQUALITY_OPERATORS_HPP */
