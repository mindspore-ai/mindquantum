/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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

#ifndef FORMAT_EIGEN_MATRICES_HPP
#define FORMAT_EIGEN_MATRICES_HPP

#include <Eigen/Dense>
#include <Eigen/SparseCore>

#include <fmt/format.h>
#include <fmt/ostream.h>

template <typename scalar_t, int options, typename index_t>
struct fmt::formatter<Eigen::SparseMatrix<scalar_t, options, index_t>> : fmt::ostream_formatter {};

template <typename scalar_t, int rows, int cols, int options, int max_rows, int max_cols>
struct fmt::formatter<Eigen::Matrix<scalar_t, rows, cols, options, max_rows, max_cols>> : fmt::ostream_formatter {};

#endif /* FORMAT_EIGEN_MATRICES_HPP */
