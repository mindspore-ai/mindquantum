/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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

#ifndef INCLUDE_SIMULATOR_MPS_LINALG_SVD_HPP
#define INCLUDE_SIMULATOR_MPS_LINALG_SVD_HPP

#include <Eigen/Dense>
#include <Eigen/SVD>

namespace mindquantum::sim::mps::linalg {
// Thin SVD wrapper.
//
// Today: Eigen BDCSVD (thin U/V).
// Future: optional LAPACK gesdd backend with a shared API.
template <typename scalar_t>
class ThinSvd {
 public:
    using Mat = Eigen::Matrix<scalar_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using Svd = Eigen::BDCSVD<Mat>;

    ThinSvd() = default;

    explicit ThinSvd(Eigen::Index rows, Eigen::Index cols, unsigned int opts) : svd_(rows, cols, opts) {
        rows_cap_ = rows;
        cols_cap_ = cols;
        opts_ = opts;
    }

    void EnsureCapacity(Eigen::Index rows, Eigen::Index cols, unsigned int opts) {
        if (rows <= rows_cap_ && cols <= cols_cap_ && opts == opts_) {
            return;
        }
        svd_ = Svd(rows, cols, opts);
        rows_cap_ = rows;
        cols_cap_ = cols;
        opts_ = opts;
    }

    void Compute(const Mat& a, unsigned int opts) {
        EnsureCapacity(a.rows(), a.cols(), opts);
        svd_.compute(a, opts);
    }

    const auto& singularValues() const {
        return svd_.singularValues();
    }
    const auto& matrixU() const {
        return svd_.matrixU();
    }
    const auto& matrixV() const {
        return svd_.matrixV();
    }

 private:
    Svd svd_;
    Eigen::Index rows_cap_ = 0;
    Eigen::Index cols_cap_ = 0;
    unsigned int opts_ = 0;
};
}  // namespace mindquantum::sim::mps::linalg

#endif  // INCLUDE_SIMULATOR_MPS_LINALG_SVD_HPP

