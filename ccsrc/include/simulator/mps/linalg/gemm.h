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

#ifndef INCLUDE_SIMULATOR_MPS_LINALG_GEMM_HPP
#define INCLUDE_SIMULATOR_MPS_LINALG_GEMM_HPP

#include <Eigen/Dense>

namespace mindquantum::sim::mps::linalg {
// Minimal GEMM wrapper.
//
// Design note (MPS_SIMULATOR_DESIGN.md):
// - This layer exists so we can switch the hot-path to BLAS GEMM when available,
//   while keeping an Eigen fallback for portability.
template <typename DerivedA, typename DerivedB, typename DerivedC>
inline void Gemm(const Eigen::MatrixBase<DerivedA>& a, const Eigen::MatrixBase<DerivedB>& b,
                 Eigen::MatrixBase<DerivedC>* c) {
    c->derived().noalias() = a.derived() * b.derived();
}
}  // namespace mindquantum::sim::mps::linalg

#endif  // INCLUDE_SIMULATOR_MPS_LINALG_GEMM_HPP

