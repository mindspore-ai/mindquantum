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

#ifndef INCLUDE_SIMULATOR_MPS_DETAIL_CPU_MPS_POLICY_HPP
#define INCLUDE_SIMULATOR_MPS_DETAIL_CPU_MPS_POLICY_HPP

#include <complex>
#include <type_traits>

#include "math/tensor/traits.h"

namespace mindquantum::sim::mps::detail {
template <typename calc_t_>
struct CPUMPSPolicy {
    using calc_type = calc_t_;
    using complex_t = std::complex<calc_type>;
    using py_qs_data_t = complex_t;

    static constexpr tensor::TDtype kDType = tensor::to_dtype_v<complex_t>;
    static constexpr calc_type kEpsRel = std::is_same_v<calc_type, float> ? static_cast<calc_type>(1e-6)
                                                                          : static_cast<calc_type>(1e-12);
    static constexpr calc_type kLambdaFloor = kEpsRel;
};
}  // namespace mindquantum::sim::mps::detail

#endif  // INCLUDE_SIMULATOR_MPS_DETAIL_CPU_MPS_POLICY_HPP
