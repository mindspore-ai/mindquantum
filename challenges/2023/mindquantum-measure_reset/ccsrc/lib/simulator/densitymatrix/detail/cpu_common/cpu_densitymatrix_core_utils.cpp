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
#include "config/openmp.h"
#include "core/utils.h"
#include "math/pr/parameter_resolver.h"
#include "simulator/utils.h"
#ifdef __x86_64__
#    include "simulator/densitymatrix/detail/cpu_densitymatrix_avx_double_policy.h"
#    include "simulator/densitymatrix/detail/cpu_densitymatrix_avx_float_policy.h"
#elif defined(__amd64)
#    include "simulator/densitymatrix/detail/cpu_densitymatrix_arm_double_policy.h"
#    include "simulator/densitymatrix/detail/cpu_densitymatrix_arm_float_policy.h"
#endif
#include "simulator/densitymatrix/detail/cpu_densitymatrix_policy.h"

namespace mindquantum::sim::densitymatrix::detail {
template <typename derived_, typename calc_type_>
void CPUDensityMatrixPolicyBase<derived_, calc_type_>::SwapValue(qs_data_p_t qs, index_t x0, index_t y0, index_t x1,
                                                                 index_t y1, qs_data_t coeff) {
    if (x0 >= y0) {
        qs_data_t tmp = qs[IdxMap(x0, y0)];
        qs[IdxMap(x0, y0)] = coeff * GetValue(qs, x1, y1);
        SetValue(qs, x1, y1, coeff * tmp);
    } else {
        qs_data_t tmp = std::conj(qs[IdxMap(y0, x0)]);
        qs[IdxMap(y0, x0)] = std::conj(coeff * GetValue(qs, x1, y1));
        SetValue(qs, x1, y1, coeff * tmp);
    }
}

#ifdef __x86_64__
template struct CPUDensityMatrixPolicyBase<CPUDensityMatrixPolicyAvxFloat, float>;
template struct CPUDensityMatrixPolicyBase<CPUDensityMatrixPolicyAvxDouble, double>;
#elif defined(__amd64)
template struct CPUDensityMatrixPolicyBase<CPUDensityMatrixPolicyArmFloat, float>;
template struct CPUDensityMatrixPolicyBase<CPUDensityMatrixPolicyArmDouble, double>;
#endif
}  // namespace mindquantum::sim::densitymatrix::detail
