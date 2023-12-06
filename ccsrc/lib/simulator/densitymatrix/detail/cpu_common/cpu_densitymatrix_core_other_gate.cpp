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
#include "math/pr/parameter_resolver.h"
#ifdef __x86_64__
#    include "simulator/densitymatrix/detail/cpu_densitymatrix_avx_double_policy.h"
#    include "simulator/densitymatrix/detail/cpu_densitymatrix_avx_float_policy.h"
#elif defined(__amd64)
#    include "simulator/densitymatrix/detail/cpu_densitymatrix_arm_double_policy.h"
#    include "simulator/densitymatrix/detail/cpu_densitymatrix_arm_float_policy.h"
#endif
#include "ops/gates.h"
#include "simulator/densitymatrix/detail/cpu_densitymatrix_policy.h"
namespace mindquantum::sim::densitymatrix::detail {
template <typename derived_, typename calc_type_>
void CPUDensityMatrixPolicyBase<derived_, calc_type_>::ApplyH(qs_data_p_t* qs_p, const qbits_t& objs,
                                                              const qbits_t& ctrls, index_t dim) {
    matrix_t m{{M_SQRT1_2, M_SQRT1_2}, {M_SQRT1_2, -M_SQRT1_2}};
    derived::ApplySingleQubitMatrix(*qs_p, qs_p, objs[0], ctrls, m, dim);
}
template <typename derived_, typename calc_type_>
void CPUDensityMatrixPolicyBase<derived_, calc_type_>::ApplySX(qs_data_p_t* qs_p, const qbits_t& objs,
                                                               const qbits_t& ctrls, index_t dim) {
    py_qs_data_t a = py_qs_data_t(0.5, 0.5);
    py_qs_data_t b = std::conj(a);
    matrix_t m{{a, b}, {b, a}};
    derived::ApplySingleQubitMatrix(*qs_p, qs_p, objs[0], ctrls, m, dim);
}
template <typename derived_, typename calc_type_>
void CPUDensityMatrixPolicyBase<derived_, calc_type_>::ApplySXdag(qs_data_p_t* qs_p, const qbits_t& objs,
                                                                  const qbits_t& ctrls, index_t dim) {
    py_qs_data_t a = py_qs_data_t(0.5, -0.5);
    py_qs_data_t b = std::conj(a);
    matrix_t m{{a, b}, {b, a}};
    derived::ApplySingleQubitMatrix(*qs_p, qs_p, objs[0], ctrls, m, dim);
}

template <typename derived_, typename calc_type_>
void CPUDensityMatrixPolicyBase<derived_, calc_type_>::ApplyGP(qs_data_p_t* qs_p, const qbits_t& objs,
                                                               const qbits_t& ctrls, calc_type val, index_t dim,
                                                               bool diff) {
    auto c = std::exp(std::complex<calc_type>(0, -val));
    std::vector<std::vector<py_qs_data_t>> m = {{c, 0}, {0, c}};
    derived::ApplySingleQubitMatrix(*qs_p, qs_p, objs[0], ctrls, m, dim);
}

#ifdef __x86_64__
template struct CPUDensityMatrixPolicyBase<CPUDensityMatrixPolicyAvxFloat, float>;
template struct CPUDensityMatrixPolicyBase<CPUDensityMatrixPolicyAvxDouble, double>;
#elif defined(__amd64)
template struct CPUDensityMatrixPolicyBase<CPUDensityMatrixPolicyArmFloat, float>;
template struct CPUDensityMatrixPolicyBase<CPUDensityMatrixPolicyArmDouble, double>;
#endif
}  // namespace mindquantum::sim::densitymatrix::detail
