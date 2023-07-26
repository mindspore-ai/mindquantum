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
#include "config/openmp.hpp"

#include "core/utils.hpp"
#include "math/pr/parameter_resolver.hpp"
#ifdef __x86_64__
#    include "simulator/densitymatrix/detail/cpu_densitymatrix_avx_double_policy.hpp"
#    include "simulator/densitymatrix/detail/cpu_densitymatrix_avx_float_policy.hpp"
#elif defined(__amd64)
#    include "simulator/densitymatrix/detail/cpu_densitymatrix_arm_double_policy.hpp"
#    include "simulator/densitymatrix/detail/cpu_densitymatrix_arm_float_policy.hpp"
#endif
#include "simulator/densitymatrix/detail/cpu_densitymatrix_policy.hpp"

namespace mindquantum::sim::densitymatrix::detail {
template <typename derived_, typename calc_type_>
auto CPUDensityMatrixPolicyBase<derived_, calc_type_>::DiagonalConditionalCollect(const qs_data_p_t& qs, index_t mask,
                                                                                  index_t condi, index_t dim)
    -> calc_type {
    if (qs == nullptr) {
        if ((0 & mask) == condi) {
            return 1.0;
        }
        return 0.0;
    }
    // collect diagonal amplitude with index mask satisfied condition.
    calc_type res_real = 0;
    // clang-format off
    THRESHOLD_OMP(
        MQ_DO_PRAGMA(omp parallel
            for schedule(static) reduction(+: res_real)), dim, DimTh,
                     for (omp::idx_t i = 0; i <static_cast<omp::idx_t>(dim); i++) {
                         if ((i & mask) == condi) {
                             res_real += qs[IdxMap(i, i)].real();
                         }
                     });
    // clang-format on
    return res_real;
}

template <typename derived_, typename calc_type_>
template <class binary_op>
void CPUDensityMatrixPolicyBase<derived_, calc_type_>::ConditionalBinary(const qs_data_p_t& src, qs_data_p_t* des_p,
                                                                         index_t mask, index_t condi,
                                                                         qs_data_t succ_coeff, qs_data_t fail_coeff,
                                                                         index_t dim, const binary_op& op) {
    auto& des = *des_p;
    if (des == nullptr) {
        des = derived::InitState(dim);
    }
    if (src == nullptr) {
        if ((0 & mask) == condi) {
            des[0] = op(1.0, succ_coeff);
        } else {
            des[0] = op(1.0, fail_coeff);
        }
    } else {
        // if index mask satisfied condition, multiply by succe_coeff, otherwise multiply fail_coeff
        THRESHOLD_OMP_FOR(
            dim, DimTh, for (omp::idx_t i = 0; i < static_cast<omp::idx_t>(dim); i++) {
                auto _i_0 = IdxMap(i, 0);
                if ((i & mask) == condi) {
                    for (index_t j = 0; j <= i; j++) {
                        if ((j & mask) == condi) {
                            des[_i_0 + j] = op(src[_i_0 + j], succ_coeff);
                        } else {
                            des[_i_0 + j] = op(src[_i_0 + j], fail_coeff);
                        }
                    }
                } else {
                    for (index_t j = 0; j <= i; j++) {
                        des[_i_0 + j] = op(src[_i_0 + j], fail_coeff);
                    }
                }
            })
    }
}

template <typename derived_, typename calc_type_>
template <index_t mask, index_t condi, class binary_op>
void CPUDensityMatrixPolicyBase<derived_, calc_type_>::ConditionalBinary(const qs_data_p_t& src, qs_data_p_t* des_p,
                                                                         qs_data_t succ_coeff, qs_data_t fail_coeff,
                                                                         index_t dim, const binary_op& op) {
    auto& des = *des_p;
    if (des == nullptr) {
        des = derived::InitState(dim);
    }
    if (src == nullptr) {
        if ((0 & mask) == condi) {
            des[0] = op(1.0, succ_coeff);
        } else {
            des[0] = op(1.0, fail_coeff);
        }
    } else {
        // if index mask satisfied condition, multiply by succe_coeff, otherwise multiply fail_coeff
        THRESHOLD_OMP_FOR(
            dim, DimTh, for (omp::idx_t i = 0; i < static_cast<omp::idx_t>(dim); i++) {
                auto _i_0 = IdxMap(i, 0);
                if ((i & mask) == condi) {
                    for (index_t j = 0; j <= i; j++) {
                        if ((j & mask) == condi) {
                            des[_i_0 + j] = op(src[_i_0 + j], succ_coeff);
                        } else {
                            des[_i_0 + j] = op(src[_i_0 + j], fail_coeff);
                        }
                    }
                } else {
                    for (index_t j = 0; j <= i; j++) {
                        des[_i_0 + j] = op(src[_i_0 + j], fail_coeff);
                    }
                }
            })
    }
}

template <typename derived_, typename calc_type_>
void CPUDensityMatrixPolicyBase<derived_, calc_type_>::QSMulValue(const qs_data_p_t& src, qs_data_p_t* des_p,
                                                                  qs_data_t value, index_t dim) {
    derived::template ConditionalBinary<0, 0>(src, des_p, value, 0, dim, std::multiplies<qs_data_t>());
}
template <typename derived_, typename calc_type_>
void CPUDensityMatrixPolicyBase<derived_, calc_type_>::ConditionalAdd(const qs_data_p_t& src, qs_data_p_t* des_p,
                                                                      index_t mask, index_t condi, qs_data_t succ_coeff,
                                                                      qs_data_t fail_coeff, index_t dim) {
    derived::template ConditionalBinary(src, des_p, mask, condi, succ_coeff, fail_coeff, dim, std::plus<qs_data_t>());
}
template <typename derived_, typename calc_type_>
void CPUDensityMatrixPolicyBase<derived_, calc_type_>::ConditionalMinus(const qs_data_p_t& src, qs_data_p_t* des_p,
                                                                        index_t mask, index_t condi,
                                                                        qs_data_t succ_coeff, qs_data_t fail_coeff,
                                                                        index_t dim) {
    derived::template ConditionalBinary(src, des_p, mask, condi, succ_coeff, fail_coeff, dim, std::minus<qs_data_t>());
}
template <typename derived_, typename calc_type_>
void CPUDensityMatrixPolicyBase<derived_, calc_type_>::ConditionalMul(const qs_data_p_t& src, qs_data_p_t* des_p,
                                                                      index_t mask, index_t condi, qs_data_t succ_coeff,
                                                                      qs_data_t fail_coeff, index_t dim) {
    derived::template ConditionalBinary(src, des_p, mask, condi, succ_coeff, fail_coeff, dim,
                                        std::multiplies<qs_data_t>());
}
template <typename derived_, typename calc_type_>
void CPUDensityMatrixPolicyBase<derived_, calc_type_>::ConditionalDiv(const qs_data_p_t& src, qs_data_p_t* des_p,
                                                                      index_t mask, index_t condi, qs_data_t succ_coeff,
                                                                      qs_data_t fail_coeff, index_t dim) {
    derived::template ConditionalBinary(src, des_p, mask, condi, succ_coeff, fail_coeff, dim,
                                        std::divides<qs_data_t>());
}

#ifdef __x86_64__
template struct CPUDensityMatrixPolicyBase<CPUDensityMatrixPolicyAvxFloat, float>;
template struct CPUDensityMatrixPolicyBase<CPUDensityMatrixPolicyAvxDouble, double>;
#elif defined(__amd64)
template struct CPUDensityMatrixPolicyBase<CPUDensityMatrixPolicyArmFloat, float>;
template struct CPUDensityMatrixPolicyBase<CPUDensityMatrixPolicyArmDouble, double>;
#endif
}  // namespace mindquantum::sim::densitymatrix::detail
