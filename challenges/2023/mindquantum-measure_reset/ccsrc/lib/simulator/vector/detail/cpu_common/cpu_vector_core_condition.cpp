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
#include <functional>

#include "config/openmp.h"
#include "math/pr/parameter_resolver.h"

#ifdef __x86_64__
#    include "simulator/vector/detail/cpu_vector_avx_double_policy.h"
#    include "simulator/vector/detail/cpu_vector_avx_float_policy.h"
#elif defined(__amd64)
#    include "simulator/vector/detail/cpu_vector_arm_double_policy.h"
#    include "simulator/vector/detail/cpu_vector_arm_float_policy.h"
#endif
#include "simulator/vector/detail/cpu_vector_policy.h"
namespace mindquantum::sim::vector::detail {
template <typename derived_, typename calc_type_>
template <class binary_op>
void CPUVectorPolicyBase<derived_, calc_type_>::ConditionalBinary(const qs_data_p_t& src, qs_data_p_t* des_p, index_t mask, index_t condi, qs_data_t succ_coeff, qs_data_t fail_coeff, index_t dim, const binary_op& op) {
    // if index mask satisfied condition, multiply by success_coeff, otherwise multiply fail_coeff
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
        THRESHOLD_OMP_FOR(
            dim, DimTh, for (omp::idx_t i = 0; i < static_cast<omp::idx_t>(dim); i++) {
                if ((i & mask) == condi) {
                    des[i] = op(src[i], succ_coeff);
                } else {
                    des[i] = op(src[i], fail_coeff);
                }
            })
    }
}
//new ConditionalBinary for reset
template <typename derived_, typename calc_type_>
template <class binary_op>
void CPUVectorPolicyBase<derived_, calc_type_>::ConditionalBinary(const qs_data_p_t& src, qs_data_p_t* des_p, index_t mask, index_t condi, qs_data_t succ_coeff, qs_data_t fail_coeff, index_t dim, const binary_op& op, bool reset) {
    // if index mask satisfied condition, multiply by success_coeff, otherwise multiply fail_coeff, the function is only suitable for measuregate. reset the qubit to |0> is not a unitary operation, so the fail_coeff must to be zero.
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
    } 
    else 
    {
        if (reset && condi != 0)
            {
            THRESHOLD_OMP_FOR(
            dim, DimTh, for (omp::idx_t i = 0; i < static_cast<omp::idx_t>(dim); i++) {
                if ((i & mask) == condi) {
                    des[i] = op(src[i], fail_coeff);
                } else {
                    des[i] = op(src[i + mask], succ_coeff);
                }
            })
            }
            else
            {THRESHOLD_OMP_FOR(
                dim, DimTh, for (omp::idx_t i = 0; i < static_cast<omp::idx_t>(dim); i++) {
                    if ((i & mask) == condi) {
                        des[i] = op(src[i], succ_coeff);
                    } else {
                        des[i] = op(src[i], fail_coeff);
                    }
                })
            }
    }
}

template <typename derived_, typename calc_type_>
template <index_t mask, index_t condi, class binary_op>
void CPUVectorPolicyBase<derived_, calc_type_>::ConditionalBinary(const qs_data_p_t& src, qs_data_p_t* des_p, qs_data_t succ_coeff, qs_data_t fail_coeff, index_t dim, const binary_op& op) {
    // if index mask satisfied condition, multiply by success_coeff, otherwise multiply fail_coeff
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
        THRESHOLD_OMP_FOR(
            dim, DimTh, for (omp::idx_t i = 0; i < static_cast<omp::idx_t>(dim); i++) {
                if ((i & mask) == condi) {
                    des[i] = op(src[i], succ_coeff);
                } else {
                    des[i] = op(src[i], fail_coeff);
                }
            })
    }
}


template <typename derived_, typename calc_type_>
void CPUVectorPolicyBase<derived_, calc_type_>::QSMulValue(const qs_data_p_t& src, qs_data_p_t* des_p, qs_data_t value,
                                                           index_t dim) {
    derived::template ConditionalBinary<0, 0>(src, des_p, value, 0, dim, std::multiplies<qs_data_t>());
}

template <typename derived_, typename calc_type_>
void CPUVectorPolicyBase<derived_, calc_type_>::ConditionalAdd(const qs_data_p_t& src, qs_data_p_t* des_p, index_t mask,
                                                               index_t condi, qs_data_t succ_coeff,
                                                               qs_data_t fail_coeff, index_t dim) {
    derived::template ConditionalBinary(src, des_p, mask, condi, succ_coeff, fail_coeff, dim, std::plus<qs_data_t>());
}
template <typename derived_, typename calc_type_>
void CPUVectorPolicyBase<derived_, calc_type_>::ConditionalMinus(const qs_data_p_t& src, qs_data_p_t* des_p,
                                                                 index_t mask, index_t condi, qs_data_t succ_coeff,
                                                                 qs_data_t fail_coeff, index_t dim) {
    derived::template ConditionalBinary(src, des_p, mask, condi, succ_coeff, fail_coeff, dim, std::minus<qs_data_t>());
}

template <typename derived_, typename calc_type_>
void CPUVectorPolicyBase<derived_, calc_type_>::ConditionalMul(const qs_data_p_t& src, qs_data_p_t* des_p, index_t mask,
                                                               index_t condi, qs_data_t succ_coeff,
                                                               qs_data_t fail_coeff, index_t dim) {
    derived::template ConditionalBinary(src, des_p, mask, condi, succ_coeff, fail_coeff, dim,
                                        std::multiplies<qs_data_t>());
}

template <typename derived_, typename calc_type_>
void CPUVectorPolicyBase<derived_, calc_type_>::ConditionalMul(const qs_data_p_t& src, qs_data_p_t* des_p, index_t mask, index_t condi, qs_data_t succ_coeff, qs_data_t fail_coeff, index_t dim, bool reset) 
{
    derived::template ConditionalBinary(src, des_p, mask, condi, succ_coeff, fail_coeff, dim, std::multiplies<qs_data_t>(), reset);
}


template <typename derived_, typename calc_type_>
void CPUVectorPolicyBase<derived_, calc_type_>::ConditionalDiv(const qs_data_p_t& src, qs_data_p_t* des_p, index_t mask,
                                                               index_t condi, qs_data_t succ_coeff,
                                                               qs_data_t fail_coeff, index_t dim) {
    derived::template ConditionalBinary(src, des_p, mask, condi, succ_coeff, fail_coeff, dim,
                                        std::divides<qs_data_t>());
}

template <typename derived_, typename calc_type_>
auto CPUVectorPolicyBase<derived_, calc_type_>::ConditionalCollect(const qs_data_p_t& qs, index_t mask, index_t condi,
                                                                   bool abs, index_t dim) -> qs_data_t {
    // collect amplitude with index mask satisfied condition.
    calc_type res_real = 0, res_imag = 0;
    if (qs == nullptr) {
        if (abs) {
            if ((0 & mask) == condi) {
                res_real += qs[0].real() * qs[0].real() + qs[0].imag() * qs[0].imag();
            }
        } else {
            if ((0 & mask) == condi) {
                res_real += qs[0].real();
                res_imag += qs[0].imag();
            }
        }
        return qs_data_t(res_real, res_imag);
    }
    if (abs) {
        THRESHOLD_OMP(
            MQ_DO_PRAGMA(omp parallel for schedule(static) reduction(+: res_real)), dim, DimTh,
                         for (omp::idx_t i = 0; i < static_cast<omp::idx_t>(dim); i++) {
                             if ((i & mask) == condi) {
                                 res_real += qs[i].real() * qs[i].real() + qs[i].imag() * qs[i].imag();
                             }
                         });
    } else {
        THRESHOLD_OMP(
            MQ_DO_PRAGMA(omp parallel for schedule(static) reduction(+: res_real, res_imag)), dim, DimTh,
                         for (omp::idx_t i = 0; i < static_cast<omp::idx_t>(dim); i++) {
                             if ((i & mask) == condi) {
                                 res_real += qs[i].real();
                                 res_imag += qs[i].imag();
                             }
                         });
    }
    return qs_data_t(res_real, res_imag);
}

#ifdef __x86_64__
template struct CPUVectorPolicyBase<CPUVectorPolicyAvxFloat, float>;
template struct CPUVectorPolicyBase<CPUVectorPolicyAvxDouble, double>;
#elif defined(__amd64)
template struct CPUVectorPolicyBase<CPUVectorPolicyArmFloat, float>;
template struct CPUVectorPolicyBase<CPUVectorPolicyArmDouble, double>;
#endif
}  // namespace mindquantum::sim::vector::detail
