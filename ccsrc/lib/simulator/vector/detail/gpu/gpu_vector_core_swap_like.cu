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
#include <thrust/transform_reduce.h>

#include "config/openmp.h"
#include "simulator/utils.h"
#include "simulator/vector/detail/gpu_vector_double_policy.cuh"
#include "simulator/vector/detail/gpu_vector_float_policy.cuh"
#include "simulator/vector/detail/gpu_vector_policy.cuh"
#include "thrust/device_ptr.h"
#include "thrust/functional.h"
#include "thrust/inner_product.h"

namespace mindquantum::sim::vector::detail {

template <typename derived_, typename calc_type_>
void GPUVectorPolicyBase<derived_, calc_type_>::ApplySWAP(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls,
                                                          index_t dim) {
    auto& qs = *qs_p;
    if (qs == nullptr) {
        qs = derived::InitState(dim);
    }
    DoubleQubitGateMask mask(objs, ctrls);
    thrust::counting_iterator<index_t> l(0);
    auto obj_high_mask = mask.obj_high_mask;
    auto obj_rev_high_mask = mask.obj_rev_high_mask;
    auto obj_low_mask = mask.obj_low_mask;
    auto obj_rev_low_mask = mask.obj_rev_low_mask;
    auto obj_min_mask = mask.obj_min_mask;
    auto obj_max_mask = mask.obj_max_mask;
    auto ctrl_mask = mask.ctrl_mask;
    if (!mask.ctrl_mask) {
        thrust::for_each(l, l + dim / 4, [=] __device__(index_t l) {
            index_t i;
            SHIFT_BIT_TWO(obj_low_mask, obj_rev_low_mask, obj_high_mask, obj_rev_high_mask, l, i);
            auto tmp = qs[i + obj_min_mask];
            qs[i + obj_min_mask] = qs[i + obj_max_mask];
            qs[i + obj_max_mask] = tmp;
        });
    } else {
        thrust::for_each(l, l + dim / 4, [=] __device__(index_t l) {
            index_t i;
            SHIFT_BIT_TWO(obj_low_mask, obj_rev_low_mask, obj_high_mask, obj_rev_high_mask, l, i);
            if ((i & ctrl_mask) == ctrl_mask) {
                auto tmp = qs[i + obj_min_mask];
                qs[i + obj_min_mask] = qs[i + obj_max_mask];
                qs[i + obj_max_mask] = tmp;
            }
        });
    }
}

template <typename derived_, typename calc_type_>
void GPUVectorPolicyBase<derived_, calc_type_>::ApplyISWAP(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls,
                                                           bool daggered, index_t dim) {
    auto& qs = *qs_p;
    if (qs == nullptr) {
        qs = derived::InitState(dim);
    }
    DoubleQubitGateMask mask(objs, ctrls);
    thrust::counting_iterator<index_t> l(0);
    auto obj_high_mask = mask.obj_high_mask;
    auto obj_rev_high_mask = mask.obj_rev_high_mask;
    auto obj_low_mask = mask.obj_low_mask;
    auto obj_rev_low_mask = mask.obj_rev_low_mask;
    auto obj_min_mask = mask.obj_min_mask;
    auto obj_max_mask = mask.obj_max_mask;
    auto ctrl_mask = mask.ctrl_mask;
    calc_type frac = 1.0;
    if (daggered) {
        frac = -1.0;
    }
    if (!mask.ctrl_mask) {
        thrust::for_each(l, l + dim / 4, [=] __device__(index_t l) {
            index_t i;
            SHIFT_BIT_TWO(obj_low_mask, obj_rev_low_mask, obj_high_mask, obj_rev_high_mask, l, i);
            auto tmp = qs[i + obj_min_mask];
            qs[i + obj_min_mask] = frac * qs[i + obj_max_mask] * qs_data_t(0, 1);
            qs[i + obj_max_mask] = frac * tmp * qs_data_t(0, 1);
        });
    } else {
        thrust::for_each(l, l + dim / 4, [=] __device__(index_t l) {
            index_t i;
            SHIFT_BIT_TWO(obj_low_mask, obj_rev_low_mask, obj_high_mask, obj_rev_high_mask, l, i);
            if ((i & ctrl_mask) == ctrl_mask) {
                auto tmp = qs[i + obj_min_mask];
                qs[i + obj_min_mask] = frac * qs[i + obj_max_mask] * qs_data_t(0, 1);
                qs[i + obj_max_mask] = frac * tmp * qs_data_t(0, 1);
            }
        });
    }
}

template <typename derived_, typename calc_type_>
void GPUVectorPolicyBase<derived_, calc_type_>::ApplySWAPalpha(qs_data_p_t* qs_p, const qbits_t& objs,
                                                               const qbits_t& ctrls, calc_type val, index_t dim,
                                                               bool diff) {
    auto& qs = *qs_p;
    if (qs == nullptr) {
        qs = derived::InitState(dim);
    }
    DoubleQubitGateMask mask(objs, ctrls);
    thrust::counting_iterator<index_t> l(0);
    auto obj_high_mask = mask.obj_high_mask;
    auto obj_rev_high_mask = mask.obj_rev_high_mask;
    auto obj_low_mask = mask.obj_low_mask;
    auto obj_rev_low_mask = mask.obj_rev_low_mask;
    auto obj_min_mask = mask.obj_min_mask;
    auto obj_max_mask = mask.obj_max_mask;
    auto ctrl_mask = mask.ctrl_mask;
    auto obj_mask = mask.obj_mask;
    auto e = std::exp(IMAGE_I * static_cast<calc_type_>(M_PI) * val);
    auto a = (static_cast<calc_type_>(1) + e) / static_cast<calc_type_>(2);
    auto b = (static_cast<calc_type_>(1) - e) / static_cast<calc_type_>(2);
    if (diff) {
        if (!mask.ctrl_mask) {
            thrust::for_each(l, l + dim / 4, [=] __device__(index_t l) {
                index_t i;
                SHIFT_BIT_TWO(obj_low_mask, obj_rev_low_mask, obj_high_mask, obj_rev_high_mask, l, i);
                auto j = i + obj_min_mask;
                auto k = i + obj_max_mask;
                auto m = i + obj_mask;
                auto tmp_j = qs[j];
                auto tmp_k = qs[k];
                qs[j] = a * tmp_j + b * tmp_k;
                qs[k] = b * tmp_j + a * tmp_k;
            });
        } else {
            thrust::for_each(l, l + dim / 4, [=] __device__(index_t l) {
                index_t i;
                SHIFT_BIT_TWO(obj_low_mask, obj_rev_low_mask, obj_high_mask, obj_rev_high_mask, l, i);
                if ((i & ctrl_mask) == ctrl_mask) {
                    auto j = i + obj_min_mask;
                    auto k = i + obj_max_mask;
                    auto m = i + obj_mask;
                    auto tmp_j = qs[j];
                    auto tmp_k = qs[k];
                    qs[j] = a * tmp_j + b * tmp_k;
                    qs[k] = b * tmp_j + a * tmp_k;
                }
            });
        }
    } else {
        a = IMAGE_I * static_cast<calc_type_>(M_PI_2) * e;
        b = IMAGE_MI * static_cast<calc_type_>(M_PI_2) * e;
        if (!mask.ctrl_mask) {
            thrust::for_each(l, l + dim / 4, [=] __device__(index_t l) {
                index_t i;
                SHIFT_BIT_TWO(obj_low_mask, obj_rev_low_mask, obj_high_mask, obj_rev_high_mask, l, i);
                auto j = i + obj_min_mask;
                auto k = i + obj_max_mask;
                auto m = i + obj_mask;
                auto tmp_j = qs[j];
                auto tmp_k = qs[k];
                qs[i] = 0;
                qs[m] = 0;
                qs[j] = a * tmp_j + b * tmp_k;
                qs[k] = b * tmp_j + a * tmp_k;
            });
        } else {
            thrust::for_each(l, l + dim / 4, [=] __device__(index_t l) {
                index_t i;
                SHIFT_BIT_TWO(obj_low_mask, obj_rev_low_mask, obj_high_mask, obj_rev_high_mask, l, i);
                if ((i & ctrl_mask) == ctrl_mask) {
                    auto j = i + obj_min_mask;
                    auto k = i + obj_max_mask;
                    auto m = i + obj_mask;
                    auto tmp_j = qs[j];
                    auto tmp_k = qs[k];
                    qs[i] = 0;
                    qs[m] = 0;
                    qs[j] = a * tmp_j + b * tmp_k;
                    qs[k] = b * tmp_j + a * tmp_k;
                }
            });
            derived::SetToZeroExcept(&qs, ctrl_mask, dim);
        }
    }
}

template struct GPUVectorPolicyBase<GPUVectorPolicyFloat, float>;
template struct GPUVectorPolicyBase<GPUVectorPolicyDouble, double>;

}  // namespace mindquantum::sim::vector::detail
