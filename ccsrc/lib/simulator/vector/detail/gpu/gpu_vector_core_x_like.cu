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
#include "simulator/vector/detail/cuquantum_vector_double_policy.cuh"
#include "simulator/vector/detail/cuquantum_vector_float_policy.cuh"
#include "simulator/vector/detail/gpu_vector_double_policy.cuh"
#include "simulator/vector/detail/gpu_vector_float_policy.cuh"
#include "simulator/vector/detail/gpu_vector_policy.cuh"
#include "thrust/device_ptr.h"
#include "thrust/functional.h"
#include "thrust/inner_product.h"

namespace mindquantum::sim::vector::detail {
template <typename derived_, typename calc_type_>
void GPUVectorPolicyBase<derived_, calc_type_>::ApplyXLike(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls,
                                                           qs_data_t v1, qs_data_t v2, index_t dim) {
    auto& qs = *qs_p;
    if (qs == nullptr) {
        qs = derived::InitState(dim);
    }
    SingleQubitGateMask mask(objs, ctrls);
    thrust::counting_iterator<index_t> l(0);
    auto obj_high_mask = mask.obj_high_mask;
    auto obj_low_mask = mask.obj_low_mask;
    auto obj_mask = mask.obj_mask;
    auto ctrl_mask = mask.ctrl_mask;
    if (!mask.ctrl_mask) {
        thrust::for_each(l, l + dim / 2, [=] __device__(index_t l) {
            auto i = ((l & obj_high_mask) << 1) + (l & obj_low_mask);
            auto j = i + obj_mask;
            auto tmp = qs[i];
            qs[i] = qs[j] * v1;
            qs[j] = tmp * v2;
        });
    } else {
        thrust::for_each(l, l + dim / 2, [=] __device__(index_t l) {
            auto i = ((l & obj_high_mask) << 1) + (l & obj_low_mask);
            if ((i & ctrl_mask) == ctrl_mask) {
                auto j = i + obj_mask;
                auto tmp = qs[i];
                qs[i] = qs[j] * v1;
                qs[j] = tmp * v2;
            }
        });
    }
}

template <typename derived_, typename calc_type_>
void GPUVectorPolicyBase<derived_, calc_type_>::ApplyX(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls,
                                                       index_t dim) {
    derived::ApplyXLike(qs_p, objs, ctrls, 1, 1, dim);
}

template <typename derived_, typename calc_type_>
void GPUVectorPolicyBase<derived_, calc_type_>::ApplyY(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls,
                                                       index_t dim) {
    derived::ApplyXLike(qs_p, objs, ctrls, qs_data_t(0, -1), qs_data_t(0, 1), dim);
}

template struct GPUVectorPolicyBase<CuQuantumVectorPolicyFloat, float>;
template struct GPUVectorPolicyBase<CuQuantumVectorPolicyDouble, double>;
template struct GPUVectorPolicyBase<GPUVectorPolicyFloat, float>;
template struct GPUVectorPolicyBase<GPUVectorPolicyDouble, double>;

}  // namespace mindquantum::sim::vector::detail
