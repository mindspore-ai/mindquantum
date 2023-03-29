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
#include <thrust/transform_reduce.h>

#include "config/openmp.hpp"

#include "simulator/utils.hpp"
#include "simulator/vector/detail/gpu_vector_double_policy.cuh"
#include "simulator/vector/detail/gpu_vector_float_policy.cuh"
#include "simulator/vector/detail/gpu_vector_policy.cuh"
#include "thrust/device_ptr.h"
#include "thrust/device_vector.h"
#include "thrust/functional.h"
#include "thrust/inner_product.h"
namespace mindquantum::sim::vector::detail {
template <typename derived_, typename calc_type_>
void GPUVectorPolicyBase<derived_, calc_type_>::ApplyNQubitsMatrix(qs_data_p_t src, qs_data_p_t des,
                                                                   const qbits_t& objs, const qbits_t& ctrls,
                                                                   const std::vector<std::vector<py_qs_data_t>>& gate,
                                                                   index_t dim) {
    size_t n_qubit = objs.size();
    size_t m_dim = (1UL << n_qubit);
    size_t ctrl_mask = 0;
    for (auto& i : ctrls) {
        ctrl_mask |= 1UL << i;
    }
    std::vector<size_t> obj_masks{};
    for (size_t i = 0; i < m_dim; i++) {
        size_t n = 0;
        size_t mask_j = 0;
        for (size_t j = i; j != 0; j >>= 1) {
            if (j & 1) {
                mask_j += 1UL << objs[n];
            }
            n += 1;
        }
        obj_masks.push_back(mask_j);
    }
    auto obj_mask = obj_masks.back();
    thrust::device_vector<size_t> device_obj_masks = obj_masks;
    auto device_obj_masks_ptr = thrust::raw_pointer_cast(device_obj_masks.data());
    thrust::device_vector<qs_data_t> device_gate;
    for (auto& m : gate) {
        for (auto v : m) {
            device_gate.push_back(v);
        }
    }
    auto device_gate_ptr = thrust::raw_pointer_cast(device_gate.data());

    thrust::counting_iterator<size_t> l(0);
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, sizeof(qs_data_t) * m_dim);
    thrust::for_each(l, l + dim, [=] __device__(size_t l) {
        if (((l & ctrl_mask) == ctrl_mask) && ((l & obj_mask) == 0)) {
            qs_data_p_t tmp_des = (qs_data_p_t) malloc(sizeof(qs_data_t) * m_dim);
            for (size_t i = 0; i < m_dim; i++) {
                qs_data_t tmp = 0;
                for (size_t j = 0; j < m_dim; j++) {
                    tmp += device_gate_ptr[i * m_dim + j] * src[device_obj_masks_ptr[j] | l];
                }
                tmp_des[i] = tmp;
            }
            for (size_t i = 0; i < m_dim; i++) {
                des[device_obj_masks_ptr[i] | l] = tmp_des[i];
            }
            free(tmp_des);
        }
    });
}

template <typename derived_, typename calc_type_>
void GPUVectorPolicyBase<derived_, calc_type_>::ApplyTwoQubitsMatrix(qs_data_p_t src, qs_data_p_t des,
                                                                     const qbits_t& objs, const qbits_t& ctrls,
                                                                     const std::vector<std::vector<py_qs_data_t>>& m,
                                                                     index_t dim) {
    DoubleQubitGateMask mask(objs, ctrls);
    qs_data_t m00 = m[0][0];
    qs_data_t m01 = m[0][1];
    qs_data_t m02 = m[0][2];
    qs_data_t m03 = m[0][3];
    qs_data_t m10 = m[1][0];
    qs_data_t m11 = m[1][1];
    qs_data_t m12 = m[1][2];
    qs_data_t m13 = m[1][3];
    qs_data_t m20 = m[2][0];
    qs_data_t m21 = m[2][1];
    qs_data_t m22 = m[2][2];
    qs_data_t m23 = m[2][3];
    qs_data_t m30 = m[3][0];
    qs_data_t m31 = m[3][1];
    qs_data_t m32 = m[3][2];
    qs_data_t m33 = m[3][3];
    auto obj_high_mask = mask.obj_high_mask;
    auto obj_rev_high_mask = mask.obj_rev_high_mask;
    auto obj_low_mask = mask.obj_low_mask;
    auto obj_rev_low_mask = mask.obj_rev_low_mask;
    size_t mask1 = (1UL << objs[0]);
    size_t mask2 = (1UL << objs[1]);
    auto ctrl_mask = mask.ctrl_mask;
    auto obj_mask = mask.obj_mask;
    thrust::counting_iterator<size_t> l(0);
    if (!mask.ctrl_mask) {
        thrust::for_each(l, l + (dim / 4), [=] __device__(size_t l) {
            index_t i;
            SHIFT_BIT_TWO(obj_low_mask, obj_rev_low_mask, obj_high_mask, obj_rev_high_mask, l, i);
            auto j = i + mask1;
            auto k = i + mask2;
            auto m = i + obj_mask;
            auto v00 = m00 * src[i] + m01 * src[j] + m02 * src[k] + m03 * src[m];
            auto v01 = m10 * src[i] + m11 * src[j] + m12 * src[k] + m13 * src[m];
            auto v10 = m20 * src[i] + m21 * src[j] + m22 * src[k] + m23 * src[m];
            auto v11 = m30 * src[i] + m31 * src[j] + m32 * src[k] + m33 * src[m];
            src[i] = v00;
            src[j] = v01;
            src[k] = v10;
            src[m] = v11;
        });
    } else {
        thrust::for_each(l, l + (dim / 4), [=] __device__(size_t l) {
            index_t i;
            SHIFT_BIT_TWO(obj_low_mask, obj_rev_low_mask, obj_high_mask, obj_rev_high_mask, l, i);
            if ((i & ctrl_mask) == ctrl_mask) {
                auto j = i + mask1;
                auto k = i + mask2;
                auto m = i + obj_mask;
                auto v00 = m00 * src[i] + m01 * src[j] + m02 * src[k] + m03 * src[m];
                auto v01 = m10 * src[i] + m11 * src[j] + m12 * src[k] + m13 * src[m];
                auto v10 = m20 * src[i] + m21 * src[j] + m22 * src[k] + m23 * src[m];
                auto v11 = m30 * src[i] + m31 * src[j] + m32 * src[k] + m33 * src[m];
                src[i] = v00;
                src[j] = v01;
                src[k] = v10;
                src[m] = v11;
            }
        });
    }
}
template <typename derived_, typename calc_type_>
void GPUVectorPolicyBase<derived_, calc_type_>::ApplySingleQubitMatrix(qs_data_p_t src, qs_data_p_t des,
                                                                       qbit_t obj_qubit, const qbits_t& ctrls,
                                                                       const std::vector<std::vector<py_qs_data_t>>& m,
                                                                       index_t dim) {
    SingleQubitGateMask mask({obj_qubit}, ctrls);
    qs_data_t m00 = m[0][0];
    qs_data_t m01 = m[0][1];
    qs_data_t m10 = m[1][0];
    qs_data_t m11 = m[1][1];
    auto obj_high_mask = mask.obj_high_mask;
    auto obj_low_mask = mask.obj_low_mask;
    auto obj_mask = mask.obj_mask;
    auto ctrl_mask = mask.ctrl_mask;
    thrust::counting_iterator<size_t> l(0);
    if (!mask.ctrl_mask) {
        thrust::for_each(l, l + (dim / 2), [=] __device__(size_t l) {
            auto i = ((l & obj_high_mask) << 1) + (l & obj_low_mask);
            auto j = i + obj_mask;
            auto a = m00 * src[i] + m01 * src[j];
            auto b = m10 * src[i] + m11 * src[j];
            des[i] = a;
            des[j] = b;
        });
    } else {
        thrust::for_each(l, l + (dim / 2), [=] __device__(size_t l) {
            auto i = ((l & obj_high_mask) << 1) + (l & obj_low_mask);
            if ((i & ctrl_mask) == ctrl_mask) {
                auto j = i + obj_mask;
                auto a = m00 * src[i] + m01 * src[j];
                auto b = m10 * src[i] + m11 * src[j];
                des[i] = a;
                des[j] = b;
            }
        });
    }
}

template <typename derived_, typename calc_type_>
void GPUVectorPolicyBase<derived_, calc_type_>::ApplyMatrixGate(qs_data_p_t src, qs_data_p_t des, const qbits_t& objs,
                                                                const qbits_t& ctrls,
                                                                const std::vector<std::vector<py_qs_data_t>>& m,
                                                                index_t dim) {
    if (objs.size() == 1) {
        derived::ApplySingleQubitMatrix(src, des, objs[0], ctrls, m, dim);
    } else if (objs.size() == 2) {
        derived::ApplyTwoQubitsMatrix(src, des, objs, ctrls, m, dim);
    } else {
        derived::ApplyNQubitsMatrix(src, des, objs, ctrls, m, dim);
    }
}

template struct GPUVectorPolicyBase<GPUVectorPolicyFloat, float>;
template struct GPUVectorPolicyBase<GPUVectorPolicyDouble, double>;

}  // namespace mindquantum::sim::vector::detail
