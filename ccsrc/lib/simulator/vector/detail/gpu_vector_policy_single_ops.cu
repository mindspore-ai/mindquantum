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
#include <cmath>

#include <cassert>
#include <complex>
#include <stdexcept>

#include <thrust/transform_reduce.h>

#include "simulator/types.hpp"
#include "simulator/utils.hpp"
#include "simulator/vector/detail/gpu_vector_policy.cuh"
#include "thrust/device_ptr.h"
#include "thrust/functional.h"
#include "thrust/inner_product.h"

namespace mindquantum::sim::vector::detail {
void GPUVectorPolicyBase::ApplyTwoQubitsMatrix(qs_data_p_t src, qs_data_p_t des, const qbits_t& objs,
                                               const qbits_t& ctrls, const std::vector<std::vector<py_qs_data_t>>& m,
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
    auto obj_min_mask = mask.obj_min_mask;
    auto obj_max_mask = mask.obj_max_mask;
    auto ctrl_mask = mask.ctrl_mask;
    auto obj_mask = mask.obj_mask;
    thrust::counting_iterator<size_t> l(0);
    if (!mask.ctrl_mask) {
        thrust::for_each(l, l + (dim / 4), [=] __device__(size_t l) {
            index_t i;
            SHIFT_BIT_TWO(obj_low_mask, obj_rev_low_mask, obj_high_mask, obj_rev_high_mask, l, i);
            auto j = i + obj_min_mask;
            auto k = i + obj_max_mask;
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
                auto j = i + obj_min_mask;
                auto k = i + obj_max_mask;
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
void GPUVectorPolicyBase::ApplySingleQubitMatrix(qs_data_p_t src, qs_data_p_t des, qbit_t obj_qubit,
                                                 const qbits_t& ctrls, const std::vector<std::vector<py_qs_data_t>>& m,
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


void GPUVectorPolicyBase::ApplyMatrixGate(qs_data_p_t src, qs_data_p_t des, const qbits_t& objs, const qbits_t& ctrls,
                                          const std::vector<std::vector<py_qs_data_t>>& m, index_t dim) {
    if (objs.size() == 1) {
        ApplySingleQubitMatrix(src, des, objs[0], ctrls, m, dim);
    } else if (objs.size() == 2) {
        ApplyTwoQubitsMatrix(src, des, objs, ctrls, m, dim);
    } else {
        throw std::runtime_error("Can not custom " + std::to_string(objs.size()) + " qubits gate for gpu backend.");
    }
}

void GPUVectorPolicyBase::ApplyH(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls, index_t dim) {
    std::vector<std::vector<py_qs_data_t>> m{{M_SQRT1_2, M_SQRT1_2}, {M_SQRT1_2, -M_SQRT1_2}};
    ApplySingleQubitMatrix(qs, qs, objs[0], ctrls, m, dim);
}

void GPUVectorPolicyBase::ApplyRX(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls, calc_type val, index_t dim,
                                  bool diff) {
    SingleQubitGateMask mask(objs, ctrls);
    auto a = std::cos(val / 2);
    auto b = -std::sin(val / 2);
    if (diff) {
        a = -0.5 * std::sin(val / 2);
        b = -0.5 * std::cos(val / 2);
    }
    std::vector<std::vector<py_qs_data_t>> m{{{a, 0}, {0, b}}, {{0, b}, {a, 0}}};
    ApplySingleQubitMatrix(qs, qs, objs[0], ctrls, m, dim);
    if (diff && mask.ctrl_mask) {
        SetToZeroExcept(qs, mask.ctrl_mask, dim);
    }
}

void GPUVectorPolicyBase::ApplyRY(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls, calc_type val, index_t dim,
                                  bool diff) {
    SingleQubitGateMask mask(objs, ctrls);
    auto a = std::cos(val / 2);
    auto b = std::sin(val / 2);
    if (diff) {
        a = -0.5 * std::sin(val / 2);
        b = 0.5 * std::cos(val / 2);
    }
    std::vector<std::vector<py_qs_data_t>> m{{{a, 0}, {-b, 0}}, {{b, 0}, {a, 0}}};
    ApplySingleQubitMatrix(qs, qs, objs[0], ctrls, m, dim);
    if (diff && mask.ctrl_mask) {
        SetToZeroExcept(qs, mask.ctrl_mask, dim);
    }
}

void GPUVectorPolicyBase::ApplyRZ(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls, calc_type val, index_t dim,
                                  bool diff) {
    SingleQubitGateMask mask(objs, ctrls);
    auto a = std::cos(val / 2);
    auto b = std::sin(val / 2);
    if (diff) {
        a = -0.5 * std::sin(val / 2);
        b = 0.5 * std::cos(val / 2);
    }
    std::vector<std::vector<py_qs_data_t>> m{{{a, -b}, {0, 0}}, {{0, 0}, {a, b}}};
    ApplySingleQubitMatrix(qs, qs, objs[0], ctrls, m, dim);
    if (diff && mask.ctrl_mask) {
        SetToZeroExcept(qs, mask.ctrl_mask, dim);
    }
}

}  // namespace mindquantum::sim::vector::detail
