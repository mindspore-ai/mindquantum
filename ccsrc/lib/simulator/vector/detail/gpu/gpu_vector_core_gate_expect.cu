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
#include "thrust/device_vector.h"
#include "thrust/functional.h"
#include "thrust/inner_product.h"

namespace mindquantum::sim::vector::detail {
template <typename derived_, typename calc_type_>
auto GPUVectorPolicyBase<derived_, calc_type_>::ExpectDiffRPS(const qs_data_p_t& bra_out, const qs_data_p_t& ket_out,
                                                              const PauliMask& mask, Index ctrl_mask, calc_type val,
                                                              index_t dim) -> qs_data_t {
    auto bra = bra_out;
    auto ket = ket_out;
    bool will_free_bra = false, will_free_ket = false;
    if (bra == nullptr) {
        bra = derived::InitState(dim);
        will_free_bra = true;
    }
    if (ket == nullptr) {
        ket = derived::InitState(dim);
        will_free_ket = true;
    }
    qs_data_t res = 0.0;
    auto a = -std::sin(val / 2) / 2;
    auto b = std::cos(val / 2) / 2;
    thrust::counting_iterator<size_t> i(0);
    auto mask_f = mask.mask_x | mask.mask_y;
    auto mask_z = mask.mask_z;
    auto mask_y = mask.mask_y;
    auto num_y = mask.num_y;
    auto m_i = qs_data_t(0, 1);
    if (!ctrl_mask) {
        res = thrust::transform_reduce(
            i, i + dim,
            MQ_PROCLAIM_RETURN_TYPE(qs_data_t, [=] __device__(size_t i) {
                auto j = i ^ mask_f;
                if (i <= j) {
                    auto axis2power = __popcll(i & mask_z);
                    auto axis3power = __popcll(i & mask_y);
                    auto idx = (num_y + 2 * axis3power + 2 * axis2power) & 3;
                    auto c = qs_data_t(1, 0);
                    if (idx == 1) {
                        c = qs_data_t(0, 1);
                    } else if (idx == 2) {
                        c = qs_data_t(-1, 0);
                    } else if (idx == 3) {
                        c = qs_data_t(0, -1);
                    }
                    if (i == j) {
                        auto this_res = thrust::conj(bra[i]) * ket[i] * (a - m_i * b * c);
                        return this_res;
                    }
                    auto this_res = thrust::conj(bra[i]) * (ket[i] * a - m_i * ket[j] * b / c);
                    this_res += thrust::conj(bra[j]) * (ket[j] * a - m_i * ket[i] * b * c);
                    return this_res;
                } else {
                    return qs_data_t(0, 0);
                }
            }),
            qs_data_t(0, 0), thrust::plus<qs_data_t>());
    } else {
        res = thrust::transform_reduce(
            i, i + dim,
            MQ_PROCLAIM_RETURN_TYPE(qs_data_t, [=] __device__(size_t i) {
                if ((i & ctrl_mask) == ctrl_mask) {
                    auto j = i ^ mask_f;
                    if (i <= j) {
                        auto axis2power = __popcll(i & mask_z);
                        auto axis3power = __popcll(i & mask_y);
                        auto idx = (num_y + 2 * axis3power + 2 * axis2power) & 3;
                        auto c = qs_data_t(1, 0);
                        if (idx == 1) {
                            c = qs_data_t(0, 1);
                        } else if (idx == 2) {
                            c = qs_data_t(-1, 0);
                        } else if (idx == 3) {
                            c = qs_data_t(0, -1);
                        }
                        if (i == j) {
                            auto this_res = thrust::conj(bra[i]) * ket[i] * (a - m_i * b * c);
                            return this_res;
                        }
                        auto this_res = thrust::conj(bra[i]) * (ket[i] * a - m_i * ket[j] * b / c);
                        this_res += thrust::conj(bra[j]) * (ket[j] * a - m_i * ket[i] * b * c);
                        return this_res;
                    } else {
                        return qs_data_t(0, 0);
                    }
                } else {
                    return qs_data_t(0, 0);
                }
            }),
            qs_data_t(0, 0), thrust::plus<qs_data_t>());
    }
    if (will_free_bra) {
        derived::FreeState(&bra);
    }
    if (will_free_ket) {
        derived::FreeState(&ket);
    }
    return res;
}

template <typename derived_, typename calc_type_>
auto GPUVectorPolicyBase<derived_, calc_type_>::ExpectDiffNQubitsMatrix(const qs_data_p_t& bra_out,
                                                                        const qs_data_p_t& ket_out, const qbits_t& objs,
                                                                        const qbits_t& ctrls,
                                                                        const std::vector<py_qs_datas_t>& gate,
                                                                        index_t dim) -> qs_data_t {
    auto bra = bra_out;
    auto ket = ket_out;
    bool will_free_bra = false, will_free_ket = false;
    if (bra == nullptr) {
        bra = derived::InitState(dim);
        will_free_bra = true;
    }
    if (ket == nullptr) {
        ket = derived::InitState(dim);
        will_free_ket = true;
    }
    size_t n_qubit = objs.size();
    size_t m_dim = (static_cast<uint64_t>(1) << n_qubit);
    size_t ctrl_mask = 0;
    for (auto& i : ctrls) {
        ctrl_mask |= static_cast<uint64_t>(1) << i;
    }
    std::vector<size_t> obj_masks{};
    for (size_t i = 0; i < m_dim; i++) {
        size_t n = 0;
        size_t mask_j = 0;
        for (size_t j = i; j != 0; j >>= 1) {
            if (j & 1) {
                mask_j += static_cast<uint64_t>(1) << objs[n];
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
    auto res = thrust::transform_reduce(
        l, l + dim,
        [=] __host__ __device__(size_t l) {
            qs_data_t res = 0;
            if (((l & ctrl_mask) == ctrl_mask) && ((l & obj_mask) == 0)) {
                for (size_t i = 0; i < m_dim; i++) {
                    qs_data_t tmp = 0;
                    for (size_t j = 0; j < m_dim; j++) {
                        tmp += device_gate_ptr[i * m_dim + j] * ket[device_obj_masks_ptr[j] | l];
                    }
                    res += thrust::conj(bra[device_obj_masks_ptr[i] | l]) * tmp;
                }
            }
            return res;
        },
        qs_data_t(0, 0), thrust::plus<qs_data_t>());
    if (will_free_bra) {
        derived::FreeState(&bra);
    }
    if (will_free_ket) {
        derived::FreeState(&ket);
    }
    return res;
}
template <typename derived_, typename calc_type_>
auto GPUVectorPolicyBase<derived_, calc_type_>::ExpectDiffTwoQubitsMatrix(const qs_data_p_t& bra_out,
                                                                          const qs_data_p_t& ket_out,
                                                                          const qbits_t& objs, const qbits_t& ctrls,
                                                                          const std::vector<py_qs_datas_t>& m,
                                                                          index_t dim) -> qs_data_t {
    auto bra = bra_out;
    auto ket = ket_out;
    bool will_free_bra = false, will_free_ket = false;
    if (bra == nullptr) {
        bra = derived::InitState(dim);
        will_free_bra = true;
    }
    if (ket == nullptr) {
        ket = derived::InitState(dim);
        will_free_ket = true;
    }
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
    thrust::counting_iterator<size_t> l(0);
    auto obj_high_mask = mask.obj_high_mask;
    auto obj_rev_high_mask = mask.obj_rev_high_mask;
    auto obj_low_mask = mask.obj_low_mask;
    auto obj_rev_low_mask = mask.obj_rev_low_mask;
    auto obj_mask = mask.obj_mask;
    auto obj_min_mask = mask.obj_min_mask;
    auto obj_max_mask = mask.obj_max_mask;
    auto ctrl_mask = mask.ctrl_mask;
    qs_data_t res = 0.0;
    if (!mask.ctrl_mask) {
        res = thrust::transform_reduce(
            l, l + dim / 4,
            [=] __host__ __device__(size_t l) {
                index_t i;
                SHIFT_BIT_TWO(obj_low_mask, obj_rev_low_mask, obj_high_mask, obj_rev_high_mask, l, i);
                auto m = i + obj_mask;
                auto j = i + obj_min_mask;
                auto k = i + obj_max_mask;
                auto v00 = m00 * ket[i] + m01 * ket[j] + m02 * ket[k] + m03 * ket[m];
                auto v01 = m10 * ket[i] + m11 * ket[j] + m12 * ket[k] + m13 * ket[m];
                auto v10 = m20 * ket[i] + m21 * ket[j] + m22 * ket[k] + m23 * ket[m];
                auto v11 = m30 * ket[i] + m31 * ket[j] + m32 * ket[k] + m33 * ket[m];
                auto this_res = thrust::conj(bra[i]) * v00;
                this_res += thrust::conj(bra[j]) * v01;
                this_res += thrust::conj(bra[k]) * v10;
                this_res += thrust::conj(bra[m]) * v11;
                return this_res;
            },
            qs_data_t(0, 0), thrust::plus<qs_data_t>());
    } else {
        res = thrust::transform_reduce(
            l, l + dim / 4,
            [=] __host__ __device__(size_t l) {
                index_t i;
                SHIFT_BIT_TWO(obj_low_mask, obj_rev_low_mask, obj_high_mask, obj_rev_high_mask, l, i);
                if ((i & ctrl_mask) != ctrl_mask) {
                    return qs_data_t(0, 0);
                }
                auto m = i + obj_mask;
                auto j = i + obj_min_mask;
                auto k = i + obj_max_mask;
                auto v00 = m00 * ket[i] + m01 * ket[j] + m02 * ket[k] + m03 * ket[m];
                auto v01 = m10 * ket[i] + m11 * ket[j] + m12 * ket[k] + m13 * ket[m];
                auto v10 = m20 * ket[i] + m21 * ket[j] + m22 * ket[k] + m23 * ket[m];
                auto v11 = m30 * ket[i] + m31 * ket[j] + m32 * ket[k] + m33 * ket[m];
                auto this_res = thrust::conj(bra[i]) * v00;
                this_res += thrust::conj(bra[j]) * v01;
                this_res += thrust::conj(bra[k]) * v10;
                this_res += thrust::conj(bra[m]) * v11;
                return this_res;
            },
            qs_data_t(0, 0), thrust::plus<qs_data_t>());
    }
    if (will_free_bra) {
        derived::FreeState(&bra);
    }
    if (will_free_ket) {
        derived::FreeState(&ket);
    }
    return res;
}

template <typename derived_, typename calc_type_>
auto GPUVectorPolicyBase<derived_, calc_type_>::ExpectDiffSingleQubitMatrix(const qs_data_p_t& bra_out,
                                                                            const qs_data_p_t& ket_out,
                                                                            const qbits_t& objs, const qbits_t& ctrls,
                                                                            const std::vector<py_qs_datas_t>& m,
                                                                            index_t dim) -> qs_data_t {
    auto bra = bra_out;
    auto ket = ket_out;
    bool will_free_bra = false, will_free_ket = false;
    if (bra == nullptr) {
        bra = derived::InitState(dim);
        will_free_bra = true;
    }
    if (ket == nullptr) {
        ket = derived::InitState(dim);
        will_free_ket = true;
    }
    SingleQubitGateMask mask(objs, ctrls);
    qs_data_t m00 = m[0][0];
    qs_data_t m01 = m[0][1];
    qs_data_t m10 = m[1][0];
    qs_data_t m11 = m[1][1];
    thrust::counting_iterator<size_t> l(0);
    auto obj_high_mask = mask.obj_high_mask;
    auto obj_low_mask = mask.obj_low_mask;
    auto obj_mask = mask.obj_mask;
    auto ctrl_mask = mask.ctrl_mask;
    qs_data_t res = 0.0;
    if (!mask.ctrl_mask) {
        res = thrust::transform_reduce(
            l, l + dim / 2,
            [=] __host__ __device__(size_t l) {
                auto i = ((l & obj_high_mask) << 1) + (l & obj_low_mask);
                auto j = i + obj_mask;
                auto t1 = m00 * ket[i] + m01 * ket[j];
                auto t2 = m10 * ket[i] + m11 * ket[j];
                auto this_res = thrust::conj(bra[i]) * t1 + thrust::conj(bra[j]) * t2;
                return this_res;
            },
            qs_data_t(0, 0), thrust::plus<qs_data_t>());
    } else if (mask.ctrl_qubits.size() == 1) {
        index_t ctrl_low = static_cast<uint64_t>(0);
        for (qbit_t i = 0; i < mask.ctrl_qubits[0]; i++) {
            ctrl_low = (ctrl_low << 1) + 1;
        }
        index_t first_low_mask = mask.obj_low_mask;
        index_t second_low_mask = ctrl_low;
        if (mask.obj_low_mask > ctrl_low) {
            first_low_mask = ctrl_low;
            second_low_mask = mask.obj_low_mask;
        }
        auto first_high_mask = ~first_low_mask;
        auto second_high_mask = ~second_low_mask;
        res = thrust::transform_reduce(
            l, l + dim / 4,
            [=] __host__ __device__(size_t l) {
                auto i = ((l & first_high_mask) << 1) + (l & first_low_mask);
                i = ((i & second_high_mask) << 1) + (i & second_low_mask) + ctrl_mask;
                auto j = i + obj_mask;
                auto t1 = m00 * ket[i] + m01 * ket[j];
                auto t2 = m10 * ket[i] + m11 * ket[j];
                auto this_res = thrust::conj(bra[i]) * t1 + thrust::conj(bra[j]) * t2;
                return this_res;
            },
            qs_data_t(0, 0), thrust::plus<qs_data_t>());
    } else {
        res = thrust::transform_reduce(
            l, l + dim / 2,
            [=] __host__ __device__(size_t l) {
                auto i = ((l & obj_high_mask) << 1) + (l & obj_low_mask);
                if ((i & ctrl_mask) != ctrl_mask) {
                    return qs_data_t(0, 0);
                }
                auto j = i + obj_mask;
                auto t1 = m00 * ket[i] + m01 * ket[j];
                auto t2 = m10 * ket[i] + m11 * ket[j];
                auto this_res = thrust::conj(bra[i]) * t1 + thrust::conj(bra[j]) * t2;
                return this_res;
            },
            qs_data_t(0, 0), thrust::plus<qs_data_t>());
    }
    if (will_free_bra) {
        derived::FreeState(&bra);
    }
    if (will_free_ket) {
        derived::FreeState(&ket);
    }
    return res;
}

template <typename derived_, typename calc_type_>
auto GPUVectorPolicyBase<derived_, calc_type_>::ExpectDiffMatrixGate(const qs_data_p_t& bra, const qs_data_p_t& ket,
                                                                     const qbits_t& objs, const qbits_t& ctrls,
                                                                     const std::vector<py_qs_datas_t>& m, index_t dim)
    -> qs_data_t {
    if (objs.size() == 1) {
        return derived::ExpectDiffSingleQubitMatrix(bra, ket, objs, ctrls, m, dim);
    }
    if (objs.size() == 2) {
        return derived::ExpectDiffTwoQubitsMatrix(bra, ket, objs, ctrls, m, dim);
    }
    return derived::ExpectDiffNQubitsMatrix(bra, ket, objs, ctrls, m, dim);
}

template <typename derived_, typename calc_type_>
auto GPUVectorPolicyBase<derived_, calc_type_>::ExpectDiffRX(const qs_data_p_t& bra, const qs_data_p_t& ket,
                                                             const qbits_t& objs, const qbits_t& ctrls, calc_type val,
                                                             index_t dim) -> qs_data_t {
    auto c = static_cast<calc_type>(-0.5 * std::sin(val / 2));
    auto is = static_cast<calc_type>(0.5 * std::cos(val / 2)) * qs_data_t(0, -1);
    std::vector<py_qs_datas_t> gate = {{c, is}, {is, c}};
    return derived::ExpectDiffSingleQubitMatrix(bra, ket, objs, ctrls, gate, dim);
}

template <typename derived_, typename calc_type_>
auto GPUVectorPolicyBase<derived_, calc_type_>::ExpectDiffRY(const qs_data_p_t& bra, const qs_data_p_t& ket,
                                                             const qbits_t& objs, const qbits_t& ctrls, calc_type val,
                                                             index_t dim) -> qs_data_t {
    SingleQubitGateMask mask(objs, ctrls);
    auto c = static_cast<calc_type>(-0.5 * std::sin(val / 2));
    auto s = static_cast<calc_type>(0.5 * std::cos(val / 2));
    std::vector<py_qs_datas_t> gate = {{c, -s}, {s, c}};
    return derived::ExpectDiffSingleQubitMatrix(bra, ket, objs, ctrls, gate, dim);
}

template <typename derived_, typename calc_type_>
auto GPUVectorPolicyBase<derived_, calc_type_>::ExpectDiffRZ(const qs_data_p_t& bra, const qs_data_p_t& ket,
                                                             const qbits_t& objs, const qbits_t& ctrls, calc_type val,
                                                             index_t dim) -> qs_data_t {
    SingleQubitGateMask mask(objs, ctrls);
    auto c = static_cast<calc_type>(-0.5 * std::sin(val / 2));
    auto s = static_cast<calc_type>(0.5 * std::cos(val / 2));
    auto e0 = c + qs_data_t(0, -1) * s;
    auto e1 = c + qs_data_t(0, 1) * s;
    std::vector<py_qs_datas_t> gate = {{e0, 0}, {0, e1}};
    return derived::ExpectDiffSingleQubitMatrix(bra, ket, objs, ctrls, gate, dim);
}

template <typename derived_, typename calc_type_>
auto GPUVectorPolicyBase<derived_, calc_type_>::ExpectDiffRxx(const qs_data_p_t& bra_out, const qs_data_p_t& ket_out,
                                                              const qbits_t& objs, const qbits_t& ctrls, calc_type val,
                                                              index_t dim) -> qs_data_t {
    auto bra = bra_out;
    auto ket = ket_out;
    bool will_free_bra = false, will_free_ket = false;
    if (bra == nullptr) {
        bra = derived::InitState(dim);
        will_free_bra = true;
    }
    if (ket == nullptr) {
        ket = derived::InitState(dim);
        will_free_ket = true;
    }
    DoubleQubitGateMask mask(objs, ctrls);
    auto c = static_cast<calc_type>(-std::sin(val / 2) / 2);
    auto s = static_cast<calc_type>(std::cos(val / 2) / 2) * qs_data_t(0, -1);
    thrust::counting_iterator<size_t> l(0);
    auto obj_high_mask = mask.obj_high_mask;
    auto obj_rev_high_mask = mask.obj_rev_high_mask;
    auto obj_low_mask = mask.obj_low_mask;
    auto obj_rev_low_mask = mask.obj_rev_low_mask;
    auto obj_mask = mask.obj_mask;
    auto obj_min_mask = mask.obj_min_mask;
    auto obj_max_mask = mask.obj_max_mask;
    auto ctrl_mask = mask.ctrl_mask;
    qs_data_t res = 0.0;
    if (!mask.ctrl_mask) {
        res = thrust::transform_reduce(
            l, l + dim / 4,
            [=] __host__ __device__(size_t l) {
                index_t i;
                SHIFT_BIT_TWO(obj_low_mask, obj_rev_low_mask, obj_high_mask, obj_rev_high_mask, l, i);
                auto m = i + obj_mask;
                auto j = i + obj_min_mask;
                auto k = i + obj_max_mask;
                auto v00 = c * ket[i] + s * ket[m];
                auto v01 = c * ket[j] + s * ket[k];
                auto v10 = c * ket[k] + s * ket[j];
                auto v11 = c * ket[m] + s * ket[i];
                auto this_res = thrust::conj(bra[i]) * v00;
                this_res += thrust::conj(bra[j]) * v01;
                this_res += thrust::conj(bra[k]) * v10;
                this_res += thrust::conj(bra[m]) * v11;
                return this_res;
            },
            qs_data_t(0, 0), thrust::plus<qs_data_t>());
    } else {
        res = thrust::transform_reduce(
            l, l + dim / 4,
            [=] __host__ __device__(size_t l) {
                index_t i;
                SHIFT_BIT_TWO(obj_low_mask, obj_rev_low_mask, obj_high_mask, obj_rev_high_mask, l, i);
                if ((i & ctrl_mask) != ctrl_mask) {
                    return qs_data_t(0, 0);
                }
                auto m = i + obj_mask;
                auto j = i + obj_min_mask;
                auto k = i + obj_max_mask;
                auto v00 = c * ket[i] + s * ket[m];
                auto v01 = c * ket[j] + s * ket[k];
                auto v10 = c * ket[k] + s * ket[j];
                auto v11 = c * ket[m] + s * ket[i];
                auto this_res = thrust::conj(bra[i]) * v00;
                this_res += thrust::conj(bra[j]) * v01;
                this_res += thrust::conj(bra[k]) * v10;
                this_res += thrust::conj(bra[m]) * v11;
                return this_res;
            },
            qs_data_t(0, 0), thrust::plus<qs_data_t>());
    }
    if (will_free_bra) {
        derived::FreeState(&bra);
    }
    if (will_free_ket) {
        derived::FreeState(&ket);
    }
    return res;
}

template <typename derived_, typename calc_type_>
auto GPUVectorPolicyBase<derived_, calc_type_>::ExpectDiffRxy(const qs_data_p_t& bra_out, const qs_data_p_t& ket_out,
                                                              const qbits_t& objs, const qbits_t& ctrls, calc_type val,
                                                              index_t dim) -> qs_data_t {
    auto bra = bra_out;
    auto ket = ket_out;
    bool will_free_bra = false, will_free_ket = false;
    if (bra == nullptr) {
        bra = derived::InitState(dim);
        will_free_bra = true;
    }
    if (ket == nullptr) {
        ket = derived::InitState(dim);
        will_free_ket = true;
    }
    DoubleQubitGateMask mask(objs, ctrls);
    auto c = static_cast<calc_type>(-std::sin(val / 2) / 2);
    auto s = static_cast<calc_type>(std::cos(val / 2) / 2);
    thrust::counting_iterator<size_t> l(0);
    auto obj_high_mask = mask.obj_high_mask;
    auto obj_rev_high_mask = mask.obj_rev_high_mask;
    auto obj_low_mask = mask.obj_low_mask;
    auto obj_rev_low_mask = mask.obj_rev_low_mask;
    auto obj_mask = mask.obj_mask;
    auto obj_min_mask = mask.obj_min_mask;
    auto obj_max_mask = mask.obj_max_mask;
    auto ctrl_mask = mask.ctrl_mask;
    qs_data_t res = 0.0;
    if (!mask.ctrl_mask) {
        res = thrust::transform_reduce(
            l, l + dim / 4,
            [=] __host__ __device__(size_t l) {
                index_t i;
                SHIFT_BIT_TWO(obj_low_mask, obj_rev_low_mask, obj_high_mask, obj_rev_high_mask, l, i);
                auto m = i + obj_mask;
                auto j = i + obj_min_mask;
                auto k = i + obj_max_mask;
                auto v00 = c * ket[i] - s * ket[m];
                auto v01 = c * ket[j] - s * ket[k];
                auto v10 = c * ket[k] + s * ket[j];
                auto v11 = c * ket[m] + s * ket[i];
                auto this_res = thrust::conj(bra[i]) * v00;
                this_res += thrust::conj(bra[j]) * v01;
                this_res += thrust::conj(bra[k]) * v10;
                this_res += thrust::conj(bra[m]) * v11;
                return this_res;
            },
            qs_data_t(0, 0), thrust::plus<qs_data_t>());
    } else {
        res = thrust::transform_reduce(
            l, l + dim / 4,
            [=] __host__ __device__(size_t l) {
                index_t i;
                SHIFT_BIT_TWO(obj_low_mask, obj_rev_low_mask, obj_high_mask, obj_rev_high_mask, l, i);
                if ((i & ctrl_mask) != ctrl_mask) {
                    return qs_data_t(0, 0);
                }
                auto m = i + obj_mask;
                auto j = i + obj_min_mask;
                auto k = i + obj_max_mask;
                auto v00 = c * ket[i] - s * ket[m];
                auto v01 = c * ket[j] - s * ket[k];
                auto v10 = c * ket[k] + s * ket[j];
                auto v11 = c * ket[m] + s * ket[i];
                auto this_res = thrust::conj(bra[i]) * v00;
                this_res += thrust::conj(bra[j]) * v01;
                this_res += thrust::conj(bra[k]) * v10;
                this_res += thrust::conj(bra[m]) * v11;
                return this_res;
            },
            qs_data_t(0, 0), thrust::plus<qs_data_t>());
    }
    if (will_free_bra) {
        derived::FreeState(&bra);
    }
    if (will_free_ket) {
        derived::FreeState(&ket);
    }
    return res;
}

template <typename derived_, typename calc_type_>
auto GPUVectorPolicyBase<derived_, calc_type_>::ExpectDiffRxz(const qs_data_p_t& bra_out, const qs_data_p_t& ket_out,
                                                              const qbits_t& objs, const qbits_t& ctrls, calc_type val,
                                                              index_t dim) -> qs_data_t {
    auto bra = bra_out;
    auto ket = ket_out;
    bool will_free_bra = false, will_free_ket = false;
    if (bra == nullptr) {
        bra = derived::InitState(dim);
        will_free_bra = true;
    }
    if (ket == nullptr) {
        ket = derived::InitState(dim);
        will_free_ket = true;
    }
    DoubleQubitGateMask mask(objs, ctrls);
    auto c = static_cast<calc_type>(-std::sin(val / 2) / 2);
    auto s = static_cast<calc_type>(std::cos(val / 2) / 2) * qs_data_t(0, -1);
    thrust::counting_iterator<size_t> l(0);
    auto obj_high_mask = mask.obj_high_mask;
    auto obj_rev_high_mask = mask.obj_rev_high_mask;
    auto obj_low_mask = mask.obj_low_mask;
    auto obj_rev_low_mask = mask.obj_rev_low_mask;
    auto obj_mask = mask.obj_mask;
    auto obj_min_mask = mask.obj_min_mask;
    auto obj_max_mask = mask.obj_max_mask;
    auto ctrl_mask = mask.ctrl_mask;
    qs_data_t res = 0.0;
    if (!mask.ctrl_mask) {
        res = thrust::transform_reduce(
            l, l + dim / 4,
            [=] __host__ __device__(size_t l) {
                index_t i;
                SHIFT_BIT_TWO(obj_low_mask, obj_rev_low_mask, obj_high_mask, obj_rev_high_mask, l, i);
                auto m = i + obj_mask;
                auto j = i + obj_min_mask;
                auto k = i + obj_max_mask;
                auto v00 = c * ket[i] + s * ket[j];
                auto v01 = c * ket[j] + s * ket[i];
                auto v10 = c * ket[k] - s * ket[m];
                auto v11 = c * ket[m] - s * ket[k];
                auto this_res = thrust::conj(bra[i]) * v00;
                this_res += thrust::conj(bra[j]) * v01;
                this_res += thrust::conj(bra[k]) * v10;
                this_res += thrust::conj(bra[m]) * v11;
                return this_res;
            },
            qs_data_t(0, 0), thrust::plus<qs_data_t>());
    } else {
        res = thrust::transform_reduce(
            l, l + dim / 4,
            [=] __host__ __device__(size_t l) {
                index_t i;
                SHIFT_BIT_TWO(obj_low_mask, obj_rev_low_mask, obj_high_mask, obj_rev_high_mask, l, i);
                if ((i & ctrl_mask) != ctrl_mask) {
                    return qs_data_t(0, 0);
                }
                auto m = i + obj_mask;
                auto j = i + obj_min_mask;
                auto k = i + obj_max_mask;
                auto v00 = c * ket[i] + s * ket[j];
                auto v01 = c * ket[j] + s * ket[i];
                auto v10 = c * ket[k] - s * ket[m];
                auto v11 = c * ket[m] - s * ket[k];
                auto this_res = thrust::conj(bra[i]) * v00;
                this_res += thrust::conj(bra[j]) * v01;
                this_res += thrust::conj(bra[k]) * v10;
                this_res += thrust::conj(bra[m]) * v11;
                return this_res;
            },
            qs_data_t(0, 0), thrust::plus<qs_data_t>());
    }
    if (will_free_bra) {
        derived::FreeState(&bra);
    }
    if (will_free_ket) {
        derived::FreeState(&ket);
    }
    return res;
}

template <typename derived_, typename calc_type_>
auto GPUVectorPolicyBase<derived_, calc_type_>::ExpectDiffRyz(const qs_data_p_t& bra_out, const qs_data_p_t& ket_out,
                                                              const qbits_t& objs, const qbits_t& ctrls, calc_type val,
                                                              index_t dim) -> qs_data_t {
    auto bra = bra_out;
    auto ket = ket_out;
    bool will_free_bra = false, will_free_ket = false;
    if (bra == nullptr) {
        bra = derived::InitState(dim);
        will_free_bra = true;
    }
    if (ket == nullptr) {
        ket = derived::InitState(dim);
        will_free_ket = true;
    }
    DoubleQubitGateMask mask(objs, ctrls);
    auto c = static_cast<calc_type>(-std::sin(val / 2) / 2);
    auto s = static_cast<calc_type>(std::cos(val / 2) / 2);
    thrust::counting_iterator<size_t> l(0);
    auto obj_high_mask = mask.obj_high_mask;
    auto obj_rev_high_mask = mask.obj_rev_high_mask;
    auto obj_low_mask = mask.obj_low_mask;
    auto obj_rev_low_mask = mask.obj_rev_low_mask;
    auto obj_mask = mask.obj_mask;
    auto obj_min_mask = mask.obj_min_mask;
    auto obj_max_mask = mask.obj_max_mask;
    auto ctrl_mask = mask.ctrl_mask;
    qs_data_t res = 0.0;
    if (!mask.ctrl_mask) {
        res = thrust::transform_reduce(
            l, l + dim / 4,
            [=] __host__ __device__(size_t l) {
                index_t i;
                SHIFT_BIT_TWO(obj_low_mask, obj_rev_low_mask, obj_high_mask, obj_rev_high_mask, l, i);
                auto m = i + obj_mask;
                auto j = i + obj_min_mask;
                auto k = i + obj_max_mask;
                auto v00 = c * ket[i] - s * ket[j];
                auto v01 = c * ket[j] + s * ket[i];
                auto v10 = c * ket[k] + s * ket[m];
                auto v11 = c * ket[m] - s * ket[k];
                auto this_res = thrust::conj(bra[i]) * v00;
                this_res += thrust::conj(bra[j]) * v01;
                this_res += thrust::conj(bra[k]) * v10;
                this_res += thrust::conj(bra[m]) * v11;
                return this_res;
            },
            qs_data_t(0, 0), thrust::plus<qs_data_t>());
    } else {
        res = thrust::transform_reduce(
            l, l + dim / 4,
            [=] __host__ __device__(size_t l) {
                index_t i;
                SHIFT_BIT_TWO(obj_low_mask, obj_rev_low_mask, obj_high_mask, obj_rev_high_mask, l, i);
                if ((i & ctrl_mask) != ctrl_mask) {
                    return qs_data_t(0, 0);
                }
                auto m = i + obj_mask;
                auto j = i + obj_min_mask;
                auto k = i + obj_max_mask;
                auto v00 = c * ket[i] - s * ket[j];
                auto v01 = c * ket[j] + s * ket[i];
                auto v10 = c * ket[k] + s * ket[m];
                auto v11 = c * ket[m] - s * ket[k];
                auto this_res = thrust::conj(bra[i]) * v00;
                this_res += thrust::conj(bra[j]) * v01;
                this_res += thrust::conj(bra[k]) * v10;
                this_res += thrust::conj(bra[m]) * v11;
                return this_res;
            },
            qs_data_t(0, 0), thrust::plus<qs_data_t>());
    }
    if (will_free_bra) {
        derived::FreeState(&bra);
    }
    if (will_free_ket) {
        derived::FreeState(&ket);
    }
    return res;
}

template <typename derived_, typename calc_type_>
auto GPUVectorPolicyBase<derived_, calc_type_>::ExpectDiffRyy(const qs_data_p_t& bra_out, const qs_data_p_t& ket_out,
                                                              const qbits_t& objs, const qbits_t& ctrls, calc_type val,
                                                              index_t dim) -> qs_data_t {
    auto bra = bra_out;
    auto ket = ket_out;
    bool will_free_bra = false, will_free_ket = false;
    if (bra == nullptr) {
        bra = derived::InitState(dim);
        will_free_bra = true;
    }
    if (ket == nullptr) {
        ket = derived::InitState(dim);
        will_free_ket = true;
    }
    DoubleQubitGateMask mask(objs, ctrls);
    auto c = static_cast<calc_type>(-std::sin(val / 2) / 2);
    auto s = static_cast<calc_type>(std::cos(val / 2) / 2) * qs_data_t(0, 1);
    thrust::counting_iterator<size_t> l(0);
    auto obj_high_mask = mask.obj_high_mask;
    auto obj_rev_high_mask = mask.obj_rev_high_mask;
    auto obj_low_mask = mask.obj_low_mask;
    auto obj_rev_low_mask = mask.obj_rev_low_mask;
    auto obj_mask = mask.obj_mask;
    auto obj_min_mask = mask.obj_min_mask;
    auto obj_max_mask = mask.obj_max_mask;
    auto ctrl_mask = mask.ctrl_mask;
    qs_data_t res = 0.0;
    if (!mask.ctrl_mask) {
        res = thrust::transform_reduce(
            l, l + dim / 4,
            [=] __host__ __device__(size_t l) {
                index_t i;
                SHIFT_BIT_TWO(obj_low_mask, obj_rev_low_mask, obj_high_mask, obj_rev_high_mask, l, i);
                auto m = i + obj_mask;
                auto j = i + obj_min_mask;
                auto k = i + obj_max_mask;
                auto v00 = c * ket[i] + s * ket[m];
                auto v01 = c * ket[j] - s * ket[k];
                auto v10 = c * ket[k] - s * ket[j];
                auto v11 = c * ket[m] + s * ket[i];
                auto this_res = thrust::conj(bra[i]) * v00;
                this_res += thrust::conj(bra[j]) * v01;
                this_res += thrust::conj(bra[k]) * v10;
                this_res += thrust::conj(bra[m]) * v11;
                return this_res;
            },
            qs_data_t(0, 0), thrust::plus<qs_data_t>());
    } else {
        res = thrust::transform_reduce(
            l, l + dim / 4,
            [=] __host__ __device__(size_t l) {
                index_t i;
                SHIFT_BIT_TWO(obj_low_mask, obj_rev_low_mask, obj_high_mask, obj_rev_high_mask, l, i);
                if ((i & ctrl_mask) != ctrl_mask) {
                    return qs_data_t(0, 0);
                }
                auto m = i + obj_mask;
                auto j = i + obj_min_mask;
                auto k = i + obj_max_mask;
                auto v00 = c * ket[i] + s * ket[m];
                auto v01 = c * ket[j] - s * ket[k];
                auto v10 = c * ket[k] - s * ket[j];
                auto v11 = c * ket[m] + s * ket[i];
                auto this_res = thrust::conj(bra[i]) * v00;
                this_res += thrust::conj(bra[j]) * v01;
                this_res += thrust::conj(bra[k]) * v10;
                this_res += thrust::conj(bra[m]) * v11;
                return this_res;
            },
            qs_data_t(0, 0), thrust::plus<qs_data_t>());
    }
    if (will_free_bra) {
        derived::FreeState(&bra);
    }
    if (will_free_ket) {
        derived::FreeState(&ket);
    }
    return res;
}

template <typename derived_, typename calc_type_>
auto GPUVectorPolicyBase<derived_, calc_type_>::ExpectDiffRzz(const qs_data_p_t& bra_out, const qs_data_p_t& ket_out,
                                                              const qbits_t& objs, const qbits_t& ctrls, calc_type val,
                                                              index_t dim) -> qs_data_t {
    auto bra = bra_out;
    auto ket = ket_out;
    bool will_free_bra = false, will_free_ket = false;
    if (bra == nullptr) {
        bra = derived::InitState(dim);
        will_free_bra = true;
    }
    if (ket == nullptr) {
        ket = derived::InitState(dim);
        will_free_ket = true;
    }
    DoubleQubitGateMask mask(objs, ctrls);
    auto c = static_cast<calc_type>(-std::sin(val / 2) / 2);
    auto s = static_cast<calc_type>(std::cos(val / 2) / 2);
    auto e = c + qs_data_t(0, 1) * s;
    auto me = c + qs_data_t(0, -1) * s;
    thrust::counting_iterator<size_t> l(0);
    auto obj_high_mask = mask.obj_high_mask;
    auto obj_rev_high_mask = mask.obj_rev_high_mask;
    auto obj_low_mask = mask.obj_low_mask;
    auto obj_rev_low_mask = mask.obj_rev_low_mask;
    auto obj_mask = mask.obj_mask;
    auto obj_min_mask = mask.obj_min_mask;
    auto obj_max_mask = mask.obj_max_mask;
    auto ctrl_mask = mask.ctrl_mask;
    qs_data_t res = 0.0;
    if (!mask.ctrl_mask) {
        res = thrust::transform_reduce(
            l, l + dim / 4,
            [=] __host__ __device__(size_t l) {
                index_t i;
                SHIFT_BIT_TWO(obj_low_mask, obj_rev_low_mask, obj_high_mask, obj_rev_high_mask, l, i);
                auto m = i + obj_mask;
                auto j = i + obj_min_mask;
                auto k = i + obj_max_mask;
                auto this_res = thrust::conj(bra[i]) * ket[i] * me;
                this_res += thrust::conj(bra[j]) * ket[j] * e;
                this_res += thrust::conj(bra[k]) * ket[k] * e;
                this_res += thrust::conj(bra[m]) * ket[m] * me;
                return this_res;
            },
            qs_data_t(0, 0), thrust::plus<qs_data_t>());
    } else {
        res = thrust::transform_reduce(
            l, l + dim / 4,
            [=] __host__ __device__(size_t l) {
                index_t i;
                SHIFT_BIT_TWO(obj_low_mask, obj_rev_low_mask, obj_high_mask, obj_rev_high_mask, l, i);
                if ((i & ctrl_mask) != ctrl_mask) {
                    return qs_data_t(0, 0);
                }
                auto m = i + obj_mask;
                auto j = i + obj_min_mask;
                auto k = i + obj_max_mask;
                auto this_res = thrust::conj(bra[i]) * ket[i] * me;
                this_res += thrust::conj(bra[j]) * ket[j] * e;
                this_res += thrust::conj(bra[k]) * ket[k] * e;
                this_res += thrust::conj(bra[m]) * ket[m] * me;
                return this_res;
            },
            qs_data_t(0, 0), thrust::plus<qs_data_t>());
    }
    if (will_free_bra) {
        derived::FreeState(&bra);
    }
    if (will_free_ket) {
        derived::FreeState(&ket);
    }
    return res;
}

template <typename derived_, typename calc_type_>
auto GPUVectorPolicyBase<derived_, calc_type_>::ExpectDiffGivens(const qs_data_p_t& bra_out, const qs_data_p_t& ket_out,
                                                                 const qbits_t& objs, const qbits_t& ctrls,
                                                                 calc_type val, index_t dim) -> qs_data_t {
    auto bra = bra_out;
    auto ket = ket_out;
    bool will_free_bra = false, will_free_ket = false;
    if (bra == nullptr) {
        bra = derived::InitState(dim);
        will_free_bra = true;
    }
    if (ket == nullptr) {
        ket = derived::InitState(dim);
        will_free_ket = true;
    }
    DoubleQubitGateMask mask(objs, ctrls);
    auto c = -std::sin(val);
    auto s = std::cos(val);
    thrust::counting_iterator<size_t> l(0);
    auto obj_high_mask = mask.obj_high_mask;
    auto obj_rev_high_mask = mask.obj_rev_high_mask;
    auto obj_low_mask = mask.obj_low_mask;
    auto obj_rev_low_mask = mask.obj_rev_low_mask;
    auto obj_min_mask = mask.obj_min_mask;
    auto obj_max_mask = mask.obj_max_mask;
    auto ctrl_mask = mask.ctrl_mask;
    qs_data_t res = 0.0;
    if (!mask.ctrl_mask) {
        res = thrust::transform_reduce(
            l, l + dim / 4,
            [=] __host__ __device__(size_t l) {
                index_t i;
                SHIFT_BIT_TWO(obj_low_mask, obj_rev_low_mask, obj_high_mask, obj_rev_high_mask, l, i);
                auto j = i + obj_min_mask;
                auto k = i + obj_max_mask;
                auto v01 = c * ket[j] - s * ket[k];
                auto v10 = s * ket[j] + c * ket[k];
                auto this_res = thrust::conj(bra[j]) * v01;
                this_res += thrust::conj(bra[k]) * v10;
                return this_res;
            },
            qs_data_t(0, 0), thrust::plus<qs_data_t>());
    } else {
        res = thrust::transform_reduce(
            l, l + dim / 4,
            [=] __host__ __device__(size_t l) {
                index_t i;
                SHIFT_BIT_TWO(obj_low_mask, obj_rev_low_mask, obj_high_mask, obj_rev_high_mask, l, i);
                if ((i & ctrl_mask) != ctrl_mask) {
                    return qs_data_t(0, 0);
                }
                auto j = i + obj_min_mask;
                auto k = i + obj_max_mask;
                auto v01 = c * ket[j] - s * ket[k];
                auto v10 = s * ket[j] + c * ket[k];
                auto this_res = thrust::conj(bra[j]) * v01;
                this_res += thrust::conj(bra[k]) * v10;
                return this_res;
            },
            qs_data_t(0, 0), thrust::plus<qs_data_t>());
    }
    if (will_free_bra) {
        derived::FreeState(&bra);
    }
    if (will_free_ket) {
        derived::FreeState(&ket);
    }
    return res;
}

template <typename derived_, typename calc_type_>
auto GPUVectorPolicyBase<derived_, calc_type_>::ExpectDiffGP(const qs_data_p_t& bra, const qs_data_p_t& ket,
                                                             const qbits_t& objs, const qbits_t& ctrls, calc_type val,
                                                             index_t dim) -> qs_data_t {
    SingleQubitGateMask mask(objs, ctrls);
    auto e = std::complex<calc_type>(0, -1);
    e *= std::exp(std::complex<calc_type>(0, -val));
    std::vector<py_qs_datas_t> gate = {{e, 0}, {0, e}};
    return derived::ExpectDiffSingleQubitMatrix(bra, ket, objs, ctrls, gate, dim);
}

template <typename derived_, typename calc_type_>
auto GPUVectorPolicyBase<derived_, calc_type_>::ExpectDiffPS(const qs_data_p_t& bra_out, const qs_data_p_t& ket_out,
                                                             const qbits_t& objs, const qbits_t& ctrls, calc_type val,
                                                             index_t dim) -> qs_data_t {
    auto bra = bra_out;
    auto ket = ket_out;
    bool will_free_bra = false, will_free_ket = false;
    if (bra == nullptr) {
        bra = derived::InitState(dim);
        will_free_bra = true;
    }
    if (ket == nullptr) {
        ket = derived::InitState(dim);
        will_free_ket = true;
    }
    SingleQubitGateMask mask(objs, ctrls);
    auto e = static_cast<calc_type>(-std::sin(val)) + qs_data_t(0, 1) * static_cast<calc_type>(std::cos(val));
    thrust::counting_iterator<size_t> l(0);
    auto obj_high_mask = mask.obj_high_mask;
    auto obj_low_mask = mask.obj_low_mask;
    auto obj_mask = mask.obj_mask;
    auto ctrl_mask = mask.ctrl_mask;
    qs_data_t res = 0.0;
    if (!(mask.ctrl_mask)) {
        res = thrust::transform_reduce(
            l, l + dim / 2,
            [=] __host__ __device__(size_t l) {
                auto i = ((l & obj_high_mask) << 1) + (l & obj_low_mask);
                auto j = i + obj_mask;
                auto this_res = thrust::conj(bra[j]) * ket[j] * e;
                return this_res;
            },
            qs_data_t(0, 0), thrust::plus<qs_data_t>());
    } else if (mask.ctrl_qubits.size() == 1) {
        index_t ctrl_low = static_cast<uint64_t>(0);
        for (qbit_t i = 0; i < mask.ctrl_qubits[0]; i++) {
            ctrl_low = (ctrl_low << 1) + 1;
        }
        index_t first_low_mask = mask.obj_low_mask;
        index_t second_low_mask = ctrl_low;
        if (mask.obj_low_mask > ctrl_low) {
            first_low_mask = ctrl_low;
            second_low_mask = mask.obj_low_mask;
        }
        auto first_high_mask = ~first_low_mask;
        auto second_high_mask = ~second_low_mask;
        res = thrust::transform_reduce(
            l, l + dim / 4,
            [=] __host__ __device__(size_t l) {
                auto i = ((l & first_high_mask) << 1) + (l & first_low_mask);
                i = ((i & second_high_mask) << 1) + (i & second_low_mask) + ctrl_mask;
                auto j = i + obj_mask;
                auto this_res = thrust::conj(bra[j]) * ket[j] * e;
                return this_res;
            },
            qs_data_t(0, 0), thrust::plus<qs_data_t>());
    } else {
        res = thrust::transform_reduce(
            l, l + dim / 2,
            [=] __host__ __device__(size_t l) {
                auto i = ((l & obj_high_mask) << 1) + (l & obj_low_mask);
                if ((i & ctrl_mask) != ctrl_mask) {
                    return qs_data_t(0, 0);
                }
                auto j = i + obj_mask;
                auto this_res = thrust::conj(bra[j]) * ket[j] * e;
                return this_res;
            },
            qs_data_t(0, 0), thrust::plus<qs_data_t>());
    }
    if (will_free_bra) {
        derived::FreeState(&bra);
    }
    if (will_free_ket) {
        derived::FreeState(&ket);
    }
    return res;
}

template <typename derived_, typename calc_type_>
auto GPUVectorPolicyBase<derived_, calc_type_>::ExpectDiffSWAPalpha(const qs_data_p_t& bra_out,
                                                                    const qs_data_p_t& ket_out, const qbits_t& objs,
                                                                    const qbits_t& ctrls, calc_type val, index_t dim)
    -> qs_data_t {
    auto bra = bra_out;
    auto ket = ket_out;
    bool will_free_bra = false, will_free_ket = false;
    if (bra == nullptr) {
        bra = derived::InitState(dim);
        will_free_bra = true;
    }
    if (ket == nullptr) {
        ket = derived::InitState(dim);
        will_free_ket = true;
    }
    DoubleQubitGateMask mask(objs, ctrls);
    auto image_i = py_qs_data_t(0, 1);
    auto image_mi = py_qs_data_t(0, -1);
    auto e = std::exp(image_i * static_cast<calc_type_>(M_PI) * val);
    qs_data_t a = image_i * static_cast<calc_type_>(M_PI_2) * e;
    qs_data_t b = image_mi * static_cast<calc_type_>(M_PI_2) * e;
    thrust::counting_iterator<size_t> l(0);
    auto obj_high_mask = mask.obj_high_mask;
    auto obj_rev_high_mask = mask.obj_rev_high_mask;
    auto obj_low_mask = mask.obj_low_mask;
    auto obj_rev_low_mask = mask.obj_rev_low_mask;
    auto obj_min_mask = mask.obj_min_mask;
    auto obj_max_mask = mask.obj_max_mask;
    auto ctrl_mask = mask.ctrl_mask;
    qs_data_t res = 0.0;
    if (!mask.ctrl_mask) {
        res = thrust::transform_reduce(
            l, l + dim / 4,
            [=] __host__ __device__(size_t l) {
                index_t i;
                SHIFT_BIT_TWO(obj_low_mask, obj_rev_low_mask, obj_high_mask, obj_rev_high_mask, l, i);
                auto j = i + obj_min_mask;
                auto k = i + obj_max_mask;
                auto v01 = a * ket[j] + b * ket[k];
                auto v10 = a * ket[k] + b * ket[j];
                auto this_res = thrust::conj(bra[j]) * v01;
                this_res += thrust::conj(bra[k]) * v10;
                return this_res;
            },
            qs_data_t(0, 0), thrust::plus<qs_data_t>());
    } else {
        res = thrust::transform_reduce(
            l, l + dim / 4,
            [=] __host__ __device__(size_t l) {
                index_t i;
                SHIFT_BIT_TWO(obj_low_mask, obj_rev_low_mask, obj_high_mask, obj_rev_high_mask, l, i);
                if ((i & ctrl_mask) != ctrl_mask) {
                    return qs_data_t(0, 0);
                }
                auto j = i + obj_min_mask;
                auto k = i + obj_max_mask;
                auto v01 = a * ket[j] + b * ket[k];
                auto v10 = a * ket[k] + b * ket[j];
                auto this_res = thrust::conj(bra[j]) * v01;
                this_res += thrust::conj(bra[k]) * v10;
                return this_res;
            },
            qs_data_t(0, 0), thrust::plus<qs_data_t>());
    }
    if (will_free_bra) {
        derived::FreeState(&bra);
    }
    if (will_free_ket) {
        derived::FreeState(&ket);
    }
    return res;
};

template struct GPUVectorPolicyBase<CuQuantumVectorPolicyFloat, float>;
template struct GPUVectorPolicyBase<CuQuantumVectorPolicyDouble, double>;
template struct GPUVectorPolicyBase<GPUVectorPolicyFloat, float>;
template struct GPUVectorPolicyBase<GPUVectorPolicyDouble, double>;

}  // namespace mindquantum::sim::vector::detail
