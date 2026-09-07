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
#include "math/pr/parameter_resolver.h"
#include "simulator/utils.h"

#ifdef __x86_64__
#    include "simulator/vector/detail/cpu_vector_avx_double_policy.h"
#    include "simulator/vector/detail/cpu_vector_avx_float_policy.h"
#elif defined(__amd64)
#    include "simulator/vector/detail/cpu_vector_arm_double_policy.h"
#    include "simulator/vector/detail/cpu_vector_arm_float_policy.h"
#endif
#include "simulator/vector/detail/cpu_vector_matrix_gate_utils.h"
#include "simulator/vector/detail/cpu_vector_policy.h"
namespace mindquantum::sim::vector::detail {
namespace {
template <size_t static_gate_dim, typename qs_data_t, typename py_qs_data_t>
void ApplyNQubitsMatrixKernel(const qs_data_t* src, qs_data_t* des, const qbits_t& objs, const qbits_t& ctrls,
                              const std::vector<std::vector<py_qs_data_t>>& gate, index_t dim) {
    const auto prepared = matrix_gate::PrepareMatrixGate<static_gate_dim, qs_data_t>(objs, ctrls, gate, dim);

    const auto apply_block = [&](index_t block, qs_data_t* input) {
        const auto base = prepared.bit_expander.Expand(block) | prepared.ctrl_mask;
        for (size_t col = 0; col < prepared.gate_dim; ++col) {
            input[col] = src[base | prepared.obj_masks[col]];
        }
        for (size_t row = 0; row < prepared.gate_dim; ++row) {
            qs_data_t value = 0;
            const auto* gate_row = prepared.gate_flat.data() + row * prepared.gate_dim;
            for (size_t col = 0; col < prepared.gate_dim; ++col) {
                value += gate_row[col] * input[col];
            }
            des[base | prepared.obj_masks[row]] = value;
        }
    };
    matrix_gate::ForEachMatrixGateBlock<static_gate_dim, qs_data_t>(dim, prepared.block_count, prepared.gate_dim,
                                                                    apply_block);
}
}  // namespace

template <typename derived_, typename calc_type_>
void CPUVectorPolicyBase<derived_, calc_type_>::ApplyNQubitsMatrix(const qs_data_p_t& src_out, qs_data_p_t* des_p,
                                                                   const qbits_t& objs, const qbits_t& ctrls,
                                                                   const std::vector<std::vector<py_qs_data_t>>& gate,
                                                                   index_t dim) {
    size_t n_qubit = objs.size();
    matrix_gate::ValidateNQubitsMatrixInputs(objs, ctrls, gate, dim);
    auto& des = *des_p;
    if (des == nullptr) {
        des = derived::InitState(dim);
    }
    qs_data_p_t src;
    bool will_free = false;
    if (src_out == nullptr) {
        src = derived::InitState(dim);
        will_free = true;
    } else {
        src = src_out;
    }
    auto apply_kernel = [&](auto static_gate_dim) {
        ApplyNQubitsMatrixKernel<decltype(static_gate_dim)::value>(src, des, objs, ctrls, gate, dim);
    };
    matrix_gate::DispatchStaticNQubits(n_qubit, apply_kernel);
    if (will_free) {
        derived::FreeState(&src);
    }
}

template <typename derived_, typename calc_type_>
void CPUVectorPolicyBase<derived_, calc_type_>::ApplyTwoQubitsMatrix(const qs_data_p_t& src_out, qs_data_p_t* des_p,
                                                                     const qbits_t& objs, const qbits_t& ctrls,
                                                                     const std::vector<std::vector<py_qs_data_t>>& gate,
                                                                     index_t dim) {
    auto& des = *des_p;
    if (des == nullptr) {
        des = derived::InitState(dim);
    }
    qs_data_p_t src;
    bool will_free = false;
    if (src_out == nullptr) {
        src = derived::InitState(dim);
        will_free = true;
    } else {
        src = src_out;
    }
    DoubleQubitGateMask mask(objs, ctrls);
    size_t mask1 = (static_cast<uint64_t>(1) << objs[0]);
    size_t mask2 = (static_cast<uint64_t>(1) << objs[1]);
    if (!mask.ctrl_mask) {
        // clang-format off
        MQ_OMP_PARALLEL_FOR(
            dim, for (omp::idx_t l = 0; l < static_cast<omp::idx_t>(dim / 4); l++) {
                omp::idx_t i;
                SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask, mask.obj_high_mask, mask.obj_rev_high_mask, l,
                              i);
                auto j = i + mask1;
                auto k = i + mask2;
                auto m = i + mask.obj_mask;
                auto v00 = gate[0][0] * src[i] + gate[0][1] * src[j] + gate[0][2] * src[k] + gate[0][3] * src[m];
                auto v01 = gate[1][0] * src[i] + gate[1][1] * src[j] + gate[1][2] * src[k] + gate[1][3] * src[m];
                auto v10 = gate[2][0] * src[i] + gate[2][1] * src[j] + gate[2][2] * src[k] + gate[2][3] * src[m];
                auto v11 = gate[3][0] * src[i] + gate[3][1] * src[j] + gate[3][2] * src[k] + gate[3][3] * src[m];
                des[i] = v00;
                des[j] = v01;
                des[k] = v10;
                des[m] = v11;
            })
        // clang-format on
    } else {
        MQ_OMP_PARALLEL_FOR(
            dim, for (omp::idx_t l = 0; l < static_cast<omp::idx_t>(dim / 4); l++) {
                omp::idx_t i;
                SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask, mask.obj_high_mask, mask.obj_rev_high_mask, l,
                              i);
                if ((i & mask.ctrl_mask) == mask.ctrl_mask) {
                    auto j = i + mask1;
                    auto k = i + mask2;
                    auto m = i + mask.obj_mask;
                    auto v00 = gate[0][0] * src[i] + gate[0][1] * src[j] + gate[0][2] * src[k] + gate[0][3] * src[m];
                    auto v01 = gate[1][0] * src[i] + gate[1][1] * src[j] + gate[1][2] * src[k] + gate[1][3] * src[m];
                    auto v10 = gate[2][0] * src[i] + gate[2][1] * src[j] + gate[2][2] * src[k] + gate[2][3] * src[m];
                    auto v11 = gate[3][0] * src[i] + gate[3][1] * src[j] + gate[3][2] * src[k] + gate[3][3] * src[m];
                    des[i] = v00;
                    des[j] = v01;
                    des[k] = v10;
                    des[m] = v11;
                }
            })
    }
    if (will_free) {
        derived::FreeState(&src);
    }
}
template <typename derived_, typename calc_type_>
void CPUVectorPolicyBase<derived_, calc_type_>::ApplySingleQubitMatrix(const qs_data_p_t& src_out, qs_data_p_t* des_p,
                                                                       qbit_t obj_qubit, const qbits_t& ctrls,
                                                                       const std::vector<std::vector<py_qs_data_t>>& m,
                                                                       index_t dim) {
    auto& des = (*des_p);
    if (des == nullptr) {
        des = derived::InitState(dim);
    }
    qs_data_p_t src;
    bool will_free = false;
    if (src_out == nullptr) {
        src = derived::InitState(dim);
        will_free = true;
    } else {
        src = src_out;
    }
    SingleQubitGateMask mask({obj_qubit}, ctrls);
    if (!mask.ctrl_mask) {
        MQ_OMP_PARALLEL_FOR(
            dim, for (omp::idx_t l = 0; l < static_cast<omp::idx_t>(dim / 2); l++) {
                auto i = ((l & mask.obj_high_mask) << 1) + (l & mask.obj_low_mask);
                auto j = i + mask.obj_mask;
                auto t1 = m[0][0] * src[i] + m[0][1] * src[j];
                auto t2 = m[1][0] * src[i] + m[1][1] * src[j];
                des[i] = t1;
                des[j] = t2;
            })
    } else {
        MQ_OMP_PARALLEL_FOR(
            dim, for (omp::idx_t l = 0; l < static_cast<omp::idx_t>(dim / 2); l++) {
                auto i = ((l & mask.obj_high_mask) << 1) + (l & mask.obj_low_mask);
                if ((i & mask.ctrl_mask) == mask.ctrl_mask) {
                    auto j = i + mask.obj_mask;
                    auto t1 = m[0][0] * src[i] + m[0][1] * src[j];
                    auto t2 = m[1][0] * src[i] + m[1][1] * src[j];
                    des[i] = t1;
                    des[j] = t2;
                }
            });
    }
    if (will_free) {
        derived::FreeState(&src);
    }
}

template <typename derived_, typename calc_type_>
void CPUVectorPolicyBase<derived_, calc_type_>::ApplyMatrixGate(const qs_data_p_t& src, qs_data_p_t* des_p,
                                                                const qbits_t& objs, const qbits_t& ctrls,
                                                                const std::vector<std::vector<py_qs_data_t>>& m,
                                                                index_t dim) {
    if (objs.size() == 1) {
        derived::ApplySingleQubitMatrix(src, des_p, objs[0], ctrls, m, dim);
    } else if (objs.size() == 2) {
        derived::ApplyTwoQubitsMatrix(src, des_p, objs, ctrls, m, dim);
    } else {
        derived::ApplyNQubitsMatrix(src, des_p, objs, ctrls, m, dim);
    }
}

#ifdef __x86_64__
template struct CPUVectorPolicyBase<CPUVectorPolicyAvxFloat, float>;
template struct CPUVectorPolicyBase<CPUVectorPolicyAvxDouble, double>;
#elif defined(__amd64)
template struct CPUVectorPolicyBase<CPUVectorPolicyArmFloat, float>;
template struct CPUVectorPolicyBase<CPUVectorPolicyArmDouble, double>;
#endif
}  // namespace mindquantum::sim::vector::detail
