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

#include "core/parameter_resolver.hpp"
#include "simulator/utils.hpp"
#ifdef __x86_64__
#    include "simulator/vector/detail/cpu_vector_avx_double_policy.hpp"
#    include "simulator/vector/detail/cpu_vector_avx_float_policy.hpp"
#elif defined(__amd64)
#    include "simulator/vector/detail/cpu_vector_arm_double_policy.hpp"
#    include "simulator/vector/detail/cpu_vector_arm_float_policy.hpp"
#endif
#include "simulator/vector/detail/cpu_vector_policy.hpp"
namespace mindquantum::sim::vector::detail {
template <typename derived_, typename calc_type_>
void CPUVectorPolicyBase<derived_, calc_type_>::ApplyTwoQubitsMatrix(qs_data_p_t src, qs_data_p_t des,
                                                                     const qbits_t& objs, const qbits_t& ctrls,
                                                                     const std::vector<std::vector<py_qs_data_t>>& gate,
                                                                     index_t dim) {
    DoubleQubitGateMask mask(objs, ctrls);
    if (!mask.ctrl_mask) {
        // clang-format off
        THRESHOLD_OMP_FOR(dim, DimTh,
            for (omp::idx_t l = 0; l < (dim / 4); l++) {
                omp::idx_t i;
                SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask, mask.obj_high_mask, mask.obj_rev_high_mask, l,
                              i);
                auto j = i + mask.obj_min_mask;
                auto k = i + mask.obj_max_mask;
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
        THRESHOLD_OMP_FOR(
            dim, DimTh, for (omp::idx_t l = 0; l < (dim / 4); l++) {
                omp::idx_t i;
                SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask, mask.obj_high_mask, mask.obj_rev_high_mask, l,
                              i);
                if ((i & mask.ctrl_mask) == mask.ctrl_mask) {
                    auto j = i + mask.obj_min_mask;
                    auto k = i + mask.obj_max_mask;
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
}
template <typename derived_, typename calc_type_>
void CPUVectorPolicyBase<derived_, calc_type_>::ApplySingleQubitMatrix(qs_data_p_t src, qs_data_p_t des,
                                                                       qbit_t obj_qubit, const qbits_t& ctrls,
                                                                       const std::vector<std::vector<py_qs_data_t>>& m,
                                                                       index_t dim) {
    SingleQubitGateMask mask({obj_qubit}, ctrls);
    if (!mask.ctrl_mask) {
        THRESHOLD_OMP_FOR(
            dim, DimTh, for (omp::idx_t l = 0; l < (dim / 2); l++) {
                auto i = ((l & mask.obj_high_mask) << 1) + (l & mask.obj_low_mask);
                auto j = i + mask.obj_mask;
                auto t1 = m[0][0] * src[i] + m[0][1] * src[j];
                auto t2 = m[1][0] * src[i] + m[1][1] * src[j];
                des[i] = t1;
                des[j] = t2;
            })
    } else {
        THRESHOLD_OMP_FOR(
            dim, DimTh, for (omp::idx_t l = 0; l < (dim / 2); l++) {
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
}

template <typename derived_, typename calc_type_>
void CPUVectorPolicyBase<derived_, calc_type_>::ApplyMatrixGate(qs_data_p_t src, qs_data_p_t des, const qbits_t& objs,
                                                                const qbits_t& ctrls,
                                                                const std::vector<std::vector<py_qs_data_t>>& m,
                                                                index_t dim) {
    if (objs.size() == 1) {
        ApplySingleQubitMatrix(src, des, objs[0], ctrls, m, dim);
    } else if (objs.size() == 2) {
        ApplyTwoQubitsMatrix(src, des, objs, ctrls, m, dim);
    } else {
        throw std::runtime_error("Can not custom " + std::to_string(objs.size()) + " qubits gate for cpu backend.");
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
