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

#include "math/pr/parameter_resolver.hpp"
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
void CPUVectorPolicyBase<derived_, calc_type_>::ApplyNQubitsMatrix(qs_data_p_t src, qs_data_p_t des,
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
    for (auto& o : obj_masks) {
        std::cout << o << std::endl;
    }
    THRESHOLD_OMP_FOR(
        dim, DimTh, for (omp::idx_t l = 0; l < dim; l++) {
            if (((l & ctrl_mask) == ctrl_mask) && ((l & obj_mask) == 0)) {
                std::vector<qs_data_t> res_tmp;
                for (size_t i = 0; i < m_dim; i++) {
                    qs_data_t tmp = 0;
                    for (size_t j = 0; j < m_dim; j++) {
                        tmp += gate[i][j] * src[obj_masks[j] | l];
                    }
                    res_tmp.push_back(tmp);
                }
                for (size_t i = 0; i < m_dim; i++) {
                    des[obj_masks[i] | l] = res_tmp[i];
                }
            }
        })
}

template <typename derived_, typename calc_type_>
void CPUVectorPolicyBase<derived_, calc_type_>::ApplyTwoQubitsMatrix(qs_data_p_t src, qs_data_p_t des,
                                                                     const qbits_t& objs, const qbits_t& ctrls,
                                                                     const std::vector<std::vector<py_qs_data_t>>& gate,
                                                                     index_t dim) {
    DoubleQubitGateMask mask(objs, ctrls);
    size_t mask1 = (1UL << objs[0]);
    size_t mask2 = (1UL << objs[1]);
    if (!mask.ctrl_mask) {
        // clang-format off
        THRESHOLD_OMP_FOR(dim, DimTh,
            for (omp::idx_t l = 0; l < (dim / 4); l++) {
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
        THRESHOLD_OMP_FOR(
            dim, DimTh, for (omp::idx_t l = 0; l < (dim / 4); l++) {
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
        derived::ApplySingleQubitMatrix(src, des, objs[0], ctrls, m, dim);
    } else if (objs.size() == 2) {
        derived::ApplyTwoQubitsMatrix(src, des, objs, ctrls, m, dim);
    } else {
        derived::ApplyNQubitsMatrix(src, des, objs, ctrls, m, dim);
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
