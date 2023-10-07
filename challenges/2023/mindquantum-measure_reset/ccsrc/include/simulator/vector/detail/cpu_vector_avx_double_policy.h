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
#ifndef INCLUDE_VECTOR_DETAIL_CPU_VECTOR_AVX_DOUBLE_POLICY_HPP
#define INCLUDE_VECTOR_DETAIL_CPU_VECTOR_AVX_DOUBLE_POLICY_HPP
#include <vector>

#include "simulator/cintrin.h"
#include "simulator/vector/detail/cpu_vector_policy.h"
#define INTRIN_M2_dot_V2(ket, i, j, mm, mmt, res)                                                                      \
    do {                                                                                                               \
        __m256d v[2];                                                                                                  \
        v[0] = load2((ket) + (i));                                                                                     \
        v[1] = load2((ket) + (j));                                                                                     \
        res = add(mul(v[0], (mm)[0], (mmt)[0]), mul(v[1], (mm)[1], (mmt)[1]));                                         \
    } while (0)

#define INTRIN_Conj_V2_dot_V2(v2_bra, m256_v2, i, j, neg, res)                                                         \
    do {                                                                                                               \
        __m256d y;                                                                                                     \
        y = load((v2_bra) + (i), (v2_bra) + (j));                                                                      \
        res = _mm256_mul_pd(mul(_mm256_mul_pd(m256_v2, neg), y, _mm256_mul_pd(_mm256_permute_pd(y, 5), neg)), neg);    \
    } while (0)

#define INTRIN_m256_to_host(device_res, host_res)                                                                      \
    _mm256_storeu2_m128d(reinterpret_cast<calc_type*>(host_res), (reinterpret_cast<calc_type*>(host_res)) + 2,         \
                         device_res);

#define INTRIN_m256_to_host2(device_res, host_res_first, host_res_second)                                              \
    _mm256_storeu2_m128d(reinterpret_cast<calc_type*>(host_res_second),                                                \
                         (reinterpret_cast<calc_type*>(host_res_first)), device_res);  // NOLINT

#define INTRIN_gene_2d_mm_and_mmt(matrix, mm, mm_len, mmt, mmt_len, neg)                                               \
    do {                                                                                                               \
        for (unsigned i = 0; i < mm_len; i++) {                                                                        \
            (mm)[i] = load(&matrix[0][i], &matrix[1][i]);                                                              \
        }                                                                                                              \
        for (unsigned i = 0; i < mmt_len; ++i) {                                                                       \
            auto badc = _mm256_permute_pd((mm)[i], 5);                                                                 \
            (mmt)[i] = _mm256_mul_pd(badc, neg);                                                                       \
        }                                                                                                              \
    } while (0)

namespace mindquantum::sim::vector::detail {
struct CPUVectorPolicyAvxDouble : public CPUVectorPolicyBase<CPUVectorPolicyAvxDouble, double> {
    using gate_matrix_t = std::vector<std::vector<qs_data_t>>;
    static void ApplySingleQubitMatrix(const qs_data_p_t& src, qs_data_p_t* des_p, qbit_t obj_qubit,
                                       const qbits_t& ctrls, const std::vector<std::vector<py_qs_data_t>>& m,
                                       index_t dim);
    static qs_data_t ExpectDiffSingleQubitMatrix(const qs_data_p_t& bra, const qs_data_p_t& ket, const qbits_t& objs,
                                                 const qbits_t& ctrls, const VVT<py_qs_data_t>& m, index_t dim);
};
}  // namespace mindquantum::sim::vector::detail
#endif
