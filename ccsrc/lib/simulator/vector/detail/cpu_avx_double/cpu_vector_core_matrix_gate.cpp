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
#include "simulator/vector/detail/cpu_vector_avx_double_policy.hpp"

namespace mindquantum::sim::vector::detail {
void CPUVectorPolicyAvxDouble::ApplySingleQubitMatrix(qs_data_p_t src, qs_data_p_t des, qbit_t obj_qubit,
                                                      const qbits_t& ctrls,
                                                      const std::vector<std::vector<py_qs_data_t>>& m, index_t dim) {
    SingleQubitGateMask mask({obj_qubit}, ctrls);
    gate_matrix_t gate = {{m[0][0], m[0][1]}, {m[1][0], m[1][1]}};
    __m256d neg = _mm256_setr_pd(1.0, -1.0, 1.0, -1.0);
    __m256d mm[2];
    __m256d mmt[2];
    INTRIN_gene_2d_mm_and_mmt(gate, mm, mmt, neg);
    if (!mask.ctrl_mask) {
        THRESHOLD_OMP_FOR(
            dim, DimTh, for (omp::idx_t l = 0; l < (dim / 2); l++) {
                auto i = ((l & mask.obj_high_mask) << 1) + (l & mask.obj_low_mask);
                auto j = i + mask.obj_mask;
                __m256d mul_res;
                INTRIN_M2_dot_V2(src, i, j, mm, mmt, mul_res);
                INTRIN_m256_to_host2(mul_res, des + i, des + j);
            })
    } else {
        THRESHOLD_OMP_FOR(
            dim, DimTh, for (omp::idx_t l = 0; l < (dim / 2); l++) {
                auto i = ((l & mask.obj_high_mask) << 1) + (l & mask.obj_low_mask);
                if ((i & mask.ctrl_mask) == mask.ctrl_mask) {
                    auto j = i + mask.obj_mask;
                    __m256d mul_res;
                    INTRIN_M2_dot_V2(src, i, j, mm, mmt, mul_res);
                    INTRIN_m256_to_host2(mul_res, des + i, des + j);
                }
            });
    }
}
}  // namespace mindquantum::sim::vector::detail
