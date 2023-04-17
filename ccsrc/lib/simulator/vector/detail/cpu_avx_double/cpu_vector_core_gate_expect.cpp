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
auto CPUVectorPolicyAvxDouble::ExpectDiffSingleQubitMatrix(qs_data_p_t bra, qs_data_p_t ket, const qbits_t& objs,
                                                           const qbits_t& ctrls, const VVT<py_qs_data_t>& m,
                                                           index_t dim) -> qs_data_t {
    SingleQubitGateMask mask(objs, ctrls);
    gate_matrix_t gate = {{m[0][0], m[0][1]}, {m[1][0], m[1][1]}};
    __m256d neg = _mm256_setr_pd(1.0, -1.0, 1.0, -1.0);
    __m256d mm[2];
    __m256d mmt[2];
    INTRIN_gene_2d_mm_and_mmt(gate, mm, mmt, neg);
    calc_type res_real = 0, res_imag = 0;
    if (!mask.ctrl_mask) {
        // clang-format off
        THRESHOLD_OMP(
            MQ_DO_PRAGMA(omp parallel for reduction(+:res_real, res_imag) schedule(static)), dim, DimTh,
                for (omp::idx_t l = 0; l < (dim / 2); l++) {
                    auto i = ((l & mask.obj_high_mask) << 1) + (l & mask.obj_low_mask);
                    auto j = i + mask.obj_mask;
                    __m256d mul_res;
                    INTRIN_M2_dot_V2(ket, i, j, mm, mmt, mul_res);
                    __m256d res;
                    INTRIN_Conj_V2_dot_V2(bra, mul_res, i, j, neg, res);
                    qs_data_t ress[2];
                    INTRIN_m256_to_host(res, ress);
                    res_real += ress[0].real() + ress[1].real();
                    res_imag += ress[0].imag() + ress[1].imag();
                });

        // clang-format on
    } else {
        if (mask.ctrl_qubits.size() == 1) {
            index_t ctrl_low = 0UL;
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
            // clang-format off
            THRESHOLD_OMP(
                MQ_DO_PRAGMA(omp parallel for reduction(+:res_real, res_imag) schedule(static)), dim, DimTh,
                    for (omp::idx_t l = 0; l < (dim / 4); l++) {
                        auto i = ((l & first_high_mask) << 1) + (l & first_low_mask);
                        i = ((i & second_high_mask) << 1) + (i & second_low_mask) + mask.ctrl_mask;
                        auto j = i + mask.obj_mask;
                        __m256d mul_res;
                        INTRIN_M2_dot_V2(ket, i, j, mm, mmt, mul_res);
                        __m256d res;
                        INTRIN_Conj_V2_dot_V2(bra, mul_res, i, j, neg, res);
                        qs_data_t ress[2];
                        INTRIN_m256_to_host(res, ress);
                        res_real += ress[0].real() + ress[1].real();
                        res_imag += ress[0].imag() + ress[1].imag();
                    });

            // clang-format on
        } else {
            // clang-format off
            THRESHOLD_OMP(
                MQ_DO_PRAGMA(omp parallel for reduction(+:res_real, res_imag) schedule(static)), dim, DimTh,
                    for (omp::idx_t l = 0; l < (dim / 2); l++) {
                        auto i = ((l & mask.obj_high_mask) << 1) + (l & mask.obj_low_mask);
                        if ((i & mask.ctrl_mask) == mask.ctrl_mask) {
                            auto j = i + mask.obj_mask;
                            __m256d mul_res;
                            INTRIN_M2_dot_V2(ket, i, j, mm, mmt, mul_res);
                            __m256d res;
                            INTRIN_Conj_V2_dot_V2(bra, mul_res, i, j, neg, res);
                            qs_data_t ress[2];
                            INTRIN_m256_to_host(res, ress);
                            res_real += ress[0].real() + ress[1].real();
                            res_imag += ress[0].imag() + ress[1].imag();
                        }
                    });

            // clang-format on
        }
    }
    return {res_real, res_imag};
};
}  // namespace mindquantum::sim::vector::detail
