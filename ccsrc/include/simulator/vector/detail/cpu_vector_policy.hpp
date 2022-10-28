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

#ifndef INCLUDE_VECTOR_DETAIL_CPU_VECTOR_POLICY_HPP
#define INCLUDE_VECTOR_DETAIL_CPU_VECTOR_POLICY_HPP

#include <complex>
#include <cstddef>
#include <iostream>
#include <memory>
#include <vector>

#include "core/mq_base_types.hpp"
#include "core/sparse/csrhdmatrix.hpp"
#include "simulator/types.hpp"

#ifdef INTRIN
#    include "projectq/backends/_sim/_cppkernels/intrin/alignedallocator.hpp"
#    include "projectq/backends/_sim/_cppkernels/intrin/cintrin.hpp"

namespace mindquantum::sim::vector::intrin {
#    define INTRIN_M2_dot_V2(ket, i, j, mm, mmt, res)                                                                  \
        do {                                                                                                           \
            __m256d v[2];                                                                                              \
            v[0] = load2(ket + (i));                                                                                   \
            v[1] = load2(ket + (j));                                                                                   \
            res = add(mul(v[0], mm[0], mmt[0]), mul(v[1], mm[1], mmt[1]));                                             \
        } while (0)

#    define INTRIN_Conj_V2_dot_V2(v2_bra, m256_v2, i, j, neg, res)                                                     \
        do {                                                                                                           \
            __m256d y;                                                                                                 \
            y = load(v2_bra + (i), v2_bra + (j));                                                                      \
            res = _mm256_mul_pd(mul(_mm256_mul_pd(m256_v2, neg), y, _mm256_mul_pd(_mm256_permute_pd(y, 5), neg)),      \
                                neg);                                                                                  \
        } while (0)

#    define INTRIN_m256_to_host(device_res, host_res)                                                                  \
        _mm256_storeu2_m128d(reinterpret_cast<calc_type*>(host_res), (reinterpret_cast<calc_type*>(host_res)) + 2,     \
                             device_res);

#    define INTRIN_m256_to_host2(device_res, host_res_first, host_res_second)                                          \
        _mm256_storeu2_m128d(reinterpret_cast<calc_type*>(host_res_second),                                            \
                             (reinterpret_cast<calc_type*>(host_res_first)), device_res);  // NOLINT

#    define INTRIN_gene_2d_mm_and_mmt(matrix, mm, mmt, neg)                                                            \
        do {                                                                                                           \
            mm[0] = load(&matrix[0][0], &matrix[1][0]);                                                                \
            mm[1] = load(&matrix[0][1], &matrix[1][1]);                                                                \
            for (unsigned i = 0; i < 2; ++i) {                                                                         \
                auto badc = _mm256_permute_pd(mm[i], 5);                                                               \
                mmt[i] = _mm256_mul_pd(badc, neg);                                                                     \
            }                                                                                                          \
        } while (0)
}  // namespace mindquantum::sim::vector::intrin
#endif
namespace mindquantum::sim::vector::detail {
struct CPUVectorPolicyBase {
    using qs_data_t = std::complex<calc_type>;
    using qs_data_p_t = qs_data_t*;
    using py_qs_data_t = std::complex<calc_type>;
    using py_qs_datas_t = std::vector<py_qs_data_t>;
    static constexpr index_t DimTh = 1UL << 13;

#ifdef INTRIN
    using gate_matrix_t = std::vector<std::vector<qs_data_t, aligned_allocator<qs_data_t, 64>>>;
#endif
    static constexpr qs_data_t IMAGE_MI = {0, -1};
    static constexpr qs_data_t IMAGE_I = {0, 1};

    // basic
    // ========================================================================================================

    static qs_data_p_t InitState(index_t dim, bool zero_state = true);
    static void Reset(qs_data_p_t qs, index_t dim);
    static void FreeState(qs_data_p_t qs);
    static void Display(qs_data_p_t qs, qbit_t n_qubits, qbit_t q_limit = 10);
    static void SetToZeroExcept(qs_data_p_t qs, index_t ctrl_mask, index_t dim);
    template <index_t mask, index_t condi, class binary_op>
    static void ConditionalBinary(qs_data_p_t src, qs_data_p_t des, qs_data_t succ_coeff, qs_data_t fail_coeff,
                                  index_t dim, const binary_op& op);
    template <class binary_op>
    static void ConditionalBinary(qs_data_p_t src, qs_data_p_t des, index_t mask, index_t condi, qs_data_t succ_coeff,
                                  qs_data_t fail_coeff, index_t dim, const binary_op& op);
    static void ConditionalAdd(qs_data_p_t src, qs_data_p_t des, index_t mask, index_t condi, qs_data_t succ_coeff,
                               qs_data_t fail_coeff, index_t dim);
    static void ConditionalMinus(qs_data_p_t src, qs_data_p_t des, index_t mask, index_t condi, qs_data_t succ_coeff,
                                 qs_data_t fail_coeff, index_t dim);
    static void ConditionalMul(qs_data_p_t src, qs_data_p_t des, index_t mask, index_t condi, qs_data_t succ_coeff,
                               qs_data_t fail_coeff, index_t dim);
    static void ConditionalDiv(qs_data_p_t src, qs_data_p_t des, index_t mask, index_t condi, qs_data_t succ_coeff,
                               qs_data_t fail_coeff, index_t dim);
    static void QSMulValue(qs_data_p_t src, qs_data_p_t des, qs_data_t value, index_t dim);
    static qs_data_t ConditionalCollect(qs_data_p_t qs, index_t mask, index_t condi, bool abs, index_t dim);
    static py_qs_datas_t GetQS(qs_data_p_t qs, index_t dim);
    static void SetQS(qs_data_p_t qs, const py_qs_datas_t& qs_out, index_t dim);
    static qs_data_p_t ApplyTerms(qs_data_p_t qs, const std::vector<PauliTerm<calc_type>>& ham, index_t dim);
    static qs_data_p_t Copy(qs_data_p_t qs, index_t dim);
    template <index_t mask, index_t condi>
    static py_qs_data_t ConditionVdot(qs_data_p_t bra, qs_data_p_t ket, index_t dim);
    static py_qs_data_t OneStateVdot(qs_data_p_t bra, qs_data_p_t ket, qbit_t obj_qubit, index_t dim);
    static py_qs_data_t ZeroStateVdot(qs_data_p_t bra, qs_data_p_t ket, qbit_t obj_qubit, index_t dim);
    static py_qs_data_t Vdot(qs_data_p_t bra, qs_data_p_t ket, index_t dim);
    static qs_data_p_t CsrDotVec(const std::shared_ptr<sparse::CsrHdMatrix<calc_type>>& a, qs_data_p_t vec,
                                 index_t dim);
    static qs_data_p_t CsrDotVec(const std::shared_ptr<sparse::CsrHdMatrix<calc_type>>& a,
                                 const std::shared_ptr<sparse::CsrHdMatrix<calc_type>>& b, qs_data_p_t vec,
                                 index_t dim);
    // X like operator
    // ========================================================================================================

    static void ApplyXLike(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls, qs_data_t v1, qs_data_t v2,
                           index_t dim);
    static void ApplyX(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls, index_t dim);
    static void ApplyY(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls, index_t dim);

    // Z like operator
    // ========================================================================================================

    static void ApplyZLike(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls, qs_data_t val, index_t dim);
    static void ApplyZ(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls, index_t dim);
    // The crazy code spell check in CI do not allow apply s to name following API, even I set the filter file.
    static void ApplySGate(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls, index_t dim);
    static void ApplyT(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls, index_t dim);
    static void ApplySdag(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls, index_t dim);
    static void ApplyTdag(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls, index_t dim);
    static void ApplyPS(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls, calc_type val, index_t dim,
                        bool diff = false);

    // Single qubit operator
    // ========================================================================================================

    static void ApplySingleQubitMatrix(qs_data_p_t src, qs_data_p_t des, qbit_t obj_qubit, const qbits_t& ctrls,
                                       const std::vector<std::vector<py_qs_data_t>>& m, index_t dim);
    static void ApplyTwoQubitsMatrix(qs_data_p_t src, qs_data_p_t des, const qbits_t& objs, const qbits_t& ctrls,
                                     const std::vector<std::vector<py_qs_data_t>>& m, index_t dim);
    static void ApplyMatrixGate(qs_data_p_t src, qs_data_p_t des, const qbits_t& objs, const qbits_t& ctrls,
                                const std::vector<std::vector<py_qs_data_t>>& m, index_t dim);
    static void ApplyH(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls, index_t dim);
    static void ApplyGP(qs_data_p_t qs, qbit_t obj_qubit, const qbits_t& ctrls, calc_type val, index_t dim,
                        bool diff = false);
    static void ApplyRX(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls, calc_type val, index_t dim,
                        bool diff = false);
    static void ApplyRY(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls, calc_type val, index_t dim,
                        bool diff = false);
    static void ApplyRZ(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls, calc_type val, index_t dim,
                        bool diff = false);

    static void ApplySWAP(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls, index_t dim);
    static void ApplyISWAP(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls, bool daggered, index_t dim);
    static void ApplyXX(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls, calc_type val, index_t dim,
                        bool diff = false);
    static void ApplyYY(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls, calc_type val, index_t dim,
                        bool diff = false);
    static void ApplyZZ(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls, calc_type val, index_t dim,
                        bool diff = false);

    // gate_expec
    // ========================================================================================================
    static qs_data_t ExpectDiffSingleQubitMatrix(qs_data_p_t bra, qs_data_p_t ket, const qbits_t& objs,
                                                 const qbits_t& ctrls, const std::vector<py_qs_datas_t>& m,
                                                 index_t dim);
    static qs_data_t ExpectDiffTwoQubitsMatrix(qs_data_p_t bra, qs_data_p_t ket, const qbits_t& objs,
                                               const qbits_t& ctrls, const std::vector<py_qs_datas_t>& m, index_t dim);
    static qs_data_t ExpectDiffMatrixGate(qs_data_p_t bra, qs_data_p_t ket, const qbits_t& objs, const qbits_t& ctrls,
                                          const std::vector<py_qs_datas_t>& m, index_t dim);
    static qs_data_t ExpectDiffRX(qs_data_p_t bra, qs_data_p_t ket, const qbits_t& objs, const qbits_t& ctrls,
                                  calc_type val, index_t dim);
    static qs_data_t ExpectDiffRY(qs_data_p_t bra, qs_data_p_t ket, const qbits_t& objs, const qbits_t& ctrls,
                                  calc_type val, index_t dim);
    static qs_data_t ExpectDiffRZ(qs_data_p_t bra, qs_data_p_t ket, const qbits_t& objs, const qbits_t& ctrls,
                                  calc_type val, index_t dim);
    static qs_data_t ExpectDiffXX(qs_data_p_t bra, qs_data_p_t ket, const qbits_t& objs, const qbits_t& ctrls,
                                  calc_type val, index_t dim);
    static qs_data_t ExpectDiffYY(qs_data_p_t bra, qs_data_p_t ket, const qbits_t& objs, const qbits_t& ctrls,
                                  calc_type val, index_t dim);
    static qs_data_t ExpectDiffZZ(qs_data_p_t bra, qs_data_p_t ket, const qbits_t& objs, const qbits_t& ctrls,
                                  calc_type val, index_t dim);
    static qs_data_t ExpectDiffPS(qs_data_p_t bra, qs_data_p_t ket, const qbits_t& objs, const qbits_t& ctrls,
                                  calc_type val, index_t dim);
    static qs_data_t ExpectDiffGP(qs_data_p_t bra, qs_data_p_t ket, const qbits_t& objs, const qbits_t& ctrls,
                                  calc_type val, index_t dim);
};
}  // namespace mindquantum::sim::vector::detail

#endif
