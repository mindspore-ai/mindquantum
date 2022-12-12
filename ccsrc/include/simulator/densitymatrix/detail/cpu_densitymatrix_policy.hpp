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

#ifndef INCLUDE_DENSITYMATRIX_DETAIL_CPU_DENSITYMATRIX_POLICY_HPP
#define INCLUDE_DENSITYMATRIX_DETAIL_CPU_DENSITYMATRIX_POLICY_HPP

#include <complex>
#include <cstddef>
#include <iostream>
#include <vector>

#include "core/mq_base_types.hpp"
#include "simulator/types.hpp"

namespace mindquantum::sim::densitymatrix::detail {
struct CPUDensityMatrixPolicyBase {
    using qs_data_t = std::complex<calc_type>;
    using qs_data_p_t = qs_data_t*;
    using py_qs_data_t = std::complex<calc_type>;
    using py_qs_datas_t = std::vector<py_qs_data_t>;
    using matrix_t = std::vector<py_qs_datas_t>;
    static constexpr index_t DimTh = 1UL << 8;
    static constexpr qs_data_t IMAGE_MI = {0, -1};
    static constexpr qs_data_t IMAGE_I = {0, 1};

    // basic
    // ========================================================================================================

    static index_t IdxMap(index_t x, index_t y);
    static qs_data_t GetValue(qs_data_p_t qs, index_t x, index_t y);
    static void SetValue(qs_data_p_t qs, index_t x, index_t y, qs_data_t data);
    static void SelfMultiply(qs_data_p_t qs, index_t x, index_t y, qs_data_t data);
    static void SwapValue(qs_data_p_t qs, index_t x0, index_t y0, index_t x1, index_t y1, qs_data_t coeff);

    static qs_data_p_t InitState(index_t dim, bool zero_state = true);
    static void Reset(qs_data_p_t qs, index_t dim, bool zero_state = false);
    static void FreeState(qs_data_p_t qs);
    static void Display(qs_data_p_t qs, qbit_t n_qubits, qbit_t q_limit = 10);
    static void SetToZeroExcept(qs_data_p_t qs, index_t ctrl_mask, index_t dim);
    static matrix_t GetQS(qs_data_p_t qs, index_t dim);
    static void SetQS(qs_data_p_t qs, const py_qs_datas_t& qs_out, index_t dim);
    static void CopyQS(qs_data_p_t qs, const qs_data_p_t qs_out, index_t dim);
    static qs_data_p_t Copy(qs_data_p_t qs, index_t dim);
    static calc_type Purity(qs_data_p_t qs, index_t dim);
    static bool IsPure(qs_data_p_t qs, index_t dim);
    static py_qs_datas_t PureStateVector(qs_data_p_t qs, index_t dim);
    static void ApplyTerms(qs_data_p_t qs, const std::vector<PauliTerm<calc_type>>& ham, index_t dim);
    static calc_type DiagonalConditionalCollect(qs_data_p_t qs, index_t mask, index_t condi, index_t dim);
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
    static qs_data_p_t HamiltonianMatrix(const std::vector<PauliTerm<calc_type>>& ham, index_t dim);
    static qs_data_t GetExpectation(qs_data_p_t qs, const std::vector<PauliTerm<calc_type>>& ham, index_t dim);
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

    // Qubit operator
    // ========================================================================================================

    static void ApplySingleQubitMatrix(qs_data_p_t src, qs_data_p_t des, qbit_t obj_qubit, const qbits_t& ctrls,
                                       const matrix_t& m, index_t dim);
    static void ApplyH(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls, index_t dim);
    static void ApplyRX(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls, calc_type val, index_t dim,
                        bool diff = false);
    static void ApplyRY(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls, calc_type val, index_t dim,
                        bool diff = false);
    static void ApplyRZ(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls, calc_type val, index_t dim,
                        bool diff = false);

    static void ApplyTwoQubitsMatrix(qs_data_p_t src, qs_data_p_t des, const qbits_t& objs, const qbits_t& ctrls,
                                     const matrix_t& m, index_t dim);
    static void ApplySWAP(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls, index_t dim);
    static void ApplyISWAP(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls, index_t dim);
    static void ApplyXX(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls, calc_type val, index_t dim,
                        bool diff = false);
    static void ApplyYY(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls, calc_type val, index_t dim,
                        bool diff = false);
    static void ApplyZZ(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls, calc_type val, index_t dim,
                        bool diff = false);
    static void ApplyMatrixGate(qs_data_p_t src, qs_data_p_t des, const qbits_t& objs, const qbits_t& ctrls,
                                const matrix_t& m, index_t dim);
    // Channel operator
    // ========================================================================================================
    static void ApplySingleQubitChannel(qs_data_p_t src, qs_data_p_t des, qbit_t obj_qubit,
                                        const VT<matrix_t>& kraus_set, index_t dim);

    static void ApplyAmplitudeDamping(qs_data_p_t qs, const qbits_t& objs, calc_type gamma, index_t dim);
    static void ApplyPhaseDamping(qs_data_p_t qs, const qbits_t& objs, calc_type gamma, index_t dim);
    static void ApplyPauli(qs_data_p_t qs, const qbits_t& objs, const VT<calc_type>& probs, index_t dim);
    static void ApplyKraus(qs_data_p_t qs, const qbits_t& objs, const VT<matrix_t>& kraus_set, index_t dim);
    static void ApplyHermitianAmplitudeDamping(qs_data_p_t qs, const qbits_t& objs, calc_type gamma, index_t dim);

    // gate_expec
    // ========================================================================================================
    static qs_data_t ExpectDiffSingleQubitMatrix(qs_data_p_t qs, qs_data_p_t ham_matrix, const qbits_t& objs,
                                                 const qbits_t& ctrls, const matrix_t& m, index_t dim);
    static qs_data_t ExpectDiffTwoQubitsMatrix(qs_data_p_t qs, qs_data_p_t ham_matrix, const qbits_t& objs,
                                               const qbits_t& ctrls, const matrix_t& m, index_t dim);
    static qs_data_t ExpectDiffMatrixGate(qs_data_p_t qs, qs_data_p_t ham_matrix, const qbits_t& objs,
                                          const qbits_t& ctrls, const matrix_t& m, index_t dim);
    static qs_data_t ExpectDiffRX(qs_data_p_t bra, qs_data_p_t ket, const qbits_t& objs, const qbits_t& ctrls,
                                  index_t dim);
    static qs_data_t ExpectDiffRY(qs_data_p_t bra, qs_data_p_t ket, const qbits_t& objs, const qbits_t& ctrls,
                                  index_t dim);
    static qs_data_t ExpectDiffRZ(qs_data_p_t bra, qs_data_p_t ket, const qbits_t& objs, const qbits_t& ctrls,
                                  index_t dim);
    static qs_data_t ExpectDiffXX(qs_data_p_t bra, qs_data_p_t ket, const qbits_t& objs, const qbits_t& ctrls,
                                  index_t dim);
    static qs_data_t ExpectDiffYY(qs_data_p_t bra, qs_data_p_t ket, const qbits_t& objs, const qbits_t& ctrls,
                                  index_t dim);
    static qs_data_t ExpectDiffZZ(qs_data_p_t bra, qs_data_p_t ket, const qbits_t& objs, const qbits_t& ctrls,
                                  index_t dim);
    static qs_data_t ExpectDiffPS(qs_data_p_t bra, qs_data_p_t ket, const qbits_t& objs, const qbits_t& ctrls,
                                  index_t dim);
};
}  // namespace mindquantum::sim::densitymatrix::detail

#endif
