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

#ifndef INCLUDE_DENSITYMATRIX_DETAIL_CPU_DENSITYMATRIX_POLICY_HPP
#define INCLUDE_DENSITYMATRIX_DETAIL_CPU_DENSITYMATRIX_POLICY_HPP

#include <complex>
#include <cstddef>
#include <iostream>
#include <memory>
#include <vector>

#include "config/openmp.h"
#include "config/type_promotion.h"
#include "core/mq_base_types.h"
#include "core/utils.h"
#include "math/tensor/ops_cpu/utils.h"
#include "ops/hamiltonian.h"

// Warning: only correct when x >= y
#define IdxMap(x, y)       (((x) * ((x) + 1)) / 2 + (y))
#define GetValue(qs, x, y) (((x) >= (y)) ? ((qs)[IdxMap((x), (y))]) : (std::conj((qs)[IdxMap((y), (x))])))
#define SetValue(qs, x, y, data)                                                                                       \
    (((x) >= (y)) ? ((qs)[IdxMap((x), (y))] = (data)) : ((qs)[IdxMap((y), (x))] = std::conj((data))))
#define SelfMultiply(qs, x, y, data)                                                                                   \
    (((x) >= (y)) ? ((qs)[IdxMap((x), (y))] *= (data)) : ((qs)[IdxMap((y), (x))] *= std::conj((data))))

namespace mindquantum::sim::densitymatrix::detail {
struct CPUDensityMatrixPolicyAvxFloat;
struct CPUDensityMatrixPolicyAvxDouble;

template <typename derived_, typename calc_type_>
struct CPUDensityMatrixPolicyBase {
    using derived = derived_;
    using calc_type = calc_type_;
    using qs_data_t = std::complex<calc_type>;
    using qs_data_p_t = qs_data_t*;
    using py_qs_data_t = std::complex<calc_type>;
    using py_qs_datas_t = std::vector<py_qs_data_t>;
    using matrix_t = std::vector<py_qs_datas_t>;
    static constexpr index_t DimTh = static_cast<uint64_t>(1) << 8;
    static constexpr qs_data_t IMAGE_MI = {0, -1};
    static constexpr qs_data_t IMAGE_I = {0, 1};
    static constexpr qs_data_t HALF_MI{0, -0.5};
    static constexpr qs_data_t HALF_I{0, 0.5};

    // basic
    // ========================================================================================================

    static void SwapValue(qs_data_p_t qs, index_t x0, index_t y0, index_t x1, index_t y1, qs_data_t coeff);

    static qs_data_p_t InitState(index_t dim, bool zero_state = true);
    static void Reset(qs_data_p_t* qs_p);
    static void FreeState(qs_data_p_t* qs_p);
    static void Display(const qs_data_p_t& qs, qbit_t n_qubits, qbit_t q_limit = 10);
    static void SetToZeroExcept(qs_data_p_t* qs_p, index_t ctrl_mask, index_t dim);
    static matrix_t GetQS(const qs_data_p_t& qs, index_t dim);
    static void SetQS(qs_data_p_t* qs_p, const py_qs_datas_t& vec_out, index_t dim);
    static void SetDM(qs_data_p_t* qs_p, const matrix_t& mat_out, index_t dim);
    static void CopyQS(qs_data_p_t* qs_des, const qs_data_p_t& qs_src, index_t dim);
    static qs_data_p_t Copy(const qs_data_p_t& qs, index_t dim);
    static calc_type Purity(const qs_data_p_t& qs, index_t dim);
    static matrix_t GetPartialTrace(const qs_data_p_t& qs, const qbits_t& objs, index_t dim);
    static py_qs_datas_t PureStateVector(const qs_data_p_t& qs, index_t dim);
    static void ApplyTerms(qs_data_p_t* qs_p, const std::vector<PauliTerm<calc_type>>& ham, index_t dim);
    static void ApplyPauliString(qs_data_p_t* qs_p, const PauliMask& pauli_mask, Index ctrl_mask, index_t dim);
    static void ApplyPauliStringNoCtrl(qs_data_p_t* qs_p, const PauliMask& pauli_mask, index_t dim);
    static void ApplyPauliStringWithCtrl(qs_data_p_t* qs_p, const PauliMask& pauli_mask, Index ctrl_mask, index_t dim);
    static void ApplyCsr(qs_data_p_t* qs_p, const std::shared_ptr<sparse::CsrHdMatrix<calc_type>>& a, index_t dim);
    static calc_type DiagonalConditionalCollect(const qs_data_p_t& qs, index_t mask, index_t condi, index_t dim);

    template <index_t mask, index_t condi, class binary_op>
    static void ConditionalBinary(const qs_data_p_t& src, qs_data_p_t* des_p, qs_data_t succ_coeff,
                                  qs_data_t fail_coeff, index_t dim, const binary_op& op);
    template <class binary_op>
    static void ConditionalBinary(const qs_data_p_t& src, qs_data_p_t* des_p, index_t mask, index_t condi,
                                  qs_data_t succ_coeff, qs_data_t fail_coeff, index_t dim, const binary_op& op);
    static void ConditionalAdd(const qs_data_p_t& src, qs_data_p_t* des_p, index_t mask, index_t condi,
                               qs_data_t succ_coeff, qs_data_t fail_coeff, index_t dim);
    static void ConditionalMinus(const qs_data_p_t& src, qs_data_p_t* des_p, index_t mask, index_t condi,
                                 qs_data_t succ_coeff, qs_data_t fail_coeff, index_t dim);
    static void ConditionalMul(const qs_data_p_t& src, qs_data_p_t* des_p, index_t mask, index_t condi,
                               qs_data_t succ_coeff, qs_data_t fail_coeff, index_t dim);
    static void ConditionalDiv(const qs_data_p_t& src, qs_data_p_t* des_p, index_t mask, index_t condi,
                               qs_data_t succ_coeff, qs_data_t fail_coeff, index_t dim);
    static void QSMulValue(const qs_data_p_t& src, qs_data_p_t* des_p, qs_data_t value, index_t dim);
    static qs_data_p_t HamiltonianMatrix(const Hamiltonian<calc_type>& ham, index_t dim);
    static qs_data_p_t TermsToMatrix(const std::vector<PauliTerm<calc_type>>& ham, index_t dim);
    static qs_data_p_t CsrToMatrix(const std::shared_ptr<sparse::CsrHdMatrix<calc_type>>& a, index_t dim);
    static qs_data_t ExpectationOfTerms(const qs_data_p_t& qs, const std::vector<PauliTerm<calc_type>>& ham,
                                        index_t dim);
    static qs_data_t ExpectationOfCsr(const qs_data_p_t& qs_out,
                                      const std::shared_ptr<sparse::CsrHdMatrix<calc_type>>& a, index_t dim);
    // X like operator
    // ========================================================================================================

    static void ApplyXLike(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls, qs_data_t v1, qs_data_t v2,
                           index_t dim);
    static void ApplyX(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls, index_t dim);
    static void ApplyY(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls, index_t dim);

    // Z like operator
    // ========================================================================================================

    static void ApplyZLike(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls, qs_data_t val, index_t dim);
    static void ApplyZ(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls, index_t dim);
    static void ApplyRPS(qs_data_p_t* qs_p, const PauliMask& pauli_mask, Index ctrl_mask, calc_type val, index_t dim,
                         bool diff = false);
    static void ApplyRPSNoCtrl(qs_data_p_t* qs_p, const PauliMask& pauli_mask, calc_type val, index_t dim,
                               bool diff = false);
    static void ApplyRPSWithCtrl(qs_data_p_t* qs_p, const PauliMask& pauli_mask, Index ctrl_mask, calc_type val,
                                 index_t dim, bool diff = false);
    // The crazy code spell check in CI do not allow apply s to name following API, even I set the filter file.
    static void ApplySGate(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls, index_t dim);
    static void ApplyT(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls, index_t dim);
    static void ApplySdag(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls, index_t dim);
    static void ApplyTdag(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls, index_t dim);
    static void ApplyPS(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls, calc_type val, index_t dim,
                        bool diff = false);

    // Qubit operator
    // ========================================================================================================

    static void ApplySingleQubitMatrix(const qs_data_p_t& src, qs_data_p_t* des_p, qbit_t obj_qubit,
                                       const qbits_t& ctrls, const matrix_t& m, index_t dim);
    static void ApplyH(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls, index_t dim);
    static void ApplySX(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls, index_t dim);
    static void ApplySXdag(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls, index_t dim);
    static void ApplyGP(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls, calc_type val, index_t dim,
                        bool diff = false);
    static void ApplyRX(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls, calc_type val, index_t dim,
                        bool diff = false);
    static void ApplyRY(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls, calc_type val, index_t dim,
                        bool diff = false);
    static void ApplyRZ(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls, calc_type val, index_t dim,
                        bool diff = false);

    static void ApplyTwoQubitsMatrix(const qs_data_p_t& src, qs_data_p_t* des_p, const qbits_t& objs,
                                     const qbits_t& ctrls, const matrix_t& m, index_t dim);
    static void ApplyTwoQubitsMatrixNoCtrl(const qs_data_p_t& src, qs_data_p_t* des_p, const qbits_t& objs,
                                           const qbits_t& ctrls, const matrix_t& m, index_t dim);
    static void ApplyTwoQubitsMatrixCtrl(const qs_data_p_t& src, qs_data_p_t* des_p, const qbits_t& objs,
                                         const qbits_t& ctrls, const matrix_t& m, index_t dim);
    static void ApplyNQubitsMatrix(const qs_data_p_t& src, qs_data_p_t* des_p, const qbits_t& objs,
                                   const qbits_t& ctrls, const matrix_t& m, index_t dim);
    static void NQubitsMatrixMul(const qs_data_p_t& src, const qs_data_p_t& des, std::vector<size_t> obj_masks,
                                 size_t ctrl_mask, const matrix_t& gate, index_t dim, size_t m_dim);
    static void ApplySWAP(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls, index_t dim);
    static void ApplyISWAP(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls, bool daggered, index_t dim);
    static void ApplyISWAPNoCtrl(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls, bool daggered,
                                 index_t dim);
    static void ApplyISWAPCtrl(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls, bool daggered,
                               index_t dim);
    static void ApplySWAPalpha(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls, calc_type val, index_t dim,
                               bool diff = false);
    static void ApplyGivens(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls, calc_type val, index_t dim,
                            bool diff = false);
    static void ApplyRxx(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls, calc_type val, index_t dim,
                         bool diff = false);
    static void ApplyRxxNoCtrl(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls, index_t dim, calc_type c,
                               qs_data_t s);
    static void ApplyRxxCtrl(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls, index_t dim, calc_type c,
                             qs_data_t s, bool diff);
    static void ApplyRyy(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls, calc_type val, index_t dim,
                         bool diff = false);
    static void ApplyRyyNoCtrl(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls, index_t dim, calc_type c,
                               qs_data_t s);
    static void ApplyRyyCtrl(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls, index_t dim, calc_type c,
                             qs_data_t s, bool diff);
    static void ApplyRzz(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls, calc_type val, index_t dim,
                         bool diff = false);
    static void ApplyRzzNoCtrl(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls, index_t dim, calc_type c,
                               calc_type s);
    static void ApplyRzzCtrl(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls, index_t dim, calc_type c,
                             calc_type s, bool diff);
    static void ApplyRxy(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls, calc_type val, index_t dim,
                         bool diff = false);
    static void ApplyRxyNoCtrl(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls, index_t dim, calc_type c,
                               qs_data_t s);
    static void ApplyRxyCtrl(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls, index_t dim, calc_type c,
                             qs_data_t s, bool diff);
    static void ApplyRxz(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls, calc_type val, index_t dim,
                         bool diff = false);
    static void ApplyRxzNoCtrl(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls, index_t dim, calc_type c,
                               qs_data_t s);
    static void ApplyRxzCtrl(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls, index_t dim, calc_type c,
                             qs_data_t s, bool diff);
    static void ApplyRyz(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls, calc_type val, index_t dim,
                         bool diff = false);
    static void ApplyRyzNoCtrl(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls, index_t dim, calc_type c,
                               qs_data_t s);
    static void ApplyRyzCtrl(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls, index_t dim, calc_type c,
                             qs_data_t s, bool diff);
    static void ApplyMatrixGate(const qs_data_p_t& src, qs_data_p_t* des_p, const qbits_t& objs, const qbits_t& ctrls,
                                const matrix_t& m, index_t dim);
    // Channel operator
    // ========================================================================================================
    static void ApplySingleQubitChannel(const qs_data_p_t& src, qs_data_p_t* des_p, qbit_t obj_qubit,
                                        const VT<matrix_t>& kraus_set, index_t dim);

    static void ApplyAmplitudeDamping(qs_data_p_t* qs_p, const qbits_t& objs, calc_type gamma, bool daggered,
                                      index_t dim);
    static void ApplyPhaseDamping(qs_data_p_t* qs_p, const qbits_t& objs, calc_type gamma, index_t dim);
    static void ApplyPauli(qs_data_p_t* qs_p, const qbits_t& objs, const VT<double>& probs, index_t dim);
    static void ApplyDepolarizing(qs_data_p_t* qs_p, const qbits_t& objs, calc_type prob, index_t dim);
    static void ApplyKraus(qs_data_p_t* qs_p, const qbits_t& objs, const VT<matrix_t>& kraus_set, index_t dim);
    static void ApplyThermalRelaxation(qs_data_p_t* qs_p, const qbits_t& objs, calc_type t1, calc_type t2,
                                       calc_type gate_time, index_t dim);

    // gate_expec
    // ========================================================================================================
    static qs_data_t ExpectDiffRPS(const qs_data_p_t& qs, const qs_data_p_t& ham_matrix, const PauliMask& pauli_mask,
                                   Index ctrl_mask, index_t dim);
    static qs_data_t ExpectDiffSingleQubitMatrix(const qs_data_p_t& qs_out, const qs_data_p_t& ham_matrix_out,
                                                 const qbits_t& objs, const qbits_t& ctrls, const matrix_t& gate_m,
                                                 const matrix_t& diff_m, index_t dim);
    static qs_data_t ExpectDiffTwoQubitsMatrix(const qs_data_p_t& qs_out, const qs_data_p_t& ham_matrix_out,
                                               const qbits_t& objs, const qbits_t& ctrls, const matrix_t& gate_m,
                                               const matrix_t& diff_m, index_t dim);
    static qs_data_t ExpectDiffNQubitsMatrix(const qs_data_p_t& qs_out, const qs_data_p_t& ham_matrix_out,
                                             const qbits_t& objs, const qbits_t& ctrls, const matrix_t& gate_m,
                                             const matrix_t& diff_m, index_t dim);
    static qs_data_t ExpectDiffMatrixGate(const qs_data_p_t& qs, const qs_data_p_t& ham_matrix, const qbits_t& objs,
                                          const qbits_t& ctrls, const matrix_t& diff_m, const matrix_t& herm_m,
                                          index_t dim);
    static qs_data_t ExpectDiffRX(const qs_data_p_t& qs_out, const qs_data_p_t& ham_matrix_out, const qbits_t& objs,
                                  const qbits_t& ctrls, index_t dim);
    static qs_data_t ExpectDiffRY(const qs_data_p_t& qs_out, const qs_data_p_t& ham_matrix_out, const qbits_t& objs,
                                  const qbits_t& ctrls, index_t dim);
    static qs_data_t ExpectDiffRZ(const qs_data_p_t& qs_out, const qs_data_p_t& ham_matrix_out, const qbits_t& objs,
                                  const qbits_t& ctrls, index_t dim);
    static qs_data_t ExpectDiffPS(const qs_data_p_t& qs_out, const qs_data_p_t& ham_matrix_out, const qbits_t& objs,
                                  const qbits_t& ctrls, index_t dim);
    static qs_data_t ExpectDiffGP(const qs_data_p_t& qs_out, const qs_data_p_t& ham_matrix_out, const qbits_t& objs,
                                  const qbits_t& ctrls, index_t dim);
    static qs_data_t ExpectDiffU3Theta(const qs_data_p_t& qs_out, const qs_data_p_t& ham_matrix_out,
                                       const qbits_t& objs, const qbits_t& ctrls, calc_type phi, index_t dim);
    static qs_data_t ExpectDiffU3Phi(const qs_data_p_t& qs_out, const qs_data_p_t& ham_matrix_out, const qbits_t& objs,
                                     const qbits_t& ctrls, index_t dim);
    static qs_data_t ExpectDiffRxx(const qs_data_p_t& qs_out, const qs_data_p_t& ham_matrix_out, const qbits_t& objs,
                                   const qbits_t& ctrls, index_t dim);
    static qs_data_t ExpectDiffRyy(const qs_data_p_t& qs_out, const qs_data_p_t& ham_matrix_out, const qbits_t& objs,
                                   const qbits_t& ctrls, index_t dim);
    static qs_data_t ExpectDiffRzz(const qs_data_p_t& qs_out, const qs_data_p_t& ham_matrix_out, const qbits_t& objs,
                                   const qbits_t& ctrls, index_t dim);
    static qs_data_t ExpectDiffRxy(const qs_data_p_t& qs_out, const qs_data_p_t& ham_matrix_out, const qbits_t& objs,
                                   const qbits_t& ctrls, index_t dim);
    static qs_data_t ExpectDiffRxz(const qs_data_p_t& qs_out, const qs_data_p_t& ham_matrix_out, const qbits_t& objs,
                                   const qbits_t& ctrls, index_t dim);
    static qs_data_t ExpectDiffRyz(const qs_data_p_t& qs_out, const qs_data_p_t& ham_matrix_out, const qbits_t& objs,
                                   const qbits_t& ctrls, index_t dim);
    static qs_data_t ExpectDiffSWAPalpha(const qs_data_p_t& qs_out, const qs_data_p_t& ham_matrix_out,
                                         const qbits_t& objs, const qbits_t& ctrls, index_t dim);
    static qs_data_t ExpectDiffGivens(const qs_data_p_t& qs_out, const qs_data_p_t& ham_matrix_out, const qbits_t& objs,
                                      const qbits_t& ctrls, index_t dim);
    static qs_data_t ExpectDiffFSimTheta(const qs_data_p_t& qs_out, const qs_data_p_t& ham_matrix_out,
                                         const qbits_t& objs, const qbits_t& ctrls, index_t dim);
    static qs_data_t ExpectDiffFSimPhi(const qs_data_p_t& qs_out, const qs_data_p_t& ham_matrix_out,
                                       const qbits_t& objs, const qbits_t& ctrls, index_t dim);
};

template <typename policy_src, typename policy_des>
struct CastTo {
    static typename policy_des::qs_data_p_t cast(typename policy_src::qs_data_p_t qs, size_t dim) {
        if (qs == nullptr) {
            return nullptr;
        }
        if constexpr (std::is_same_v<policy_src, policy_des>) {
            return policy_des::Copy(qs, dim);
        }
        auto des = policy_des::InitState(dim, false);
        auto caster = tensor::cast_value<typename policy_src::calc_type, typename policy_des::calc_type>();
        THRESHOLD_OMP_FOR(
            dim, policy_des::DimTh, for (omp::idx_t i = 0; i < static_cast<omp::idx_t>((dim * dim + dim) / 2); i++) {
                des[i] = typename policy_des::qs_data_t{caster(std::real(qs[i])), caster(std::imag(qs[i]))};
            })
        return des;
    }
};
}  // namespace mindquantum::sim::densitymatrix::detail

#endif
