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

#include "config/openmp.hpp"
#include "config/type_promotion.hpp"

#include "core/mq_base_types.hpp"
#include "core/utils.hpp"
#include "math/tensor/traits.hpp"
#include "simulator/types.hpp"

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
    static constexpr index_t DimTh = 1UL << 8;
    static constexpr qs_data_t IMAGE_MI = {0, -1};
    static constexpr qs_data_t IMAGE_I = {0, 1};
    static constexpr qs_data_t HALF_MI{0, -0.5};
    static constexpr qs_data_t HALF_I{0, 0.5};

    // basic
    // ========================================================================================================

    static index_t IdxMap(index_t x, index_t y);
    static qs_data_t GetValue(qs_data_p_t qs, index_t x, index_t y);
    static void SetValue(qs_data_p_t qs, index_t x, index_t y, qs_data_t data);
    static void SelfMultiply(qs_data_p_t qs, index_t x, index_t y, qs_data_t data);
    static void SwapValue(qs_data_p_t qs, index_t x0, index_t y0, index_t x1, index_t y1, qs_data_t coeff);

    static qs_data_p_t InitState(index_t dim, bool zero_state = true);
    static void Reset(qs_data_p_t qs, index_t dim, bool zero_state = true);
    static void FreeState(qs_data_p_t qs);
    static void Display(qs_data_p_t qs, qbit_t n_qubits, qbit_t q_limit = 10);
    static void SetToZeroExcept(qs_data_p_t qs, index_t ctrl_mask, index_t dim);
    static matrix_t GetQS(qs_data_p_t qs, index_t dim);
    static void SetQS(qs_data_p_t qs, const py_qs_datas_t& vec_out, index_t dim);
    static void SetDM(qs_data_p_t qs, const matrix_t& mat_out, index_t dim);
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
    static void ApplyTwoQubitsMatrixNoCtrl(qs_data_p_t src, qs_data_p_t des, const qbits_t& objs, const qbits_t& ctrls,
                                           const matrix_t& m, index_t dim);
    static void ApplyTwoQubitsMatrixCtrl(qs_data_p_t src, qs_data_p_t des, const qbits_t& objs, const qbits_t& ctrls,
                                         const matrix_t& m, index_t dim);
    static void ApplySWAP(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls, index_t dim);
    static void ApplyISWAP(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls, index_t dim);
    static void ApplyRxx(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls, calc_type val, index_t dim,
                         bool diff = false);
    static void ApplyRxxNoCtrl(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls, index_t dim, calc_type c,
                               qs_data_t s);
    static void ApplyRxxCtrl(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls, index_t dim, calc_type c,
                             qs_data_t s, bool diff);
    static void ApplyRyy(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls, calc_type val, index_t dim,
                         bool diff = false);
    static void ApplyRyyNoCtrl(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls, index_t dim, calc_type c,
                               qs_data_t s);
    static void ApplyRyyCtrl(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls, index_t dim, calc_type c,
                             qs_data_t s, bool diff);
    static void ApplyRzz(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls, calc_type val, index_t dim,
                         bool diff = false);
    static void ApplyRzzNoCtrl(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls, index_t dim, calc_type c,
                               calc_type s);
    static void ApplyRzzCtrl(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls, index_t dim, calc_type c,
                             calc_type s, bool diff);
    static void ApplyMatrixGate(qs_data_p_t src, qs_data_p_t des, const qbits_t& objs, const qbits_t& ctrls,
                                const matrix_t& m, index_t dim);
    // Channel operator
    // ========================================================================================================
    static void ApplySingleQubitChannel(qs_data_p_t src, qs_data_p_t des, qbit_t obj_qubit,
                                        const VT<matrix_t>& kraus_set, index_t dim);

    static void ApplyAmplitudeDamping(qs_data_p_t qs, const qbits_t& objs, calc_type gamma, bool daggered, index_t dim);
    static void ApplyPhaseDamping(qs_data_p_t qs, const qbits_t& objs, calc_type gamma, index_t dim);
    static void ApplyPauli(qs_data_p_t qs, const qbits_t& objs, const VT<double>& probs, index_t dim);
    static void ApplyKraus(qs_data_p_t qs, const qbits_t& objs, const VT<matrix_t>& kraus_set, index_t dim);

    // gate_expec
    // ========================================================================================================
    static qs_data_t ExpectDiffSingleQubitMatrix(qs_data_p_t qs, qs_data_p_t ham_matrix, const qbits_t& objs,
                                                 const qbits_t& ctrls, const matrix_t& gate_m, const matrix_t& diff_m,
                                                 index_t dim);
    static qs_data_t ExpectDiffTwoQubitsMatrix(qs_data_p_t qs, qs_data_p_t ham_matrix, const qbits_t& objs,
                                               const qbits_t& ctrls, const matrix_t& gate_m, const matrix_t& diff_m,
                                               index_t dim);
    static qs_data_t ExpectDiffMatrixGate(qs_data_p_t qs, qs_data_p_t ham_matrix, const qbits_t& objs,
                                          const qbits_t& ctrls, const matrix_t& diff_m, const matrix_t& herm_m,
                                          index_t dim);
    static qs_data_t ExpectDiffRX(qs_data_p_t qs, qs_data_p_t ham_matrix, const qbits_t& objs, const qbits_t& ctrls,
                                  index_t dim);
    static qs_data_t ExpectDiffRY(qs_data_p_t qs, qs_data_p_t ham_matrix, const qbits_t& objs, const qbits_t& ctrls,
                                  index_t dim);
    static qs_data_t ExpectDiffRZ(qs_data_p_t qs, qs_data_p_t ham_matrix, const qbits_t& objs, const qbits_t& ctrls,
                                  index_t dim);
    static qs_data_t ExpectDiffPS(qs_data_p_t qs, qs_data_p_t ham_matrix, const qbits_t& objs, const qbits_t& ctrls,
                                  index_t dim);
    static qs_data_t ExpectDiffU3Theta(qs_data_p_t qs, qs_data_p_t ham_matrix, const qbits_t& objs,
                                       const qbits_t& ctrls, calc_type phi, index_t dim);
    static qs_data_t ExpectDiffU3Phi(qs_data_p_t qs, qs_data_p_t ham_matrix, const qbits_t& objs, const qbits_t& ctrls,
                                     index_t dim);
    static qs_data_t ExpectDiffRxx(qs_data_p_t qs, qs_data_p_t ham_matrix, const qbits_t& objs, const qbits_t& ctrls,
                                   index_t dim);
    static qs_data_t ExpectDiffRyy(qs_data_p_t qs, qs_data_p_t ham_matrix, const qbits_t& objs, const qbits_t& ctrls,
                                   index_t dim);
    static qs_data_t ExpectDiffRzz(qs_data_p_t qs, qs_data_p_t ham_matrix, const qbits_t& objs, const qbits_t& ctrls,
                                   index_t dim);
    static qs_data_t ExpectDiffFSimTheta(qs_data_p_t qs, qs_data_p_t ham_matrix, const qbits_t& objs,
                                         const qbits_t& ctrls, index_t dim);
    static qs_data_t ExpectDiffFSimPhi(qs_data_p_t qs, qs_data_p_t ham_matrix, const qbits_t& objs,
                                       const qbits_t& ctrls, index_t dim);
};

template <typename policy_src, typename policy_des>
struct CastTo {
    static constexpr tensor::TDtype src_dtype = policy_src::dtype;
    static constexpr tensor::TDtype des_dtype = policy_des::dtype;
    static typename policy_des::qs_data_p_t cast(typename policy_src::qs_data_p_t qs, size_t dim) {
        if constexpr (std::is_same_v<policy_src, policy_des>) {
            return policy_des::Copy(qs, dim);
        }
        auto des = policy_des::InitState(dim, false);
        THRESHOLD_OMP_FOR(
            dim, policy_des::DimTh, for (omp::idx_t i = 0; i < (dim * dim + dim) / 2; i++) {
                des[i] = typename policy_des::qs_data_t{std::real(qs[i]), std::imag(qs[i])};
            })
        return des;
    }
};
}  // namespace mindquantum::sim::densitymatrix::detail

#endif
