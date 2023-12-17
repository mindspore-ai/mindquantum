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

#ifndef INCLUDE_VECTOR_DETAIL_GPU_VECTOR_POLICY_CUH
#define INCLUDE_VECTOR_DETAIL_GPU_VECTOR_POLICY_CUH

#include <cstddef>
#include <iostream>
#include <map>
#include <memory>
#include <vector>

#include <thrust/transform_reduce.h>

#include "core/mq_base_types.h"
#include "core/sparse/csrhdmatrix.h"
#include "math/tensor/traits.h"
#include "thrust/complex.h"
#include "thrust/functional.h"

namespace mindquantum::sim::vector::detail {
struct GPUVectorPolicyFloat;
struct GPUVectorPolicyDouble;

template <typename derived_, typename calc_type_>
struct GPUVectorPolicyBase {
    using derived = derived_;
    using calc_type = calc_type_;
    using qs_data_t = thrust::complex<calc_type>;
    using qs_data_p_t = qs_data_t*;
    using py_qs_data_t = std::complex<calc_type>;
    using py_qs_datas_t = std::vector<py_qs_data_t>;
    static qs_data_p_t InitState(index_t dim, bool zero_state = true);
    static void Reset(qs_data_p_t* qs_p);
    static void FreeState(qs_data_p_t* qs_p);
    static void Display(const qs_data_p_t& qs, qbit_t n_qubits, qbit_t q_limit = 10);
    static void SetToZeroExcept(qs_data_p_t* qs_p, index_t ctrl_mask, index_t dim);
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
    static qs_data_t ConditionalCollect(const qs_data_p_t& qs, index_t mask, index_t condi, bool abs, index_t dim);
    static py_qs_datas_t GetQS(const qs_data_p_t& qs, index_t dim);
    static void SetQS(qs_data_p_t* qs_p, const py_qs_datas_t& qs_out, index_t dim);
    static qs_data_p_t ApplyTerms(qs_data_p_t* qs_p, const std::vector<PauliTerm<calc_type>>& ham, index_t dim);
    static void ApplyPauliString(qs_data_p_t* qs_p, const PauliMask& pauli_mask, Index ctrl_mask, index_t dim);
    static py_qs_data_t ExpectationOfTerms(const qs_data_p_t& bra, const qs_data_p_t& ket,
                                           const std::vector<PauliTerm<calc_type>>& ham, index_t dim);
    static qs_data_p_t Copy(const qs_data_p_t& qs, index_t dim);
    template <index_t mask, index_t condi>
    static py_qs_data_t ConditionVdot(const qs_data_p_t& bra, const qs_data_p_t& ket, index_t dim);
    static py_qs_data_t OneStateVdot(const qs_data_p_t& bra, const qs_data_p_t& ket, qbit_t obj_qubit, index_t dim);
    static py_qs_data_t ZeroStateVdot(const qs_data_p_t& bra, const qs_data_p_t& ket, qbit_t obj_qubit, index_t dim);
    static py_qs_data_t Vdot(const qs_data_p_t& bra, const qs_data_p_t& ket, index_t dim);
    static qs_data_p_t CsrDotVec(const std::shared_ptr<sparse::CsrHdMatrix<calc_type>>& a, const qs_data_p_t& vec,
                                 index_t dim);
    static qs_data_p_t CsrDotVec(const std::shared_ptr<sparse::CsrHdMatrix<calc_type>>& a,
                                 const std::shared_ptr<sparse::CsrHdMatrix<calc_type>>& b, const qs_data_p_t& vec,
                                 index_t dim);
    static py_qs_data_t ExpectationOfCsr(const std::shared_ptr<sparse::CsrHdMatrix<calc_type>>& a,
                                         const qs_data_p_t& bra, const qs_data_p_t& ket, index_t dim);
    static py_qs_data_t ExpectationOfCsr(const std::shared_ptr<sparse::CsrHdMatrix<calc_type>>& a,
                                         const std::shared_ptr<sparse::CsrHdMatrix<calc_type>>& b,
                                         const qs_data_p_t& bra, const qs_data_p_t& ket, index_t dim);
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
    static void ApplySGate(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls, index_t dim);
    static void ApplyT(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls, index_t dim);
    static void ApplySdag(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls, index_t dim);
    static void ApplyTdag(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls, index_t dim);
    static void ApplyPS(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls, calc_type val, index_t dim,
                        bool diff = false);
    static void ApplyILike(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls, qs_data_t v1, qs_data_t v2,
                           index_t dim);

    // Single qubit operator
    // ========================================================================================================

    static void ApplySingleQubitMatrix(const qs_data_p_t& src, qs_data_p_t* des_p, qbit_t obj_qubit,
                                       const qbits_t& ctrls, const std::vector<std::vector<py_qs_data_t>>& m,
                                       index_t dim);
    static void ApplyTwoQubitsMatrix(const qs_data_p_t& src, qs_data_p_t* des_p, const qbits_t& objs,
                                     const qbits_t& ctrls, const std::vector<std::vector<py_qs_data_t>>& m,
                                     index_t dim);
    static void ApplyNQubitsMatrix(const qs_data_p_t& src, qs_data_p_t* des_p, const qbits_t& objs,
                                   const qbits_t& ctrls, const std::vector<std::vector<py_qs_data_t>>& m, index_t dim);
    static void ApplyMatrixGate(const qs_data_p_t& src, qs_data_p_t* des_p, const qbits_t& objs, const qbits_t& ctrls,
                                const std::vector<std::vector<py_qs_data_t>>& m, index_t dim);
    static void ApplyH(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls, index_t dim);
    static void ApplySX(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls, index_t dim);
    static void ApplySXdag(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls, index_t dim);
    static void ApplyGP(qs_data_p_t* qs_p, qbit_t obj_qubit, const qbits_t& ctrls, calc_type val, index_t dim,
                        bool diff = false);
    static void ApplyRX(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls, calc_type val, index_t dim,
                        bool diff = false);
    static void ApplyRY(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls, calc_type val, index_t dim,
                        bool diff = false);
    static void ApplyRZ(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls, calc_type val, index_t dim,
                        bool diff = false);
    static void ApplyRPS(qs_data_p_t* qs_p, const PauliMask& pauli_mask, Index ctrl_mask, calc_type val, index_t dim,
                         bool diff = false);
    static void ApplySWAP(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls, index_t dim);
    static void ApplyISWAP(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls, bool daggered, index_t dim);
    static void ApplySWAPalpha(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls, calc_type val, index_t dim,
                               bool diff = false);
    static void ApplyRxx(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls, calc_type val, index_t dim,
                         bool diff = false);
    static void ApplyRyy(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls, calc_type val, index_t dim,
                         bool diff = false);
    static void ApplyRzz(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls, calc_type val, index_t dim,
                         bool diff = false);
    static void ApplyRxy(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls, calc_type val, index_t dim,
                         bool diff = false);
    static void ApplyRxz(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls, calc_type val, index_t dim,
                         bool diff = false);
    static void ApplyRyz(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls, calc_type val, index_t dim,
                         bool diff = false);
    static void ApplyGivens(qs_data_p_t* qs_p, const qbits_t& objs, const qbits_t& ctrls, calc_type val, index_t dim,
                         bool diff = false);
    // gate_expec
    // ========================================================================================================
    static qs_data_t ExpectDiffRPS(const qs_data_p_t& bra, const qs_data_p_t& ket, const PauliMask& pauli_mask,
                                   Index ctrl_mask, calc_type val, index_t dim);
    static qs_data_t ExpectDiffSingleQubitMatrix(const qs_data_p_t& bra, const qs_data_p_t& ket, const qbits_t& objs,
                                                 const qbits_t& ctrls, const std::vector<py_qs_datas_t>& m,
                                                 index_t dim);
    static qs_data_t ExpectDiffTwoQubitsMatrix(const qs_data_p_t& bra, const qs_data_p_t& ket, const qbits_t& objs,
                                               const qbits_t& ctrls, const std::vector<py_qs_datas_t>& m, index_t dim);
    static qs_data_t ExpectDiffNQubitsMatrix(const qs_data_p_t& bra, const qs_data_p_t& ket, const qbits_t& objs,
                                             const qbits_t& ctrls, const std::vector<py_qs_datas_t>& m, index_t dim);
    static qs_data_t ExpectDiffMatrixGate(const qs_data_p_t& bra, const qs_data_p_t& ket, const qbits_t& objs,
                                          const qbits_t& ctrls, const std::vector<py_qs_datas_t>& m, index_t dim);
    static qs_data_t ExpectDiffRX(const qs_data_p_t& bra, const qs_data_p_t& ket, const qbits_t& objs,
                                  const qbits_t& ctrls, calc_type val, index_t dim);
    static qs_data_t ExpectDiffRY(const qs_data_p_t& bra, const qs_data_p_t& ket, const qbits_t& objs,
                                  const qbits_t& ctrls, calc_type val, index_t dim);
    static qs_data_t ExpectDiffRZ(const qs_data_p_t& bra, const qs_data_p_t& ket, const qbits_t& objs,
                                  const qbits_t& ctrls, calc_type val, index_t dim);
    static qs_data_t ExpectDiffRxx(const qs_data_p_t& bra, const qs_data_p_t& ket, const qbits_t& objs,
                                   const qbits_t& ctrls, calc_type val, index_t dim);
    static qs_data_t ExpectDiffRyy(const qs_data_p_t& bra, const qs_data_p_t& ket, const qbits_t& objs,
                                   const qbits_t& ctrls, calc_type val, index_t dim);
    static qs_data_t ExpectDiffRzz(const qs_data_p_t& bra, const qs_data_p_t& ket, const qbits_t& objs,
                                   const qbits_t& ctrls, calc_type val, index_t dim);
    static qs_data_t ExpectDiffRxy(const qs_data_p_t& bra, const qs_data_p_t& ket, const qbits_t& objs,
                                   const qbits_t& ctrls, calc_type val, index_t dim);
    static qs_data_t ExpectDiffRxz(const qs_data_p_t& bra, const qs_data_p_t& ket, const qbits_t& objs,
                                   const qbits_t& ctrls, calc_type val, index_t dim);
    static qs_data_t ExpectDiffRyz(const qs_data_p_t& bra, const qs_data_p_t& ket, const qbits_t& objs,
                                   const qbits_t& ctrls, calc_type val, index_t dim);
    static qs_data_t ExpectDiffGivens(const qs_data_p_t& bra, const qs_data_p_t& ket, const qbits_t& objs,
                                   const qbits_t& ctrls, calc_type val, index_t dim);
    static qs_data_t ExpectDiffSWAPalpha(const qs_data_p_t& bra, const qs_data_p_t& ket, const qbits_t& objs,
                                         const qbits_t& ctrls, calc_type val, index_t dim);
    static qs_data_t ExpectDiffPS(const qs_data_p_t& bra, const qs_data_p_t& ket, const qbits_t& objs,
                                  const qbits_t& ctrls, calc_type val, index_t dim);
    static qs_data_t ExpectDiffGP(const qs_data_p_t& bra, const qs_data_p_t& ket, const qbits_t& objs,
                                  const qbits_t& ctrls, calc_type val, index_t dim);
    static calc_type GroundStateOfZZs(const std::map<index_t, calc_type>& masks_value, qbit_t n_qubits);
};

template <typename qs_data_t>
struct conj_a_dot_b : public thrust::binary_function<qs_data_t, qs_data_t, qs_data_t> {
    __host__ __device__ qs_data_t operator()(qs_data_t a, qs_data_t b) {
        return thrust::conj(a) * b;
    }
};

template <typename qs_data_t, typename qs_data_p_t, typename calc_type>
__global__ void ApplyTerm(qs_data_p_t des, qs_data_p_t src, calc_type coeff, index_t num_y, index_t mask_y,
                          index_t mask_z, index_t mask_f, index_t dim) {
    index_t index = threadIdx.x + blockIdx.x * blockDim.x;
    index_t stride = blockDim.x * gridDim.x;
    for (index_t i = index; i < dim; i += stride) {
        auto j = (i ^ mask_f);
        if (i <= j) {
            auto axis2power = __popcll(i & mask_z);
            auto axis3power = __popcll(i & mask_y);
            auto idx = (num_y + 2 * axis3power + 2 * axis2power) & 3;
            auto c = qs_data_t(1, 0);
            if (idx == 1) {
                c = qs_data_t(0, 1);
            } else if (idx == 2) {
                c = qs_data_t(-1, 0);
            } else if (idx == 3) {
                c = qs_data_t(0, -1);
            }
            des[j] += src[i] * coeff * c;
            if (i != j) {
                des[i] += src[j] * coeff / c;
            }
        }
    }
}

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
        thrust::counting_iterator<index_t> i(0);
        thrust::for_each(i, i + dim, [=] __device__(index_t i) {
            des[i] = typename policy_des::qs_data_t{qs[i].real(), qs[i].imag()};
        });
        return des;
    }
};
}  // namespace mindquantum::sim::vector::detail

#endif
