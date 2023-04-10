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

#include "core/sparse/algo.hpp"
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
auto CPUVectorPolicyBase<derived_, calc_type_>::Vdot(qs_data_p_t bra, qs_data_p_t ket, index_t dim) -> py_qs_data_t {
    calc_type res_real = 0, res_imag = 0;
    // clang-format off
    THRESHOLD_OMP(
        MQ_DO_PRAGMA(omp parallel for reduction(+:res_real, res_imag) schedule(static)), dim, DimTh,
            for (omp::idx_t i = 0; i < dim; i++) {
                res_real += bra[i].real() * ket[i].real() + bra[i].imag() * ket[i].imag();
                res_imag += bra[i].real() * ket[i].imag() - bra[i].imag() * ket[i].real();
            })
    // clang-format on
    return {res_real, res_imag};
}

template <typename derived_, typename calc_type_>
template <index_t mask, index_t condi>
auto CPUVectorPolicyBase<derived_, calc_type_>::ConditionVdot(qs_data_p_t bra, qs_data_p_t ket, index_t dim)
    -> py_qs_data_t {
    calc_type res_real = 0, res_imag = 0;
    // clang-format off
    THRESHOLD_OMP(
        MQ_DO_PRAGMA(omp parallel for reduction(+:res_real, res_imag) schedule(static)), dim, DimTh,
            for (omp::idx_t i = 0; i < dim; i++) {
                if ((i & mask) == condi) {
                    res_real += bra[i].real() * ket[i].real() + bra[i].imag() * ket[i].imag();
                    res_imag += bra[i].real() * ket[i].imag() - bra[i].imag() * ket[i].real();
                }
            })
    // clang-format on
    return {res_real, res_imag};
}

template <typename derived_, typename calc_type_>
auto CPUVectorPolicyBase<derived_, calc_type_>::OneStateVdot(qs_data_p_t bra, qs_data_p_t ket, qbit_t obj_qubit,
                                                             index_t dim) -> py_qs_data_t {
    SingleQubitGateMask mask({obj_qubit}, {});
    calc_type res_real = 0, res_imag = 0;
    // clang-format off
    THRESHOLD_OMP(
        MQ_DO_PRAGMA(omp parallel for reduction(+:res_real, res_imag) schedule(static)), dim, DimTh,
            for (omp::idx_t l = 0; l < (dim / 2); l++) {
                auto i = ((l & mask.obj_high_mask) << 1) + (l & mask.obj_low_mask) + mask.obj_mask;
                res_real += bra[i].real() * ket[i].real() + bra[i].imag() * ket[i].imag();
                res_imag += bra[i].real() * ket[i].imag() - bra[i].imag() * ket[i].real();
            })
    // clang-format on
    return {res_real, res_imag};
}

template <typename derived_, typename calc_type_>
auto CPUVectorPolicyBase<derived_, calc_type_>::ZeroStateVdot(qs_data_p_t bra, qs_data_p_t ket, qbit_t obj_qubit,
                                                              index_t dim) -> py_qs_data_t {
    SingleQubitGateMask mask({obj_qubit}, {});
    calc_type res_real = 0, res_imag = 0;
    // clang-format off
    THRESHOLD_OMP(
        MQ_DO_PRAGMA(omp parallel for reduction(+:res_real, res_imag) schedule(static)), dim, DimTh,
            for (omp::idx_t l = 0; l < (dim / 2); l++) {
                auto i = ((l & mask.obj_high_mask) << 1) + (l & mask.obj_low_mask);
                res_real += bra[i].real() * ket[i].real() + bra[i].imag() * ket[i].imag();
                res_imag += bra[i].real() * ket[i].imag() - bra[i].imag() * ket[i].real();
            })
    // clang-format on
    return {res_real, res_imag};
}

template <typename derived_, typename calc_type_>
auto CPUVectorPolicyBase<derived_, calc_type_>::CsrDotVec(const std::shared_ptr<sparse::CsrHdMatrix<calc_type>>& a,
                                                          qs_data_p_t vec, index_t dim) -> qs_data_p_t {
    if (dim != a->dim_) {
        throw std::runtime_error("Sparse hamiltonian size not match with quantum state size.");
    }
    auto out = sparse::Csr_Dot_Vec<calc_type, calc_type>(a, reinterpret_cast<calc_type*>(vec));
    return reinterpret_cast<qs_data_p_t>(out);
}

template <typename derived_, typename calc_type_>
auto CPUVectorPolicyBase<derived_, calc_type_>::CsrDotVec(const std::shared_ptr<sparse::CsrHdMatrix<calc_type>>& a,
                                                          const std::shared_ptr<sparse::CsrHdMatrix<calc_type>>& b,
                                                          qs_data_p_t vec, index_t dim) -> qs_data_p_t {
    if ((dim != a->dim_) || (dim != b->dim_)) {
        throw std::runtime_error("Sparse hamiltonian size not match with quantum state size.");
    }
    auto out = sparse::Csr_Dot_Vec<calc_type, calc_type>(a, b, reinterpret_cast<calc_type*>(vec));
    return reinterpret_cast<qs_data_p_t>(out);
}

#ifdef __x86_64__
template struct CPUVectorPolicyBase<CPUVectorPolicyAvxFloat, float>;
template struct CPUVectorPolicyBase<CPUVectorPolicyAvxDouble, double>;
#elif defined(__amd64)
template struct CPUVectorPolicyBase<CPUVectorPolicyArmFloat, float>;
template struct CPUVectorPolicyBase<CPUVectorPolicyArmDouble, double>;
#endif
}  // namespace mindquantum::sim::vector::detail
