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
#include <limits>
#include <stdexcept>

#include "config/details/macros.hpp"
#include "config/openmp.hpp"
#include "config/type_promotion.hpp"

#include "core/utils.hpp"
#include "math/pr/parameter_resolver.hpp"
#include "simulator/types.hpp"
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
auto CPUVectorPolicyBase<derived_, calc_type_>::InitState(index_t dim, bool zero_state) -> qs_data_p_t {
    if (dim == 0 || dim > (~0UL)) {
        throw std::runtime_error("Dimension too large.");
    }
    auto qs = reinterpret_cast<qs_data_p_t>(calloc(dim, sizeof(qs_data_t)));
    if (zero_state) {
        qs[0] = 1;
    }
    return qs;
}

template <typename derived_, typename calc_type_>
void CPUVectorPolicyBase<derived_, calc_type_>::Reset(qs_data_p_t qs, index_t dim) {
    THRESHOLD_OMP_FOR(
        dim, DimTh, for (omp::idx_t i = 0; i < dim; i++) { qs[i] = 0; })
    qs[0] = 1;
}

template <typename derived_, typename calc_type_>
void CPUVectorPolicyBase<derived_, calc_type_>::FreeState(qs_data_p_t qs) {
    if (qs != nullptr) {
        free(qs);
    }
}

template <typename derived_, typename calc_type_>
void CPUVectorPolicyBase<derived_, calc_type_>::Display(qs_data_p_t qs, qbit_t n_qubits, qbit_t q_limit) {
    if (n_qubits > q_limit) {
        n_qubits = q_limit;
    }
    std::cout << n_qubits << " qubits cpu simulator (little endian)." << std::endl;
    for (index_t i = 0; i < (1UL << n_qubits); i++) {
        std::cout << "(" << qs[i].real() << ", " << qs[i].imag() << ")" << std::endl;
    }
}

template <typename derived_, typename calc_type_>
void CPUVectorPolicyBase<derived_, calc_type_>::SetToZeroExcept(qs_data_p_t qs, index_t ctrl_mask, index_t dim) {
    THRESHOLD_OMP_FOR(
        dim, DimTh, for (omp::idx_t i = 0; i < dim; i++) {
            if ((i & ctrl_mask) != ctrl_mask) {
                qs[i] = 0;
            }
        })
}

template <typename derived_, typename calc_type_>
auto CPUVectorPolicyBase<derived_, calc_type_>::Copy(qs_data_p_t qs, index_t dim) -> qs_data_p_t {
    qs_data_p_t out = derived::InitState(dim, false);
    THRESHOLD_OMP_FOR(
        dim, DimTh, for (omp::idx_t i = 0; i < dim; i++) { out[i] = qs[i]; })
    return out;
};

template <typename derived_, typename calc_type_>
auto CPUVectorPolicyBase<derived_, calc_type_>::GetQS(qs_data_p_t qs, index_t dim) -> VT<py_qs_data_t> {
    VT<py_qs_data_t> out(dim);
    THRESHOLD_OMP_FOR(
        dim, DimTh, for (omp::idx_t i = 0; i < dim; i++) { out[i] = qs[i]; })
    return out;
}

template <typename derived_, typename calc_type_>
void CPUVectorPolicyBase<derived_, calc_type_>::SetQS(qs_data_p_t qs, const VT<py_qs_data_t>& qs_out, index_t dim) {
    if (qs_out.size() != dim) {
        throw std::invalid_argument("state size not match");
    }
    THRESHOLD_OMP_FOR(
        dim, DimTh, for (omp::idx_t i = 0; i < dim; i++) { qs[i] = qs_out[i]; })
}

template <typename derived_, typename calc_type_>
auto CPUVectorPolicyBase<derived_, calc_type_>::ApplyTerms(qs_data_p_t qs, const std::vector<PauliTerm<calc_type>>& ham,
                                                           index_t dim) -> qs_data_p_t {
    qs_data_p_t out = derived::InitState(dim, false);
    for (const auto& [pauli_string, coeff_] : ham) {
        auto mask = GenPauliMask(pauli_string);
        auto mask_f = mask.mask_x | mask.mask_y;
        auto coeff = coeff_;
        THRESHOLD_OMP_FOR(
            dim, DimTh, for (omp::idx_t i = 0; i < dim; i++) {
                auto j = (i ^ mask_f);
                if (i <= j) {
                    auto axis2power = CountOne(static_cast<int64_t>(i & mask.mask_z));  // -1
                    auto axis3power = CountOne(static_cast<int64_t>(i & mask.mask_y));  // -1j
                    auto c = ComplexCast<double, calc_type>::apply(
                        POLAR[static_cast<char>((mask.num_y + 2 * axis3power + 2 * axis2power) & 3)]);
                    out[j] += qs[i] * coeff * c;
                    if (i != j) {
                        out[i] += qs[j] * coeff / c;
                    }
                }
            })
    }
    return out;
};

template <typename derived_, typename calc_type>
auto CPUVectorPolicyBase<derived_, calc_type>::GroundStateOfZZs(const std::map<index_t, calc_type>& masks_value,
                                                                qbit_t n_qubits) -> calc_type {
    calc_type result = std::numeric_limits<calc_type>::max();
    auto dim = 1UL << n_qubits;
    THRESHOLD_OMP(
        MQ_DO_PRAGMA(omp parallel for reduction(min:result) schedule(static)), dim, DimTh,
                     for (omp::idx_t i = 0; i < (1UL << n_qubits); i++) {
                         calc_type ith_energy = 0;
                         for (auto& [mask, coeff] : masks_value) {
                             if (CountOne(static_cast<int64_t>(i & mask)) & 1) {
                                 ith_energy -= coeff;
                             } else {
                                 ith_energy += coeff;
                             }
                         }
                         result = std::min(result, ith_energy);
                     });
    return result;
}

#ifdef __x86_64__
template struct CPUVectorPolicyBase<CPUVectorPolicyAvxFloat, float>;
template struct CPUVectorPolicyBase<CPUVectorPolicyAvxDouble, double>;
#elif defined(__amd64)
template struct CPUVectorPolicyBase<CPUVectorPolicyArmFloat, float>;
template struct CPUVectorPolicyBase<CPUVectorPolicyArmDouble, double>;
#endif
}  // namespace mindquantum::sim::vector::detail
