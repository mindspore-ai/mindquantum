/**
 * Copyright (c) Huawei Technologies Co.,  Ltd. 2022. All rights reserved.
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
#include <cstddef>
#include <limits>
#include <stdexcept>

#include "config/details/macros.hpp"
#include "config/openmp.hpp"
#include "config/type_promotion.hpp"

#include "core/utils.hpp"
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
void CPUVectorPolicyBase<derived_, calc_type_>::Reset(qs_data_p_t* qs_p) {
    derived::FreeState(qs_p);
}

template <typename derived_, typename calc_type_>
void CPUVectorPolicyBase<derived_, calc_type_>::FreeState(qs_data_p_t* qs_p) {
    auto& qs = (*qs_p);
    if (qs != nullptr) {
        free(qs);
        qs = nullptr;
    }
}

template <typename derived_, typename calc_type_>
void CPUVectorPolicyBase<derived_, calc_type_>::Display(const qs_data_p_t& qs, qbit_t n_qubits, qbit_t q_limit) {
    if (n_qubits > q_limit) {
        n_qubits = q_limit;
    }
    std::cout << n_qubits << " qubits cpu simulator (little endian)." << std::endl;
    if (qs == nullptr) {
        std::cout << "(" << 1 << ", " << 0 << ")" << std::endl;
        for (index_t i = 0; i < (1UL << n_qubits) - 1; i++) {
            std::cout << "(" << 0 << ", " << 0 << ")" << std::endl;
        }
    } else {
        for (index_t i = 0; i < (1UL << n_qubits); i++) {
            std::cout << "(" << qs[i].real() << ", " << qs[i].imag() << ")" << std::endl;
        }
    }
}

template <typename derived_, typename calc_type_>
void CPUVectorPolicyBase<derived_, calc_type_>::SetToZeroExcept(qs_data_p_t* qs_p, index_t ctrl_mask, index_t dim) {
    auto& qs = *qs_p;
    if (qs == nullptr) {
        qs = derived::InitState(dim);
    }
    THRESHOLD_OMP_FOR(
        dim, DimTh, for (omp::idx_t i = 0; i < static_cast<omp::idx_t>(dim); i++) {
            if ((i & ctrl_mask) != ctrl_mask) {
                qs[i] = 0;
            }
        })
}

template <typename derived_, typename calc_type_>
auto CPUVectorPolicyBase<derived_, calc_type_>::Copy(const qs_data_p_t& qs, index_t dim) -> qs_data_p_t {
    qs_data_p_t out = nullptr;
    if (qs != nullptr) {
        out = derived::InitState(dim, false);
        THRESHOLD_OMP_FOR(
            dim, DimTh, for (omp::idx_t i = 0; i < static_cast<omp::idx_t>(dim); i++) { out[i] = qs[i]; })
    }
    return out;
};

template <typename derived_, typename calc_type_>
auto CPUVectorPolicyBase<derived_, calc_type_>::GetQS(const qs_data_p_t& qs, index_t dim) -> VT<py_qs_data_t> {
    VT<py_qs_data_t> out(dim);
    if (qs != nullptr) {
        THRESHOLD_OMP_FOR(
            dim, DimTh, for (omp::idx_t i = 0; i < static_cast<omp::idx_t>(dim); i++) { out[i] = qs[i]; })
    } else {
        out[0] = 1.0;
    }
    return out;
}

template <typename derived_, typename calc_type_>
void CPUVectorPolicyBase<derived_, calc_type_>::SetQS(qs_data_p_t* qs_p, const VT<py_qs_data_t>& qs_out, index_t dim) {
    auto& qs = (*qs_p);
    if (qs_out.size() != dim) {
        throw std::invalid_argument("state size not match");
    }
    if (qs == nullptr) {
        qs = derived::InitState(dim, false);
    }
    THRESHOLD_OMP_FOR(
        dim, DimTh, for (omp::idx_t i = 0; i < static_cast<omp::idx_t>(dim); i++) { qs[i] = qs_out[i]; })
}

template <typename derived_, typename calc_type_>
auto CPUVectorPolicyBase<derived_, calc_type_>::ApplyTerms(qs_data_p_t* qs_p,
                                                           const std::vector<PauliTerm<calc_type>>& ham, index_t dim)
    -> qs_data_p_t {
    auto& qs = (*qs_p);
    if (qs == nullptr) {
        qs = derived::InitState(dim);
    }
    qs_data_p_t out = derived::InitState(dim, false);
    for (const auto& [pauli_string, coeff_] : ham) {
        auto mask = GenPauliMask(pauli_string);
        auto mask_f = mask.mask_x | mask.mask_y;
        auto coeff = coeff_;
        THRESHOLD_OMP_FOR(
            dim, DimTh, for (omp::idx_t i = 0; i < static_cast<omp::idx_t>(dim); i++) {
                auto j = (i ^ mask_f);
                if (i <= j) {
                    auto axis2power = CountOne(i & mask.mask_z);  // -1
                    auto axis3power = CountOne(i & mask.mask_y);  // -1j
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

template <typename derived_, typename calc_type_>
auto CPUVectorPolicyBase<derived_, calc_type_>::ExpectationOfTerms(const qs_data_p_t& bra_out,
                                                                   const qs_data_p_t& ket_out,
                                                                   const std::vector<PauliTerm<calc_type>>& ham,
                                                                   index_t dim) -> py_qs_data_t {
    auto bra = bra_out;
    auto ket = ket_out;
    bool will_free_bra = false, will_free_ket = false;
    if (bra == nullptr) {
        bra = derived::InitState(dim);
        will_free_bra = true;
    }
    if (ket == nullptr) {
        ket = derived::InitState(dim);
        will_free_ket = true;
    }
    py_qs_data_t out = 0.0;
    for (const auto& [pauli_string, coeff_] : ham) {
        auto mask = GenPauliMask(pauli_string);
        auto mask_f = mask.mask_x | mask.mask_y;
        auto coeff = coeff_;
        calc_type res_real = 0, res_imag = 0;
        // clang-format off
        THRESHOLD_OMP(
            MQ_DO_PRAGMA(omp parallel for reduction(+:res_real, res_imag) schedule(static)), dim, DimTh,
                for (omp::idx_t i = 0; i < static_cast<omp::idx_t>(dim); i++) {
                    auto j = (i ^ mask_f);
                    if (i <= j) {
                        auto axis2power = CountOne(i & mask.mask_z);  // -1
                        auto axis3power = CountOne(i & mask.mask_y);  // -1j
                        auto c = ComplexCast<double, calc_type>::apply(
                            POLAR[static_cast<char>((mask.num_y + 2 * axis3power + 2 * axis2power) & 3)]);
                        auto tmp = std::conj(bra[j]) * ket[i] * coeff * c;
                        if (i != j) {
                            tmp += std::conj(bra[i]) * ket[j] * coeff / c;
                        }
                        res_real += std::real(tmp);
                        res_imag += std::imag(tmp);
                    }
                })
        // clang-format on
        out += py_qs_data_t(res_real, res_imag);
    }
    if (will_free_bra) {
        derived::FreeState(&bra);
    }
    if (will_free_ket) {
        derived::FreeState(&ket);
    }
    return out;
}

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
                             if (CountOne(i & mask) & 1) {
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
