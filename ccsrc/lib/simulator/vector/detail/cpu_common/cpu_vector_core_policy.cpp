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
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <stdexcept>

#include "config/details/macros.h"
#include "config/openmp.h"
#include "config/type_promotion.h"
#include "core/mq_base_types.h"
#include "core/utils.h"
#include "math/pr/parameter_resolver.h"
#include "simulator/utils.h"
#ifdef __x86_64__
#    include "simulator/vector/detail/cpu_vector_avx_double_policy.h"
#    include "simulator/vector/detail/cpu_vector_avx_float_policy.h"
#elif defined(__amd64)
#    include "simulator/vector/detail/cpu_vector_arm_double_policy.h"
#    include "simulator/vector/detail/cpu_vector_arm_float_policy.h"
#endif
#include "simulator/vector/detail/cpu_vector_policy.h"
namespace mindquantum::sim::vector::detail {
template <typename derived_, typename calc_type_>
auto CPUVectorPolicyBase<derived_, calc_type_>::InitState(index_t dim, bool zero_state) -> qs_data_p_t {
    if (dim == 0 || dim > (~static_cast<uint64_t>(0))) {
        throw std::runtime_error("Dimension too large.");
    }
    auto qs = reinterpret_cast<qs_data_p_t>(calloc(dim, sizeof(qs_data_t)));
    if (qs == nullptr) {
        throw std::runtime_error("Allocate memory for quantum state failed.");
    }
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
        for (index_t i = 0; i < (static_cast<uint64_t>(1) << n_qubits) - 1; i++) {
            std::cout << "(" << 0 << ", " << 0 << ")" << std::endl;
        }
    } else {
        for (index_t i = 0; i < (static_cast<uint64_t>(1) << n_qubits); i++) {
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
        THRESHOLD_OMP_FOR(dim, DimTh, for (omp::idx_t i = 0; i < static_cast<omp::idx_t>(dim); i++) { out[i] = qs[i]; })
    }
    return out;
};

template <typename derived_, typename calc_type_>
auto CPUVectorPolicyBase<derived_, calc_type_>::GetQS(const qs_data_p_t& qs, index_t dim) -> VT<py_qs_data_t> {
    VT<py_qs_data_t> out(dim);
    if (qs != nullptr) {
        THRESHOLD_OMP_FOR(dim, DimTh, for (omp::idx_t i = 0; i < static_cast<omp::idx_t>(dim); i++) { out[i] = qs[i]; })
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
    THRESHOLD_OMP_FOR(dim, DimTh, for (omp::idx_t i = 0; i < static_cast<omp::idx_t>(dim); i++) { qs[i] = qs_out[i]; })
}

template <typename derived_, typename calc_type_>
auto CPUVectorPolicyBase<derived_, calc_type_>::ApplyTerms(qs_data_p_t* qs_p,
                                                           const std::vector<PauliTerm<calc_type>>& ham,
                                                           index_t dim) -> qs_data_p_t {
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
void CPUVectorPolicyBase<derived_, calc_type_>::ApplyPauliString(qs_data_p_t* qs_p, const PauliMask& mask,
                                                                 Index ctrl_mask, index_t dim) {
    auto& qs = (*qs_p);
    if (qs == nullptr) {
        qs = derived::InitState(dim);
    }
    auto mask_f = mask.mask_x | mask.mask_y;
    if (ctrl_mask == 0) {
        THRESHOLD_OMP_FOR(
            dim, DimTh, for (omp::idx_t i = 0; i < static_cast<omp::idx_t>(dim); i++) {
                auto j = (i ^ mask_f);
                if (i <= j) {
                    auto axis2power = CountOne(i & mask.mask_z);  // -1
                    auto axis3power = CountOne(i & mask.mask_y);  // -1j
                    auto c = ComplexCast<double, calc_type>::apply(
                        POLAR[static_cast<char>((mask.num_y + 2 * axis3power + 2 * axis2power) & 3)]);
                    if (i == j) {
                        qs[i] = qs[i] * c;
                    } else {
                        auto tmp = qs[j];
                        qs[j] = qs[i] * c;
                        qs[i] = tmp / c;
                    }
                }
            })
    } else {
        THRESHOLD_OMP_FOR(
            dim, DimTh, for (omp::idx_t i = 0; i < static_cast<omp::idx_t>(dim); i++) {
                if ((i & ctrl_mask) == ctrl_mask) {
                    auto j = (i ^ mask_f);
                    if (i <= j) {
                        auto axis2power = CountOne(i & mask.mask_z);  // -1
                        auto axis3power = CountOne(i & mask.mask_y);  // -1j
                        auto c = ComplexCast<double, calc_type>::apply(
                            POLAR[static_cast<char>((mask.num_y + 2 * axis3power + 2 * axis2power) & 3)]);
                        if (i == j) {
                            qs[i] = qs[i] * c;
                        } else {
                            auto tmp = qs[j];
                            qs[j] = qs[i] * c;
                            qs[i] = tmp / c;
                        }
                    }
                }
            })
    }
}

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
    auto dim = static_cast<uint64_t>(1) << n_qubits;
    THRESHOLD_OMP(
        MQ_DO_PRAGMA(omp parallel for reduction(min:result) schedule(static)), dim, DimTh,
                     for (omp::idx_t i = 0; i < (static_cast<uint64_t>(1) << n_qubits); i++) {
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

template <typename T>
void ShowVec(const std::vector<T>& a) {
    for (auto i : a) {
        std::cout << i << ", ";
    }
    std::cout << std::endl;
}

template <typename derived_, typename calc_type>
VT<calc_type> CPUVectorPolicyBase<derived_, calc_type>::GetCumulativeProbs(const qs_data_p_t& qs_out, index_t dim) {
    auto qs = qs_out;
    bool will_free = false;
    if (qs == nullptr) {
        qs = derived_::InitState(dim);
        will_free = true;
    }
    VT<calc_type> prob(dim);
    // Can be optimized by parallel prefix sum algorithm.
    prob[0] = qs[0].real() * qs[0].real() + qs[0].imag() * qs[0].imag();
    for (size_t i = 1; i < dim; ++i) {
        prob[i] = qs[i].real() * qs[i].real() + qs[i].imag() * qs[i].imag() + prob[i - 1];
    }

    prob[dim - 1] = 1.0;
    if (will_free) {
        free(qs);
    }
    return prob;
}

template <typename derived_, typename calc_type>
VT<unsigned> CPUVectorPolicyBase<derived_, calc_type>::LowerBound(const VT<calc_type>& cum_prob,
                                                                  const VT<calc_type>& sampled_probs) {
    size_t samp_size = sampled_probs.size();
    VT<unsigned> out(samp_size);
    size_t samp_idx = 0, dist_idx = 0;
    while (true) {
        if (samp_idx >= samp_size) {
            break;
        }
        if (sampled_probs[samp_idx] < cum_prob[dist_idx]) {
            out[samp_idx] = dist_idx;
            samp_idx += 1;
        } else {
            dist_idx += 1;
        }
    }
    return out;
}

template <typename derived_, typename calc_type_>
auto CPUVectorPolicyBase<derived_, calc_type_>::GetReducedDensityMatrix(const qs_data_p_t& qs,
                                                                        const qbits_t& kept_qubits,
                                                                        index_t dim) -> VVT<py_qs_data_t> {
    if (qs != nullptr) {
        size_t n_qubits = static_cast<size_t>(std::log2(dim));
        size_t n_kept = kept_qubits.size();
        size_t dim_kept = (1UL << n_kept);

        qbits_t sorted_qubits = kept_qubits;
        std::sort(sorted_qubits.begin(), sorted_qubits.end());

        VVT<py_qs_data_t> rho(dim_kept, VT<py_qs_data_t>(dim_kept, 0.0));

        for (size_t i = 0; i < dim; i++) {
            size_t i_kept = 0;
            for (size_t k = 0; k < n_kept; k++) {
                if ((i >> sorted_qubits[k]) & 1) {
                    i_kept |= (1UL << k);
                }
            }
            for (size_t j = 0; j < dim; j++) {
                size_t j_kept = 0;
                for (size_t k = 0; k < n_kept; k++) {
                    if ((j >> sorted_qubits[k]) & 1) {
                        j_kept |= (1UL << k);
                    }
                }
                bool same_traced = true;
                for (size_t k = 0; k < n_qubits; k++) {
                    if (std::find(sorted_qubits.begin(), sorted_qubits.end(), k) == sorted_qubits.end()) {
                        if (((i >> k) & 1) != ((j >> k) & 1)) {
                            same_traced = false;
                            break;
                        }
                    }
                }
                if (same_traced) {
                    rho[i_kept][j_kept] += qs[i] * std::conj(qs[j]);
                }
            }
        }
        return rho;
    } else {
        size_t dim_kept = (1UL << kept_qubits.size());
        VVT<py_qs_data_t> rho(dim_kept, VT<py_qs_data_t>(dim_kept, 0.0));
        rho[0][0] = 1.0;
        return rho;
    }
}

#ifdef __x86_64__
template struct CPUVectorPolicyBase<CPUVectorPolicyAvxFloat, float>;
template struct CPUVectorPolicyBase<CPUVectorPolicyAvxDouble, double>;
#elif defined(__amd64)
template struct CPUVectorPolicyBase<CPUVectorPolicyArmFloat, float>;
template struct CPUVectorPolicyBase<CPUVectorPolicyArmDouble, double>;
#endif
}  // namespace mindquantum::sim::vector::detail
