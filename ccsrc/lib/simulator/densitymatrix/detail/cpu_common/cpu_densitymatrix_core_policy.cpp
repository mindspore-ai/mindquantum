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
#include <stdexcept>

#include "config/openmp.hpp"

#include "core/utils.hpp"
#include "math/pr/parameter_resolver.hpp"
#include "simulator/types.hpp"
#include "simulator/utils.hpp"
#ifdef __x86_64__
#    include "simulator/densitymatrix/detail/cpu_densitymatrix_avx_double_policy.hpp"
#    include "simulator/densitymatrix/detail/cpu_densitymatrix_avx_float_policy.hpp"
#elif defined(__amd64)
#    include "simulator/densitymatrix/detail/cpu_densitymatrix_arm_double_policy.hpp"
#    include "simulator/densitymatrix/detail/cpu_densitymatrix_arm_float_policy.hpp"
#endif
#include "simulator/densitymatrix/detail/cpu_densitymatrix_policy.hpp"

namespace mindquantum::sim::densitymatrix::detail {
template <typename derived_, typename calc_type_>
auto CPUDensityMatrixPolicyBase<derived_, calc_type_>::InitState(index_t dim, bool zero_state) -> qs_data_p_t {
    index_t n_elements = (dim * dim + dim) / 2;
    auto qs = reinterpret_cast<qs_data_p_t>(calloc(n_elements, sizeof(qs_data_t)));
    if (zero_state) {
        qs[0] = 1;
    }
    return qs;
}

template <typename derived_, typename calc_type_>
void CPUDensityMatrixPolicyBase<derived_, calc_type_>::Reset(qs_data_p_t* qs_p, index_t dim, bool zero_state) {
    derived::FreeState(qs_p);
}

template <typename derived_, typename calc_type_>
void CPUDensityMatrixPolicyBase<derived_, calc_type_>::FreeState(qs_data_p_t* qs_p) {
    auto& qs = (*qs_p);
    if (qs != nullptr) {
        free(qs);
        qs = nullptr;
    }
}

template <typename derived_, typename calc_type_>
void CPUDensityMatrixPolicyBase<derived_, calc_type_>::Display(const qs_data_p_t& qs, qbit_t n_qubits, qbit_t q_limit) {
    if (n_qubits > q_limit) {
        n_qubits = q_limit;
    }
    index_t dim = 1UL << n_qubits;
    std::cout << n_qubits << " qubits cpu simulator (little endian)." << std::endl;
    if (qs == nullptr) {
        std::cout << "(" << 1.0 << ", " << 0.0 << "),";
        for (index_t i = 1; i < dim; i++) {
            std::cout << "(" << 0.0 << ", " << 0.0 << "),";
        }
        std::cout << std::endl;
        for (index_t i = 1; i < dim; i++) {
            for (index_t j = 0; j < dim; j++) {
                std::cout << "(" << 0.0 << ", " << 0.0 << "),";
            }
            std::cout << std::endl;
        }
    } else {
        for (index_t i = 0; i < dim; i++) {
            for (index_t j = 0; j <= i; j++) {
                std::cout << "(" << qs[IdxMap(i, j)].real() << ", " << qs[IdxMap(i, j)].imag() << ")"
                          << ",";
            }
            for (index_t j = i + 1; j < dim; j++) {
                std::cout << "(" << qs[IdxMap(j, i)].real() << ", " << -qs[IdxMap(j, i)].imag() << ")"
                          << ",";
            }
            std::cout << std::endl;
        }
    }
}

template <typename derived_, typename calc_type_>
void CPUDensityMatrixPolicyBase<derived_, calc_type_>::SetToZeroExcept(qs_data_p_t* qs_p, index_t ctrl_mask,
                                                                       index_t dim) {
    auto& qs = *qs_p;
    if (qs == nullptr) {
        qs = derived::InitState(dim);
    }
    THRESHOLD_OMP_FOR(
        dim, DimTh, for (omp::idx_t i = 0; i < dim; i++) {
            if ((i & ctrl_mask) != ctrl_mask) {
                for (index_t j = 0; j < (dim / 2); j++) {
                    if ((j & ctrl_mask) != ctrl_mask) {
                        qs[IdxMap(i, j)] = 0;
                    }
                }
            }
        })
}

template <typename derived_, typename calc_type_>
auto CPUDensityMatrixPolicyBase<derived_, calc_type_>::Copy(const qs_data_p_t& qs, index_t dim) -> qs_data_p_t {
    qs_data_p_t out = nullptr;
    if (qs != nullptr) {
        out = derived::InitState(dim, false);
        THRESHOLD_OMP_FOR(
            dim, DimTh, for (omp::idx_t i = 0; i < (dim * dim + dim) / 2; i++) { out[i] = qs[i]; })
    }
    return out;
}

template <typename derived_, typename calc_type_>
auto CPUDensityMatrixPolicyBase<derived_, calc_type_>::GetQS(const qs_data_p_t& qs, index_t dim) -> matrix_t {
    matrix_t out(dim, py_qs_datas_t(dim));
    if (qs == nullptr) {
        out[0][0] = 1.0;
        return out;
    }
    THRESHOLD_OMP_FOR(
        dim, DimTh, for (omp::idx_t i = 0; i < dim; i++) {
            for (index_t j = 0; j < i; j++) {
                out[i][j] = qs[IdxMap(i, j)];
            }
            for (index_t j = i; j < dim; j++) {
                out[i][j] = std::conj(qs[IdxMap(j, i)]);
            }
        })
    return out;
}

template <typename derived_, typename calc_type_>
void CPUDensityMatrixPolicyBase<derived_, calc_type_>::SetQS(qs_data_p_t* qs_p, const py_qs_datas_t& vec_out,
                                                             index_t dim) {
    if (vec_out.size() != dim) {
        throw std::invalid_argument("state size not match");
    }
    auto& qs = *qs_p;
    if (qs == nullptr) {
        qs = derived::InitState(dim);
    }
    THRESHOLD_OMP_FOR(
        dim, DimTh, for (omp::idx_t i = 0; i < dim; i++) {
            for (index_t j = 0; j <= i; j++) {
                qs[IdxMap(i, j)] = vec_out[i] * vec_out[j];
            }
        })
}

template <typename derived_, typename calc_type_>
void CPUDensityMatrixPolicyBase<derived_, calc_type_>::SetDM(qs_data_p_t* qs_p, const matrix_t& mat_out, index_t dim) {
    if (mat_out[0].size() != dim) {
        throw std::invalid_argument("state size not match");
    }
    auto& qs = *qs_p;
    if (qs == nullptr) {
        qs = derived::InitState(dim);
    }
    THRESHOLD_OMP_FOR(
        dim, DimTh, for (omp::idx_t i = 0; i < dim; i++) {
            for (index_t j = 0; j <= i; j++) {
                qs[IdxMap(i, j)] = mat_out[i][j];
            }
        })
}

template <typename derived_, typename calc_type_>
void CPUDensityMatrixPolicyBase<derived_, calc_type_>::CopyQS(qs_data_p_t* qs_des, const qs_data_p_t& qs_src,
                                                              index_t dim) {
    auto& qs = *qs_des;
    if (qs == nullptr) {
        qs = derived::InitState(dim);
    }
    if (qs_src == nullptr) {
        qs[0] = 1.0;
    } else {
        THRESHOLD_OMP_FOR(
            dim, DimTh, for (omp::idx_t i = 0; i < (dim * dim + dim) / 2; i++) { qs[i] = qs_src[i]; })
    }
}

template <typename derived_, typename calc_type_>
auto CPUDensityMatrixPolicyBase<derived_, calc_type_>::Purity(const qs_data_p_t& qs, index_t dim) -> calc_type {
    if (qs == nullptr) {
        return 1.0;
    }
    calc_type p = 0;
    THRESHOLD_OMP(
        MQ_DO_PRAGMA(omp parallel for schedule(static) reduction(+: p)), dim, DimTh,
                     for (omp::idx_t i = 0; i < (dim * dim + dim) / 2; i++) { p += 2 * std::norm(qs[i]); })
    THRESHOLD_OMP(
        MQ_DO_PRAGMA(omp parallel for schedule(static) reduction(+: p)), dim, DimTh,
                     for (omp::idx_t i = 0; i < dim; i++) { p += -std::norm(qs[IdxMap(i, i)]); })
    return p;
}

template <typename derived_, typename calc_type_>
bool CPUDensityMatrixPolicyBase<derived_, calc_type_>::IsPure(const qs_data_p_t& qs, index_t dim) {
    auto p = Purity(qs, dim);
    if (std::abs(p - 1) < 1e-8) {
        return true;
    } else {
        return false;
    }
}

template <typename derived_, typename calc_type_>
auto CPUDensityMatrixPolicyBase<derived_, calc_type_>::PureStateVector(const qs_data_p_t& qs, index_t dim)
    -> py_qs_datas_t {
    if (!IsPure(qs, dim)) {
        throw(std::runtime_error("PureStateVector(): Cannot transform mixed density matrix to vector."));
    }
    if (qs == nullptr) {
        py_qs_datas_t qs_vector(dim, 0.0);
        qs_vector[0] = 1.0;
        return qs_vector;
    }
    py_qs_datas_t qs_vector(dim);
    index_t base;
    calc_type base_value;
    for (index_t i = 0; i < dim; i++) {
        if (qs[IdxMap(i, i)].real() > 1e-8) {
            base = i;
            base_value = std::sqrt(qs[IdxMap(i, i)].real());
            qs_vector[i] = base_value;
            break;
        }
    }
    THRESHOLD_OMP_FOR(
        dim, DimTh, for (omp::idx_t i = base + 1; i < dim; i++) { qs_vector[i] = qs[IdxMap(i, base)] / base_value; })
    return qs_vector;
}

template <typename derived_, typename calc_type_>
void CPUDensityMatrixPolicyBase<derived_, calc_type_>::ApplyTerms(qs_data_p_t* qs_p,
                                                                  const std::vector<PauliTerm<calc_type>>& ham,
                                                                  index_t dim) {
    auto& qs = *qs_p;
    if (qs == nullptr) {
        qs = derived::InitState(dim);
    }
    matrix_t tmp(dim, VT<qs_data_t>(dim));
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
                    for (index_t col = 0; col < dim; col++) {
                        tmp[j][col] += GetValue(qs, i, col) * coeff * c;
                    }
                    if (i != j) {
                        for (index_t col = 0; col < dim; col++) {
                            tmp[i][col] += GetValue(qs, j, col) * coeff / c;
                        }
                    }
                }
            })
    }
    Reset(qs_p, dim, false);
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
                    for (index_t row = 0; row <= i; row++) {
                        qs[IdxMap(i, row)] += std::conj(tmp[row][j] * coeff * c);
                    }
                    if (i != j) {
                        for (index_t row = 0; row <= j; row++) {
                            qs[IdxMap(j, row)] += std::conj(tmp[row][i] * coeff / c);
                        }
                    }
                }
            })
    }
}

#ifdef __x86_64__
template struct CPUDensityMatrixPolicyBase<CPUDensityMatrixPolicyAvxFloat, float>;
template struct CPUDensityMatrixPolicyBase<CPUDensityMatrixPolicyAvxDouble, double>;
#elif defined(__amd64)
template struct CPUDensityMatrixPolicyBase<CPUDensityMatrixPolicyArmFloat, float>;
template struct CPUDensityMatrixPolicyBase<CPUDensityMatrixPolicyArmDouble, double>;
#endif
}  // namespace mindquantum::sim::densitymatrix::detail
