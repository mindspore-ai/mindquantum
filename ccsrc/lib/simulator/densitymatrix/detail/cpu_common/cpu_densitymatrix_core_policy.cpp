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
#include <stdexcept>
#include <vector>

#include "config/openmp.h"
#include "core/mq_base_types.h"
#include "core/utils.h"
#include "math/pr/parameter_resolver.h"
#include "simulator/utils.h"
#ifdef __x86_64__
#    include "simulator/densitymatrix/detail/cpu_densitymatrix_avx_double_policy.h"
#    include "simulator/densitymatrix/detail/cpu_densitymatrix_avx_float_policy.h"
#elif defined(__amd64)
#    include "simulator/densitymatrix/detail/cpu_densitymatrix_arm_double_policy.h"
#    include "simulator/densitymatrix/detail/cpu_densitymatrix_arm_float_policy.h"
#endif
#include "simulator/densitymatrix/detail/cpu_densitymatrix_policy.h"

namespace mindquantum::sim::densitymatrix::detail {
template <typename derived_, typename calc_type_>
auto CPUDensityMatrixPolicyBase<derived_, calc_type_>::InitState(index_t dim, bool zero_state) -> qs_data_p_t {
    index_t n_elements = (dim * dim + dim) / 2;
    auto qs = reinterpret_cast<qs_data_p_t>(calloc(n_elements, sizeof(qs_data_t)));
    if (qs == nullptr) {
        throw std::runtime_error("Allocate memory for quantum state failed.");
    }
    if (zero_state) {
        qs[0] = 1;
    }
    return qs;
}

template <typename derived_, typename calc_type_>
void CPUDensityMatrixPolicyBase<derived_, calc_type_>::Reset(qs_data_p_t* qs_p) {
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
    index_t dim = static_cast<uint64_t>(1) << n_qubits;
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
        dim, DimTh, for (omp::idx_t i = 0; i < static_cast<omp::idx_t>(dim); i++) {
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
            dim, DimTh,
            for (omp::idx_t i = 0; i < static_cast<omp::idx_t>((dim * dim + dim) / 2); i++) { out[i] = qs[i]; })
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
        dim, DimTh, for (omp::idx_t i = 0; i < static_cast<omp::idx_t>(dim); i++) {
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
        dim, DimTh, for (omp::idx_t i = 0; i < static_cast<omp::idx_t>(dim); i++) {
            for (index_t j = 0; j <= i; j++) {
                qs[IdxMap(i, j)] = vec_out[i] * std::conj(vec_out[j]);
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
        dim, DimTh, for (omp::idx_t i = 0; i < static_cast<omp::idx_t>(dim); i++) {
            for (index_t j = 0; j <= i; j++) {
                qs[IdxMap(i, j)] = mat_out[i][j];
            }
        })
}

template <typename derived_, typename calc_type_>
void CPUDensityMatrixPolicyBase<derived_, calc_type_>::CopyQS(qs_data_p_t* qs_des, const qs_data_p_t& qs_src,
                                                              index_t dim) {
    auto& qs = *qs_des;
    if (qs_src == nullptr) {
        FreeState(&qs);
    } else {
        if (qs == nullptr) {
            qs = derived::InitState(dim);
        }
        THRESHOLD_OMP_FOR(
            dim, DimTh,
            for (omp::idx_t i = 0; i < static_cast<omp::idx_t>((dim * dim + dim) / 2); i++) { qs[i] = qs_src[i]; })
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
                     for (omp::idx_t i = 0; i < static_cast<omp::idx_t>((dim * dim + dim) / 2);
                          i++) { p += 2 * std::norm(qs[i]); })
    THRESHOLD_OMP(
        MQ_DO_PRAGMA(omp parallel for schedule(static) reduction(+: p)), dim, DimTh,
                     for (omp::idx_t i = 0; i < static_cast<omp::idx_t>(dim);
                          i++) { p += -std::norm(qs[IdxMap(i, i)]); })
    return p;
}

template <typename derived_, typename calc_type_>
auto CPUDensityMatrixPolicyBase<derived_, calc_type_>::GetPartialTrace(const qs_data_p_t& qs, const qbits_t& objs,
                                                                       index_t dim) -> matrix_t {
    qs_data_p_t tmp = qs;
    bool will_free = false;
    if (tmp == nullptr) {
        tmp = derived::InitState(dim);
        will_free = true;
    }
    qs_data_p_t out = nullptr;
    index_t out_dim = dim;
    qbit_t offset;
    for (Index i = 0; i < objs.size(); i++) {
        offset = 0;
        for (Index j = 0; j < i; j++) {
            if (objs[i] > objs[j]) {
                offset += 1;
            }
        }
        SingleQubitGateMask mask({objs[i] - offset}, {});
        out_dim = out_dim >> 1;
        out = derived::InitState(out_dim, false);
        THRESHOLD_OMP_FOR(
            dim, DimTh, for (omp::idx_t a = 0; a < static_cast<omp::idx_t>(out_dim); a++) {  // loop on the row
                auto r0 = ((a & mask.obj_high_mask) << 1) + (a & mask.obj_low_mask);
                auto r1 = r0 + mask.obj_mask;
                for (index_t b = 0; b <= a; b++) {  // loop on the column
                    auto c0 = ((b & mask.obj_high_mask) << 1) + (b & mask.obj_low_mask);
                    auto c1 = c0 + mask.obj_mask;
                    auto value = tmp[IdxMap(r0, c0)] + tmp[IdxMap(r1, c1)];
                    out[IdxMap(a, b)] = value;
                }
            })
        if (i != 0) {
            FreeState(&tmp);
        } else if (will_free) {
            FreeState(&tmp);
        }
        tmp = out;
    }
    auto res = GetQS(out, out_dim);
    FreeState(&out);
    return res;
}

template <typename derived_, typename calc_type_>
auto CPUDensityMatrixPolicyBase<derived_, calc_type_>::PureStateVector(const qs_data_p_t& qs, index_t dim)
    -> py_qs_datas_t {
    auto p = Purity(qs, dim);
    if (1 - p > 1e-6) {
        throw(std::runtime_error("PureStateVector(): Cannot transform mixed density matrix to vector."));
    }
    if (qs == nullptr) {
        py_qs_datas_t qs_vector(dim, 0.0);
        qs_vector[0] = 1.0;
        return qs_vector;
    }
    py_qs_datas_t qs_vector(dim);
    index_t base = 0;
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
        dim, DimTh, for (omp::idx_t i = base + 1; i < static_cast<omp::idx_t>(dim); i++) {
            qs_vector[i] = qs[IdxMap(i, base)] / base_value;
        })
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
            dim, DimTh, for (omp::idx_t i = 0; i < static_cast<omp::idx_t>(dim); i++) {
                auto j = (i ^ mask_f);
                if (i <= j) {
                    auto axis2power = CountOne(i & mask.mask_z);  // -1
                    auto axis3power = CountOne(i & mask.mask_y);  // -1j
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
    Reset(qs_p);
    qs = derived::InitState(dim, false);
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

template <typename derived_, typename calc_type_>
void CPUDensityMatrixPolicyBase<derived_, calc_type_>::ApplyPauliString(qs_data_p_t* qs_p, const PauliMask& mask,
                                                                        Index ctrl_mask, index_t dim) {
    if (!ctrl_mask) {
        derived_::ApplyPauliStringNoCtrl(qs_p, mask, dim);
    } else {
        derived_::ApplyPauliStringWithCtrl(qs_p, mask, ctrl_mask, dim);
    }
}
template <typename derived_, typename calc_type_>
void CPUDensityMatrixPolicyBase<derived_, calc_type_>::ApplyPauliStringNoCtrl(qs_data_p_t* qs_p, const PauliMask& mask,
                                                                              index_t dim) {
    auto& qs = *qs_p;
    if (qs == nullptr) {
        qs = derived::InitState(dim);
    }
    auto mask_f = mask.mask_x | mask.mask_y;
    auto origin = derived_::Copy(*qs_p, dim);
    THRESHOLD_OMP_FOR(
        dim, DimTh, for (omp::idx_t r_i = 0; r_i < static_cast<omp::idx_t>(dim); r_i++) {
            auto r_j = (r_i ^ mask_f);
            if (r_j <= r_i) {
                for (index_t c_i = 0; c_i <= r_i; c_i++) {
                    auto c_j = (c_i ^ mask_f);
                    if (c_j >= c_i) {
                        auto r_axis2power = CountOne(r_i & mask.mask_z);  // -1
                        auto r_axis3power = CountOne(r_i & mask.mask_y);  // -1j
                        auto r_c = ComplexCast<double, calc_type>::apply(
                            POLAR[static_cast<char>((mask.num_y + 2 * r_axis3power + 2 * r_axis2power) & 3)]);
                        auto c_axis2power = CountOne(c_i & mask.mask_z);  // -1
                        auto c_axis3power = CountOne(c_i & mask.mask_y);  // -1j
                        auto c_c = std::conj(ComplexCast<double, calc_type>::apply(
                            POLAR[static_cast<char>((mask.num_y + 2 * c_axis3power + 2 * c_axis2power) & 3)]));
                        auto m_ri_ci = GetValue(origin, r_i, c_i);
                        auto m_ri_cj = GetValue(origin, r_i, c_j);
                        auto m_rj_ci = GetValue(origin, r_j, c_i);
                        auto m_rj_cj = GetValue(origin, r_j, c_j);
                        qs[IdxMap(r_i, c_i)] = m_rj_cj / r_c / c_c;
                        if ((r_j >= c_i) && (r_i != r_j)) {
                            qs[IdxMap(r_j, c_i)] = m_ri_cj * r_c / c_c;
                        }
                        if ((c_j <= r_i) && (c_i != c_j)) {
                            qs[IdxMap(r_i, c_j)] = m_rj_ci / r_c * c_c;
                        }
                        if ((r_j >= c_j) && (r_i != r_j) && (c_i != c_j)) {
                            qs[IdxMap(r_j, c_j)] = m_ri_ci * r_c * c_c;
                        }
                    }
                }
            }
        })
    derived_::FreeState(&origin);
}
template <typename derived_, typename calc_type_>
void CPUDensityMatrixPolicyBase<derived_, calc_type_>::ApplyPauliStringWithCtrl(qs_data_p_t* qs_p,
                                                                                const PauliMask& mask, Index ctrl_mask,
                                                                                index_t dim) {
    auto& qs = *qs_p;
    if (qs == nullptr) {
        qs = derived::InitState(dim);
    }
    auto mask_f = mask.mask_x | mask.mask_y;
    auto origin = derived_::Copy(*qs_p, dim);
    THRESHOLD_OMP_FOR(
        dim, DimTh, for (omp::idx_t r_i = 0; r_i < static_cast<omp::idx_t>(dim); r_i++) {
            bool r_ctrl = ((r_i & ctrl_mask) == ctrl_mask);
            auto r_j = r_ctrl ? (r_i ^ mask_f) : r_i;
            if (r_j <= r_i) {
                for (index_t c_i = 0; c_i <= r_i; c_i++) {
                    bool c_ctrl = ((c_i & ctrl_mask) == ctrl_mask);
                    auto c_j = c_ctrl ? (c_i ^ mask_f) : c_i;
                    if (c_j >= c_i) {
                        auto r_axis2power = CountOne(r_i & mask.mask_z);  // -1
                        auto r_axis3power = CountOne(r_i & mask.mask_y);  // -1j
                        auto r_c = ComplexCast<double, calc_type>::apply(
                            POLAR[static_cast<char>((mask.num_y + 2 * r_axis3power + 2 * r_axis2power) & 3)]);
                        auto c_axis2power = CountOne(c_i & mask.mask_z);  // -1
                        auto c_axis3power = CountOne(c_i & mask.mask_y);  // -1j
                        auto c_c = std::conj(ComplexCast<double, calc_type>::apply(
                            POLAR[static_cast<char>((mask.num_y + 2 * c_axis3power + 2 * c_axis2power) & 3)]));
                        r_c = r_ctrl ? r_c : 1;
                        c_c = c_ctrl ? c_c : 1;
                        auto m_ri_ci = GetValue(origin, r_i, c_i);
                        auto m_ri_cj = GetValue(origin, r_i, c_j);
                        auto m_rj_ci = GetValue(origin, r_j, c_i);
                        auto m_rj_cj = GetValue(origin, r_j, c_j);
                        qs[IdxMap(r_i, c_i)] = m_rj_cj / r_c / c_c;
                        if ((r_j >= c_i) && (r_i != r_j)) {
                            qs[IdxMap(r_j, c_i)] = m_ri_cj * r_c / c_c;
                        }
                        if ((c_j <= r_i) && (c_i != c_j)) {
                            qs[IdxMap(r_i, c_j)] = m_rj_ci / r_c * c_c;
                        }
                        if ((r_j >= c_j) && (r_i != r_j) && (c_i != c_j)) {
                            qs[IdxMap(r_j, c_j)] = m_ri_ci * r_c * c_c;
                        }
                    }
                }
            }
        })
    derived_::FreeState(&origin);
}

template <typename derived_, typename calc_type_>
void CPUDensityMatrixPolicyBase<derived_, calc_type_>::ApplyCsr(
    qs_data_p_t* qs_p, const std::shared_ptr<sparse::CsrHdMatrix<calc_type>>& a, index_t dim) {
    if (dim != a->dim_) {
        throw std::runtime_error("Sparse hamiltonian size not match with quantum state size.");
    }
    auto& qs = *qs_p;
    if (qs == nullptr) {
        qs = derived::InitState(dim);
    }
    matrix_t tmp(dim, VT<qs_data_t>(dim));
    auto data = a->data_;
    auto indptr = a->indptr_;
    auto indices = a->indices_;
    THRESHOLD_OMP_FOR(
        dim, DimTh, for (omp::idx_t i = 0; i < dim; i++) {
            for (index_t j = 0; j < dim; j++) {
                qs_data_t sum = {0.0, 0.0};
                for (index_t k = indptr[j]; k < indptr[j + 1]; k++) {
                    sum += data[k] * GetValue(qs, indices[k], i);
                }
                tmp[j][i] = sum;
            }
        })
    THRESHOLD_OMP_FOR(
        dim, DimTh, for (omp::idx_t i = 0; i < dim; i++) {
            auto start = indptr[i];
            auto end = indptr[i + 1];
            for (index_t j = i; j < dim; j++) {
                qs_data_t sum = {0.0, 0.0};
                for (index_t k = start; k < end; k++) {
                    sum += std::conj(data[k]) * tmp[j][indices[k]];
                }
                qs[IdxMap(j, i)] = sum;
            }
        })
}

#ifdef __x86_64__
template struct CPUDensityMatrixPolicyBase<CPUDensityMatrixPolicyAvxFloat, float>;
template struct CPUDensityMatrixPolicyBase<CPUDensityMatrixPolicyAvxDouble, double>;
#elif defined(__amd64)
template struct CPUDensityMatrixPolicyBase<CPUDensityMatrixPolicyArmFloat, float>;
template struct CPUDensityMatrixPolicyBase<CPUDensityMatrixPolicyArmDouble, double>;
#endif
}  // namespace mindquantum::sim::densitymatrix::detail
