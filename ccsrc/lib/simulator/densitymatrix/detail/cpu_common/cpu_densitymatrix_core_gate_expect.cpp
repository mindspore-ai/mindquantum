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
auto CPUDensityMatrixPolicyBase<derived_, calc_type_>::ExpectDiffRPS(const qs_data_p_t& qs_out,
                                                                     const qs_data_p_t& ham_matrix_out,
                                                                     const PauliMask& mask, Index ctrl_mask,
                                                                     index_t dim) -> qs_data_t {
    qs_data_p_t qs;
    bool will_free_qs = false;
    if (qs_out == nullptr) {
        qs = derived::InitState(dim);
        will_free_qs = true;
    } else {
        qs = qs_out;
    }
    qs_data_p_t ham_matrix;
    bool will_free_ham = false;
    if (ham_matrix_out == nullptr) {
        ham_matrix = derived::InitState(dim);
        will_free_ham = true;
    } else {
        ham_matrix = ham_matrix_out;
    }
    calc_type e_r = 0, e_i = 0;
    auto mask_f = mask.mask_x | mask.mask_y;
    // clang-format off
    if (!ctrl_mask) {
        THRESHOLD_OMP(
            MQ_DO_PRAGMA(omp parallel for reduction(+:e_r, e_i) schedule(static)), dim, DimTh,
                for (omp::idx_t i = 0; i < dim; i++) {
                    auto j = i ^ mask_f;
                    if (j <= i) {
                        auto r_axis2power = CountOne(i & mask.mask_z);  // -1
                        auto r_axis3power = CountOne(i & mask.mask_y);  // -1j
                        auto c = ComplexCast<double, calc_type>::apply(
                            POLAR[static_cast<char>((mask.num_y + 2 * r_axis3power + 2 * r_axis2power) & 3)]);
                        qs_data_t sum = {0.0, 0.0};
                        for (index_t k = 0; k < dim; k++) {
                            sum += GetValue(qs, i, k) * GetValue(ham_matrix, k, j);
                        }
                        sum *= (c * HALF_MI);
                        e_r += sum.real();
                        e_i += sum.imag();
                        if (i != j) {
                            qs_data_t sum_j = {0.0, 0.0};
                            for (index_t k = 0; k < dim; k++) {
                                sum_j += GetValue(qs, j, k) * GetValue(ham_matrix, k, i);
                            }
                            sum_j *= (HALF_MI / c);
                            e_r += sum_j.real();
                            e_i += sum_j.imag();
                        }
                    }
                })
    } else {
        THRESHOLD_OMP(
            MQ_DO_PRAGMA(omp parallel for reduction(+:e_r, e_i) schedule(static)), dim, DimTh,
                for (omp::idx_t i = 0; i < dim; i++) {
                    if ((i & ctrl_mask) == ctrl_mask) {
                        auto j = i ^ mask_f;
                        if (j <= i) {
                            auto r_axis2power = CountOne(i & mask.mask_z);  // -1
                            auto r_axis3power = CountOne(i & mask.mask_y);  // -1j
                            auto c = ComplexCast<double, calc_type>::apply(
                                POLAR[static_cast<char>((mask.num_y + 2 * r_axis3power + 2 * r_axis2power) & 3)]);
                            qs_data_t sum = {0.0, 0.0};
                            for (index_t k = 0; k < dim; k++) {
                                sum += GetValue(qs, i, k) * GetValue(ham_matrix, k, j);
                            }
                            sum *= (c * HALF_MI);
                            e_r += sum.real();
                            e_i += sum.imag();
                            if (i != j) {
                                qs_data_t sum_j = {0.0, 0.0};
                                for (index_t k = 0; k < dim; k++) {
                                    sum_j += GetValue(qs, j, k) * GetValue(ham_matrix, k, i);
                                }
                                sum_j *= (HALF_MI / c);
                                e_r += sum_j.real();
                                e_i += sum_j.imag();
                            }
                        }
                    }
                })
    }
    // clang-format on
    if (will_free_qs) {
        derived::FreeState(&qs);
    }
    if (will_free_ham) {
        derived::FreeState(&ham_matrix);
    }
    return qs_data_t(e_r, e_i);
}

template <typename derived_, typename calc_type_>
auto CPUDensityMatrixPolicyBase<derived_, calc_type_>::HamiltonianMatrix(const Hamiltonian<calc_type>& ham, index_t dim)
    -> qs_data_p_t {
    if (ham.how_to_ == ORIGIN) {
        return TermsToMatrix(ham.ham_, dim);
    } else if (ham.how_to_ == BACKEND) {
        return CsrToMatrix(sparse::Csr_Plus_Csr<calc_type>(ham.ham_sparse_main_, ham.ham_sparse_second_), dim);
    } else {
        return CsrToMatrix(ham.ham_sparse_main_, dim);
    }
}

template <typename derived_, typename calc_type_>
auto CPUDensityMatrixPolicyBase<derived_, calc_type_>::TermsToMatrix(const std::vector<PauliTerm<calc_type>>& ham,
                                                                     index_t dim) -> qs_data_p_t {
    qs_data_p_t out = InitState(dim, false);
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
                    out[IdxMap(j, i)] += coeff * c;
                }
            })
    }
    return out;
}

template <typename derived_, typename calc_type_>
auto CPUDensityMatrixPolicyBase<derived_, calc_type_>::CsrToMatrix(
    const std::shared_ptr<sparse::CsrHdMatrix<calc_type>>& a, index_t dim) -> qs_data_p_t {
    if (dim != a->dim_) {
        throw std::runtime_error("Sparse hamiltonian size not match with quantum state size.");
    }
    qs_data_p_t out = InitState(dim, false);
    auto data = a->data_;
    auto indptr = a->indptr_;
    auto indices = a->indices_;
    THRESHOLD_OMP_FOR(
        dim, DimTh, for (omp::idx_t i = 0; i < dim; i++) {
            for (index_t j = indptr[i]; j < indptr[i + 1]; j++) {
                if (i >= indices[j]) {
                    out[IdxMap(i, indices[j])] = data[j];
                }
            }
        })
    return out;
}

template <typename derived_, typename calc_type_>
auto CPUDensityMatrixPolicyBase<derived_, calc_type_>::ExpectationOfTerms(const qs_data_p_t& qs_out,
                                                                          const std::vector<PauliTerm<calc_type>>& ham,
                                                                          index_t dim) -> qs_data_t {
    qs_data_p_t qs;
    bool will_free = false;
    if (qs_out == nullptr) {
        qs = derived::InitState(dim);
        will_free = true;
    } else {
        qs = qs_out;
    }
    calc_type e_r = 0, e_i = 0;
    for (const auto& [pauli_string, coeff_] : ham) {
        auto mask = GenPauliMask(pauli_string);
        auto mask_f = mask.mask_x | mask.mask_y;
        auto coeff = coeff_;
        // clang-format off
        THRESHOLD_OMP(
            MQ_DO_PRAGMA(omp parallel for reduction(+:e_r, e_i) schedule(static)), dim, DimTh,
                for (omp::idx_t i = 0; i <static_cast<omp::idx_t>(dim); i++) {
                    auto j = (i ^ mask_f);
                    if (i <= j) {
                        auto axis2power = CountOne(i & mask.mask_z);  // -1
                        auto axis3power = CountOne(i & mask.mask_y);  // -1j
                        auto c = ComplexCast<double, calc_type>::apply(
                            POLAR[static_cast<char>((mask.num_y + 2 * axis3power + 2 * axis2power) & 3)]);
                        auto e = std::conj(qs[IdxMap(j, i)]) * coeff * c;
                        if (i != j) {
                            e += qs[IdxMap(j, i)] * coeff / c;
                        }
                        e_r += std::real(e);
                        e_i += std::imag(e);
                    }
                })
        // clang-format on
    }
    if (will_free) {
        derived::FreeState(&qs);
    }
    return qs_data_t(e_r, e_i);
}

template <typename derived_, typename calc_type_>
auto CPUDensityMatrixPolicyBase<derived_, calc_type_>::ExpectationOfCsr(
    const qs_data_p_t& qs_out, const std::shared_ptr<sparse::CsrHdMatrix<calc_type>>& a, index_t dim) -> qs_data_t {
    qs_data_p_t qs;
    bool will_free = false;
    if (qs_out == nullptr) {
        qs = derived::InitState(dim);
        will_free = true;
    } else {
        qs = qs_out;
    }
    calc_type e_r = 0, e_i = 0;
    auto data = a->data_;
    auto indptr = a->indptr_;
    auto indices = a->indices_;
    // clang-format off
    THRESHOLD_OMP(
        MQ_DO_PRAGMA(omp parallel for reduction(+:e_r, e_i) schedule(static)), dim, DimTh,
            for (omp::idx_t i = 0; i < dim; i++) {
                qs_data_t sum = {0.0, 0.0};
                for (index_t j = indptr[i]; j < indptr[i + 1]; j++) {
                    sum += data[j] * GetValue(qs, indices[j], i);
                }
                e_r += sum.real();
                e_i += sum.imag();
            })
    // clang-format on
    if (will_free) {
        derived::FreeState(&qs);
    }
    return qs_data_t(e_r, e_i);
}

template <typename derived_, typename calc_type_>
auto CPUDensityMatrixPolicyBase<derived_, calc_type_>::ExpectDiffSingleQubitMatrix(
    const qs_data_p_t& qs_out, const qs_data_p_t& ham_matrix_out, const qbits_t& objs, const qbits_t& ctrls,
    const matrix_t& gate_m, const matrix_t& diff_m, index_t dim) -> qs_data_t {
    qs_data_p_t qs;
    bool will_free = false;
    if (qs_out == nullptr) {
        qs = derived::InitState(dim);
        will_free = true;
    } else {
        qs = qs_out;
    }
    qs_data_p_t ham_matrix;
    bool will_free_ham = false;
    if (ham_matrix_out == nullptr) {
        ham_matrix = derived::InitState(dim);
        will_free_ham = true;
    } else {
        ham_matrix = ham_matrix_out;
    }
    // G = Tr(m \rho H), where m = U' \dot \dagger{U}
    qs_data_t m00 = diff_m[0][0] * std::conj(gate_m[0][0]) + diff_m[0][1] * std::conj(gate_m[0][1]);
    qs_data_t m01 = diff_m[0][0] * std::conj(gate_m[1][0]) + diff_m[0][1] * std::conj(gate_m[1][1]);
    qs_data_t m10 = diff_m[1][0] * std::conj(gate_m[0][0]) + diff_m[1][1] * std::conj(gate_m[0][1]);
    qs_data_t m11 = diff_m[1][0] * std::conj(gate_m[1][0]) + diff_m[1][1] * std::conj(gate_m[1][1]);
    SingleQubitGateMask mask(objs, ctrls);
    calc_type res_real = 0, res_imag = 0;
    if (!mask.ctrl_mask) {
        // clang-format off
        THRESHOLD_OMP(
            MQ_DO_PRAGMA(omp parallel for reduction(+:res_real, res_imag) schedule(static)), dim, DimTh,
                for (omp::idx_t a = 0; a < static_cast<omp::idx_t>(dim / 2); a++) {
                    auto i = ((a & mask.obj_high_mask) << 1) + (a & mask.obj_low_mask);
                    auto j = i + mask.obj_mask;
                    qs_data_t this_res = 0;
                    for (index_t col = 0; col < dim; col++) {
                        this_res += (m00 * GetValue(qs, i, col) + m01 * GetValue(qs, j, col))
                                    * GetValue(ham_matrix, col, i);
                        this_res += (m10 * GetValue(qs, i, col) + m11 * GetValue(qs, j, col))
                                    * GetValue(ham_matrix, col, j);
                    }
                    res_real += this_res.real();
                    res_imag += this_res.imag();
                });
        // clang-format on
    } else {
        // clang-format off
        THRESHOLD_OMP(
            MQ_DO_PRAGMA(omp parallel for reduction(+:res_real, res_imag) schedule(static)), dim, DimTh,
                for (omp::idx_t a = 0; a < static_cast<omp::idx_t>(dim / 2); a++) {
                    auto i = ((a & mask.obj_high_mask) << 1) + (a & mask.obj_low_mask);
                    if ((i & mask.ctrl_mask) == mask.ctrl_mask) {
                        auto j = i + mask.obj_mask;
                        qs_data_t this_res = 0;
                        for (index_t col = 0; col < dim; col++) {
                            this_res += (m00 * GetValue(qs, i, col) + m01 * GetValue(qs, j, col))
                                        * GetValue(ham_matrix, col, i);
                            this_res += (m10 * GetValue(qs, i, col) + m11 * GetValue(qs, j, col))
                                        * GetValue(ham_matrix, col, j);
                        }
                        res_real += this_res.real();
                        res_imag += this_res.imag();
                    }
                });
        // clang-format on
    }
    if (will_free) {
        derived::FreeState(&qs);
    }
    if (will_free_ham) {
        derived::FreeState(&ham_matrix);
    }
    return {res_real, res_imag};
};

template <typename derived_, typename calc_type_>
auto CPUDensityMatrixPolicyBase<derived_, calc_type_>::ExpectDiffTwoQubitsMatrix(
    const qs_data_p_t& qs_out, const qs_data_p_t& ham_matrix_out, const qbits_t& objs, const qbits_t& ctrls,
    const matrix_t& gate_m, const matrix_t& diff_m, index_t dim) -> qs_data_t {
    qs_data_p_t qs;
    bool will_free = false;
    if (qs_out == nullptr) {
        qs = derived::InitState(dim);
        will_free = true;
    } else {
        qs = qs_out;
    }
    qs_data_p_t ham_matrix;
    bool will_free_ham = false;
    if (ham_matrix_out == nullptr) {
        ham_matrix = derived::InitState(dim);
        will_free_ham = true;
    } else {
        ham_matrix = ham_matrix_out;
    }
    // G = Tr(m \rho H), where m = U' \dot \dagger{U}
    matrix_t m(4, std::vector<qs_data_t>(4));
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            for (int k = 0; k < 4; k++) {
                m[i][j] += diff_m[i][k] * std::conj(gate_m[j][k]);
            }
        }
    }
    DoubleQubitGateMask mask(objs, ctrls);
    calc_type res_real = 0, res_imag = 0;
    if (!mask.ctrl_mask) {
        // clang-format off
        THRESHOLD_OMP(
            MQ_DO_PRAGMA(omp parallel for reduction(+:res_real, res_imag) schedule(static)), dim, DimTh,
                for (omp::idx_t a = 0; a < static_cast<omp::idx_t>(dim / 4); a++) {
                    index_t r0;  // row index of reduced matrix entry
                    SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask, mask.obj_high_mask, mask.obj_rev_high_mask,
                                  a, r0);
                    auto r3 = r0 + mask.obj_mask;
                    auto r1 = r0 + mask.obj_min_mask;
                    auto r2 = r0 + mask.obj_max_mask;
                    qs_data_t this_res = 0;
                    for (index_t col = 0; col < dim; col++) {
                        this_res += (m[0][0] * GetValue(qs, r0, col) + m[0][1] * GetValue(qs, r1, col)
                                     + m[0][2] * GetValue(qs, r2, col) + m[0][3] * GetValue(qs, r3, col))
                                    * GetValue(ham_matrix, col, r0);
                        this_res += (m[1][0] * GetValue(qs, r0, col) + m[1][1] * GetValue(qs, r1, col)
                                     + m[1][2] * GetValue(qs, r2, col) + m[1][3] * GetValue(qs, r3, col))
                                    * GetValue(ham_matrix, col, r1);
                        this_res += (m[2][0] * GetValue(qs, r0, col) + m[2][1] * GetValue(qs, r1, col)
                                     + m[2][2] * GetValue(qs, r2, col) + m[2][3] * GetValue(qs, r3, col))
                                    * GetValue(ham_matrix, col, r2);
                        this_res += (m[3][0] * GetValue(qs, r0, col) + m[3][1] * GetValue(qs, r1, col)
                                     + m[3][2] * GetValue(qs, r2, col) + m[3][3] * GetValue(qs, r3, col))
                                    * GetValue(ham_matrix, col, r3);
                    }
                    res_real += this_res.real();
                    res_imag += this_res.imag();
                });
        // clang-format on
    } else {
        // clang-format off
        THRESHOLD_OMP(
            MQ_DO_PRAGMA(omp parallel for reduction(+:res_real, res_imag) schedule(static)), dim, DimTh,
                for (omp::idx_t a = 0; a < static_cast<omp::idx_t>(dim / 4); a++) {
                    index_t r0;  // row index of reduced matrix entry
                    SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask, mask.obj_high_mask, mask.obj_rev_high_mask,
                                  a, r0);
                    if ((r0 & mask.ctrl_mask) == mask.ctrl_mask) {
                        auto r3 = r0 + mask.obj_mask;
                        auto r1 = r0 + mask.obj_min_mask;
                        auto r2 = r0 + mask.obj_max_mask;
                        qs_data_t this_res = 0;
                        for (index_t col = 0; col < dim; col++) {
                            this_res += (m[0][0] * GetValue(qs, r0, col) + m[0][1] * GetValue(qs, r1, col)
                                         + m[0][2] * GetValue(qs, r2, col) + m[0][3] * GetValue(qs, r3, col))
                                        * GetValue(ham_matrix, col, r0);
                            this_res += (m[1][0] * GetValue(qs, r0, col) + m[1][1] * GetValue(qs, r1, col)
                                         + m[1][2] * GetValue(qs, r2, col) + m[1][3] * GetValue(qs, r3, col))
                                        * GetValue(ham_matrix, col, r1);
                            this_res += (m[2][0] * GetValue(qs, r0, col) + m[2][1] * GetValue(qs, r1, col)
                                         + m[2][2] * GetValue(qs, r2, col) + m[2][3] * GetValue(qs, r3, col))
                                        * GetValue(ham_matrix, col, r2);
                            this_res += (m[3][0] * GetValue(qs, r0, col) + m[3][1] * GetValue(qs, r1, col)
                                         + m[3][2] * GetValue(qs, r2, col) + m[3][3] * GetValue(qs, r3, col))
                                        * GetValue(ham_matrix, col, r3);
                        }
                        res_real += this_res.real();
                        res_imag += this_res.imag();
                    }
                });
        // clang-format on
    }
    if (will_free) {
        derived::FreeState(&qs);
    }
    if (will_free_ham) {
        derived::FreeState(&ham_matrix);
    }
    return {res_real, res_imag};
};

template <typename derived_, typename calc_type_>
auto CPUDensityMatrixPolicyBase<derived_, calc_type_>::ExpectDiffNQubitsMatrix(
    const qs_data_p_t& qs_out, const qs_data_p_t& ham_matrix_out, const qbits_t& objs, const qbits_t& ctrls,
    const matrix_t& gate_m, const matrix_t& diff_m, index_t dim) -> qs_data_t {
    qs_data_p_t qs;
    bool will_free = false;
    if (qs_out == nullptr) {
        qs = derived::InitState(dim);
        will_free = true;
    } else {
        qs = qs_out;
    }
    qs_data_p_t ham_matrix;
    bool will_free_ham = false;
    if (ham_matrix_out == nullptr) {
        ham_matrix = derived::InitState(dim);
        will_free_ham = true;
    } else {
        ham_matrix = ham_matrix_out;
    }
    size_t n_qubit = objs.size();
    size_t m_dim = (static_cast<uint64_t>(1) << n_qubit);
    size_t ctrl_mask = 0;
    for (auto& i : ctrls) {
        ctrl_mask |= static_cast<uint64_t>(1) << i;
    }
    std::vector<size_t> obj_masks{};
    for (size_t i = 0; i < m_dim; i++) {
        size_t n = 0;
        size_t mask_j = 0;
        for (size_t j = i; j != 0; j >>= 1) {
            if (j & 1) {
                mask_j += static_cast<uint64_t>(1) << objs[n];
            }
            n += 1;
        }
        obj_masks.push_back(mask_j);
    }
    auto obj_mask = obj_masks.back();

    // G = Tr(m \rho H), where m = U' \dot \dagger{U}
    matrix_t m(m_dim, std::vector<qs_data_t>(m_dim));
    for (size_t i = 0; i < m_dim; i++) {
        for (size_t j = 0; j < m_dim; j++) {
            for (size_t k = 0; k < m_dim; k++) {
                m[i][j] += diff_m[i][k] * std::conj(gate_m[j][k]);
            }
        }
    }
    calc_type res_real = 0, res_imag = 0;
    THRESHOLD_OMP(
        MQ_DO_PRAGMA(omp parallel for reduction(+:res_real, res_imag) schedule(static)), dim, DimTh,
            for (omp::idx_t a = 0; a < static_cast<omp::idx_t>(dim); a++) {
                if (((a & ctrl_mask) == ctrl_mask) && ((a & obj_mask) == 0)) {
                    qs_data_t this_res = 0;
                    for (index_t col = 0; col < dim; col++) {
                        for (size_t i = 0; i < m_dim; i++) {
                            qs_data_t tmp = 0;
                            for (size_t j = 0; j < m_dim; j++) {
                                tmp += m[i][j] * GetValue(qs, obj_masks[j] | a, col);
                            }
                            this_res += tmp * GetValue(ham_matrix, col, obj_masks[i] | a);
                        }
                    }
                    res_real += this_res.real();
                    res_imag += this_res.imag();
                }
            });
    if (will_free) {
        derived::FreeState(&qs);
    }
    if (will_free_ham) {
        derived::FreeState(&ham_matrix);
    }
    return {res_real, res_imag};
};

template <typename derived_, typename calc_type_>
auto CPUDensityMatrixPolicyBase<derived_, calc_type_>::ExpectDiffMatrixGate(const qs_data_p_t& qs,
                                                                            const qs_data_p_t& ham_matrix,
                                                                            const qbits_t& objs, const qbits_t& ctrls,
                                                                            const matrix_t& diff_m,
                                                                            const matrix_t& herm_m, index_t dim)
    -> qs_data_t {
    if (objs.size() == 1) {
        return derived::ExpectDiffSingleQubitMatrix(qs, ham_matrix, objs, ctrls, diff_m, herm_m, dim);
    }
    if (objs.size() == 2) {
        return derived::ExpectDiffTwoQubitsMatrix(qs, ham_matrix, objs, ctrls, diff_m, herm_m, dim);
    }
    return derived::ExpectDiffNQubitsMatrix(qs, ham_matrix, objs, ctrls, diff_m, herm_m, dim);
}

template <typename derived_, typename calc_type_>
auto CPUDensityMatrixPolicyBase<derived_, calc_type_>::ExpectDiffRX(const qs_data_p_t& qs_out,
                                                                    const qs_data_p_t& ham_matrix_out,
                                                                    const qbits_t& objs, const qbits_t& ctrls,
                                                                    index_t dim) -> qs_data_t {
    qs_data_p_t qs;
    bool will_free = false;
    if (qs_out == nullptr) {
        qs = derived::InitState(dim);
        will_free = true;
    } else {
        qs = qs_out;
    }
    qs_data_p_t ham_matrix;
    bool will_free_ham = false;
    if (ham_matrix_out == nullptr) {
        ham_matrix = derived::InitState(dim);
        will_free_ham = true;
    } else {
        ham_matrix = ham_matrix_out;
    }
    SingleQubitGateMask mask(objs, ctrls);
    calc_type res_real = 0, res_imag = 0;
    if (!mask.ctrl_mask) {
        // clang-format off
        THRESHOLD_OMP(
            MQ_DO_PRAGMA(omp parallel for reduction(+:res_real, res_imag) schedule(static)), dim, DimTh,
                for (omp::idx_t a = 0; a < static_cast<omp::idx_t>(dim / 2); a++) {
                    auto i = ((a & mask.obj_high_mask) << 1) + (a & mask.obj_low_mask);
                    auto j = i + mask.obj_mask;
                    qs_data_t this_res = 0;
                    for (index_t col = 0; col < dim; col++) {
                        this_res += GetValue(qs, j, col) * GetValue(ham_matrix, col, i);
                        this_res += GetValue(qs, i, col) * GetValue(ham_matrix, col, j);
                    }
                    this_res *= HALF_MI;
                    res_real += this_res.real();
                    res_imag += this_res.imag();
                })
        // clang-format on
    } else {
        // clang-format off
        THRESHOLD_OMP(
            MQ_DO_PRAGMA(omp parallel for reduction(+:res_real, res_imag) schedule(static)), dim, DimTh,
                for (omp::idx_t a = 0; a < static_cast<omp::idx_t>(dim / 2); a++) {
                    auto i = ((a & mask.obj_high_mask) << 1) + (a & mask.obj_low_mask);
                    if ((i & mask.ctrl_mask) == mask.ctrl_mask) {
                        auto j = i + mask.obj_mask;
                        qs_data_t this_res = 0;
                        for (index_t col = 0; col < dim; col++) {
                            this_res += GetValue(qs, j, col) * GetValue(ham_matrix, col, i);
                            this_res += GetValue(qs, i, col) * GetValue(ham_matrix, col, j);
                        }
                        this_res *= HALF_MI;
                        res_real += this_res.real();
                        res_imag += this_res.imag();
                    }
                })
        // clang-format on
    }
    if (will_free) {
        derived::FreeState(&qs);
    }
    if (will_free_ham) {
        derived::FreeState(&ham_matrix);
    }
    return {res_real, res_imag};
};

template <typename derived_, typename calc_type_>
auto CPUDensityMatrixPolicyBase<derived_, calc_type_>::ExpectDiffRY(const qs_data_p_t& qs_out,
                                                                    const qs_data_p_t& ham_matrix_out,
                                                                    const qbits_t& objs, const qbits_t& ctrls,
                                                                    index_t dim) -> qs_data_t {
    qs_data_p_t qs;
    bool will_free = false;
    if (qs_out == nullptr) {
        qs = derived::InitState(dim);
        will_free = true;
    } else {
        qs = qs_out;
    }
    qs_data_p_t ham_matrix;
    bool will_free_ham = false;
    if (ham_matrix_out == nullptr) {
        ham_matrix = derived::InitState(dim);
        will_free_ham = true;
    } else {
        ham_matrix = ham_matrix_out;
    }
    SingleQubitGateMask mask(objs, ctrls);
    calc_type res_real = 0, res_imag = 0;
    if (!mask.ctrl_mask) {
        // clang-format off
        THRESHOLD_OMP(
            MQ_DO_PRAGMA(omp parallel for reduction(+:res_real, res_imag) schedule(static)), dim, DimTh,
                for (omp::idx_t a = 0; a < static_cast<omp::idx_t>(dim / 2); a++) {
                    auto i = ((a & mask.obj_high_mask) << 1) + (a & mask.obj_low_mask);
                    auto j = i + mask.obj_mask;
                    qs_data_t this_res = 0;
                    for (index_t col = 0; col < dim; col++) {
                        this_res += -GetValue(qs, j, col) * GetValue(ham_matrix, col, i);
                        this_res += GetValue(qs, i, col) * GetValue(ham_matrix, col, j);
                    }
                    this_res *= static_cast<calc_type>(0.5);
                    res_real += this_res.real();
                    res_imag += this_res.imag();
                })
        // clang-format on
    } else {
        // clang-format off
        THRESHOLD_OMP(
            MQ_DO_PRAGMA(omp parallel for reduction(+:res_real, res_imag) schedule(static)), dim, DimTh,
                for (omp::idx_t a = 0; a < static_cast<omp::idx_t>(dim / 2); a++) {
                    auto i = ((a & mask.obj_high_mask) << 1) + (a & mask.obj_low_mask);
                    if ((i & mask.ctrl_mask) == mask.ctrl_mask) {
                        auto j = i + mask.obj_mask;
                        qs_data_t this_res = 0;
                        for (index_t col = 0; col < dim; col++) {
                            this_res += -GetValue(qs, j, col) * GetValue(ham_matrix, col, i);
                            this_res += GetValue(qs, i, col) * GetValue(ham_matrix, col, j);
                        }
                        this_res *= static_cast<calc_type>(0.5);
                        res_real += this_res.real();
                        res_imag += this_res.imag();
                    }
                })
        // clang-format on
    }
    if (will_free) {
        derived::FreeState(&qs);
    }
    if (will_free_ham) {
        derived::FreeState(&ham_matrix);
    }
    return {res_real, res_imag};
};

template <typename derived_, typename calc_type_>
auto CPUDensityMatrixPolicyBase<derived_, calc_type_>::ExpectDiffRZ(const qs_data_p_t& qs_out,
                                                                    const qs_data_p_t& ham_matrix_out,
                                                                    const qbits_t& objs, const qbits_t& ctrls,
                                                                    index_t dim) -> qs_data_t {
    qs_data_p_t qs;
    bool will_free = false;
    if (qs_out == nullptr) {
        qs = derived::InitState(dim);
        will_free = true;
    } else {
        qs = qs_out;
    }
    qs_data_p_t ham_matrix;
    bool will_free_ham = false;
    if (ham_matrix_out == nullptr) {
        ham_matrix = derived::InitState(dim);
        will_free_ham = true;
    } else {
        ham_matrix = ham_matrix_out;
    }
    SingleQubitGateMask mask(objs, ctrls);
    calc_type res_real = 0, res_imag = 0;
    if (!mask.ctrl_mask) {
        // clang-format off
        THRESHOLD_OMP(
            MQ_DO_PRAGMA(omp parallel for reduction(+:res_real, res_imag) schedule(static)), dim, DimTh,
                for (omp::idx_t a = 0; a < static_cast<omp::idx_t>(dim / 2); a++) {
                    auto i = ((a & mask.obj_high_mask) << 1) + (a & mask.obj_low_mask);
                    auto j = i + mask.obj_mask;
                    qs_data_t this_res = 0;
                    for (index_t col = 0; col < dim; col++) {
                        this_res += -GetValue(qs, i, col) * GetValue(ham_matrix, col, i);
                        this_res += GetValue(qs, j, col) * GetValue(ham_matrix, col, j);
                    }
                    this_res *= HALF_I;
                    res_real += this_res.real();
                    res_imag += this_res.imag();
                })
        // clang-format on
    } else {
        // clang-format off
        THRESHOLD_OMP(
            MQ_DO_PRAGMA(omp parallel for reduction(+:res_real, res_imag) schedule(static)), dim, DimTh,
                for (omp::idx_t a = 0; a < static_cast<omp::idx_t>(dim / 2); a++) {
                    auto i = ((a & mask.obj_high_mask) << 1) + (a & mask.obj_low_mask);
                    if ((i & mask.ctrl_mask) == mask.ctrl_mask) {
                        auto j = i + mask.obj_mask;
                        qs_data_t this_res = 0;
                        for (index_t col = 0; col < dim; col++) {
                            this_res += -GetValue(qs, i, col) * GetValue(ham_matrix, col, i);
                            this_res += GetValue(qs, j, col) * GetValue(ham_matrix, col, j);
                        }
                        this_res *= HALF_I;
                        res_real += this_res.real();
                        res_imag += this_res.imag();
                    }
                })
        // clang-format on
    }
    if (will_free) {
        derived::FreeState(&qs);
    }
    if (will_free_ham) {
        derived::FreeState(&ham_matrix);
    }
    return {res_real, res_imag};
};

template <typename derived_, typename calc_type_>
auto CPUDensityMatrixPolicyBase<derived_, calc_type_>::ExpectDiffPS(const qs_data_p_t& qs_out,
                                                                    const qs_data_p_t& ham_matrix_out,
                                                                    const qbits_t& objs, const qbits_t& ctrls,
                                                                    index_t dim) -> qs_data_t {
    qs_data_p_t qs;
    bool will_free = false;
    if (qs_out == nullptr) {
        qs = derived::InitState(dim);
        will_free = true;
    } else {
        qs = qs_out;
    }
    qs_data_p_t ham_matrix;
    bool will_free_ham = false;
    if (ham_matrix_out == nullptr) {
        ham_matrix = derived::InitState(dim);
        will_free_ham = true;
    } else {
        ham_matrix = ham_matrix_out;
    }
    SingleQubitGateMask mask(objs, ctrls);
    calc_type res_real = 0, res_imag = 0;
    if (!mask.ctrl_mask) {
        // clang-format off
        THRESHOLD_OMP(
            MQ_DO_PRAGMA(omp parallel for reduction(+:res_real, res_imag) schedule(static)), dim, DimTh,
                for (omp::idx_t a = 0; a < static_cast<omp::idx_t>(dim / 2); a++) {
                    auto i = ((a & mask.obj_high_mask) << 1) + (a & mask.obj_low_mask);
                    auto j = i + mask.obj_mask;
                    qs_data_t this_res = 0;
                    for (index_t col = 0; col < dim; col++) {
                        this_res += GetValue(qs, j, col) * GetValue(ham_matrix, col, j);
                    }
                    this_res *= IMAGE_I;
                    res_real += this_res.real();
                    res_imag += this_res.imag();
                })
        // clang-format on
    } else {
        // clang-format off
        THRESHOLD_OMP(
            MQ_DO_PRAGMA(omp parallel for reduction(+:res_real, res_imag) schedule(static)), dim, DimTh,
                for (omp::idx_t a = 0; a < static_cast<omp::idx_t>(dim / 2); a++) {
                    auto i = ((a & mask.obj_high_mask) << 1) + (a & mask.obj_low_mask);
                    if ((i & mask.ctrl_mask) == mask.ctrl_mask) {
                        auto j = i + mask.obj_mask;
                        qs_data_t this_res = 0;
                        for (index_t col = 0; col < dim; col++) {
                            this_res += GetValue(qs, j, col) * GetValue(ham_matrix, col, j);
                        }
                        this_res *= IMAGE_I;
                        res_real += this_res.real();
                        res_imag += this_res.imag();
                    }
                })
        // clang-format on
    }
    if (will_free) {
        derived::FreeState(&qs);
    }
    if (will_free_ham) {
        derived::FreeState(&ham_matrix);
    }
    return {res_real, res_imag};
};

template <typename derived_, typename calc_type_>
auto CPUDensityMatrixPolicyBase<derived_, calc_type_>::ExpectDiffGP(const qs_data_p_t& qs_out,
                                                                    const qs_data_p_t& ham_matrix_out,
                                                                    const qbits_t& objs, const qbits_t& ctrls,
                                                                    index_t dim) -> qs_data_t {
    qs_data_p_t qs;
    bool will_free = false;
    if (qs_out == nullptr) {
        qs = derived::InitState(dim);
        will_free = true;
    } else {
        qs = qs_out;
    }
    qs_data_p_t ham_matrix;
    bool will_free_ham = false;
    if (ham_matrix_out == nullptr) {
        ham_matrix = derived::InitState(dim);
        will_free_ham = true;
    } else {
        ham_matrix = ham_matrix_out;
    }
    SingleQubitGateMask mask(objs, ctrls);
    calc_type res_real = 0, res_imag = 0;
    if (!mask.ctrl_mask) {
        // clang-format off
        THRESHOLD_OMP(
            MQ_DO_PRAGMA(omp parallel for reduction(+:res_real, res_imag) schedule(static)), dim, DimTh,
                for (omp::idx_t a = 0; a < (dim / 2); a++) {
                    auto i = ((a & mask.obj_high_mask) << 1) + (a & mask.obj_low_mask);
                    auto j = i + mask.obj_mask;
                    qs_data_t this_res = 0;
                    for (index_t col = 0; col < dim; col++) {
                        this_res += GetValue(qs, i, col) * GetValue(ham_matrix, col, i)
                                    + GetValue(qs, j, col) * GetValue(ham_matrix, col, j);
                    }
                    this_res *= IMAGE_MI;
                    res_real += this_res.real();
                    res_imag += this_res.imag();
                })
        // clang-format on
    } else {
        // clang-format off
        THRESHOLD_OMP(
            MQ_DO_PRAGMA(omp parallel for reduction(+:res_real, res_imag) schedule(static)), dim, DimTh,
                for (omp::idx_t a = 0; a < (dim / 2); a++) {
                    auto i = ((a & mask.obj_high_mask) << 1) + (a & mask.obj_low_mask);
                    if ((i & mask.ctrl_mask) == mask.ctrl_mask) {
                        auto j = i + mask.obj_mask;
                        qs_data_t this_res = 0;
                        for (index_t col = 0; col < dim; col++) {
                            this_res += GetValue(qs, i, col) * GetValue(ham_matrix, col, i)
                                        + GetValue(qs, j, col) * GetValue(ham_matrix, col, j);
                        }
                        this_res *= IMAGE_MI;
                        res_real += this_res.real();
                        res_imag += this_res.imag();
                    }
                })
        // clang-format on
    }
    if (will_free) {
        derived::FreeState(&qs);
    }
    if (will_free_ham) {
        derived::FreeState(&ham_matrix);
    }
    return {res_real, res_imag};
};

template <typename derived_, typename calc_type_>
auto CPUDensityMatrixPolicyBase<derived_, calc_type_>::ExpectDiffU3Theta(const qs_data_p_t& qs_out,
                                                                         const qs_data_p_t& ham_matrix_out,
                                                                         const qbits_t& objs, const qbits_t& ctrls,
                                                                         calc_type phi, index_t dim) -> qs_data_t {
    qs_data_p_t qs;
    bool will_free = false;
    if (qs_out == nullptr) {
        qs = derived::InitState(dim);
        will_free = true;
    } else {
        qs = qs_out;
    }
    qs_data_p_t ham_matrix;
    bool will_free_ham = false;
    if (ham_matrix_out == nullptr) {
        ham_matrix = derived::InitState(dim);
        will_free_ham = true;
    } else {
        ham_matrix = ham_matrix_out;
    }
    qs_data_t e_phi = static_cast<calc_type>(0.5) * std::exp(std::complex<calc_type>(0, phi));
    qs_data_t e_m_phi = static_cast<calc_type>(-0.5) * std::exp(std::complex<calc_type>(0, -phi));
    SingleQubitGateMask mask(objs, ctrls);
    calc_type res_real = 0, res_imag = 0;
    if (!mask.ctrl_mask) {
        // clang-format off
        THRESHOLD_OMP(
            MQ_DO_PRAGMA(omp parallel for reduction(+:res_real, res_imag) schedule(static)), dim, DimTh,
                for (omp::idx_t a = 0; a < static_cast<omp::idx_t>(dim / 2); a++) {
                    auto i = ((a & mask.obj_high_mask) << 1) + (a & mask.obj_low_mask);
                    auto j = i + mask.obj_mask;
                    qs_data_t this_res = 0;
                    for (index_t col = 0; col < dim; col++) {
                        this_res += e_m_phi * GetValue(qs, j, col) * GetValue(ham_matrix, col, i);
                        this_res += e_phi * GetValue(qs, i, col) * GetValue(ham_matrix, col, j);
                    }
                    res_real += this_res.real();
                    res_imag += this_res.imag();
                })
        // clang-format on
    } else {
        // clang-format off
        THRESHOLD_OMP(
            MQ_DO_PRAGMA(omp parallel for reduction(+:res_real, res_imag) schedule(static)), dim, DimTh,
                for (omp::idx_t a = 0; a < static_cast<omp::idx_t>(dim / 2); a++) {
                    auto i = ((a & mask.obj_high_mask) << 1) + (a & mask.obj_low_mask);
                    if ((i & mask.ctrl_mask) == mask.ctrl_mask) {
                        auto j = i + mask.obj_mask;
                        qs_data_t this_res = 0;
                        for (index_t col = 0; col < dim; col++) {
                            this_res += e_m_phi * GetValue(qs, j, col) * GetValue(ham_matrix, col, i);
                            this_res += e_phi * GetValue(qs, i, col) * GetValue(ham_matrix, col, j);
                        }
                        res_real += this_res.real();
                        res_imag += this_res.imag();
                    }
                })
        // clang-format on
    }
    if (will_free) {
        derived::FreeState(&qs);
    }
    if (will_free_ham) {
        derived::FreeState(&ham_matrix);
    }
    return {res_real, res_imag};
};

template <typename derived_, typename calc_type_>
auto CPUDensityMatrixPolicyBase<derived_, calc_type_>::ExpectDiffU3Phi(const qs_data_p_t& qs_out,
                                                                       const qs_data_p_t& ham_matrix_out,
                                                                       const qbits_t& objs, const qbits_t& ctrls,
                                                                       index_t dim) -> qs_data_t {
    qs_data_p_t qs;
    bool will_free = false;
    if (qs_out == nullptr) {
        qs = derived::InitState(dim);
        will_free = true;
    } else {
        qs = qs_out;
    }
    qs_data_p_t ham_matrix;
    bool will_free_ham = false;
    if (ham_matrix_out == nullptr) {
        ham_matrix = derived::InitState(dim);
        will_free_ham = true;
    } else {
        ham_matrix = ham_matrix_out;
    }
    SingleQubitGateMask mask(objs, ctrls);
    calc_type res_real = 0, res_imag = 0;
    if (!mask.ctrl_mask) {
        // clang-format off
        THRESHOLD_OMP(
            MQ_DO_PRAGMA(omp parallel for reduction(+:res_real, res_imag) schedule(static)), dim, DimTh,
                for (omp::idx_t a = 0; a < static_cast<omp::idx_t>(dim / 2); a++) {
                    auto i = ((a & mask.obj_high_mask) << 1) + (a & mask.obj_low_mask);
                    auto j = i + mask.obj_mask;
                    qs_data_t this_res = 0;
                    for (index_t col = 0; col < dim; col++) {
                        this_res += GetValue(qs, j, col) * GetValue(ham_matrix, col, j);
                    }
                    this_res *= IMAGE_I;
                    res_real += this_res.real();
                    res_imag += this_res.imag();
                })
        // clang-format on
    } else {
        // clang-format off
        THRESHOLD_OMP(
            MQ_DO_PRAGMA(omp parallel for reduction(+:res_real, res_imag) schedule(static)), dim, DimTh,
                for (omp::idx_t a = 0; a < static_cast<omp::idx_t>(dim / 2); a++) {
                    auto i = ((a & mask.obj_high_mask) << 1) + (a & mask.obj_low_mask);
                    if ((i & mask.ctrl_mask) == mask.ctrl_mask) {
                        auto j = i + mask.obj_mask;
                        qs_data_t this_res = 0;
                        for (index_t col = 0; col < dim; col++) {
                            this_res += GetValue(qs, j, col) * GetValue(ham_matrix, col, j);
                        }
                        this_res *= IMAGE_I;
                        res_real += this_res.real();
                        res_imag += this_res.imag();
                    }
                })
        // clang-format on
    }
    if (will_free) {
        derived::FreeState(&qs);
    }
    if (will_free_ham) {
        derived::FreeState(&ham_matrix);
    }
    return {res_real, res_imag};
};

template <typename derived_, typename calc_type_>
auto CPUDensityMatrixPolicyBase<derived_, calc_type_>::ExpectDiffRxx(const qs_data_p_t& qs_out,
                                                                     const qs_data_p_t& ham_matrix_out,
                                                                     const qbits_t& objs, const qbits_t& ctrls,
                                                                     index_t dim) -> qs_data_t {
    qs_data_p_t qs;
    bool will_free = false;
    if (qs_out == nullptr) {
        qs = derived::InitState(dim);
        will_free = true;
    } else {
        qs = qs_out;
    }
    qs_data_p_t ham_matrix;
    bool will_free_ham = false;
    if (ham_matrix_out == nullptr) {
        ham_matrix = derived::InitState(dim);
        will_free_ham = true;
    } else {
        ham_matrix = ham_matrix_out;
    }
    DoubleQubitGateMask mask(objs, ctrls);
    calc_type res_real = 0, res_imag = 0;
    if (!mask.ctrl_mask) {
        // clang-format off
        THRESHOLD_OMP(
            MQ_DO_PRAGMA(omp parallel for reduction(+:res_real, res_imag) schedule(static)), dim, DimTh,
                for (omp::idx_t l = 0; l < static_cast<omp::idx_t>(dim / 4); l++) {
                    index_t r0;
                    SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask, mask.obj_high_mask, mask.obj_rev_high_mask,
                                  l, r0);
                    auto r3 = r0 + mask.obj_mask;
                    auto r1 = r0 + mask.obj_min_mask;
                    auto r2 = r0 + mask.obj_max_mask;
                    qs_data_t this_res = 0;
                    for (index_t col = 0; col < dim; col++) {
                        this_res += GetValue(qs, r3, col) * GetValue(ham_matrix, col, r0)
                                    + GetValue(qs, r2, col) * GetValue(ham_matrix, col, r1)
                                    + GetValue(qs, r1, col) * GetValue(ham_matrix, col, r2)
                                    + GetValue(qs, r0, col) * GetValue(ham_matrix, col, r3);
                    }
                    this_res *= HALF_MI;
                    res_real += this_res.real();
                    res_imag += this_res.imag();
                })
        // clang-format on
    } else {
        // clang-format off
        THRESHOLD_OMP(
            MQ_DO_PRAGMA(omp parallel for reduction(+:res_real, res_imag) schedule(static)), dim, DimTh,
                for (omp::idx_t l = 0; l < static_cast<omp::idx_t>(dim / 4); l++) {
                    index_t r0;
                    SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask, mask.obj_high_mask, mask.obj_rev_high_mask,
                                  l, r0);
                    if ((r0 & mask.ctrl_mask) == mask.ctrl_mask) {
                        auto r3 = r0 + mask.obj_mask;
                        auto r1 = r0 + mask.obj_min_mask;
                        auto r2 = r0 + mask.obj_max_mask;
                        qs_data_t this_res = 0;
                        for (index_t col = 0; col < dim; col++) {
                            this_res += GetValue(qs, r3, col) * GetValue(ham_matrix, col, r0)
                                        + GetValue(qs, r2, col) * GetValue(ham_matrix, col, r1)
                                        + GetValue(qs, r1, col) * GetValue(ham_matrix, col, r2)
                                        + GetValue(qs, r0, col) * GetValue(ham_matrix, col, r3);
                        }
                        this_res *= HALF_MI;
                        res_real += this_res.real();
                        res_imag += this_res.imag();
                    }
                })
        // clang-format on
    }
    if (will_free) {
        derived::FreeState(&qs);
    }
    if (will_free_ham) {
        derived::FreeState(&ham_matrix);
    }
    return {res_real, res_imag};
};

template <typename derived_, typename calc_type_>
auto CPUDensityMatrixPolicyBase<derived_, calc_type_>::ExpectDiffRyy(const qs_data_p_t& qs_out,
                                                                     const qs_data_p_t& ham_matrix_out,
                                                                     const qbits_t& objs, const qbits_t& ctrls,
                                                                     index_t dim) -> qs_data_t {
    qs_data_p_t qs;
    bool will_free = false;
    if (qs_out == nullptr) {
        qs = derived::InitState(dim);
        will_free = true;
    } else {
        qs = qs_out;
    }
    qs_data_p_t ham_matrix;
    bool will_free_ham = false;
    if (ham_matrix_out == nullptr) {
        ham_matrix = derived::InitState(dim);
        will_free_ham = true;
    } else {
        ham_matrix = ham_matrix_out;
    }
    DoubleQubitGateMask mask(objs, ctrls);
    calc_type res_real = 0, res_imag = 0;
    if (!mask.ctrl_mask) {
        // clang-format off
        THRESHOLD_OMP(
            MQ_DO_PRAGMA(omp parallel for reduction(+:res_real, res_imag) schedule(static)), dim, DimTh,
                for (omp::idx_t l = 0; l < static_cast<omp::idx_t>(dim / 4); l++) {
                    index_t r0;
                    SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask, mask.obj_high_mask, mask.obj_rev_high_mask,
                                  l, r0);
                    auto r3 = r0 + mask.obj_mask;
                    auto r1 = r0 + mask.obj_min_mask;
                    auto r2 = r0 + mask.obj_max_mask;
                    qs_data_t this_res = 0;
                    for (index_t col = 0; col < dim; col++) {
                        this_res += GetValue(qs, r3, col) * GetValue(ham_matrix, col, r0)
                                    - GetValue(qs, r2, col) * GetValue(ham_matrix, col, r1)
                                    - GetValue(qs, r1, col) * GetValue(ham_matrix, col, r2)
                                    + GetValue(qs, r0, col) * GetValue(ham_matrix, col, r3);
                    }
                    this_res *= HALF_I;
                    res_real += this_res.real();
                    res_imag += this_res.imag();
                })
        // clang-format on
    } else {
        // clang-format off
        THRESHOLD_OMP(
            MQ_DO_PRAGMA(omp parallel for reduction(+:res_real, res_imag) schedule(static)), dim, DimTh,
                for (omp::idx_t l = 0; l < static_cast<omp::idx_t>(dim / 4); l++) {
                    index_t r0;
                    SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask, mask.obj_high_mask, mask.obj_rev_high_mask,
                                  l, r0);
                    if ((r0 & mask.ctrl_mask) == mask.ctrl_mask) {
                        auto r3 = r0 + mask.obj_mask;
                        auto r1 = r0 + mask.obj_min_mask;
                        auto r2 = r0 + mask.obj_max_mask;
                        qs_data_t this_res = 0;
                        for (index_t col = 0; col < dim; col++) {
                            this_res += GetValue(qs, r3, col) * GetValue(ham_matrix, col, r0)
                                        - GetValue(qs, r2, col) * GetValue(ham_matrix, col, r1)
                                        - GetValue(qs, r1, col) * GetValue(ham_matrix, col, r2)
                                        + GetValue(qs, r0, col) * GetValue(ham_matrix, col, r3);
                        }
                        this_res *= HALF_I;
                        res_real += this_res.real();
                        res_imag += this_res.imag();
                    }
                })
        // clang-format on
    }
    if (will_free) {
        derived::FreeState(&qs);
    }
    if (will_free_ham) {
        derived::FreeState(&ham_matrix);
    }
    return {res_real, res_imag};
};

template <typename derived_, typename calc_type_>
auto CPUDensityMatrixPolicyBase<derived_, calc_type_>::ExpectDiffRzz(const qs_data_p_t& qs_out,
                                                                     const qs_data_p_t& ham_matrix_out,
                                                                     const qbits_t& objs, const qbits_t& ctrls,
                                                                     index_t dim) -> qs_data_t {
    qs_data_p_t qs;
    bool will_free = false;
    if (qs_out == nullptr) {
        qs = derived::InitState(dim);
        will_free = true;
    } else {
        qs = qs_out;
    }
    qs_data_p_t ham_matrix;
    bool will_free_ham = false;
    if (ham_matrix_out == nullptr) {
        ham_matrix = derived::InitState(dim);
        will_free_ham = true;
    } else {
        ham_matrix = ham_matrix_out;
    }
    DoubleQubitGateMask mask(objs, ctrls);
    calc_type res_real = 0, res_imag = 0;
    if (!mask.ctrl_mask) {
        // clang-format off
        THRESHOLD_OMP(
            MQ_DO_PRAGMA(omp parallel for reduction(+:res_real, res_imag) schedule(static)), dim, DimTh,
                for (omp::idx_t l = 0; l < static_cast<omp::idx_t>(dim / 4); l++) {
                    index_t r0;
                    SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask, mask.obj_high_mask, mask.obj_rev_high_mask,
                                  l, r0);
                    auto r3 = r0 + mask.obj_mask;
                    auto r1 = r0 + mask.obj_min_mask;
                    auto r2 = r0 + mask.obj_max_mask;
                    qs_data_t this_res = 0;
                    for (index_t col = 0; col < dim; col++) {
                        this_res += GetValue(qs, r0, col) * GetValue(ham_matrix, col, r0)
                                    - GetValue(qs, r1, col) * GetValue(ham_matrix, col, r1)
                                    - GetValue(qs, r2, col) * GetValue(ham_matrix, col, r2)
                                    + GetValue(qs, r3, col) * GetValue(ham_matrix, col, r3);
                    }
                    this_res *= HALF_MI;
                    res_real += this_res.real();
                    res_imag += this_res.imag();
                })
        // clang-format on
    } else {
        // clang-format off
        THRESHOLD_OMP(
            MQ_DO_PRAGMA(omp parallel for reduction(+:res_real, res_imag) schedule(static)), dim, DimTh,
                for (omp::idx_t l = 0; l < static_cast<omp::idx_t>(dim / 4); l++) {
                    index_t r0;
                    SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask, mask.obj_high_mask, mask.obj_rev_high_mask,
                                  l, r0);
                    if ((r0 & mask.ctrl_mask) == mask.ctrl_mask) {
                        auto r3 = r0 + mask.obj_mask;
                        auto r1 = r0 + mask.obj_min_mask;
                        auto r2 = r0 + mask.obj_max_mask;
                        qs_data_t this_res = 0;
                        for (index_t col = 0; col < dim; col++) {
                            this_res += GetValue(qs, r0, col) * GetValue(ham_matrix, col, r0)
                                        - GetValue(qs, r1, col) * GetValue(ham_matrix, col, r1)
                                        - GetValue(qs, r2, col) * GetValue(ham_matrix, col, r2)
                                        + GetValue(qs, r3, col) * GetValue(ham_matrix, col, r3);
                        }
                        this_res *= HALF_MI;
                        res_real += this_res.real();
                        res_imag += this_res.imag();
                    }
                })
        // clang-format on
    }
    if (will_free) {
        derived::FreeState(&qs);
    }
    if (will_free_ham) {
        derived::FreeState(&ham_matrix);
    }
    return {res_real, res_imag};
};

template <typename derived_, typename calc_type_>
auto CPUDensityMatrixPolicyBase<derived_, calc_type_>::ExpectDiffRxy(const qs_data_p_t& qs_out,
                                                                     const qs_data_p_t& ham_matrix_out,
                                                                     const qbits_t& objs, const qbits_t& ctrls,
                                                                     index_t dim) -> qs_data_t {
    qs_data_p_t qs;
    bool will_free = false;
    if (qs_out == nullptr) {
        qs = derived::InitState(dim);
        will_free = true;
    } else {
        qs = qs_out;
    }
    qs_data_p_t ham_matrix;
    bool will_free_ham = false;
    if (ham_matrix_out == nullptr) {
        ham_matrix = derived::InitState(dim);
        will_free_ham = true;
    } else {
        ham_matrix = ham_matrix_out;
    }
    DoubleQubitGateMask mask(objs, ctrls);
    calc_type res_real = 0, res_imag = 0;
    if (!mask.ctrl_mask) {
        // clang-format off
        THRESHOLD_OMP(
            MQ_DO_PRAGMA(omp parallel for reduction(+:res_real, res_imag) schedule(static)), dim, DimTh,
                for (omp::idx_t l = 0; l < (dim / 4); l++) {
                    index_t r0;
                    SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask, mask.obj_high_mask, mask.obj_rev_high_mask,
                                  l, r0);
                    auto r3 = r0 + mask.obj_mask;
                    auto r1 = r0 + mask.obj_min_mask;
                    auto r2 = r0 + mask.obj_max_mask;
                    qs_data_t this_res = 0;
                    for (index_t col = 0; col < dim; col++) {
                        this_res += - GetValue(qs, r3, col) * GetValue(ham_matrix, col, r0)
                                    - GetValue(qs, r2, col) * GetValue(ham_matrix, col, r1)
                                    + GetValue(qs, r1, col) * GetValue(ham_matrix, col, r2)
                                    + GetValue(qs, r0, col) * GetValue(ham_matrix, col, r3);
                    }
                    this_res *= 0.5;
                    res_real += this_res.real();
                    res_imag += this_res.imag();
                })
        // clang-format on
    } else {
        // clang-format off
        THRESHOLD_OMP(
            MQ_DO_PRAGMA(omp parallel for reduction(+:res_real, res_imag) schedule(static)), dim, DimTh,
                for (omp::idx_t l = 0; l < (dim / 4); l++) {
                    index_t r0;
                    SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask, mask.obj_high_mask, mask.obj_rev_high_mask,
                                  l, r0);
                    if ((r0 & mask.ctrl_mask) == mask.ctrl_mask) {
                        auto r3 = r0 + mask.obj_mask;
                        auto r1 = r0 + mask.obj_min_mask;
                        auto r2 = r0 + mask.obj_max_mask;
                        qs_data_t this_res = 0;
                        for (index_t col = 0; col < dim; col++) {
                            this_res += - GetValue(qs, r3, col) * GetValue(ham_matrix, col, r0)
                                        - GetValue(qs, r2, col) * GetValue(ham_matrix, col, r1)
                                        + GetValue(qs, r1, col) * GetValue(ham_matrix, col, r2)
                                        + GetValue(qs, r0, col) * GetValue(ham_matrix, col, r3);
                        }
                        this_res *= 0.5;
                        res_real += this_res.real();
                        res_imag += this_res.imag();
                    }
                })
        // clang-format on
    }
    if (will_free) {
        derived::FreeState(&qs);
    }
    if (will_free_ham) {
        derived::FreeState(&ham_matrix);
    }
    return {res_real, res_imag};
};

template <typename derived_, typename calc_type_>
auto CPUDensityMatrixPolicyBase<derived_, calc_type_>::ExpectDiffRxz(const qs_data_p_t& qs_out,
                                                                     const qs_data_p_t& ham_matrix_out,
                                                                     const qbits_t& objs, const qbits_t& ctrls,
                                                                     index_t dim) -> qs_data_t {
    qs_data_p_t qs;
    bool will_free = false;
    if (qs_out == nullptr) {
        qs = derived::InitState(dim);
        will_free = true;
    } else {
        qs = qs_out;
    }
    qs_data_p_t ham_matrix;
    bool will_free_ham = false;
    if (ham_matrix_out == nullptr) {
        ham_matrix = derived::InitState(dim);
        will_free_ham = true;
    } else {
        ham_matrix = ham_matrix_out;
    }
    DoubleQubitGateMask mask(objs, ctrls);
    calc_type res_real = 0, res_imag = 0;
    if (!mask.ctrl_mask) {
        // clang-format off
        THRESHOLD_OMP(
            MQ_DO_PRAGMA(omp parallel for reduction(+:res_real, res_imag) schedule(static)), dim, DimTh,
                for (omp::idx_t l = 0; l < (dim / 4); l++) {
                    index_t r0;
                    SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask, mask.obj_high_mask, mask.obj_rev_high_mask,
                                  l, r0);
                    auto r3 = r0 + mask.obj_mask;
                    auto r1 = r0 + mask.obj_min_mask;
                    auto r2 = r0 + mask.obj_max_mask;
                    qs_data_t this_res = 0;
                    for (index_t col = 0; col < dim; col++) {
                        this_res += - GetValue(qs, r1, col) * GetValue(ham_matrix, col, r0)
                                    - GetValue(qs, r0, col) * GetValue(ham_matrix, col, r1)
                                    + GetValue(qs, r3, col) * GetValue(ham_matrix, col, r2)
                                    + GetValue(qs, r2, col) * GetValue(ham_matrix, col, r3);
                    }
                    this_res *= HALF_I;
                    res_real += this_res.real();
                    res_imag += this_res.imag();
                })
        // clang-format on
    } else {
        // clang-format off
        THRESHOLD_OMP(
            MQ_DO_PRAGMA(omp parallel for reduction(+:res_real, res_imag) schedule(static)), dim, DimTh,
                for (omp::idx_t l = 0; l < (dim / 4); l++) {
                    index_t r0;
                    SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask, mask.obj_high_mask, mask.obj_rev_high_mask,
                                  l, r0);
                    if ((r0 & mask.ctrl_mask) == mask.ctrl_mask) {
                        auto r3 = r0 + mask.obj_mask;
                        auto r1 = r0 + mask.obj_min_mask;
                        auto r2 = r0 + mask.obj_max_mask;
                        qs_data_t this_res = 0;
                        for (index_t col = 0; col < dim; col++) {
                            this_res += - GetValue(qs, r1, col) * GetValue(ham_matrix, col, r0)
                                        - GetValue(qs, r0, col) * GetValue(ham_matrix, col, r1)
                                        + GetValue(qs, r3, col) * GetValue(ham_matrix, col, r2)
                                        + GetValue(qs, r2, col) * GetValue(ham_matrix, col, r3);
                        }
                        this_res *= HALF_I;
                        res_real += this_res.real();
                        res_imag += this_res.imag();
                    }
                })
        // clang-format on
    }
    if (will_free) {
        derived::FreeState(&qs);
    }
    if (will_free_ham) {
        derived::FreeState(&ham_matrix);
    }
    return {res_real, res_imag};
};

template <typename derived_, typename calc_type_>
auto CPUDensityMatrixPolicyBase<derived_, calc_type_>::ExpectDiffRyz(const qs_data_p_t& qs_out,
                                                                     const qs_data_p_t& ham_matrix_out,
                                                                     const qbits_t& objs, const qbits_t& ctrls,
                                                                     index_t dim) -> qs_data_t {
    qs_data_p_t qs;
    bool will_free = false;
    if (qs_out == nullptr) {
        qs = derived::InitState(dim);
        will_free = true;
    } else {
        qs = qs_out;
    }
    qs_data_p_t ham_matrix;
    bool will_free_ham = false;
    if (ham_matrix_out == nullptr) {
        ham_matrix = derived::InitState(dim);
        will_free_ham = true;
    } else {
        ham_matrix = ham_matrix_out;
    }
    DoubleQubitGateMask mask(objs, ctrls);
    calc_type res_real = 0, res_imag = 0;
    if (!mask.ctrl_mask) {
        // clang-format off
        THRESHOLD_OMP(
            MQ_DO_PRAGMA(omp parallel for reduction(+:res_real, res_imag) schedule(static)), dim, DimTh,
                for (omp::idx_t l = 0; l < (dim / 4); l++) {
                    index_t r0;
                    SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask, mask.obj_high_mask, mask.obj_rev_high_mask,
                                  l, r0);
                    auto r3 = r0 + mask.obj_mask;
                    auto r1 = r0 + mask.obj_min_mask;
                    auto r2 = r0 + mask.obj_max_mask;
                    qs_data_t this_res = 0;
                    for (index_t col = 0; col < dim; col++) {
                        this_res += - GetValue(qs, r1, col) * GetValue(ham_matrix, col, r0)
                                    + GetValue(qs, r0, col) * GetValue(ham_matrix, col, r1)
                                    + GetValue(qs, r3, col) * GetValue(ham_matrix, col, r2)
                                    - GetValue(qs, r2, col) * GetValue(ham_matrix, col, r3);
                    }
                    this_res *= 0.5;
                    res_real += this_res.real();
                    res_imag += this_res.imag();
                })
        // clang-format on
    } else {
        // clang-format off
        THRESHOLD_OMP(
            MQ_DO_PRAGMA(omp parallel for reduction(+:res_real, res_imag) schedule(static)), dim, DimTh,
                for (omp::idx_t l = 0; l < (dim / 4); l++) {
                    index_t r0;
                    SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask, mask.obj_high_mask, mask.obj_rev_high_mask,
                                  l, r0);
                    if ((r0 & mask.ctrl_mask) == mask.ctrl_mask) {
                        auto r3 = r0 + mask.obj_mask;
                        auto r1 = r0 + mask.obj_min_mask;
                        auto r2 = r0 + mask.obj_max_mask;
                        qs_data_t this_res = 0;
                        for (index_t col = 0; col < dim; col++) {
                            this_res += - GetValue(qs, r1, col) * GetValue(ham_matrix, col, r0)
                                        + GetValue(qs, r0, col) * GetValue(ham_matrix, col, r1)
                                        + GetValue(qs, r3, col) * GetValue(ham_matrix, col, r2)
                                        - GetValue(qs, r2, col) * GetValue(ham_matrix, col, r3);
                        }
                        this_res *= 0.5;
                        res_real += this_res.real();
                        res_imag += this_res.imag();
                    }
                })
        // clang-format on
    }
    if (will_free) {
        derived::FreeState(&qs);
    }
    if (will_free_ham) {
        derived::FreeState(&ham_matrix);
    }
    return {res_real, res_imag};
};

template <typename derived_, typename calc_type_>
auto CPUDensityMatrixPolicyBase<derived_, calc_type_>::ExpectDiffSWAPalpha(const qs_data_p_t& qs_out,
                                                                           const qs_data_p_t& ham_matrix_out,
                                                                           const qbits_t& objs, const qbits_t& ctrls,
                                                                           index_t dim) -> qs_data_t {
    qs_data_p_t qs;
    bool will_free = false;
    if (qs_out == nullptr) {
        qs = derived::InitState(dim);
        will_free = true;
    } else {
        qs = qs_out;
    }
    qs_data_p_t ham_matrix;
    bool will_free_ham = false;
    if (ham_matrix_out == nullptr) {
        ham_matrix = derived::InitState(dim);
        will_free_ham = true;
    } else {
        ham_matrix = ham_matrix_out;
    }
    DoubleQubitGateMask mask(objs, ctrls);
    auto coeff = IMAGE_I * static_cast<calc_type_>(M_PI_2);
    calc_type res_real = 0, res_imag = 0;
    if (!mask.ctrl_mask) {
        // clang-format off
        THRESHOLD_OMP(
            MQ_DO_PRAGMA(omp parallel for reduction(+:res_real, res_imag) schedule(static)), dim, DimTh,
                for (omp::idx_t l = 0; l < static_cast<omp::idx_t>(dim / 4); l++) {
                    index_t r0;
                    SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask, mask.obj_high_mask, mask.obj_rev_high_mask,
                                  l, r0);
                    auto r3 = r0 + mask.obj_mask;
                    auto r1 = r0 + mask.obj_min_mask;
                    auto r2 = r0 + mask.obj_max_mask;
                    qs_data_t this_res = 0;
                    for (index_t col = 0; col < dim; col++) {
                        this_res += (GetValue(qs, r1, col) - GetValue(qs, r2, col))
                                    * (GetValue(ham_matrix, col, r1) - GetValue(ham_matrix, col, r2));
                    }
                    this_res *= coeff;
                    res_real += this_res.real();
                    res_imag += this_res.imag();
                })
        // clang-format on
    } else {
        // clang-format off
        THRESHOLD_OMP(
            MQ_DO_PRAGMA(omp parallel for reduction(+:res_real, res_imag) schedule(static)), dim, DimTh,
                for (omp::idx_t l = 0; l < static_cast<omp::idx_t>(dim / 4); l++) {
                    index_t r0;
                    SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask, mask.obj_high_mask, mask.obj_rev_high_mask,
                                  l, r0);
                    if ((r0 & mask.ctrl_mask) == mask.ctrl_mask) {
                        auto r3 = r0 + mask.obj_mask;
                        auto r1 = r0 + mask.obj_min_mask;
                        auto r2 = r0 + mask.obj_max_mask;
                        qs_data_t this_res = 0;
                        for (index_t col = 0; col < dim; col++) {
                            this_res += (GetValue(qs, r1, col) - GetValue(qs, r2, col))
                                        * (GetValue(ham_matrix, col, r1) - GetValue(ham_matrix, col, r2));
                        }
                        this_res *= coeff;
                        res_real += this_res.real();
                        res_imag += this_res.imag();
                    }
                })
        // clang-format on
    }
    if (will_free) {
        derived::FreeState(&qs);
    }
    if (will_free_ham) {
        derived::FreeState(&ham_matrix);
    }
    return {res_real, res_imag};
};

template <typename derived_, typename calc_type_>
auto CPUDensityMatrixPolicyBase<derived_, calc_type_>::ExpectDiffGivens(const qs_data_p_t& qs_out,
                                                                        const qs_data_p_t& ham_matrix_out,
                                                                        const qbits_t& objs, const qbits_t& ctrls,
                                                                        index_t dim) -> qs_data_t {
    qs_data_p_t qs;
    bool will_free = false;
    if (qs_out == nullptr) {
        qs = derived::InitState(dim);
        will_free = true;
    } else {
        qs = qs_out;
    }
    qs_data_p_t ham_matrix;
    bool will_free_ham = false;
    if (ham_matrix_out == nullptr) {
        ham_matrix = derived::InitState(dim);
        will_free_ham = true;
    } else {
        ham_matrix = ham_matrix_out;
    }
    DoubleQubitGateMask mask(objs, ctrls);
    calc_type res_real = 0, res_imag = 0;
    if (!mask.ctrl_mask) {
        // clang-format off
        THRESHOLD_OMP(
            MQ_DO_PRAGMA(omp parallel for reduction(+:res_real, res_imag) schedule(static)), dim, DimTh,
                for (omp::idx_t l = 0; l < static_cast<omp::idx_t>(dim / 4); l++) {
                    index_t r0;
                    SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask, mask.obj_high_mask, mask.obj_rev_high_mask,
                                  l, r0);
                    auto r3 = r0 + mask.obj_mask;
                    auto r1 = r0 + mask.obj_min_mask;
                    auto r2 = r0 + mask.obj_max_mask;
                    qs_data_t this_res = 0;
                    for (index_t col = 0; col < dim; col++) {
                        this_res += - GetValue(qs, r2, col) * GetValue(ham_matrix, col, r1)
                                    + GetValue(qs, r1, col) * GetValue(ham_matrix, col, r2);
                    }
                    res_real += this_res.real();
                    res_imag += this_res.imag();
                })
        // clang-format on
    } else {
        // clang-format off
        THRESHOLD_OMP(
            MQ_DO_PRAGMA(omp parallel for reduction(+:res_real, res_imag) schedule(static)), dim, DimTh,
                for (omp::idx_t l = 0; l < static_cast<omp::idx_t>(dim / 4); l++) {
                    index_t r0;
                    SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask, mask.obj_high_mask, mask.obj_rev_high_mask,
                                  l, r0);
                    if ((r0 & mask.ctrl_mask) == mask.ctrl_mask) {
                        auto r3 = r0 + mask.obj_mask;
                        auto r1 = r0 + mask.obj_min_mask;
                        auto r2 = r0 + mask.obj_max_mask;
                        qs_data_t this_res = 0;
                        for (index_t col = 0; col < dim; col++) {
                            this_res += - GetValue(qs, r2, col) * GetValue(ham_matrix, col, r1)
                                        + GetValue(qs, r1, col) * GetValue(ham_matrix, col, r2);
                        }
                        res_real += this_res.real();
                        res_imag += this_res.imag();
                    }
                })
        // clang-format on
    }
    if (will_free) {
        derived::FreeState(&qs);
    }
    if (will_free_ham) {
        derived::FreeState(&ham_matrix);
    }
    return {res_real, res_imag};
};

template <typename derived_, typename calc_type_>
auto CPUDensityMatrixPolicyBase<derived_, calc_type_>::ExpectDiffFSimTheta(const qs_data_p_t& qs_out,
                                                                           const qs_data_p_t& ham_matrix_out,
                                                                           const qbits_t& objs, const qbits_t& ctrls,
                                                                           index_t dim) -> qs_data_t {
    qs_data_p_t qs;
    bool will_free = false;
    if (qs_out == nullptr) {
        qs = derived::InitState(dim);
        will_free = true;
    } else {
        qs = qs_out;
    }
    qs_data_p_t ham_matrix;
    bool will_free_ham = false;
    if (ham_matrix_out == nullptr) {
        ham_matrix = derived::InitState(dim);
        will_free_ham = true;
    } else {
        ham_matrix = ham_matrix_out;
    }
    DoubleQubitGateMask mask(objs, ctrls);
    calc_type res_real = 0, res_imag = 0;
    if (!mask.ctrl_mask) {
        // clang-format off
        THRESHOLD_OMP(
            MQ_DO_PRAGMA(omp parallel for reduction(+:res_real, res_imag) schedule(static)), dim, DimTh,
                for (omp::idx_t l = 0; l < static_cast<omp::idx_t>(dim / 4); l++) {
                    index_t r0;
                    SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask, mask.obj_high_mask, mask.obj_rev_high_mask,
                                  l, r0);
                    auto r1 = r0 + mask.obj_min_mask;
                    auto r2 = r0 + mask.obj_max_mask;
                    qs_data_t this_res = 0;
                    for (index_t col = 0; col < dim; col++) {
                        this_res += GetValue(qs, r2, col) * GetValue(ham_matrix, col, r1);
                        this_res += GetValue(qs, r1, col) * GetValue(ham_matrix, col, r2);
                    }
                    this_res *= IMAGE_MI;
                    res_real += this_res.real();
                    res_imag += this_res.imag();
                })
        // clang-format on
    } else {
        // clang-format off
        THRESHOLD_OMP(
            MQ_DO_PRAGMA(omp parallel for reduction(+:res_real, res_imag) schedule(static)), dim, DimTh,
                for (omp::idx_t l = 0; l < static_cast<omp::idx_t>(dim / 4); l++) {
                    index_t r0;
                    SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask, mask.obj_high_mask, mask.obj_rev_high_mask,
                                  l, r0);
                    if ((r0 & mask.ctrl_mask) == mask.ctrl_mask) {
                        auto r1 = r0 + mask.obj_min_mask;
                        auto r2 = r0 + mask.obj_max_mask;
                        qs_data_t this_res = 0;
                        for (index_t col = 0; col < dim; col++) {
                            this_res += GetValue(qs, r2, col) * GetValue(ham_matrix, col, r1);
                            this_res += GetValue(qs, r1, col) * GetValue(ham_matrix, col, r2);
                        }
                        this_res *= IMAGE_MI;
                        res_real += this_res.real();
                        res_imag += this_res.imag();
                    }
                })
        // clang-format on
    }
    if (will_free) {
        derived::FreeState(&qs);
    }
    if (will_free_ham) {
        derived::FreeState(&ham_matrix);
    }
    return {res_real, res_imag};
};

template <typename derived_, typename calc_type_>
auto CPUDensityMatrixPolicyBase<derived_, calc_type_>::ExpectDiffFSimPhi(const qs_data_p_t& qs_out,
                                                                         const qs_data_p_t& ham_matrix_out,
                                                                         const qbits_t& objs, const qbits_t& ctrls,
                                                                         index_t dim) -> qs_data_t {
    qs_data_p_t qs;
    bool will_free = false;
    if (qs_out == nullptr) {
        qs = derived::InitState(dim);
        will_free = true;
    } else {
        qs = qs_out;
    }
    qs_data_p_t ham_matrix;
    bool will_free_ham = false;
    if (ham_matrix_out == nullptr) {
        ham_matrix = derived::InitState(dim);
        will_free_ham = true;
    } else {
        ham_matrix = ham_matrix_out;
    }
    DoubleQubitGateMask mask(objs, ctrls);
    calc_type res_real = 0, res_imag = 0;
    if (!mask.ctrl_mask) {
        // clang-format off
        THRESHOLD_OMP(
            MQ_DO_PRAGMA(omp parallel for reduction(+:res_real, res_imag) schedule(static)), dim, DimTh,
                for (omp::idx_t l = 0; l < static_cast<omp::idx_t>(dim / 4); l++) {
                    index_t r0;
                    SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask, mask.obj_high_mask, mask.obj_rev_high_mask,
                                  l, r0);
                    auto r3 = r0 + mask.obj_mask;
                    qs_data_t this_res = 0;
                    for (index_t col = 0; col < dim; col++) {
                        this_res += GetValue(qs, r3, col) * GetValue(ham_matrix, col, r3);
                    }
                    this_res *= IMAGE_I;
                    res_real += this_res.real();
                    res_imag += this_res.imag();
                })
        // clang-format on
    } else {
        // clang-format off
        THRESHOLD_OMP(
            MQ_DO_PRAGMA(omp parallel for reduction(+:res_real, res_imag) schedule(static)), dim, DimTh,
                for (omp::idx_t l = 0; l < static_cast<omp::idx_t>(dim / 4); l++) {
                    index_t r0;
                    SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask, mask.obj_high_mask, mask.obj_rev_high_mask,
                                  l, r0);
                    if ((r0 & mask.ctrl_mask) == mask.ctrl_mask) {
                        auto r3 = r0 + mask.obj_mask;
                        qs_data_t this_res = 0;
                        for (index_t col = 0; col < dim; col++) {
                            this_res += GetValue(qs, r3, col) * GetValue(ham_matrix, col, r3);
                        }
                        this_res *= IMAGE_I;
                        res_real += this_res.real();
                        res_imag += this_res.imag();
                    }
                })
        // clang-format on
    }
    if (will_free) {
        derived::FreeState(&qs);
    }
    if (will_free_ham) {
        derived::FreeState(&ham_matrix);
    }
    return {res_real, res_imag};
};

#ifdef __x86_64__
template struct CPUDensityMatrixPolicyBase<CPUDensityMatrixPolicyAvxFloat, float>;
template struct CPUDensityMatrixPolicyBase<CPUDensityMatrixPolicyAvxDouble, double>;
#elif defined(__amd64)
template struct CPUDensityMatrixPolicyBase<CPUDensityMatrixPolicyArmFloat, float>;
template struct CPUDensityMatrixPolicyBase<CPUDensityMatrixPolicyArmDouble, double>;
#endif
}  // namespace mindquantum::sim::densitymatrix::detail
