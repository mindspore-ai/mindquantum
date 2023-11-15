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
#define MATSUM4(a1, b1, c1, d1, a2, b2, c2, d2) ((a1) * (a2) + (b1) * (b2) + (c1) * (c2) + (d1) * (d2))
namespace mindquantum::sim::densitymatrix::detail {
template <typename derived_, typename calc_type_>
void CPUDensityMatrixPolicyBase<derived_, calc_type_>::ApplyRPS(qs_data_p_t* qs_p, const PauliMask& mask,
                                                                Index ctrl_mask, calc_type val, index_t dim,
                                                                bool diff) {
    if (!ctrl_mask) {
        derived_::ApplyRPSNoCtrl(qs_p, mask, val, dim, diff);
    } else {
        derived_::ApplyRPSWithCtrl(qs_p, mask, ctrl_mask, val, dim, diff);
    }
}

template <typename derived_, typename calc_type_>
void CPUDensityMatrixPolicyBase<derived_, calc_type_>::ApplyRPSNoCtrl(qs_data_p_t* qs_p, const PauliMask& mask,
                                                                      calc_type val, index_t dim, bool diff) {
    auto& qs = *qs_p;
    if (qs == nullptr) {
        qs = derived::InitState(dim);
    }
    auto mask_f = mask.mask_x | mask.mask_y;
    auto a = std::cos(val / 2);
    auto b = std::sin(val / 2) * IMAGE_I;
    if (diff) {
        a = -std::sin(val / 2) / 2;
        b = std::cos(val / 2) / 2 * IMAGE_I;
    }
    auto origin = derived_::Copy(*qs_p, dim);
    THRESHOLD_OMP_FOR(
        dim, DimTh, for (omp::idx_t r_i = 0; r_i < static_cast<omp::idx_t>(dim); r_i++) {
            auto r_j = (r_i ^ mask_f);
            if (r_j <= r_i) {
                for (index_t c_i = 0; c_i <= r_i; c_i++) {
                    auto c_j = (c_i ^ mask_f);
                    if (c_j >= c_i) {
                        auto r_c = ComplexCast<double, calc_type>::apply(POLAR[static_cast<char>(
                            (mask.num_y + 2 * CountOne(r_i & mask.mask_y) + 2 * CountOne(r_i & mask.mask_z)) & 3)]);
                        auto c_c = std::conj(ComplexCast<double, calc_type>::apply(POLAR[static_cast<char>(
                            (mask.num_y + 2 * CountOne(c_i & mask.mask_y) + 2 * CountOne(c_i & mask.mask_z)) & 3)]));
                        auto m_ri_ci = GetValue(origin, r_i, c_i);
                        auto m_ri_cj = GetValue(origin, r_i, c_j);
                        auto m_rj_ci = GetValue(origin, r_j, c_i);
                        auto m_rj_cj = GetValue(origin, r_j, c_j);
                        qs[IdxMap(r_i, c_i)] = MATSUM4(a * a, a * b / c_c, -a * b / r_c, -b * b / c_c / r_c, m_ri_ci,
                                                       m_ri_cj, m_rj_ci, m_rj_cj);
                        if ((r_j >= c_i) && (r_i != r_j)) {
                            qs[IdxMap(r_j, c_i)] = MATSUM4(-a * b * r_c, -b * b * r_c / c_c, a * a, a * b / c_c,
                                                           m_ri_ci, m_ri_cj, m_rj_ci, m_rj_cj);
                        }
                        if ((c_j <= r_i) && (c_i != c_j)) {
                            qs[IdxMap(r_i, c_j)] = MATSUM4(a * b * c_c, a * a, -b * b * c_c / r_c, -a * b / r_c,
                                                           m_ri_ci, m_ri_cj, m_rj_ci, m_rj_cj);
                        }
                        if ((r_j >= c_j) && (r_j != r_i) && (c_j != c_i)) {
                            qs[IdxMap(r_j, c_j)] = MATSUM4(-b * b * c_c * r_c, -a * b * r_c, a * b * c_c, a * a,
                                                           m_ri_ci, m_ri_cj, m_rj_ci, m_rj_cj);
                        }
                    }
                }
            }
        })
    derived_::FreeState(&origin);
}

template <typename derived_, typename calc_type_>
void CPUDensityMatrixPolicyBase<derived_, calc_type_>::ApplyRPSWithCtrl(qs_data_p_t* qs_p, const PauliMask& mask,
                                                                        Index ctrl_mask, calc_type val, index_t dim,
                                                                        bool diff) {
    auto& qs = *qs_p;
    if (qs == nullptr) {
        qs = derived::InitState(dim);
    }
    auto mask_f = mask.mask_x | mask.mask_y;
    auto a = std::cos(val / 2);
    auto b = std::sin(val / 2) * IMAGE_I;
    if (diff) {
        a = -std::sin(val / 2) / 2;
        b = std::cos(val / 2) / 2 * IMAGE_I;
    }
    auto origin = derived_::Copy(*qs_p, dim);
    THRESHOLD_OMP_FOR(
        dim, DimTh, for (omp::idx_t r_i = 0; r_i < static_cast<omp::idx_t>(dim); r_i++) {
            bool r_ctrl = ((r_i & ctrl_mask) == ctrl_mask);
            auto r_j = r_ctrl ? (r_i ^ mask_f) : r_i;
            qs_data_t a1 = r_ctrl * a + (1 - r_ctrl);
            qs_data_t b1 = b * static_cast<calc_type>(r_ctrl);
            if (r_j <= r_i) {
                for (index_t c_i = 0; c_i <= r_i; c_i++) {
                    bool c_ctrl = ((c_i & ctrl_mask) == ctrl_mask);
                    auto c_j = c_ctrl ? (c_i ^ mask_f) : c_i;
                    qs_data_t a2 = c_ctrl ? a : 1;
                    qs_data_t b2 = c_ctrl ? b : 0;
                    if (c_j >= c_i) {
                        auto r_c = ComplexCast<double, calc_type>::apply(POLAR[static_cast<char>(
                            (mask.num_y + 2 * CountOne(r_i & mask.mask_y) + 2 * CountOne(r_i & mask.mask_z)) & 3)]);
                        auto c_c = std::conj(ComplexCast<double, calc_type>::apply(POLAR[static_cast<char>(
                            (mask.num_y + 2 * CountOne(c_i & mask.mask_y) + 2 * CountOne(c_i & mask.mask_z)) & 3)]));
                        auto m_ri_ci = GetValue(origin, r_i, c_i);
                        auto m_ri_cj = GetValue(origin, r_i, c_j);
                        auto m_rj_ci = GetValue(origin, r_j, c_i);
                        auto m_rj_cj = GetValue(origin, r_j, c_j);
                        qs[IdxMap(r_i, c_i)] = MATSUM4(a1 * a2, a1 * b2 / c_c, -a2 * b1 / r_c, -b1 * b2 / c_c / r_c,
                                                       m_ri_ci, m_ri_cj, m_rj_ci, m_rj_cj);
                        if ((r_j >= c_i) && (r_i != r_j)) {
                            qs[IdxMap(r_j, c_i)] = MATSUM4(-a2 * b1 * r_c, -b1 * b2 * r_c / c_c, a1 * a2, a1 * b2 / c_c,
                                                           m_ri_ci, m_ri_cj, m_rj_ci, m_rj_cj);
                        }
                        if ((c_j <= r_i) && (c_i != c_j)) {
                            qs[IdxMap(r_i, c_j)] = MATSUM4(a1 * b2 * c_c, a1 * a2, -b1 * b2 * c_c / r_c, -a2 * b1 / r_c,
                                                           m_ri_ci, m_ri_cj, m_rj_ci, m_rj_cj);
                        }
                        if ((r_j >= c_j) && (r_j != r_i) && (c_j != c_i)) {
                            qs[IdxMap(r_j, c_j)] = MATSUM4(-b1 * b2 * c_c * r_c, -a2 * b1 * r_c, a1 * b2 * c_c, a1 * a2,
                                                           m_ri_ci, m_ri_cj, m_rj_ci, m_rj_cj);
                        }
                    }
                }
            }
        })
    if (diff) {
        derived::SetToZeroExcept(qs_p, ctrl_mask, dim);
    }
    derived_::FreeState(&origin);
}

template <typename derived_, typename calc_type_>
void CPUDensityMatrixPolicyBase<derived_, calc_type_>::ApplyRX(qs_data_p_t* qs_p, const qbits_t& objs,
                                                               const qbits_t& ctrls, calc_type val, index_t dim,
                                                               bool diff) {
    SingleQubitGateMask mask(objs, ctrls);
    auto a = std::cos(val / 2);
    auto b = -std::sin(val / 2);
    if (diff) {
        a = -0.5 * std::sin(val / 2);
        b = -0.5 * std::cos(val / 2);
    }
    matrix_t m{{{a, 0}, {0, b}}, {{0, b}, {a, 0}}};
    derived::ApplySingleQubitMatrix(*qs_p, qs_p, objs[0], ctrls, m, dim);
    if (diff && mask.ctrl_mask) {
        derived::SetToZeroExcept(qs_p, mask.ctrl_mask, dim);
    }
}

template <typename derived_, typename calc_type_>
void CPUDensityMatrixPolicyBase<derived_, calc_type_>::ApplyRY(qs_data_p_t* qs_p, const qbits_t& objs,
                                                               const qbits_t& ctrls, calc_type val, index_t dim,
                                                               bool diff) {
    SingleQubitGateMask mask(objs, ctrls);
    auto a = std::cos(val / 2);
    auto b = std::sin(val / 2);
    if (diff) {
        a = -0.5 * std::sin(val / 2);
        b = 0.5 * std::cos(val / 2);
    }
    matrix_t m{{{a, 0}, {-b, 0}}, {{b, 0}, {a, 0}}};
    derived::ApplySingleQubitMatrix(*qs_p, qs_p, objs[0], ctrls, m, dim);
    if (diff && mask.ctrl_mask) {
        derived::SetToZeroExcept(qs_p, mask.ctrl_mask, dim);
    }
}

template <typename derived_, typename calc_type_>
void CPUDensityMatrixPolicyBase<derived_, calc_type_>::ApplyRZ(qs_data_p_t* qs_p, const qbits_t& objs,
                                                               const qbits_t& ctrls, calc_type val, index_t dim,
                                                               bool diff) {
    SingleQubitGateMask mask(objs, ctrls);
    auto a = std::cos(val / 2);
    auto b = std::sin(val / 2);
    if (diff) {
        a = -0.5 * std::sin(val / 2);
        b = 0.5 * std::cos(val / 2);
    }
    matrix_t m{{{a, -b}, {0, 0}}, {{0, 0}, {a, b}}};
    derived::ApplySingleQubitMatrix(*qs_p, qs_p, objs[0], ctrls, m, dim);
    if (diff && mask.ctrl_mask) {
        derived::SetToZeroExcept(qs_p, mask.ctrl_mask, dim);
    }
}

template <typename derived_, typename calc_type_>
void CPUDensityMatrixPolicyBase<derived_, calc_type_>::ApplyRxx(qs_data_p_t* qs_p, const qbits_t& objs,
                                                                const qbits_t& ctrls, calc_type val, index_t dim,
                                                                bool diff) {
    DoubleQubitGateMask mask(objs, ctrls);
    auto c = static_cast<calc_type_>(std::cos(val / 2));
    auto s = static_cast<calc_type_>(std::sin(val / 2)) * IMAGE_MI;
    if (diff) {
        c = static_cast<calc_type_>(-std::sin(val / 2) / 2);
        s = static_cast<calc_type_>(std::cos(val / 2) / 2) * IMAGE_MI;
    }
    if (!mask.ctrl_mask) {
        ApplyRxxNoCtrl(qs_p, objs, ctrls, dim, c, s);
    } else {
        ApplyRxxCtrl(qs_p, objs, ctrls, dim, c, s, diff);
    }
}

template <typename derived_, typename calc_type_>
void CPUDensityMatrixPolicyBase<derived_, calc_type_>::ApplyRxxNoCtrl(qs_data_p_t* qs_p, const qbits_t& objs,
                                                                      const qbits_t& ctrls, index_t dim, calc_type c,
                                                                      qs_data_t s) {
    auto& qs = *qs_p;
    if (qs == nullptr) {
        qs = derived::InitState(dim);
    }
    DoubleQubitGateMask mask(objs, ctrls);
    THRESHOLD_OMP_FOR(
        dim, DimTh, for (omp::idx_t a = 0; a < static_cast<omp::idx_t>(dim / 4); a++) {
            VT<index_t> row(4);  // row index of reduced matrix entry
            SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask, mask.obj_high_mask, mask.obj_rev_high_mask, a,
                          row[0]);
            row[3] = row[0] + mask.obj_mask;
            row[1] = row[0] + mask.obj_min_mask;
            row[2] = row[0] + mask.obj_max_mask;
            for (index_t b = 0; b <= a; b++) {
                VT<index_t> col(4);  // column index of reduced matrix entry
                SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask, mask.obj_high_mask, mask.obj_rev_high_mask, b,
                              col[0]);
                col[3] = col[0] + mask.obj_mask;
                col[1] = col[0] + mask.obj_min_mask;
                col[2] = col[0] + mask.obj_max_mask;

                VT<VT<qs_data_t>> tmp_mat(4, VT<qs_data_t>(4));
                for (int i = 0; i < 4; i++) {
                    for (int j = 0; j < 4; j++) {
                        tmp_mat[i][j] = c * GetValue(qs, row[i], col[j]) + s * GetValue(qs, row[3 - i], col[j]);
                    }
                }
                for (int i = 0; i < 4; i++) {
                    for (int j = 0; j < 4; j++) {
                        SetValue(qs, row[i], col[j], c * tmp_mat[i][j] - s * tmp_mat[i][3 - j]);
                    }
                }
            }
        })
}

template <typename derived_, typename calc_type_>
void CPUDensityMatrixPolicyBase<derived_, calc_type_>::ApplyRxxCtrl(qs_data_p_t* qs_p, const qbits_t& objs,
                                                                    const qbits_t& ctrls, index_t dim, calc_type c,
                                                                    qs_data_t s, bool diff) {
    auto& qs = *qs_p;
    if (qs == nullptr) {
        qs = derived::InitState(dim);
    }
    DoubleQubitGateMask mask(objs, ctrls);
    THRESHOLD_OMP_FOR(
        dim, DimTh, for (omp::idx_t a = 0; a < static_cast<omp::idx_t>(dim / 4); a++) {
            VT<index_t> row(4);  // row index of reduced matrix entry
            SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask, mask.obj_high_mask, mask.obj_rev_high_mask, a,
                          row[0]);
            row[3] = row[0] + mask.obj_mask;
            row[1] = row[0] + mask.obj_min_mask;
            row[2] = row[0] + mask.obj_max_mask;
            for (index_t b = 0; b <= a; b++) {
                VT<index_t> col(4);  // column index of reduced matrix entry
                SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask, mask.obj_high_mask, mask.obj_rev_high_mask, b,
                              col[0]);
                if (((row[0] & mask.ctrl_mask) != mask.ctrl_mask)
                    && ((col[0] & mask.ctrl_mask) != mask.ctrl_mask)) {  // both not in control
                    continue;
                }
                col[3] = col[0] + mask.obj_mask;
                col[1] = col[0] + mask.obj_min_mask;
                col[2] = col[0] + mask.obj_max_mask;
                VT<VT<qs_data_t>> tmp_mat(4, VT<qs_data_t>(4));
                if ((row[0] & mask.ctrl_mask) == mask.ctrl_mask) {  // row in control
                    for (int i = 0; i < 4; i++) {
                        for (int j = 0; j < 4; j++) {
                            tmp_mat[i][j] = c * GetValue(qs, row[i], col[j]) + s * GetValue(qs, row[3 - i], col[j]);
                        }
                    }
                } else {  // row not in control
                    for (int i = 0; i < 4; i++) {
                        for (int j = 0; j < 4; j++) {
                            tmp_mat[i][j] = GetValue(qs, row[i], col[j]);
                        }
                    }
                }
                if ((col[0] & mask.ctrl_mask) == mask.ctrl_mask) {  // column in control
                    for (int i = 0; i < 4; i++) {
                        for (int j = 0; j < 4; j++) {
                            SetValue(qs, row[i], col[j], c * tmp_mat[i][j] - s * tmp_mat[i][3 - j]);
                        }
                    }
                } else {  // column not in control
                    for (int i = 0; i < 4; i++) {
                        for (int j = 0; j < 4; j++) {
                            SetValue(qs, row[i], col[j], tmp_mat[i][j]);
                        }
                    }
                }
            }
        })
    if (diff) {
        derived::SetToZeroExcept(qs_p, mask.ctrl_mask, dim);
    }
}

template <typename derived_, typename calc_type_>
void CPUDensityMatrixPolicyBase<derived_, calc_type_>::ApplyRyy(qs_data_p_t* qs_p, const qbits_t& objs,
                                                                const qbits_t& ctrls, calc_type val, index_t dim,
                                                                bool diff) {
    DoubleQubitGateMask mask(objs, ctrls);
    auto c = static_cast<calc_type_>(std::cos(val / 2));
    auto s = static_cast<calc_type_>(std::sin(val / 2)) * IMAGE_I;
    if (diff) {
        c = static_cast<calc_type_>(-std::sin(val / 2) / 2);
        s = static_cast<calc_type_>(std::cos(val / 2) / 2) * IMAGE_I;
    }
    if (!mask.ctrl_mask) {
        ApplyRyyNoCtrl(qs_p, objs, ctrls, dim, c, s);
    } else {
        ApplyRyyCtrl(qs_p, objs, ctrls, dim, c, s, diff);
    }
}

template <typename derived_, typename calc_type_>
void CPUDensityMatrixPolicyBase<derived_, calc_type_>::ApplyRyyNoCtrl(qs_data_p_t* qs_p, const qbits_t& objs,
                                                                      const qbits_t& ctrls, index_t dim, calc_type c,
                                                                      qs_data_t s) {
    auto& qs = *qs_p;
    if (qs == nullptr) {
        qs = derived::InitState(dim);
    }
    DoubleQubitGateMask mask(objs, ctrls);
    THRESHOLD_OMP_FOR(
        dim, DimTh, for (omp::idx_t a = 0; a < static_cast<omp::idx_t>(dim / 4); a++) {
            VT<index_t> row(4);  // row index of reduced matrix entry
            SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask, mask.obj_high_mask, mask.obj_rev_high_mask, a,
                          row[0]);
            row[3] = row[0] + mask.obj_mask;
            row[1] = row[0] + mask.obj_min_mask;
            row[2] = row[0] + mask.obj_max_mask;
            for (index_t b = 0; b <= a; b++) {
                VT<index_t> col(4);  // column index of reduced matrix entry
                SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask, mask.obj_high_mask, mask.obj_rev_high_mask, b,
                              col[0]);
                col[3] = col[0] + mask.obj_mask;
                col[1] = col[0] + mask.obj_min_mask;
                col[2] = col[0] + mask.obj_max_mask;
                VT<VT<qs_data_t>> tmp_mat(4, VT<qs_data_t>(4));
                for (int j = 0; j < 4; j++) {
                    tmp_mat[0][j] = c * GetValue(qs, row[0], col[j]) + s * GetValue(qs, row[3], col[j]);
                    tmp_mat[1][j] = c * GetValue(qs, row[1], col[j]) - s * GetValue(qs, row[2], col[j]);
                    tmp_mat[2][j] = c * GetValue(qs, row[2], col[j]) - s * GetValue(qs, row[1], col[j]);
                    tmp_mat[3][j] = c * GetValue(qs, row[3], col[j]) + s * GetValue(qs, row[0], col[j]);
                }
                VT<VT<qs_data_t>> res_mat(4, VT<qs_data_t>(4));
                for (int i = 0; i < 4; i++) {
                    res_mat[i][0] = c * tmp_mat[i][0] - s * tmp_mat[i][3];
                    res_mat[i][1] = c * tmp_mat[i][1] + s * tmp_mat[i][2];
                    res_mat[i][2] = c * tmp_mat[i][2] + s * tmp_mat[i][1];
                    res_mat[i][3] = c * tmp_mat[i][3] - s * tmp_mat[i][0];
                }
                for (int i = 0; i < 4; i++) {
                    for (int j = 0; j < 4; j++) {
                        SetValue(qs, row[i], col[j], res_mat[i][j]);
                    }
                }
            }
        })
}

template <typename derived_, typename calc_type_>
void CPUDensityMatrixPolicyBase<derived_, calc_type_>::ApplyRyyCtrl(qs_data_p_t* qs_p, const qbits_t& objs,
                                                                    const qbits_t& ctrls, index_t dim, calc_type c,
                                                                    qs_data_t s, bool diff) {
    auto& qs = *qs_p;
    if (qs == nullptr) {
        qs = derived::InitState(dim);
    }
    DoubleQubitGateMask mask(objs, ctrls);
    THRESHOLD_OMP_FOR(
        dim, DimTh, for (omp::idx_t a = 0; a < static_cast<omp::idx_t>(dim / 4); a++) {
            VT<index_t> row(4);  // row index of reduced matrix entry
            SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask, mask.obj_high_mask, mask.obj_rev_high_mask, a,
                          row[0]);
            row[3] = row[0] + mask.obj_mask;
            row[1] = row[0] + mask.obj_min_mask;
            row[2] = row[0] + mask.obj_max_mask;
            for (index_t b = 0; b <= a; b++) {
                VT<index_t> col(4);  // column index of reduced matrix entry
                SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask, mask.obj_high_mask, mask.obj_rev_high_mask, b,
                              col[0]);
                if (((row[0] & mask.ctrl_mask) != mask.ctrl_mask)
                    && ((col[0] & mask.ctrl_mask) != mask.ctrl_mask)) {  // both not in control
                    continue;
                }
                col[3] = col[0] + mask.obj_mask;
                col[1] = col[0] + mask.obj_min_mask;
                col[2] = col[0] + mask.obj_max_mask;
                VT<VT<qs_data_t>> tmp_mat(4, VT<qs_data_t>(4));
                if ((row[0] & mask.ctrl_mask) == mask.ctrl_mask) {  // row in control
                    for (int j = 0; j < 4; j++) {
                        tmp_mat[0][j] = c * GetValue(qs, row[0], col[j]) + s * GetValue(qs, row[3], col[j]);
                        tmp_mat[1][j] = c * GetValue(qs, row[1], col[j]) - s * GetValue(qs, row[2], col[j]);
                        tmp_mat[2][j] = c * GetValue(qs, row[2], col[j]) - s * GetValue(qs, row[1], col[j]);
                        tmp_mat[3][j] = c * GetValue(qs, row[3], col[j]) + s * GetValue(qs, row[0], col[j]);
                    }
                } else {  // row not in control
                    for (int i = 0; i < 4; i++) {
                        for (int j = 0; j < 4; j++) {
                            tmp_mat[i][j] = GetValue(qs, row[i], col[j]);
                        }
                    }
                }
                if ((col[0] & mask.ctrl_mask) == mask.ctrl_mask) {  // column in control
                    VT<VT<qs_data_t>> res_mat(4, VT<qs_data_t>(4));
                    for (int i = 0; i < 4; i++) {
                        res_mat[i][0] = c * tmp_mat[i][0] - s * tmp_mat[i][3];
                        res_mat[i][1] = c * tmp_mat[i][1] + s * tmp_mat[i][2];
                        res_mat[i][2] = c * tmp_mat[i][2] + s * tmp_mat[i][1];
                        res_mat[i][3] = c * tmp_mat[i][3] - s * tmp_mat[i][0];
                    }
                    tmp_mat.swap(res_mat);
                }  // do nothing if column not in control

                for (int i = 0; i < 4; i++) {
                    for (int j = 0; j < 4; j++) {
                        SetValue(qs, row[i], col[j], tmp_mat[i][j]);
                    }
                }
            }
        })
    if (diff) {
        derived::SetToZeroExcept(qs_p, mask.ctrl_mask, dim);
    }
}

template <typename derived_, typename calc_type_>
void CPUDensityMatrixPolicyBase<derived_, calc_type_>::ApplyRzz(qs_data_p_t* qs_p, const qbits_t& objs,
                                                                const qbits_t& ctrls, calc_type val, index_t dim,
                                                                bool diff) {
    DoubleQubitGateMask mask(objs, ctrls);
    auto c = static_cast<calc_type_>(std::cos(val / 2));
    auto s = static_cast<calc_type_>(std::sin(val / 2));
    if (diff) {
        c = static_cast<calc_type_>(-std::sin(val / 2) / 2);
        s = static_cast<calc_type_>(std::cos(val / 2) / 2);
    }
    if (!mask.ctrl_mask) {
        ApplyRzzNoCtrl(qs_p, objs, ctrls, dim, c, s);
    } else {
        ApplyRzzCtrl(qs_p, objs, ctrls, dim, c, s, diff);
    }
}

template <typename derived_, typename calc_type_>
void CPUDensityMatrixPolicyBase<derived_, calc_type_>::ApplyRzzNoCtrl(qs_data_p_t* qs_p, const qbits_t& objs,
                                                                      const qbits_t& ctrls, index_t dim, calc_type c,
                                                                      calc_type s) {
    auto& qs = *qs_p;
    if (qs == nullptr) {
        qs = derived::InitState(dim);
    }
    DoubleQubitGateMask mask(objs, ctrls);
    auto e = c + IMAGE_I * s;
    auto me = c + IMAGE_MI * s;
    auto me2 = me * me;
    auto e2 = e * e;
    THRESHOLD_OMP_FOR(
        dim, DimTh, for (omp::idx_t a = 0; a < static_cast<omp::idx_t>(dim / 4); a++) {
            VT<index_t> row(4);  // row index of reduced matrix entry
            SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask, mask.obj_high_mask, mask.obj_rev_high_mask, a,
                          row[0]);
            row[3] = row[0] + mask.obj_mask;
            row[1] = row[0] + mask.obj_min_mask;
            row[2] = row[0] + mask.obj_max_mask;
            for (index_t b = 0; b < a; b++) {
                VT<index_t> col(4);  // column index of reduced matrix entry
                SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask, mask.obj_high_mask, mask.obj_rev_high_mask, b,
                              col[0]);
                col[3] = col[0] + mask.obj_mask;
                col[1] = col[0] + mask.obj_min_mask;
                col[2] = col[0] + mask.obj_max_mask;

                SelfMultiply(qs, row[0], col[1], me2);
                SelfMultiply(qs, row[0], col[2], me2);
                SelfMultiply(qs, row[3], col[1], me2);
                SelfMultiply(qs, row[3], col[2], me2);

                SelfMultiply(qs, row[1], col[0], e2);
                SelfMultiply(qs, row[2], col[0], e2);
                SelfMultiply(qs, row[1], col[3], e2);
                SelfMultiply(qs, row[2], col[3], e2);
            }
            // diagonal case
            SelfMultiply(qs, row[1], row[0], e2);
            SelfMultiply(qs, row[2], row[0], e2);
            SelfMultiply(qs, row[3], row[1], me2);
            SelfMultiply(qs, row[3], row[2], me2);
        })
}

template <typename derived_, typename calc_type_>
void CPUDensityMatrixPolicyBase<derived_, calc_type_>::ApplyRzzCtrl(qs_data_p_t* qs_p, const qbits_t& objs,
                                                                    const qbits_t& ctrls, index_t dim, calc_type c,
                                                                    calc_type s, bool diff) {
    auto& qs = *qs_p;
    if (qs == nullptr) {
        qs = derived::InitState(dim);
    }
    DoubleQubitGateMask mask(objs, ctrls);
    auto e = c + IMAGE_I * s;
    auto me = c + IMAGE_MI * s;
    auto me2 = me * me;
    auto e2 = e * e;
    THRESHOLD_OMP_FOR(
        dim, DimTh, for (omp::idx_t a = 0; a < static_cast<omp::idx_t>(dim / 4); a++) {
            VT<index_t> row(4);  // row index of reduced matrix entry
            SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask, mask.obj_high_mask, mask.obj_rev_high_mask, a,
                          row[0]);
            row[3] = row[0] + mask.obj_mask;
            row[1] = row[0] + mask.obj_min_mask;
            row[2] = row[0] + mask.obj_max_mask;
            for (index_t b = 0; b < a; b++) {
                VT<index_t> col(4);  // column index of reduced matrix entry
                SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask, mask.obj_high_mask, mask.obj_rev_high_mask, b,
                              col[0]);
                if (((row[0] & mask.ctrl_mask) != mask.ctrl_mask)
                    && ((col[0] & mask.ctrl_mask) != mask.ctrl_mask)) {  // both not in control
                    continue;
                }
                col[3] = col[0] + mask.obj_mask;
                col[1] = col[0] + mask.obj_min_mask;
                col[2] = col[0] + mask.obj_max_mask;
                if ((row[0] & mask.ctrl_mask) == mask.ctrl_mask) {
                    if ((col[0] & mask.ctrl_mask) == mask.ctrl_mask) {  // both in control
                        SelfMultiply(qs, row[0], col[1], me2);
                        SelfMultiply(qs, row[0], col[2], me2);
                        SelfMultiply(qs, row[3], col[1], me2);
                        SelfMultiply(qs, row[3], col[2], me2);

                        SelfMultiply(qs, row[1], col[0], e2);
                        SelfMultiply(qs, row[2], col[0], e2);
                        SelfMultiply(qs, row[1], col[3], e2);
                        SelfMultiply(qs, row[2], col[3], e2);
                    } else {  // only row in control
                        for (int j = 0; j < 4; j++) {
                            SelfMultiply(qs, row[0], col[j], me);
                            SelfMultiply(qs, row[3], col[j], me);
                            SelfMultiply(qs, row[1], col[j], e);
                            SelfMultiply(qs, row[2], col[j], e);
                        }
                    }
                } else {  // only column in control
                    for (int i = 0; i < 4; i++) {
                        SelfMultiply(qs, row[i], col[0], e);
                        SelfMultiply(qs, row[i], col[3], e);
                        SelfMultiply(qs, row[i], col[1], me);
                        SelfMultiply(qs, row[i], col[2], me);
                    }
                }
            }
            // diagonal case
            if ((row[0] & mask.ctrl_mask) == mask.ctrl_mask) {
                SelfMultiply(qs, row[1], row[0], e2);
                SelfMultiply(qs, row[2], row[0], e2);
                SelfMultiply(qs, row[3], row[1], me2);
                SelfMultiply(qs, row[3], row[2], me2);
            }
        })
    if (diff) {
        derived::SetToZeroExcept(qs_p, mask.ctrl_mask, dim);
    }
}

template <typename derived_, typename calc_type_>
void CPUDensityMatrixPolicyBase<derived_, calc_type_>::ApplyRxy(qs_data_p_t* qs_p, const qbits_t& objs,
                                                                const qbits_t& ctrls, calc_type val, index_t dim,
                                                                bool diff) {
    DoubleQubitGateMask mask(objs, ctrls);
    auto c = static_cast<calc_type_>(std::cos(val / 2));
    auto s = static_cast<calc_type_>(std::sin(val / 2));
    if (diff) {
        c = static_cast<calc_type_>(-std::sin(val / 2) / 2);
        s = static_cast<calc_type_>(std::cos(val / 2) / 2);
    }
    if (!mask.ctrl_mask) {
        ApplyRxyNoCtrl(qs_p, objs, ctrls, dim, c, s);
    } else {
        ApplyRxyCtrl(qs_p, objs, ctrls, dim, c, s, diff);
    }
}

template <typename derived_, typename calc_type_>
void CPUDensityMatrixPolicyBase<derived_, calc_type_>::ApplyRxyNoCtrl(qs_data_p_t* qs_p, const qbits_t& objs,
                                                                      const qbits_t& ctrls, index_t dim, calc_type c,
                                                                      qs_data_t s) {
    auto& qs = *qs_p;
    if (qs == nullptr) {
        qs = derived::InitState(dim);
    }
    DoubleQubitGateMask mask(objs, ctrls);
    THRESHOLD_OMP_FOR(
        dim, DimTh, for (omp::idx_t a = 0; a < (dim / 4); a++) {
            VT<index_t> row(4);  // row index of reduced matrix entry
            SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask, mask.obj_high_mask, mask.obj_rev_high_mask, a,
                          row[0]);
            row[3] = row[0] + mask.obj_mask;
            row[1] = row[0] + mask.obj_min_mask;
            row[2] = row[0] + mask.obj_max_mask;
            for (index_t b = 0; b <= a; b++) {
                VT<index_t> col(4);  // column index of reduced matrix entry
                SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask, mask.obj_high_mask, mask.obj_rev_high_mask, b,
                              col[0]);
                col[3] = col[0] + mask.obj_mask;
                col[1] = col[0] + mask.obj_min_mask;
                col[2] = col[0] + mask.obj_max_mask;
                VT<VT<qs_data_t>> tmp_mat(4, VT<qs_data_t>(4));
                for (int j = 0; j < 4; j++) {
                    tmp_mat[0][j] = c * GetValue(qs, row[0], col[j]) - s * GetValue(qs, row[3], col[j]);
                    tmp_mat[1][j] = c * GetValue(qs, row[1], col[j]) - s * GetValue(qs, row[2], col[j]);
                    tmp_mat[2][j] = c * GetValue(qs, row[2], col[j]) + s * GetValue(qs, row[1], col[j]);
                    tmp_mat[3][j] = c * GetValue(qs, row[3], col[j]) + s * GetValue(qs, row[0], col[j]);
                }
                VT<VT<qs_data_t>> res_mat(4, VT<qs_data_t>(4));
                for (int i = 0; i < 4; i++) {
                    res_mat[i][0] = c * tmp_mat[i][0] - s * tmp_mat[i][3];
                    res_mat[i][1] = c * tmp_mat[i][1] - s * tmp_mat[i][2];
                    res_mat[i][2] = c * tmp_mat[i][2] + s * tmp_mat[i][1];
                    res_mat[i][3] = c * tmp_mat[i][3] + s * tmp_mat[i][0];
                }
                for (int i = 0; i < 4; i++) {
                    for (int j = 0; j < 4; j++) {
                        SetValue(qs, row[i], col[j], res_mat[i][j]);
                    }
                }
            }
        })
}

template <typename derived_, typename calc_type_>
void CPUDensityMatrixPolicyBase<derived_, calc_type_>::ApplyRxyCtrl(qs_data_p_t* qs_p, const qbits_t& objs,
                                                                    const qbits_t& ctrls, index_t dim, calc_type c,
                                                                    qs_data_t s, bool diff) {
    auto& qs = *qs_p;
    if (qs == nullptr) {
        qs = derived::InitState(dim);
    }
    DoubleQubitGateMask mask(objs, ctrls);
    THRESHOLD_OMP_FOR(
        dim, DimTh, for (omp::idx_t a = 0; a < (dim / 4); a++) {
            VT<index_t> row(4);  // row index of reduced matrix entry
            SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask, mask.obj_high_mask, mask.obj_rev_high_mask, a,
                          row[0]);
            row[3] = row[0] + mask.obj_mask;
            row[1] = row[0] + mask.obj_min_mask;
            row[2] = row[0] + mask.obj_max_mask;
            for (index_t b = 0; b <= a; b++) {
                VT<index_t> col(4);  // column index of reduced matrix entry
                SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask, mask.obj_high_mask, mask.obj_rev_high_mask, b,
                              col[0]);
                if (((row[0] & mask.ctrl_mask) != mask.ctrl_mask)
                    && ((col[0] & mask.ctrl_mask) != mask.ctrl_mask)) {  // both not in control
                    continue;
                }
                col[3] = col[0] + mask.obj_mask;
                col[1] = col[0] + mask.obj_min_mask;
                col[2] = col[0] + mask.obj_max_mask;
                VT<VT<qs_data_t>> tmp_mat(4, VT<qs_data_t>(4));
                if ((row[0] & mask.ctrl_mask) == mask.ctrl_mask) {  // row in control
                    for (int j = 0; j < 4; j++) {
                        tmp_mat[0][j] = c * GetValue(qs, row[0], col[j]) - s * GetValue(qs, row[3], col[j]);
                        tmp_mat[1][j] = c * GetValue(qs, row[1], col[j]) - s * GetValue(qs, row[2], col[j]);
                        tmp_mat[2][j] = c * GetValue(qs, row[2], col[j]) + s * GetValue(qs, row[1], col[j]);
                        tmp_mat[3][j] = c * GetValue(qs, row[3], col[j]) + s * GetValue(qs, row[0], col[j]);
                    }
                } else {  // row not in control
                    for (int i = 0; i < 4; i++) {
                        for (int j = 0; j < 4; j++) {
                            tmp_mat[i][j] = GetValue(qs, row[i], col[j]);
                        }
                    }
                }
                if ((col[0] & mask.ctrl_mask) == mask.ctrl_mask) {  // column in control
                    VT<VT<qs_data_t>> res_mat(4, VT<qs_data_t>(4));
                    for (int i = 0; i < 4; i++) {
                        res_mat[i][0] = c * tmp_mat[i][0] - s * tmp_mat[i][3];
                        res_mat[i][1] = c * tmp_mat[i][1] - s * tmp_mat[i][2];
                        res_mat[i][2] = c * tmp_mat[i][2] + s * tmp_mat[i][1];
                        res_mat[i][3] = c * tmp_mat[i][3] + s * tmp_mat[i][0];
                    }
                    tmp_mat.swap(res_mat);
                }  // do nothing if column not in control

                for (int i = 0; i < 4; i++) {
                    for (int j = 0; j < 4; j++) {
                        SetValue(qs, row[i], col[j], tmp_mat[i][j]);
                    }
                }
            }
        })
    if (diff) {
        derived::SetToZeroExcept(qs_p, mask.ctrl_mask, dim);
    }
}

template <typename derived_, typename calc_type_>
void CPUDensityMatrixPolicyBase<derived_, calc_type_>::ApplyRxz(qs_data_p_t* qs_p, const qbits_t& objs,
                                                                const qbits_t& ctrls, calc_type val, index_t dim,
                                                                bool diff) {
    DoubleQubitGateMask mask(objs, ctrls);
    auto c = static_cast<calc_type_>(std::cos(val / 2));
    auto s = static_cast<calc_type_>(std::sin(val / 2)) * IMAGE_I;
    if (diff) {
        c = static_cast<calc_type_>(-std::sin(val / 2) / 2);
        s = static_cast<calc_type_>(std::cos(val / 2) / 2) * IMAGE_I;
    }
    if (!mask.ctrl_mask) {
        ApplyRxzNoCtrl(qs_p, objs, ctrls, dim, c, s);
    } else {
        ApplyRxzCtrl(qs_p, objs, ctrls, dim, c, s, diff);
    }
}

template <typename derived_, typename calc_type_>
void CPUDensityMatrixPolicyBase<derived_, calc_type_>::ApplyRxzNoCtrl(qs_data_p_t* qs_p, const qbits_t& objs,
                                                                      const qbits_t& ctrls, index_t dim, calc_type c,
                                                                      qs_data_t s) {
    auto& qs = *qs_p;
    if (qs == nullptr) {
        qs = derived::InitState(dim);
    }
    DoubleQubitGateMask mask(objs, ctrls);
    THRESHOLD_OMP_FOR(
        dim, DimTh, for (omp::idx_t a = 0; a < (dim / 4); a++) {
            VT<index_t> row(4);  // row index of reduced matrix entry
            SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask, mask.obj_high_mask, mask.obj_rev_high_mask, a,
                          row[0]);
            row[3] = row[0] + mask.obj_mask;
            row[1] = row[0] + mask.obj_min_mask;
            row[2] = row[0] + mask.obj_max_mask;
            for (index_t b = 0; b <= a; b++) {
                VT<index_t> col(4);  // column index of reduced matrix entry
                SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask, mask.obj_high_mask, mask.obj_rev_high_mask, b,
                              col[0]);
                col[3] = col[0] + mask.obj_mask;
                col[1] = col[0] + mask.obj_min_mask;
                col[2] = col[0] + mask.obj_max_mask;
                VT<VT<qs_data_t>> tmp_mat(4, VT<qs_data_t>(4));
                for (int j = 0; j < 4; j++) {
                    tmp_mat[0][j] = c * GetValue(qs, row[0], col[j]) - s * GetValue(qs, row[1], col[j]);
                    tmp_mat[1][j] = c * GetValue(qs, row[1], col[j]) - s * GetValue(qs, row[0], col[j]);
                    tmp_mat[2][j] = c * GetValue(qs, row[2], col[j]) + s * GetValue(qs, row[3], col[j]);
                    tmp_mat[3][j] = c * GetValue(qs, row[3], col[j]) + s * GetValue(qs, row[2], col[j]);
                }
                VT<VT<qs_data_t>> res_mat(4, VT<qs_data_t>(4));
                for (int i = 0; i < 4; i++) {
                    res_mat[i][0] = c * tmp_mat[i][0] + s * tmp_mat[i][1];
                    res_mat[i][1] = c * tmp_mat[i][1] + s * tmp_mat[i][0];
                    res_mat[i][2] = c * tmp_mat[i][2] - s * tmp_mat[i][3];
                    res_mat[i][3] = c * tmp_mat[i][3] - s * tmp_mat[i][2];
                }
                for (int i = 0; i < 4; i++) {
                    for (int j = 0; j < 4; j++) {
                        SetValue(qs, row[i], col[j], res_mat[i][j]);
                    }
                }
            }
        })
}

template <typename derived_, typename calc_type_>
void CPUDensityMatrixPolicyBase<derived_, calc_type_>::ApplyRxzCtrl(qs_data_p_t* qs_p, const qbits_t& objs,
                                                                    const qbits_t& ctrls, index_t dim, calc_type c,
                                                                    qs_data_t s, bool diff) {
    auto& qs = *qs_p;
    if (qs == nullptr) {
        qs = derived::InitState(dim);
    }
    DoubleQubitGateMask mask(objs, ctrls);
    THRESHOLD_OMP_FOR(
        dim, DimTh, for (omp::idx_t a = 0; a < (dim / 4); a++) {
            VT<index_t> row(4);  // row index of reduced matrix entry
            SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask, mask.obj_high_mask, mask.obj_rev_high_mask, a,
                          row[0]);
            row[3] = row[0] + mask.obj_mask;
            row[1] = row[0] + mask.obj_min_mask;
            row[2] = row[0] + mask.obj_max_mask;
            for (index_t b = 0; b <= a; b++) {
                VT<index_t> col(4);  // column index of reduced matrix entry
                SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask, mask.obj_high_mask, mask.obj_rev_high_mask, b,
                              col[0]);
                if (((row[0] & mask.ctrl_mask) != mask.ctrl_mask)
                    && ((col[0] & mask.ctrl_mask) != mask.ctrl_mask)) {  // both not in control
                    continue;
                }
                col[3] = col[0] + mask.obj_mask;
                col[1] = col[0] + mask.obj_min_mask;
                col[2] = col[0] + mask.obj_max_mask;
                VT<VT<qs_data_t>> tmp_mat(4, VT<qs_data_t>(4));
                if ((row[0] & mask.ctrl_mask) == mask.ctrl_mask) {  // row in control
                    for (int j = 0; j < 4; j++) {
                        tmp_mat[0][j] = c * GetValue(qs, row[0], col[j]) - s * GetValue(qs, row[1], col[j]);
                        tmp_mat[1][j] = c * GetValue(qs, row[1], col[j]) - s * GetValue(qs, row[0], col[j]);
                        tmp_mat[2][j] = c * GetValue(qs, row[2], col[j]) + s * GetValue(qs, row[3], col[j]);
                        tmp_mat[3][j] = c * GetValue(qs, row[3], col[j]) + s * GetValue(qs, row[2], col[j]);
                    }
                } else {  // row not in control
                    for (int i = 0; i < 4; i++) {
                        for (int j = 0; j < 4; j++) {
                            tmp_mat[i][j] = GetValue(qs, row[i], col[j]);
                        }
                    }
                }
                if ((col[0] & mask.ctrl_mask) == mask.ctrl_mask) {  // column in control
                    VT<VT<qs_data_t>> res_mat(4, VT<qs_data_t>(4));
                    for (int i = 0; i < 4; i++) {
                        res_mat[i][0] = c * tmp_mat[i][0] + s * tmp_mat[i][1];
                        res_mat[i][1] = c * tmp_mat[i][1] + s * tmp_mat[i][0];
                        res_mat[i][2] = c * tmp_mat[i][2] - s * tmp_mat[i][3];
                        res_mat[i][3] = c * tmp_mat[i][3] - s * tmp_mat[i][2];
                    }
                    tmp_mat.swap(res_mat);
                }  // do nothing if column not in control

                for (int i = 0; i < 4; i++) {
                    for (int j = 0; j < 4; j++) {
                        SetValue(qs, row[i], col[j], tmp_mat[i][j]);
                    }
                }
            }
        })
    if (diff) {
        derived::SetToZeroExcept(qs_p, mask.ctrl_mask, dim);
    }
}

template <typename derived_, typename calc_type_>
void CPUDensityMatrixPolicyBase<derived_, calc_type_>::ApplyRyz(qs_data_p_t* qs_p, const qbits_t& objs,
                                                                const qbits_t& ctrls, calc_type val, index_t dim,
                                                                bool diff) {
    DoubleQubitGateMask mask(objs, ctrls);
    auto c = static_cast<calc_type_>(std::cos(val / 2));
    auto s = static_cast<calc_type_>(std::sin(val / 2));
    if (diff) {
        c = static_cast<calc_type_>(-std::sin(val / 2) / 2);
        s = static_cast<calc_type_>(std::cos(val / 2) / 2);
    }
    if (!mask.ctrl_mask) {
        ApplyRyzNoCtrl(qs_p, objs, ctrls, dim, c, s);
    } else {
        ApplyRyzCtrl(qs_p, objs, ctrls, dim, c, s, diff);
    }
}

template <typename derived_, typename calc_type_>
void CPUDensityMatrixPolicyBase<derived_, calc_type_>::ApplyRyzNoCtrl(qs_data_p_t* qs_p, const qbits_t& objs,
                                                                      const qbits_t& ctrls, index_t dim, calc_type c,
                                                                      qs_data_t s) {
    auto& qs = *qs_p;
    if (qs == nullptr) {
        qs = derived::InitState(dim);
    }
    DoubleQubitGateMask mask(objs, ctrls);
    THRESHOLD_OMP_FOR(
        dim, DimTh, for (omp::idx_t a = 0; a < (dim / 4); a++) {
            VT<index_t> row(4);  // row index of reduced matrix entry
            SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask, mask.obj_high_mask, mask.obj_rev_high_mask, a,
                          row[0]);
            row[3] = row[0] + mask.obj_mask;
            row[1] = row[0] + mask.obj_min_mask;
            row[2] = row[0] + mask.obj_max_mask;
            for (index_t b = 0; b <= a; b++) {
                VT<index_t> col(4);  // column index of reduced matrix entry
                SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask, mask.obj_high_mask, mask.obj_rev_high_mask, b,
                              col[0]);
                col[3] = col[0] + mask.obj_mask;
                col[1] = col[0] + mask.obj_min_mask;
                col[2] = col[0] + mask.obj_max_mask;
                VT<VT<qs_data_t>> tmp_mat(4, VT<qs_data_t>(4));
                for (int j = 0; j < 4; j++) {
                    tmp_mat[0][j] = c * GetValue(qs, row[0], col[j]) - s * GetValue(qs, row[1], col[j]);
                    tmp_mat[1][j] = c * GetValue(qs, row[1], col[j]) + s * GetValue(qs, row[0], col[j]);
                    tmp_mat[2][j] = c * GetValue(qs, row[2], col[j]) + s * GetValue(qs, row[3], col[j]);
                    tmp_mat[3][j] = c * GetValue(qs, row[3], col[j]) - s * GetValue(qs, row[2], col[j]);
                }
                VT<VT<qs_data_t>> res_mat(4, VT<qs_data_t>(4));
                for (int i = 0; i < 4; i++) {
                    res_mat[i][0] = c * tmp_mat[i][0] - s * tmp_mat[i][1];
                    res_mat[i][1] = c * tmp_mat[i][1] + s * tmp_mat[i][0];
                    res_mat[i][2] = c * tmp_mat[i][2] + s * tmp_mat[i][3];
                    res_mat[i][3] = c * tmp_mat[i][3] - s * tmp_mat[i][2];
                }
                for (int i = 0; i < 4; i++) {
                    for (int j = 0; j < 4; j++) {
                        SetValue(qs, row[i], col[j], res_mat[i][j]);
                    }
                }
            }
        })
}

template <typename derived_, typename calc_type_>
void CPUDensityMatrixPolicyBase<derived_, calc_type_>::ApplyRyzCtrl(qs_data_p_t* qs_p, const qbits_t& objs,
                                                                    const qbits_t& ctrls, index_t dim, calc_type c,
                                                                    qs_data_t s, bool diff) {
    auto& qs = *qs_p;
    if (qs == nullptr) {
        qs = derived::InitState(dim);
    }
    DoubleQubitGateMask mask(objs, ctrls);
    THRESHOLD_OMP_FOR(
        dim, DimTh, for (omp::idx_t a = 0; a < (dim / 4); a++) {
            VT<index_t> row(4);  // row index of reduced matrix entry
            SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask, mask.obj_high_mask, mask.obj_rev_high_mask, a,
                          row[0]);
            row[3] = row[0] + mask.obj_mask;
            row[1] = row[0] + mask.obj_min_mask;
            row[2] = row[0] + mask.obj_max_mask;
            for (index_t b = 0; b <= a; b++) {
                VT<index_t> col(4);  // column index of reduced matrix entry
                SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask, mask.obj_high_mask, mask.obj_rev_high_mask, b,
                              col[0]);
                if (((row[0] & mask.ctrl_mask) != mask.ctrl_mask)
                    && ((col[0] & mask.ctrl_mask) != mask.ctrl_mask)) {  // both not in control
                    continue;
                }
                col[3] = col[0] + mask.obj_mask;
                col[1] = col[0] + mask.obj_min_mask;
                col[2] = col[0] + mask.obj_max_mask;
                VT<VT<qs_data_t>> tmp_mat(4, VT<qs_data_t>(4));
                if ((row[0] & mask.ctrl_mask) == mask.ctrl_mask) {  // row in control
                    for (int j = 0; j < 4; j++) {
                        tmp_mat[0][j] = c * GetValue(qs, row[0], col[j]) - s * GetValue(qs, row[1], col[j]);
                        tmp_mat[1][j] = c * GetValue(qs, row[1], col[j]) + s * GetValue(qs, row[0], col[j]);
                        tmp_mat[2][j] = c * GetValue(qs, row[2], col[j]) + s * GetValue(qs, row[3], col[j]);
                        tmp_mat[3][j] = c * GetValue(qs, row[3], col[j]) - s * GetValue(qs, row[2], col[j]);
                    }
                } else {  // row not in control
                    for (int i = 0; i < 4; i++) {
                        for (int j = 0; j < 4; j++) {
                            tmp_mat[i][j] = GetValue(qs, row[i], col[j]);
                        }
                    }
                }
                if ((col[0] & mask.ctrl_mask) == mask.ctrl_mask) {  // column in control
                    VT<VT<qs_data_t>> res_mat(4, VT<qs_data_t>(4));
                    for (int i = 0; i < 4; i++) {
                        res_mat[i][0] = c * tmp_mat[i][0] - s * tmp_mat[i][1];
                        res_mat[i][1] = c * tmp_mat[i][1] + s * tmp_mat[i][0];
                        res_mat[i][2] = c * tmp_mat[i][2] + s * tmp_mat[i][3];
                        res_mat[i][3] = c * tmp_mat[i][3] - s * tmp_mat[i][2];
                    }
                    tmp_mat.swap(res_mat);
                }  // do nothing if column not in control

                for (int i = 0; i < 4; i++) {
                    for (int j = 0; j < 4; j++) {
                        SetValue(qs, row[i], col[j], tmp_mat[i][j]);
                    }
                }
            }
        })
    if (diff) {
        derived::SetToZeroExcept(qs_p, mask.ctrl_mask, dim);
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
