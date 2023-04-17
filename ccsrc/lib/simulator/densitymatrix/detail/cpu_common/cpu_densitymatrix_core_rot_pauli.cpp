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

#include "core/utils.hpp"
#include "math/pr/parameter_resolver.hpp"
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
void CPUDensityMatrixPolicyBase<derived_, calc_type_>::ApplyRX(qs_data_p_t qs, const qbits_t& objs,
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
    derived::ApplySingleQubitMatrix(qs, qs, objs[0], ctrls, m, dim);
    if (diff && mask.ctrl_mask) {
        derived::SetToZeroExcept(qs, mask.ctrl_mask, dim);
    }
}

template <typename derived_, typename calc_type_>
void CPUDensityMatrixPolicyBase<derived_, calc_type_>::ApplyRY(qs_data_p_t qs, const qbits_t& objs,
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
    derived::ApplySingleQubitMatrix(qs, qs, objs[0], ctrls, m, dim);
    if (diff && mask.ctrl_mask) {
        derived::SetToZeroExcept(qs, mask.ctrl_mask, dim);
    }
}

template <typename derived_, typename calc_type_>
void CPUDensityMatrixPolicyBase<derived_, calc_type_>::ApplyRZ(qs_data_p_t qs, const qbits_t& objs,
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
    derived::ApplySingleQubitMatrix(qs, qs, objs[0], ctrls, m, dim);
    if (diff && mask.ctrl_mask) {
        derived::SetToZeroExcept(qs, mask.ctrl_mask, dim);
    }
}

template <typename derived_, typename calc_type_>
void CPUDensityMatrixPolicyBase<derived_, calc_type_>::ApplyRxx(qs_data_p_t qs, const qbits_t& objs,
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
        ApplyRxxNoCtrl(qs, objs, ctrls, dim, c, s);
    } else {
        ApplyRxxCtrl(qs, objs, ctrls, dim, c, s, diff);
    }
}

template <typename derived_, typename calc_type_>
void CPUDensityMatrixPolicyBase<derived_, calc_type_>::ApplyRxxNoCtrl(qs_data_p_t qs, const qbits_t& objs,
                                                                      const qbits_t& ctrls, index_t dim, calc_type c,
                                                                      qs_data_t s) {
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
void CPUDensityMatrixPolicyBase<derived_, calc_type_>::ApplyRxxCtrl(qs_data_p_t qs, const qbits_t& objs,
                                                                    const qbits_t& ctrls, index_t dim, calc_type c,
                                                                    qs_data_t s, bool diff) {
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
        derived::SetToZeroExcept(qs, mask.ctrl_mask, dim);
    }
}

template <typename derived_, typename calc_type_>
void CPUDensityMatrixPolicyBase<derived_, calc_type_>::ApplyRyy(qs_data_p_t qs, const qbits_t& objs,
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
        ApplyRyyNoCtrl(qs, objs, ctrls, dim, c, s);
    } else {
        ApplyRyyCtrl(qs, objs, ctrls, dim, c, s, diff);
    }
}

template <typename derived_, typename calc_type_>
void CPUDensityMatrixPolicyBase<derived_, calc_type_>::ApplyRyyNoCtrl(qs_data_p_t qs, const qbits_t& objs,
                                                                      const qbits_t& ctrls, index_t dim, calc_type c,
                                                                      qs_data_t s) {
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
void CPUDensityMatrixPolicyBase<derived_, calc_type_>::ApplyRyyCtrl(qs_data_p_t qs, const qbits_t& objs,
                                                                    const qbits_t& ctrls, index_t dim, calc_type c,
                                                                    qs_data_t s, bool diff) {
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
        derived::SetToZeroExcept(qs, mask.ctrl_mask, dim);
    }
}

template <typename derived_, typename calc_type_>
void CPUDensityMatrixPolicyBase<derived_, calc_type_>::ApplyRzz(qs_data_p_t qs, const qbits_t& objs,
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
        ApplyRzzNoCtrl(qs, objs, ctrls, dim, c, s);
    } else {
        ApplyRzzCtrl(qs, objs, ctrls, dim, c, s, diff);
    }
}

template <typename derived_, typename calc_type_>
void CPUDensityMatrixPolicyBase<derived_, calc_type_>::ApplyRzzNoCtrl(qs_data_p_t qs, const qbits_t& objs,
                                                                      const qbits_t& ctrls, index_t dim, calc_type c,
                                                                      calc_type s) {
    DoubleQubitGateMask mask(objs, ctrls);
    auto e = c + IMAGE_I * s;
    auto me = c + IMAGE_MI * s;
    auto me2 = me * me;
    auto e2 = e * e;
    THRESHOLD_OMP_FOR(
        dim, DimTh, for (omp::idx_t a = 0; a < (dim / 4); a++) {
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
void CPUDensityMatrixPolicyBase<derived_, calc_type_>::ApplyRzzCtrl(qs_data_p_t qs, const qbits_t& objs,
                                                                    const qbits_t& ctrls, index_t dim, calc_type c,
                                                                    calc_type s, bool diff) {
    DoubleQubitGateMask mask(objs, ctrls);
    auto e = c + IMAGE_I * s;
    auto me = c + IMAGE_MI * s;
    auto me2 = me * me;
    auto e2 = e * e;
    THRESHOLD_OMP_FOR(
        dim, DimTh, for (omp::idx_t a = 0; a < (dim / 4); a++) {
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
        derived::SetToZeroExcept(qs, mask.ctrl_mask, dim);
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
