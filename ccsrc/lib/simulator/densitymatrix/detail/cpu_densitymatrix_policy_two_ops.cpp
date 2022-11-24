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

#include <cmath>

#include <cassert>
#include <complex>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <ratio>
#include <stdexcept>
#include <vector>

#include "config/openmp.hpp"

#include "core/utils.hpp"
#include "simulator/densitymatrix/detail/cpu_densitymatrix_policy.hpp"
#include "simulator/types.hpp"
#include "simulator/utils.hpp"

namespace mindquantum::sim::densitymatrix::detail {

void CPUDensityMatrixPolicyBase::ApplySWAP(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls, index_t dim) {
    DoubleQubitGateMask mask(objs, ctrls);
    if (!mask.ctrl_mask) {
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
                    SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask, mask.obj_high_mask, mask.obj_rev_high_mask,
                                  b, col[0]);
                    col[3] = col[0] + mask.obj_mask;
                    col[1] = col[0] + mask.obj_min_mask;
                    col[2] = col[0] + mask.obj_max_mask;

                    qs_data_t tmp;
                    tmp = GetValue(qs, row[0], col[1]);
                    SetValue(qs, row[0], col[1], GetValue(qs, row[0], col[2]));
                    SetValue(qs, row[0], col[2], tmp);

                    tmp = GetValue(qs, row[3], col[1]);
                    SetValue(qs, row[3], col[1], GetValue(qs, row[3], col[2]));
                    SetValue(qs, row[3], col[2], tmp);

                    tmp = GetValue(qs, row[1], col[0]);
                    SetValue(qs, row[1], col[0], GetValue(qs, row[2], col[0]));
                    SetValue(qs, row[2], col[0], tmp);

                    tmp = GetValue(qs, row[1], col[3]);
                    SetValue(qs, row[1], col[3], GetValue(qs, row[2], col[3]));
                    SetValue(qs, row[2], col[3], tmp);

                    tmp = GetValue(qs, row[1], col[1]);
                    SetValue(qs, row[1], col[1], GetValue(qs, row[2], col[2]));
                    SetValue(qs, row[2], col[2], tmp);

                    tmp = GetValue(qs, row[1], col[2]);
                    SetValue(qs, row[1], col[2], GetValue(qs, row[2], col[1]));
                    SetValue(qs, row[2], col[1], tmp);
                }
            })
    } else {
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
                    SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask, mask.obj_high_mask, mask.obj_rev_high_mask,
                                  b, col[0]);
                    if ((row[0] & mask.ctrl_mask) != mask.ctrl_mask) {
                        if ((col[0] & mask.ctrl_mask) != mask.ctrl_mask) {  // both not in control
                            continue;
                        }
                    }
                    col[3] = col[0] + mask.obj_mask;
                    col[1] = col[0] + mask.obj_min_mask;
                    col[2] = col[0] + mask.obj_max_mask;
                    if ((row[0] & mask.ctrl_mask) == mask.ctrl_mask) {
                        if ((col[0] & mask.ctrl_mask) == mask.ctrl_mask) {  // both in control
                            qs_data_t tmp;
                            tmp = GetValue(qs, row[0], col[1]);
                            SetValue(qs, row[0], col[1], GetValue(qs, row[0], col[2]));
                            SetValue(qs, row[0], col[2], tmp);

                            tmp = GetValue(qs, row[3], col[1]);
                            SetValue(qs, row[3], col[1], GetValue(qs, row[3], col[2]));
                            SetValue(qs, row[3], col[2], tmp);

                            tmp = GetValue(qs, row[1], col[0]);
                            SetValue(qs, row[1], col[0], GetValue(qs, row[2], col[0]));
                            SetValue(qs, row[2], col[0], tmp);

                            tmp = GetValue(qs, row[1], col[3]);
                            SetValue(qs, row[1], col[3], GetValue(qs, row[2], col[3]));
                            SetValue(qs, row[2], col[3], tmp);

                            tmp = GetValue(qs, row[1], col[1]);
                            SetValue(qs, row[1], col[1], GetValue(qs, row[2], col[2]));
                            SetValue(qs, row[2], col[2], tmp);

                            tmp = GetValue(qs, row[1], col[2]);
                            SetValue(qs, row[1], col[2], GetValue(qs, row[2], col[1]));
                            SetValue(qs, row[2], col[1], tmp);
                        } else {  // only row in control
                            for (int i = 0; i < 4; i++) {
                                auto tmp = GetValue(qs, row[1], col[i]);
                                SetValue(qs, row[1], col[i], GetValue(qs, row[2], col[i]));
                                SetValue(qs, row[2], col[i], tmp);
                            }
                        }
                    } else {  // only column in control
                        for (int i = 0; i < 4; i++) {
                            auto tmp = GetValue(qs, row[i], col[1]);
                            SetValue(qs, row[i], col[1], GetValue(qs, row[i], col[2]));
                            SetValue(qs, row[i], col[2], tmp);
                        }
                    }
                }
            })
    }
}

void CPUDensityMatrixPolicyBase::ApplyISWAP(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls, index_t dim) {
    DoubleQubitGateMask mask(objs, ctrls);
    if (!mask.ctrl_mask) {
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
                    SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask, mask.obj_high_mask, mask.obj_rev_high_mask,
                                  b, col[0]);
                    col[3] = col[0] + mask.obj_mask;
                    col[1] = col[0] + mask.obj_min_mask;
                    col[2] = col[0] + mask.obj_max_mask;

                    qs_data_t tmp;
                    tmp = GetValue(qs, row[0], col[1]);
                    SetValue(qs, row[0], col[1], IMAGE_I * GetValue(qs, row[0], col[2]));
                    SetValue(qs, row[0], col[2], IMAGE_I * tmp);

                    tmp = GetValue(qs, row[3], col[1]);
                    SetValue(qs, row[3], col[1], IMAGE_I * GetValue(qs, row[3], col[2]));
                    SetValue(qs, row[3], col[2], IMAGE_I * tmp);

                    tmp = GetValue(qs, row[1], col[0]);
                    SetValue(qs, row[1], col[0], IMAGE_I * GetValue(qs, row[2], col[0]));
                    SetValue(qs, row[2], col[0], IMAGE_I * tmp);

                    tmp = GetValue(qs, row[1], col[3]);
                    SetValue(qs, row[1], col[3], IMAGE_I * GetValue(qs, row[2], col[3]));
                    SetValue(qs, row[2], col[3], IMAGE_I * tmp);

                    tmp = GetValue(qs, row[1], col[1]);
                    SetValue(qs, row[1], col[1], -GetValue(qs, row[2], col[2]));
                    SetValue(qs, row[2], col[2], -tmp);

                    tmp = GetValue(qs, row[1], col[2]);
                    SetValue(qs, row[1], col[2], -GetValue(qs, row[2], col[1]));
                    SetValue(qs, row[2], col[1], -tmp);
                }
            })
    } else {
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
                    SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask, mask.obj_high_mask, mask.obj_rev_high_mask,
                                  b, col[0]);
                    if ((row[0] & mask.ctrl_mask) != mask.ctrl_mask) {
                        if ((col[0] & mask.ctrl_mask) != mask.ctrl_mask) {  // both not in control
                            continue;
                        }
                    }
                    col[3] = col[0] + mask.obj_mask;
                    col[1] = col[0] + mask.obj_min_mask;
                    col[2] = col[0] + mask.obj_max_mask;
                    if ((row[0] & mask.ctrl_mask) == mask.ctrl_mask) {
                        if ((col[0] & mask.ctrl_mask) == mask.ctrl_mask) {  // both in control
                            qs_data_t tmp;
                            tmp = GetValue(qs, row[0], col[1]);
                            SetValue(qs, row[0], col[1], IMAGE_I * GetValue(qs, row[0], col[2]));
                            SetValue(qs, row[0], col[2], IMAGE_I * tmp);

                            tmp = GetValue(qs, row[3], col[1]);
                            SetValue(qs, row[3], col[1], IMAGE_I * GetValue(qs, row[3], col[2]));
                            SetValue(qs, row[3], col[2], tmp);

                            tmp = GetValue(qs, row[1], col[0]);
                            SetValue(qs, row[1], col[0], IMAGE_I * GetValue(qs, row[2], col[0]));
                            SetValue(qs, row[2], col[0], IMAGE_I * tmp);

                            tmp = GetValue(qs, row[1], col[3]);
                            SetValue(qs, row[1], col[3], IMAGE_I * GetValue(qs, row[2], col[3]));
                            SetValue(qs, row[2], col[3], IMAGE_I * tmp);

                            tmp = GetValue(qs, row[1], col[1]);
                            SetValue(qs, row[1], col[1], -GetValue(qs, row[2], col[2]));
                            SetValue(qs, row[2], col[2], -tmp);

                            tmp = GetValue(qs, row[1], col[2]);
                            SetValue(qs, row[1], col[2], -GetValue(qs, row[2], col[1]));
                            SetValue(qs, row[2], col[1], -tmp);
                        } else {  // only row in control
                            for (int i = 0; i < 4; i++) {
                                auto tmp = GetValue(qs, row[1], col[i]);
                                SetValue(qs, row[1], col[i], IMAGE_I * GetValue(qs, row[2], col[i]));
                                SetValue(qs, row[2], col[i], IMAGE_I * tmp);
                            }
                        }
                    } else {  // only column in control
                        for (int i = 0; i < 4; i++) {
                            auto tmp = GetValue(qs, row[i], col[1]);
                            SetValue(qs, row[i], col[1], IMAGE_I * GetValue(qs, row[i], col[2]));
                            SetValue(qs, row[i], col[2], IMAGE_I * tmp);
                        }
                    }
                }
            })
    }
}

void CPUDensityMatrixPolicyBase::ApplyXX(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls, calc_type val,
                                         index_t dim, bool diff) {
    DoubleQubitGateMask mask(objs, ctrls);
    auto c = std::cos(val);
    auto s = std::sin(val) * IMAGE_MI;
    if (diff) {
        c = -std::sin(val);
        s = std::cos(val) * IMAGE_MI;
    }
    if (!mask.ctrl_mask) {
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
                    SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask, mask.obj_high_mask, mask.obj_rev_high_mask,
                                  b, col[0]);
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
    } else {
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
                    SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask, mask.obj_high_mask, mask.obj_rev_high_mask,
                                  b, col[0]);
                    if ((row[0] & mask.ctrl_mask) != mask.ctrl_mask) {
                        if ((col[0] & mask.ctrl_mask) != mask.ctrl_mask) {  // both not in control
                            continue;
                        }
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
            CPUDensityMatrixPolicyBase::SetToZeroExcept(qs, mask.ctrl_mask, dim);
        }
    }
}

void CPUDensityMatrixPolicyBase::ApplyYY(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls, calc_type val,
                                         index_t dim, bool diff) {
    DoubleQubitGateMask mask(objs, ctrls);
    auto c = std::cos(val);
    auto s = std::sin(val) * IMAGE_I;
    if (diff) {
        c = -std::sin(val);
        s = std::cos(val) * IMAGE_I;
    }
    if (!mask.ctrl_mask) {
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
                    SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask, mask.obj_high_mask, mask.obj_rev_high_mask,
                                  b, col[0]);
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
    } else {
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
                    SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask, mask.obj_high_mask, mask.obj_rev_high_mask,
                                  b, col[0]);
                    if ((row[0] & mask.ctrl_mask) != mask.ctrl_mask) {
                        if ((col[0] & mask.ctrl_mask) != mask.ctrl_mask) {  // both not in control
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
                }
            })
        if (diff) {
            CPUDensityMatrixPolicyBase::SetToZeroExcept(qs, mask.ctrl_mask, dim);
        }
    }
}

void CPUDensityMatrixPolicyBase::ApplyZZ(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls, calc_type val,
                                         index_t dim, bool diff) {
    DoubleQubitGateMask mask(objs, ctrls);
    auto c = std::cos(val);
    auto s = std::sin(val);
    if (diff) {
        c = -std::sin(val);
        s = std::cos(val);
    }
    auto e = c + IMAGE_I * s;
    auto me = c + IMAGE_MI * s;
    auto me2 = me * me;
    auto e2 = e * e;
    if (!mask.ctrl_mask) {
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
                    SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask, mask.obj_high_mask, mask.obj_rev_high_mask,
                                  b, col[0]);
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
            })
    } else {
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
                    SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask, mask.obj_high_mask, mask.obj_rev_high_mask,
                                  b, col[0]);
                    if ((row[0] & mask.ctrl_mask) != mask.ctrl_mask) {
                        if ((col[0] & mask.ctrl_mask) != mask.ctrl_mask) {  // both not in control
                            continue;
                        }
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
            })
        if (diff) {
            CPUDensityMatrixPolicyBase::SetToZeroExcept(qs, mask.ctrl_mask, dim);
        }
    }
}

}  // namespace mindquantum::sim::densitymatrix::detail