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
auto CPUDensityMatrixPolicyBase::ExpectDiffSingleQubitMatrix(qs_data_p_t qs, qs_data_p_t evolved_ham,
                                                             const qbits_t& objs, const qbits_t& ctrls,
                                                             const py_qs_datas_t& m, index_t dim) -> qs_data_t {
    SingleQubitGateMask mask(objs, ctrls);
    qs_data_t res;
    if (!mask.ctrl_mask) {
        THRESHOLD_OMP(
            MQ_DO_PRAGMA(omp parallel for reduction(+:res) schedule(static)), dim, DimTh, for (omp::idx_t a = 0; a < (dim / 2); a++) {
                    auto i = ((a & mask.obj_high_mask) << 1) + (a & mask.obj_low_mask);
                    auto j = i + mask.obj_mask;
                    for (index_t col = 0; col < dim; col++) {
                        if (j >= col) {
                            qs_data_t _i_col = m[0][0] * qs[IdxMap(i, col)] + m[0][1] * qs[IdxMap(j, col)];
                            qs_data_t _j_col = m[1][0] * qs[IdxMap(i, col)] + m[1][1] * qs[IdxMap(j, col)];
                            res += _i_col * std::conj(evolved_ham[IdxMap(i, col)])
                                   + _j_col * std::conj(evolved_ham[IdxMap(j, col)]);
                        } else if (i < col) {
                            qs_data_t _i_col = m[0][0] * std::conj(qs[IdxMap(col, i)])
                                               + m[0][1] * std::conj(qs[IdxMap(col, j)]);
                            qs_data_t _j_col = m[1][0] * std::conj(qs[IdxMap(col, i)])
                                               + m[1][1] * std::conj(qs[IdxMap(col, j)]);
                            res += _i_col * evolved_ham[IdxMap(col, i)] + _j_col * evolved_ham[IdxMap(col, j)];
                        } else {
                            qs_data_t _i_col = m[0][0] * qs[IdxMap(i, col)] + m[0][1] * std::conj(qs[IdxMap(col, j)]);
                            qs_data_t _j_col = m[1][0] * qs[IdxMap(i, col)] + m[1][1] * std::conj(qs[IdxMap(col, j)]);
                            res += _i_col * std::conj(evolved_ham[IdxMap(i, col)])
                                   + _j_col * evolved_ham[IdxMap(col, j)];
                        }
                    }
                });
    } else {
        if (mask.ctrl_qubits.size() == 1) {
            index_t ctrl_low = 0UL;
            for (qbit_t i = 0; i < mask.ctrl_qubits[0]; i++) {
                ctrl_low = (ctrl_low << 1) + 1;
            }
            index_t first_low_mask = mask.obj_low_mask;
            index_t second_low_mask = ctrl_low;
            if (mask.obj_low_mask > ctrl_low) {
                first_low_mask = ctrl_low;
                second_low_mask = mask.obj_low_mask;
            }
            auto first_high_mask = ~first_low_mask;
            auto second_high_mask = ~second_low_mask;

            THRESHOLD_OMP(
                MQ_DO_PRAGMA(omp parallel for reduction(+:res) schedule(static)), dim, DimTh, for (index_t a = 0; a < (dim / 2); a++) {
                        auto i = ((a & first_high_mask) << 1) + (a & first_low_mask);
                        i = ((i & second_high_mask) << 1) + (i & second_low_mask) + mask.ctrl_mask;
                        auto j = i + mask.obj_mask;
                        for (index_t b = 0; b <= a; b++) {
                            auto p = ((b & mask.obj_high_mask) << 1) + (b & mask.obj_low_mask);
                            auto q = p + mask.obj_mask;
                            qs_data_t _i_p = m[0][0] * qs[IdxMap(i, p)] + m[0][1] * qs[IdxMap(j, p)];
                            qs_data_t _j_p = m[1][0] * qs[IdxMap(i, p)] + m[1][1] * qs[IdxMap(j, p)];
                            qs_data_t _i_q;
                            qs_data_t _j_q;
                            if (i > q) {
                                _i_q = m[0][0] * qs[IdxMap(i, q)] + m[0][1] * qs[IdxMap(j, q)];
                                _j_q = m[1][0] * qs[IdxMap(i, q)] + m[1][1] * qs[IdxMap(j, q)];
                                res += _i_p * evolved_ham[IdxMap(i, p)] + _j_p * evolved_ham[IdxMap(j, p)]
                                       + _i_q * evolved_ham[IdxMap(i, q)] + _j_q * evolved_ham[IdxMap(j, q)];
                            } else {
                                _i_q = m[0][0] * std::conj(qs[IdxMap(q, i)]) + m[0][1] * qs[IdxMap(j, q)];
                                _j_q = m[1][0] * std::conj(qs[IdxMap(q, i)]) + m[1][1] * qs[IdxMap(j, q)];
                                res += _i_p * evolved_ham[IdxMap(i, p)] + _j_p * evolved_ham[IdxMap(j, p)]
                                       + _i_q * std::conj(evolved_ham[IdxMap(q, i)]) + _j_q * evolved_ham[IdxMap(j, q)];
                            }
                        }
                    });
        } else {
            THRESHOLD_OMP(
                MQ_DO_PRAGMA(omp parallel for reduction(+:res) schedule(static)), dim, DimTh, for (index_t a = 0; a < (dim / 2); a++) {
                        auto i = ((a & mask.obj_high_mask) << 1) + (a & mask.obj_low_mask);
                        if ((i & mask.ctrl_mask) == mask.ctrl_mask) {
                            auto j = i + mask.obj_mask;
                            for (index_t b = 0; b <= a; b++) {
                                auto p = ((b & mask.obj_high_mask) << 1) + (b & mask.obj_low_mask);
                                auto q = p + mask.obj_mask;
                                qs_data_t _i_p = m[0][0] * qs[IdxMap(i, p)] + m[0][1] * qs[IdxMap(j, p)];
                                qs_data_t _j_p = m[1][0] * qs[IdxMap(i, p)] + m[1][1] * qs[IdxMap(j, p)];
                                qs_data_t _i_q;
                                qs_data_t _j_q;
                                if (i > q) {
                                    _i_q = m[0][0] * qs[IdxMap(i, q)] + m[0][1] * qs[IdxMap(j, q)];
                                    _j_q = m[1][0] * qs[IdxMap(i, q)] + m[1][1] * qs[IdxMap(j, q)];
                                    res += _i_p * evolved_ham[IdxMap(i, p)] + _j_p * evolved_ham[IdxMap(j, p)]
                                           + _i_q * evolved_ham[IdxMap(i, q)] + _j_q * evolved_ham[IdxMap(j, q)];
                                } else {
                                    _i_q = m[0][0] * std::conj(qs[IdxMap(q, i)]) + m[0][1] * qs[IdxMap(j, q)];
                                    _j_q = m[1][0] * std::conj(qs[IdxMap(q, i)]) + m[1][1] * qs[IdxMap(j, q)];
                                    res += _i_p * evolved_ham[IdxMap(i, p)] + _j_p * evolved_ham[IdxMap(j, p)]
                                           + _i_q * std::conj(evolved_ham[IdxMap(q, i)])
                                           + _j_q * evolved_ham[IdxMap(j, q)];
                                }
                            }
                        }
                    });
        }
    }
    return res;
};

auto CPUDensityMatrixPolicyBase::ExpectDiffRX(qs_data_p_t bra, qs_data_p_t ket, const qbits_t& objs,
                                              const qbits_t& ctrls, calc_type val, index_t dim) -> qs_data_t {
    qs_data_t c = -0.5 * std::sin(val / 2);
    qs_data_t is = 0.5 * std::cos(val / 2) * IMAGE_MI;
    // py_qs_datas_t gate = {{c, is}, {is, c}};

    qs_data_t cosTheta = std::cos(val / 2);
    qs_data_t iSinTheta = std::sin(val / 2) * IMAGE_MI;
    py_qs_datas_t gate = {{c * cosTheta + is * iSinTheta, c * iSinTheta + is * cosTheta},
                          {c * iSinTheta + is * cosTheta, c * cosTheta + is * iSinTheta}};
    std::cout << c * cosTheta + is * iSinTheta << c * iSinTheta + is * cosTheta << std::endl;
    return CPUDensityMatrixPolicyBase::ExpectDiffSingleQubitMatrix(bra, ket, objs, ctrls, gate, dim);
};

auto CPUDensityMatrixPolicyBase::ExpectDiffRY(qs_data_p_t bra, qs_data_p_t ket, const qbits_t& objs,
                                              const qbits_t& ctrls, calc_type val, index_t dim) -> qs_data_t {
    SingleQubitGateMask mask(objs, ctrls);
    qs_data_t c = -0.5 * std::sin(val / 2);
    qs_data_t s = 0.5 * std::cos(val / 2);
    py_qs_datas_t gate = {{c, -s}, {s, c}};
    return CPUDensityMatrixPolicyBase::ExpectDiffSingleQubitMatrix(bra, ket, objs, ctrls, gate, dim);
};

auto CPUDensityMatrixPolicyBase::ExpectDiffRZ(qs_data_p_t bra, qs_data_p_t ket, const qbits_t& objs,
                                              const qbits_t& ctrls, calc_type val, index_t dim) -> qs_data_t {
    SingleQubitGateMask mask(objs, ctrls);
    qs_data_t c = -0.5 * std::sin(val / 2);
    qs_data_t s = 0.5 * std::cos(val / 2);
    qs_data_t e0 = c + IMAGE_MI * s;
    qs_data_t e1 = c + IMAGE_I * s;
    py_qs_datas_t gate = {{e0, 0}, {0, e1}};
    return CPUDensityMatrixPolicyBase::ExpectDiffSingleQubitMatrix(bra, ket, objs, ctrls, gate, dim);
};

auto CPUDensityMatrixPolicyBase::ExpectDiffPS(qs_data_p_t bra, qs_data_p_t ket, const qbits_t& objs,
                                              const qbits_t& ctrls, calc_type val, index_t dim) -> qs_data_t {
    SingleQubitGateMask mask(objs, ctrls);
    calc_type res_real = 0, res_imag = 0;

    if (!mask.ctrl_mask) {
        auto e = std::cos(val) + IMAGE_I * std::sin(val);
        // clang-format off
        THRESHOLD_OMP(
            MQ_DO_PRAGMA(omp parallel for reduction(+:res_real, res_imag) schedule(static)), dim, DimTh,
                for (omp::idx_t l = 0; l < (dim / 2); l++) {
                    auto i = ((l & mask.obj_high_mask) << 1) + (l & mask.obj_low_mask);
                    auto j = i + mask.obj_mask;
                    auto this_res = std::conj(bra[j]) * ket[j] * e;
                    res_real += this_res.real();
                    res_imag += this_res.imag();
                })
        // clang-format on
    } else {
        auto e = -std::sin(val) + IMAGE_I * std::cos(val);
        // clang-format off
        THRESHOLD_OMP(
            MQ_DO_PRAGMA(omp parallel for reduction(+:res_real, res_imag) schedule(static)), dim, DimTh,
                for (omp::idx_t l = 0; l < (dim / 2); l++) {
                    auto i = ((l & mask.obj_high_mask) << 1) + (l & mask.obj_low_mask);
                    if ((i & mask.ctrl_mask) == mask.ctrl_mask) {
                        auto j = i + mask.obj_mask;
                        auto this_res = std::conj(bra[j]) * ket[j] * e;
                        res_real += this_res.real();
                        res_imag += this_res.imag();
                    }
                })
        // clang-format on
    }
    return {res_real, res_imag};
};

auto CPUDensityMatrixPolicyBase::ExpectDiffXX(qs_data_p_t bra, qs_data_p_t ket, const qbits_t& objs,
                                              const qbits_t& ctrls, calc_type val, index_t dim) -> qs_data_t {
    DoubleQubitGateMask mask(objs, ctrls);
    auto c = -std::sin(val);
    auto s = std::cos(val) * IMAGE_MI;
    // TODO(xuxs): INTRIN
    calc_type res_real = 0, res_imag = 0;
    if (!mask.ctrl_mask) {
        THRESHOLD_OMP(
            MQ_DO_PRAGMA(omp parallel for reduction(+:res_real, res_imag) schedule(static)), dim, DimTh,
                                                    for (omp::idx_t l = 0; l < (dim / 4); l++) {
                                                        index_t i;
                                                        SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask,
                                                                      mask.obj_high_mask, mask.obj_rev_high_mask, l, i);
                                                        auto m = i + mask.obj_mask;
                                                        auto j = i + mask.obj_min_mask;
                                                        auto k = i + mask.obj_max_mask;
                                                        auto v00 = c * ket[i] + s * ket[m];
                                                        auto v01 = c * ket[j] + s * ket[k];
                                                        auto v10 = c * ket[k] + s * ket[j];
                                                        auto v11 = c * ket[m] + s * ket[i];
                                                        auto this_res = std::conj(bra[i]) * v00;
                                                        this_res += std::conj(bra[j]) * v01;
                                                        this_res += std::conj(bra[k]) * v10;
                                                        this_res += std::conj(bra[m]) * v11;
                                                        res_real += this_res.real();
                                                        res_imag += this_res.imag();
                                                    })
    } else {
        THRESHOLD_OMP(
            MQ_DO_PRAGMA(omp parallel for reduction(+:res_real, res_imag) schedule(static)), dim, DimTh,
                                                    for (omp::idx_t l = 0; l < (dim / 4); l++) {
                                                        index_t i;
                                                        SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask,
                                                                      mask.obj_high_mask, mask.obj_rev_high_mask, l, i);
                                                        if ((i & mask.ctrl_mask) == mask.ctrl_mask) {
                                                            auto m = i + mask.obj_mask;
                                                            auto j = i + mask.obj_min_mask;
                                                            auto k = i + mask.obj_max_mask;
                                                            auto v00 = c * ket[i] + s * ket[m];
                                                            auto v01 = c * ket[j] + s * ket[k];
                                                            auto v10 = c * ket[k] + s * ket[j];
                                                            auto v11 = c * ket[m] + s * ket[i];
                                                            auto this_res = std::conj(bra[i]) * v00;
                                                            this_res += std::conj(bra[j]) * v01;
                                                            this_res += std::conj(bra[k]) * v10;
                                                            this_res += std::conj(bra[m]) * v11;
                                                            res_real += this_res.real();
                                                            res_imag += this_res.imag();
                                                        }
                                                    })
    }
    return {res_real, res_imag};
};

auto CPUDensityMatrixPolicyBase::ExpectDiffYY(qs_data_p_t bra, qs_data_p_t ket, const qbits_t& objs,
                                              const qbits_t& ctrls, calc_type val, index_t dim) -> qs_data_t {
    DoubleQubitGateMask mask(objs, ctrls);
    auto c = -std::sin(val);
    auto s = std::cos(val) * IMAGE_I;
    calc_type res_real = 0, res_imag = 0;
    // TODO(xuxs): INTRIN
    if (!mask.ctrl_mask) {
        THRESHOLD_OMP(
            MQ_DO_PRAGMA(omp parallel for reduction(+:res_real, res_imag) schedule(static)), dim, DimTh,
                                                    for (omp::idx_t l = 0; l < (dim / 4); l++) {
                                                        index_t i;
                                                        SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask,
                                                                      mask.obj_high_mask, mask.obj_rev_high_mask, l, i);
                                                        auto m = i + mask.obj_mask;
                                                        auto j = i + mask.obj_min_mask;
                                                        auto k = i + mask.obj_max_mask;
                                                        auto v00 = c * ket[i] + s * ket[m];
                                                        auto v01 = c * ket[j] - s * ket[k];
                                                        auto v10 = c * ket[k] - s * ket[j];
                                                        auto v11 = c * ket[m] + s * ket[i];
                                                        auto this_res = std::conj(bra[i]) * v00;
                                                        this_res += std::conj(bra[j]) * v01;
                                                        this_res += std::conj(bra[k]) * v10;
                                                        this_res += std::conj(bra[m]) * v11;
                                                        res_real += this_res.real();
                                                        res_imag += this_res.imag();
                                                    })
    } else {
        THRESHOLD_OMP(
            MQ_DO_PRAGMA(omp parallel for reduction(+:res_real, res_imag) schedule(static)), dim, DimTh,
                                                    for (omp::idx_t l = 0; l < (dim / 4); l++) {
                                                        index_t i;
                                                        SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask,
                                                                      mask.obj_high_mask, mask.obj_rev_high_mask, l, i);
                                                        if ((i & mask.ctrl_mask) == mask.ctrl_mask) {
                                                            auto m = i + mask.obj_mask;
                                                            auto j = i + mask.obj_min_mask;
                                                            auto k = i + mask.obj_max_mask;
                                                            auto v00 = c * ket[i] + s * ket[m];
                                                            auto v01 = c * ket[j] - s * ket[k];
                                                            auto v10 = c * ket[k] - s * ket[j];
                                                            auto v11 = c * ket[m] + s * ket[i];
                                                            auto this_res = std::conj(bra[i]) * v00;
                                                            this_res += std::conj(bra[j]) * v01;
                                                            this_res += std::conj(bra[k]) * v10;
                                                            this_res += std::conj(bra[m]) * v11;
                                                            res_real += this_res.real();
                                                            res_imag += this_res.imag();
                                                        }
                                                    })
    }
    return {res_real, res_imag};
};

auto CPUDensityMatrixPolicyBase::ExpectDiffZZ(qs_data_p_t bra, qs_data_p_t ket, const qbits_t& objs,
                                              const qbits_t& ctrls, calc_type val, index_t dim) -> qs_data_t {
    DoubleQubitGateMask mask(objs, ctrls);

    auto c = -std::sin(val);
    auto s = std::cos(val);
    auto e = c + IMAGE_I * s;
    auto me = c + IMAGE_MI * s;
    calc_type res_real = 0, res_imag = 0;
    // TODO(xuxs): INTRIN
    if (!mask.ctrl_mask) {
        THRESHOLD_OMP(
            MQ_DO_PRAGMA(omp parallel for reduction(+:res_real, res_imag) schedule(static)), dim, DimTh,
                                                    for (omp::idx_t l = 0; l < (dim / 4); l++) {
                                                        index_t i;
                                                        SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask,
                                                                      mask.obj_high_mask, mask.obj_rev_high_mask, l, i);
                                                        auto m = i + mask.obj_mask;
                                                        auto j = i + mask.obj_min_mask;
                                                        auto k = i + mask.obj_max_mask;
                                                        auto this_res = std::conj(bra[i]) * ket[i] * me;
                                                        this_res += std::conj(bra[j]) * ket[j] * e;
                                                        this_res += std::conj(bra[k]) * ket[k] * e;
                                                        this_res += std::conj(bra[m]) * ket[m] * me;
                                                        res_real += this_res.real();
                                                        res_imag += this_res.imag();
                                                    })
    } else {
        THRESHOLD_OMP(
            MQ_DO_PRAGMA(omp parallel for reduction(+:res_real, res_imag) schedule(static)), dim, DimTh,
                                                    for (omp::idx_t l = 0; l < (dim / 4); l++) {
                                                        index_t i;
                                                        SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask,
                                                                      mask.obj_high_mask, mask.obj_rev_high_mask, l, i);
                                                        if ((i & mask.ctrl_mask) == mask.ctrl_mask) {
                                                            auto m = i + mask.obj_mask;
                                                            auto j = i + mask.obj_min_mask;
                                                            auto k = i + mask.obj_max_mask;
                                                            auto this_res = std::conj(bra[i]) * ket[i] * me;
                                                            this_res += std::conj(bra[j]) * ket[j] * e;
                                                            this_res += std::conj(bra[k]) * ket[k] * e;
                                                            this_res += std::conj(bra[m]) * ket[m] * me;
                                                            res_real += this_res.real();
                                                            res_imag += this_res.imag();
                                                        }
                                                    })
    }
    return {res_real, res_imag};
};
}  // namespace mindquantum::sim::densitymatrix::detail
