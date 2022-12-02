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
#include <string>
#include <vector>

#include "config/openmp.hpp"

#include "core/utils.hpp"
#include "simulator/types.hpp"
#include "simulator/utils.hpp"
#include "simulator/vector/detail/cpu_vector_policy.hpp"

namespace mindquantum::sim::vector::detail {
auto CPUVectorPolicyBase::ExpectDiffTwoQubitsMatrix(qs_data_p_t bra, qs_data_p_t ket, const qbits_t& objs,
                                                    const qbits_t& ctrls, const std::vector<py_qs_datas_t>& gate,
                                                    index_t dim) -> qs_data_t {
    DoubleQubitGateMask mask(objs, ctrls);
    calc_type res_real = 0, res_imag = 0;
    // clang-format off
    if (!mask.ctrl_mask) {
        THRESHOLD_OMP(
            MQ_DO_PRAGMA(omp parallel for reduction(+:res_real, res_imag) schedule(static)), dim, DimTh,
        for (omp::idx_t l = 0; l < (dim / 4); l++) {
            index_t i;
            SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask, mask.obj_high_mask, mask.obj_rev_high_mask,
                            l, i);
            auto j = i + mask.obj_min_mask;
            auto k = i + mask.obj_max_mask;
            auto m = i + mask.obj_mask;
            auto v00 = gate[0][0] * ket[i] + gate[0][1] * ket[j] + gate[0][2] * ket[k] + gate[0][3] * ket[m];
            auto v01 = gate[1][0] * ket[i] + gate[1][1] * ket[j] + gate[1][2] * ket[k] + gate[1][3] * ket[m];
            auto v10 = gate[2][0] * ket[i] + gate[2][1] * ket[j] + gate[2][2] * ket[k] + gate[2][3] * ket[m];
            auto v11 = gate[3][0] * ket[i] + gate[3][1] * ket[j] + gate[3][2] * ket[k] + gate[3][3] * ket[m];
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
                auto v00 = gate[0][0] * ket[i] + gate[0][1] * ket[j] + gate[0][2] * ket[k] + gate[0][3] * ket[m];
                auto v01 = gate[1][0] * ket[i] + gate[1][1] * ket[j] + gate[1][2] * ket[k] + gate[1][3] * ket[m];
                auto v10 = gate[2][0] * ket[i] + gate[2][1] * ket[j] + gate[2][2] * ket[k] + gate[2][3] * ket[m];
                auto v11 = gate[3][0] * ket[i] + gate[3][1] * ket[j] + gate[3][2] * ket[k] + gate[3][3] * ket[m];
                auto this_res = std::conj(bra[i]) * v00;
                this_res += std::conj(bra[j]) * v01;
                this_res += std::conj(bra[k]) * v10;
                this_res += std::conj(bra[m]) * v11;
                res_real += this_res.real();
                res_imag += this_res.imag();
            }
        })
    }
    // clang-format on
    return {res_real, res_imag};
};

auto CPUVectorPolicyBase::ExpectDiffSingleQubitMatrix(qs_data_p_t bra, qs_data_p_t ket, const qbits_t& objs,
                                                      const qbits_t& ctrls, const std::vector<py_qs_datas_t>& m,
                                                      index_t dim) -> qs_data_t {
    SingleQubitGateMask mask(objs, ctrls);
#ifdef INTRIN
    gate_matrix_t gate = {{m[0][0], m[0][1]}, {m[1][0], m[1][1]}};
    __m256d neg = _mm256_setr_pd(1.0, -1.0, 1.0, -1.0);
    __m256d mm[2];
    __m256d mmt[2];
    INTRIN_gene_2d_mm_and_mmt(gate, mm, mmt, neg);
#endif
    calc_type res_real = 0, res_imag = 0;
    if (!mask.ctrl_mask) {
        // clang-format off
#ifdef INTRIN
        THRESHOLD_OMP(
            MQ_DO_PRAGMA(omp parallel for reduction(+:res_real, res_imag) schedule(static)), dim, DimTh,
                for (omp::idx_t l = 0; l < (dim / 2); l++) {
                    auto i = ((l & mask.obj_high_mask) << 1) + (l & mask.obj_low_mask);
                    auto j = i + mask.obj_mask;
                    __m256d mul_res;
                    INTRIN_M2_dot_V2(ket, i, j, mm, mmt, mul_res);
                    __m256d res;
                    INTRIN_Conj_V2_dot_V2(bra, mul_res, i, j, neg, res);
                    qs_data_t ress[2];
                    INTRIN_m256_to_host(res, ress);
                    res_real += ress[0].real() + ress[1].real();
                    res_imag += ress[0].imag() + ress[1].imag();
                });
#else
        THRESHOLD_OMP(
            MQ_DO_PRAGMA(omp parallel for reduction(+:res_real, res_imag) schedule(static)), dim, DimTh,
                for (omp::idx_t l = 0; l < (dim / 2); l++) {
                    auto i = ((l & mask.obj_high_mask) << 1) + (l & mask.obj_low_mask);
                    auto j = i + mask.obj_mask;
                    auto t1 = m[0][0] * ket[i] + m[0][1] * ket[j];
                    auto t2 = m[1][0] * ket[i] + [1][1] * ket[j];
                    auto this_res = std::conj(bra[i]) * t1 + std::conj(bra[j]) * t2;
                    res_real += this_res.real();
                    res_imag += this_res.imag();
                });
#endif
        // clang-format on
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
            // clang-format off
#ifdef INTRIN
            THRESHOLD_OMP(
                MQ_DO_PRAGMA(omp parallel for reduction(+:res_real, res_imag) schedule(static)), dim, DimTh,
                    for (omp::idx_t l = 0; l < (dim / 4); l++) {
                        auto i = ((l & first_high_mask) << 1) + (l & first_low_mask);
                        i = ((i & second_high_mask) << 1) + (i & second_low_mask) + mask.ctrl_mask;
                        auto j = i + mask.obj_mask;
                        __m256d mul_res;
                        INTRIN_M2_dot_V2(ket, i, j, mm, mmt, mul_res);
                        __m256d res;
                        INTRIN_Conj_V2_dot_V2(bra, mul_res, i, j, neg, res);
                        qs_data_t ress[2];
                        INTRIN_m256_to_host(res, ress);
                        res_real += ress[0].real() + ress[1].real();
                        res_imag += ress[0].imag() + ress[1].imag();
                    });
#else
            THRESHOLD_OMP(
                MQ_DO_PRAGMA(omp parallel for reduction(+:res_real, res_imag) schedule(static)), dim, DimTh,
                    for (omp::idx_t l = 0; l < (dim / 4); l++) {
                        auto i = ((l & first_high_mask) << 1) + (l & first_low_mask);
                        i = ((i & second_high_mask) << 1) + (i & second_low_mask) + mask.ctrl_mask;
                        auto j = i + mask.obj_mask;
                        auto t1 = m[0][0] * ket[i] + m[0][1] * ket[j];
                        auto t2 = m[1][0] * ket[i] + m[1][1] * ket[j];
                        auto this_res = std::conj(bra[i]) * t1 + std::conj(bra[j]) * t2;
                        res_real += this_res.real();
                        res_imag += this_res.imag();
                    });
#endif
            // clang-format on
        } else {
            // clang-format off
#ifdef INTRIN
            THRESHOLD_OMP(
                MQ_DO_PRAGMA(omp parallel for reduction(+:res_real, res_imag) schedule(static)), dim, DimTh,
                    for (omp::idx_t l = 0; l < (dim / 2); l++) {
                        auto i = ((l & mask.obj_high_mask) << 1) + (l & mask.obj_low_mask);
                        if ((i & mask.ctrl_mask) == mask.ctrl_mask) {
                            auto j = i + mask.obj_mask;
                            __m256d mul_res;
                            INTRIN_M2_dot_V2(ket, i, j, mm, mmt, mul_res);
                            __m256d res;
                            INTRIN_Conj_V2_dot_V2(bra, mul_res, i, j, neg, res);
                            qs_data_t ress[2];
                            INTRIN_m256_to_host(res, ress);
                            res_real += ress[0].real() + ress[1].real();
                            res_imag += ress[0].imag() + ress[1].imag();
                        }
                    });
#else
            THRESHOLD_OMP(
                MQ_DO_PRAGMA(omp parallel for reduction(+:res_real, res_imag) schedule(static)), dim, DimTh,
                    for (omp::idx_t l = 0; l < (dim / 2); l++) {
                        auto i = ((l & mask.obj_high_mask) << 1) + (l & mask.obj_low_mask);
                        if ((i & mask.ctrl_mask) == mask.ctrl_mask) {
                            auto j = i + mask.obj_mask;
                            auto t1 = m[0][0] * ket[i] + m[0][1] * ket[j];
                            auto t2 = m[1][0] * ket[i] + m[1][1] * ket[j];
                            auto this_res = std::conj(bra[i]) * t1 + std::conj(bra[j]) * t2;
                            res_real += this_res.real();
                            res_imag += this_res.imag();
                        }
                    });
#endif
            // clang-format on
        }
    }
    return {res_real, res_imag};
};

auto CPUVectorPolicyBase::ExpectDiffMatrixGate(qs_data_p_t bra, qs_data_p_t ket, const qbits_t& objs,
                                               const qbits_t& ctrls, const std::vector<py_qs_datas_t>& m, index_t dim)
    -> qs_data_t {
    if (objs.size() == 1) {
        return ExpectDiffSingleQubitMatrix(bra, ket, objs, ctrls, m, dim);
    }
    if (objs.size() == 2) {
        return ExpectDiffTwoQubitsMatrix(bra, ket, objs, ctrls, m, dim);
    }
    throw std::runtime_error("Expectation of " + std::to_string(objs.size()) + " not implement for cpu backend.");
}

auto CPUVectorPolicyBase::ExpectDiffRX(qs_data_p_t bra, qs_data_p_t ket, const qbits_t& objs, const qbits_t& ctrls,
                                       calc_type val, index_t dim) -> qs_data_t {
    auto c = -0.5 * std::sin(val / 2);
    auto is = 0.5 * std::cos(val / 2) * IMAGE_MI;
    std::vector<py_qs_datas_t> gate = {{c, is}, {is, c}};
    return CPUVectorPolicyBase::ExpectDiffSingleQubitMatrix(bra, ket, objs, ctrls, gate, dim);
};

auto CPUVectorPolicyBase::ExpectDiffRY(qs_data_p_t bra, qs_data_p_t ket, const qbits_t& objs, const qbits_t& ctrls,
                                       calc_type val, index_t dim) -> qs_data_t {
    SingleQubitGateMask mask(objs, ctrls);
    auto c = -0.5 * std::sin(val / 2);
    auto s = 0.5 * std::cos(val / 2);
    std::vector<py_qs_datas_t> gate = {{c, -s}, {s, c}};
    return CPUVectorPolicyBase::ExpectDiffSingleQubitMatrix(bra, ket, objs, ctrls, gate, dim);
};

auto CPUVectorPolicyBase::ExpectDiffRZ(qs_data_p_t bra, qs_data_p_t ket, const qbits_t& objs, const qbits_t& ctrls,
                                       calc_type val, index_t dim) -> qs_data_t {
    SingleQubitGateMask mask(objs, ctrls);
    auto c = -0.5 * std::sin(val / 2);
    auto s = 0.5 * std::cos(val / 2);
    auto e0 = c + IMAGE_MI * s;
    auto e1 = c + IMAGE_I * s;
    std::vector<py_qs_datas_t> gate = {{e0, 0}, {0, e1}};
    return CPUVectorPolicyBase::ExpectDiffSingleQubitMatrix(bra, ket, objs, ctrls, gate, dim);
};

auto CPUVectorPolicyBase::ExpectDiffGP(qs_data_p_t bra, qs_data_p_t ket, const qbits_t& objs, const qbits_t& ctrls,
                                       calc_type val, index_t dim) -> qs_data_t {
    SingleQubitGateMask mask(objs, ctrls);
    auto e = std::complex<calc_type>(0, -1);
    e *= std::exp(std::complex<calc_type>(0, -val));
    std::vector<py_qs_datas_t> gate = {{e, 0}, {0, e}};
    return CPUVectorPolicyBase::ExpectDiffSingleQubitMatrix(bra, ket, objs, ctrls, gate, dim);
}

auto CPUVectorPolicyBase::ExpectDiffPS(qs_data_p_t bra, qs_data_p_t ket, const qbits_t& objs, const qbits_t& ctrls,
                                       calc_type val, index_t dim) -> qs_data_t {
    SingleQubitGateMask mask(objs, ctrls);
    calc_type res_real = 0, res_imag = 0;
    auto e = -std::sin(val) + IMAGE_I * std::cos(val);

    if (!mask.ctrl_mask) {
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

auto CPUVectorPolicyBase::ExpectDiffXX(qs_data_p_t bra, qs_data_p_t ket, const qbits_t& objs, const qbits_t& ctrls,
                                       calc_type val, index_t dim) -> qs_data_t {
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

auto CPUVectorPolicyBase::ExpectDiffYY(qs_data_p_t bra, qs_data_p_t ket, const qbits_t& objs, const qbits_t& ctrls,
                                       calc_type val, index_t dim) -> qs_data_t {
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

auto CPUVectorPolicyBase::ExpectDiffZZ(qs_data_p_t bra, qs_data_p_t ket, const qbits_t& objs, const qbits_t& ctrls,
                                       calc_type val, index_t dim) -> qs_data_t {
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
}  // namespace mindquantum::sim::vector::detail
