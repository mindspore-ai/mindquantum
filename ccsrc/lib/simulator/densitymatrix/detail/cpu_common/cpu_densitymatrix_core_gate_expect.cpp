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
auto CPUDensityMatrixPolicyBase<derived_, calc_type_>::HamiltonianMatrix(const std::vector<PauliTerm<calc_type>>& ham,
                                                                         index_t dim) -> qs_data_p_t {
    qs_data_p_t out = InitState(dim, false);
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
                    out[IdxMap(j, i)] += coeff * c;
                }
            })
    }
    return out;
}

template <typename derived_, typename calc_type_>
auto CPUDensityMatrixPolicyBase<derived_, calc_type_>::GetExpectation(qs_data_p_t qs,
                                                                      const std::vector<PauliTerm<calc_type>>& ham,
                                                                      index_t dim) -> qs_data_t {
    calc_type e_r = 0, e_i = 0;
    for (const auto& [pauli_string, coeff_] : ham) {
        auto mask = GenPauliMask(pauli_string);
        auto mask_f = mask.mask_x | mask.mask_y;
        auto coeff = coeff_;
        // clang-format off
        THRESHOLD_OMP(
            MQ_DO_PRAGMA(omp parallel for reduction(+:e_r, e_i) schedule(static)), dim, DimTh,
                for (omp::idx_t i = 0; i < dim; i++) {
                    auto j = (i ^ mask_f);
                    if (i <= j) {
                        auto axis2power = CountOne(static_cast<int64_t>(i & mask.mask_z));  // -1
                        auto axis3power = CountOne(static_cast<int64_t>(i & mask.mask_y));  // -1j
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
    return qs_data_t(e_r, e_i);
}

template <typename derived_, typename calc_type_>
auto CPUDensityMatrixPolicyBase<derived_, calc_type_>::ExpectDiffSingleQubitMatrix(
    qs_data_p_t qs, qs_data_p_t ham_matrix, const qbits_t& objs, const qbits_t& ctrls, const matrix_t& gate_m,
    const matrix_t& diff_m, index_t dim) -> qs_data_t {
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
                for (omp::idx_t a = 0; a < (dim / 2); a++) {
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
            THRESHOLD_OMP(
                MQ_DO_PRAGMA(omp parallel for reduction(+:res_real, res_imag) schedule(static)), dim, DimTh,
                    for (omp::idx_t a = 0; a < (dim / 2); a++) {
                        auto i = ((a & first_high_mask) << 1) + (a & first_low_mask);
                        i = ((i & second_high_mask) << 1) + (i & second_low_mask) + mask.ctrl_mask;
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
                    for (omp::idx_t a = 0; a < (dim / 2); a++) {
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
    }
    return {res_real, res_imag};
};

template <typename derived_, typename calc_type_>
auto CPUDensityMatrixPolicyBase<derived_, calc_type_>::ExpectDiffTwoQubitsMatrix(qs_data_p_t qs, qs_data_p_t ham_matrix,
                                                                                 const qbits_t& objs,
                                                                                 const qbits_t& ctrls,
                                                                                 const matrix_t& gate_m,
                                                                                 const matrix_t& diff_m, index_t dim)
    -> qs_data_t {
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
                for (omp::idx_t a = 0; a < (dim / 2); a++) {
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
                for (omp::idx_t a = 0; a < (dim / 2); a++) {
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
    return {res_real, res_imag};
};

template <typename derived_, typename calc_type_>
auto CPUDensityMatrixPolicyBase<derived_, calc_type_>::ExpectDiffMatrixGate(qs_data_p_t qs, qs_data_p_t ham_matrix,
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
    throw std::runtime_error("Expectation of " + std::to_string(objs.size()) + " not implement for cpu backend.");
}

template <typename derived_, typename calc_type_>
auto CPUDensityMatrixPolicyBase<derived_, calc_type_>::ExpectDiffRX(qs_data_p_t qs, qs_data_p_t ham_matrix,
                                                                    const qbits_t& objs, const qbits_t& ctrls,
                                                                    index_t dim) -> qs_data_t {
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
                for (omp::idx_t a = 0; a < (dim / 2); a++) {
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
    return {res_real, res_imag};
};

template <typename derived_, typename calc_type_>
auto CPUDensityMatrixPolicyBase<derived_, calc_type_>::ExpectDiffRY(qs_data_p_t qs, qs_data_p_t ham_matrix,
                                                                    const qbits_t& objs, const qbits_t& ctrls,
                                                                    index_t dim) -> qs_data_t {
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
                for (omp::idx_t a = 0; a < (dim / 2); a++) {
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
    return {res_real, res_imag};
};

template <typename derived_, typename calc_type_>
auto CPUDensityMatrixPolicyBase<derived_, calc_type_>::ExpectDiffRZ(qs_data_p_t qs, qs_data_p_t ham_matrix,
                                                                    const qbits_t& objs, const qbits_t& ctrls,
                                                                    index_t dim) -> qs_data_t {
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
                for (omp::idx_t a = 0; a < (dim / 2); a++) {
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
    return {res_real, res_imag};
};

template <typename derived_, typename calc_type_>
auto CPUDensityMatrixPolicyBase<derived_, calc_type_>::ExpectDiffPS(qs_data_p_t qs, qs_data_p_t ham_matrix,
                                                                    const qbits_t& objs, const qbits_t& ctrls,
                                                                    index_t dim) -> qs_data_t {
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
                for (omp::idx_t a = 0; a < (dim / 2); a++) {
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
    return {res_real, res_imag};
};

template <typename derived_, typename calc_type_>
auto CPUDensityMatrixPolicyBase<derived_, calc_type_>::ExpectDiffU3Theta(qs_data_p_t qs, qs_data_p_t ham_matrix,
                                                                         const qbits_t& objs, const qbits_t& ctrls,
                                                                         calc_type phi, index_t dim) -> qs_data_t {
    qs_data_t e_phi = static_cast<calc_type>(0.5) * std::exp(std::complex<calc_type>(0, phi));
    qs_data_t e_m_phi = static_cast<calc_type>(-0.5) * std::exp(std::complex<calc_type>(0, -phi));
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
                for (omp::idx_t a = 0; a < (dim / 2); a++) {
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
    return {res_real, res_imag};
};

template <typename derived_, typename calc_type_>
auto CPUDensityMatrixPolicyBase<derived_, calc_type_>::ExpectDiffU3Phi(qs_data_p_t qs, qs_data_p_t ham_matrix,
                                                                       const qbits_t& objs, const qbits_t& ctrls,
                                                                       index_t dim) -> qs_data_t {
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
                for (omp::idx_t a = 0; a < (dim / 2); a++) {
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
    return {res_real, res_imag};
};

template <typename derived_, typename calc_type_>
auto CPUDensityMatrixPolicyBase<derived_, calc_type_>::ExpectDiffRxx(qs_data_p_t qs, qs_data_p_t ham_matrix,
                                                                     const qbits_t& objs, const qbits_t& ctrls,
                                                                     index_t dim) -> qs_data_t {
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
    return {res_real, res_imag};
};

template <typename derived_, typename calc_type_>
auto CPUDensityMatrixPolicyBase<derived_, calc_type_>::ExpectDiffRyy(qs_data_p_t qs, qs_data_p_t ham_matrix,
                                                                     const qbits_t& objs, const qbits_t& ctrls,
                                                                     index_t dim) -> qs_data_t {
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
    return {res_real, res_imag};
};

template <typename derived_, typename calc_type_>
auto CPUDensityMatrixPolicyBase<derived_, calc_type_>::ExpectDiffRzz(qs_data_p_t qs, qs_data_p_t ham_matrix,
                                                                     const qbits_t& objs, const qbits_t& ctrls,
                                                                     index_t dim) -> qs_data_t {
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
    return {res_real, res_imag};
};

template <typename derived_, typename calc_type_>
auto CPUDensityMatrixPolicyBase<derived_, calc_type_>::ExpectDiffFSimTheta(qs_data_p_t qs, qs_data_p_t ham_matrix,
                                                                           const qbits_t& objs, const qbits_t& ctrls,
                                                                           index_t dim) -> qs_data_t {
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
    return {res_real, res_imag};
};

template <typename derived_, typename calc_type_>
auto CPUDensityMatrixPolicyBase<derived_, calc_type_>::ExpectDiffFSimPhi(qs_data_p_t qs, qs_data_p_t ham_matrix,
                                                                         const qbits_t& objs, const qbits_t& ctrls,
                                                                         index_t dim) -> qs_data_t {
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
                            this_res += GetValue(qs, r3, col) * GetValue(ham_matrix, col, r3);
                        }
                        this_res *= IMAGE_I;
                        res_real += this_res.real();
                        res_imag += this_res.imag();
                    }
                })
        // clang-format on
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
