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

auto CPUDensityMatrixPolicyBase::HamiltonianMatrix(const std::vector<PauliTerm<calc_type>>& ham, index_t dim)
    -> qs_data_p_t {
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
                    auto c = POLAR[static_cast<char>((mask.num_y + 2 * axis3power + 2 * axis2power) & 3)];
                    out[IdxMap(j, i)] = coeff * c;
                }
            })
    }
    return out;
}

auto CPUDensityMatrixPolicyBase::GetExpectation(qs_data_p_t qs, const std::vector<PauliTerm<calc_type>>& ham,
                                                index_t dim) -> qs_data_t {
    qs_data_t expectation_value = 0;
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
                    auto c = POLAR[static_cast<char>((mask.num_y + 2 * axis3power + 2 * axis2power) & 3)];
                    expectation_value += std::conj(qs[IdxMap(j, i)]) * coeff * c;
                    if (i != j) {
                        expectation_value += qs[IdxMap(j, i)] * coeff / c;
                    }
                }
            })
    }
    return expectation_value;
}

auto CPUDensityMatrixPolicyBase::ExpectDiffRX(qs_data_p_t qs, qs_data_p_t ham_matrix, const qbits_t& objs,
                                              const qbits_t& ctrls, index_t dim) -> qs_data_t {
    SingleQubitGateMask mask(objs, ctrls);
    constexpr qs_data_t HALF_MI{0, -0.5};
    qs_data_t res = 0;
    if (!mask.ctrl_mask) {
        THRESHOLD_OMP(
            MQ_DO_PRAGMA(omp parallel for reduction(+:res) schedule(static)), dim, DimTh, for (omp::idx_t a = 0; a < (dim / 2); a++) {
                    auto i = ((a & mask.obj_high_mask) << 1) + (a & mask.obj_low_mask);
                    auto j = i + mask.obj_mask;
                    for (index_t col = 0; col < dim; col++) {
                        res += HALF_MI * GetValue(qs, j, col) * GetValue(ham_matrix, col, i);
                        res += HALF_MI * GetValue(qs, i, col) * GetValue(ham_matrix, col, j);
                    }
                })
    } else {
        THRESHOLD_OMP(
            MQ_DO_PRAGMA(omp parallel for reduction(+:res) schedule(static)), dim, DimTh, for (omp::idx_t a = 0; a < (dim / 2); a++) {
                    auto i = ((a & mask.obj_high_mask) << 1) + (a & mask.obj_low_mask);
                    if ((i & mask.ctrl_mask) == mask.ctrl_mask) {
                        auto j = i + mask.obj_mask;
                        for (index_t col = 0; col < dim; col++) {
                            res += HALF_MI * GetValue(qs, j, col) * GetValue(ham_matrix, col, i);
                            res += HALF_MI * GetValue(qs, i, col) * GetValue(ham_matrix, col, j);
                        }
                    }
                })
    }
    return res;
};

auto CPUDensityMatrixPolicyBase::ExpectDiffRY(qs_data_p_t qs, qs_data_p_t ham_matrix, const qbits_t& objs,
                                              const qbits_t& ctrls, index_t dim) -> qs_data_t {
    SingleQubitGateMask mask(objs, ctrls);
    qs_data_t res = 0;
    if (!mask.ctrl_mask) {
        THRESHOLD_OMP(
            MQ_DO_PRAGMA(omp parallel for reduction(+:res) schedule(static)), dim, DimTh, for (omp::idx_t a = 0; a < (dim / 2); a++) {
                    auto i = ((a & mask.obj_high_mask) << 1) + (a & mask.obj_low_mask);
                    auto j = i + mask.obj_mask;
                    for (index_t col = 0; col < dim; col++) {
                        res += -0.5 * GetValue(qs, j, col) * GetValue(ham_matrix, col, i);
                        res += 0.5 * GetValue(qs, i, col) * GetValue(ham_matrix, col, j);
                    }
                })
    } else {
        THRESHOLD_OMP(
            MQ_DO_PRAGMA(omp parallel for reduction(+:res) schedule(static)), dim, DimTh, for (omp::idx_t a = 0; a < (dim / 2); a++) {
                    auto i = ((a & mask.obj_high_mask) << 1) + (a & mask.obj_low_mask);
                    if ((i & mask.ctrl_mask) == mask.ctrl_mask) {
                        auto j = i + mask.obj_mask;
                        for (index_t col = 0; col < dim; col++) {
                            res += -0.5 * GetValue(qs, j, col) * GetValue(ham_matrix, col, i);
                            res += 0.5 * GetValue(qs, i, col) * GetValue(ham_matrix, col, j);
                        }
                    }
                })
    }
    return res;
};

auto CPUDensityMatrixPolicyBase::ExpectDiffRZ(qs_data_p_t qs, qs_data_p_t ham_matrix, const qbits_t& objs,
                                              const qbits_t& ctrls, index_t dim) -> qs_data_t {
    constexpr qs_data_t HALF_I{0, 0.5};
    SingleQubitGateMask mask(objs, ctrls);
    qs_data_t res = 0;
    if (!mask.ctrl_mask) {
        THRESHOLD_OMP(
            MQ_DO_PRAGMA(omp parallel for reduction(+:res) schedule(static)), dim, DimTh, for (omp::idx_t a = 0; a < (dim / 2); a++) {
                    auto i = ((a & mask.obj_high_mask) << 1) + (a & mask.obj_low_mask);
                    auto j = i + mask.obj_mask;
                    for (index_t col = 0; col < dim; col++) {
                        res += -HALF_I * GetValue(qs, i, col) * GetValue(ham_matrix, col, i);
                        res += HALF_I * GetValue(qs, j, col) * GetValue(ham_matrix, col, j);
                    }
                })
    } else {
        THRESHOLD_OMP(
            MQ_DO_PRAGMA(omp parallel for reduction(+:res) schedule(static)), dim, DimTh, for (omp::idx_t a = 0; a < (dim / 2); a++) {
                    auto i = ((a & mask.obj_high_mask) << 1) + (a & mask.obj_low_mask);
                    if ((i & mask.ctrl_mask) == mask.ctrl_mask) {
                        auto j = i + mask.obj_mask;
                        for (index_t col = 0; col < dim; col++) {
                            res += -HALF_I * GetValue(qs, i, col) * GetValue(ham_matrix, col, i);
                            res += HALF_I * GetValue(qs, j, col) * GetValue(ham_matrix, col, j);
                        }
                    }
                })
    }
    return res;
};

auto CPUDensityMatrixPolicyBase::ExpectDiffPS(qs_data_p_t qs, qs_data_p_t ham_matrix, const qbits_t& objs,
                                              const qbits_t& ctrls, index_t dim) -> qs_data_t {
    SingleQubitGateMask mask(objs, ctrls);
    qs_data_t res = 0;
    if (!mask.ctrl_mask) {
        THRESHOLD_OMP(
            MQ_DO_PRAGMA(omp parallel for reduction(+:res) schedule(static)), dim, DimTh, for (omp::idx_t a = 0; a < (dim / 2); a++) {
                    auto i = ((a & mask.obj_high_mask) << 1) + (a & mask.obj_low_mask);
                    auto j = i + mask.obj_mask;
                    for (index_t col = 0; col < dim; col++) {
                        res += IMAGE_I * GetValue(qs, j, col) * GetValue(ham_matrix, col, j);
                    }
                })
    } else {
        THRESHOLD_OMP(
            MQ_DO_PRAGMA(omp parallel for reduction(+:res) schedule(static)), dim, DimTh, for (omp::idx_t a = 0; a < (dim / 2); a++) {
                    auto i = ((a & mask.obj_high_mask) << 1) + (a & mask.obj_low_mask);
                    if ((i & mask.ctrl_mask) == mask.ctrl_mask) {
                        auto j = i + mask.obj_mask;
                        for (index_t col = 0; col < dim; col++) {
                            res += IMAGE_I * GetValue(qs, j, col) * GetValue(ham_matrix, col, j);
                        }
                    }
                })
    }
    return res;
};

auto CPUDensityMatrixPolicyBase::ExpectDiffXX(qs_data_p_t qs, qs_data_p_t ham_matrix, const qbits_t& objs,
                                              const qbits_t& ctrls, index_t dim) -> qs_data_t {
    DoubleQubitGateMask mask(objs, ctrls);
    qs_data_t res = 0;
    if (!mask.ctrl_mask) {
        THRESHOLD_OMP(
            MQ_DO_PRAGMA(omp parallel for reduction(+:res) schedule(static)), dim, DimTh, for (omp::idx_t l = 0; l < (dim / 4); l++) {
                    index_t r0;
                    SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask, mask.obj_high_mask, mask.obj_rev_high_mask,
                                  l, r0);
                    auto r3 = r0 + mask.obj_mask;
                    auto r1 = r0 + mask.obj_min_mask;
                    auto r2 = r0 + mask.obj_max_mask;
                    for (index_t col = 0; col < dim; col++) {
                        res += IMAGE_MI
                               * (GetValue(qs, r3, col) * GetValue(ham_matrix, col, r0)
                                  + GetValue(qs, r2, col) * GetValue(ham_matrix, col, r1)
                                  + GetValue(qs, r1, col) * GetValue(ham_matrix, col, r2)
                                  + GetValue(qs, r0, col) * GetValue(ham_matrix, col, r3));
                    }
                })
    } else {
        THRESHOLD_OMP(
            MQ_DO_PRAGMA(omp parallel for reduction(+:res) schedule(static)), dim, DimTh, for (omp::idx_t l = 0; l < (dim / 4); l++) {
                    index_t r0;
                    SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask, mask.obj_high_mask, mask.obj_rev_high_mask,
                                  l, r0);
                    if ((r0 & mask.ctrl_mask) == mask.ctrl_mask) {
                        auto r3 = r0 + mask.obj_mask;
                        auto r1 = r0 + mask.obj_min_mask;
                        auto r2 = r0 + mask.obj_max_mask;
                        for (index_t col = 0; col < dim; col++) {
                            res += IMAGE_MI
                                   * (GetValue(qs, r3, col) * GetValue(ham_matrix, col, r0)
                                      + GetValue(qs, r2, col) * GetValue(ham_matrix, col, r1)
                                      + GetValue(qs, r1, col) * GetValue(ham_matrix, col, r2)
                                      + GetValue(qs, r0, col) * GetValue(ham_matrix, col, r3));
                        }
                    }
                })
    }
    return res;
};

auto CPUDensityMatrixPolicyBase::ExpectDiffYY(qs_data_p_t qs, qs_data_p_t ham_matrix, const qbits_t& objs,
                                              const qbits_t& ctrls, index_t dim) -> qs_data_t {
    DoubleQubitGateMask mask(objs, ctrls);
    qs_data_t res = 0;
    if (!mask.ctrl_mask) {
        THRESHOLD_OMP(
            MQ_DO_PRAGMA(omp parallel for reduction(+:res) schedule(static)), dim, DimTh, for (omp::idx_t l = 0; l < (dim / 4); l++) {
                    index_t r0;
                    SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask, mask.obj_high_mask, mask.obj_rev_high_mask,
                                  l, r0);
                    auto r3 = r0 + mask.obj_mask;
                    auto r1 = r0 + mask.obj_min_mask;
                    auto r2 = r0 + mask.obj_max_mask;
                    for (index_t col = 0; col < dim; col++) {
                        res += IMAGE_I
                               * (GetValue(qs, r3, col) * GetValue(ham_matrix, col, r0)
                                  - GetValue(qs, r2, col) * GetValue(ham_matrix, col, r1)
                                  - GetValue(qs, r1, col) * GetValue(ham_matrix, col, r2)
                                  + GetValue(qs, r0, col) * GetValue(ham_matrix, col, r3));
                    }
                })
    } else {
        THRESHOLD_OMP(
            MQ_DO_PRAGMA(omp parallel for reduction(+:res) schedule(static)), dim, DimTh, for (omp::idx_t l = 0; l < (dim / 4); l++) {
                    index_t r0;
                    SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask, mask.obj_high_mask, mask.obj_rev_high_mask,
                                  l, r0);
                    if ((r0 & mask.ctrl_mask) == mask.ctrl_mask) {
                        auto r3 = r0 + mask.obj_mask;
                        auto r1 = r0 + mask.obj_min_mask;
                        auto r2 = r0 + mask.obj_max_mask;
                        for (index_t col = 0; col < dim; col++) {
                            res += IMAGE_I
                                   * (GetValue(qs, r3, col) * GetValue(ham_matrix, col, r0)
                                      - GetValue(qs, r2, col) * GetValue(ham_matrix, col, r1)
                                      - GetValue(qs, r1, col) * GetValue(ham_matrix, col, r2)
                                      + GetValue(qs, r0, col) * GetValue(ham_matrix, col, r3));
                        }
                    }
                })
    }
    return res;
};

auto CPUDensityMatrixPolicyBase::ExpectDiffZZ(qs_data_p_t qs, qs_data_p_t ham_matrix, const qbits_t& objs,
                                              const qbits_t& ctrls, index_t dim) -> qs_data_t {
    DoubleQubitGateMask mask(objs, ctrls);
    qs_data_t res = 0;
    if (!mask.ctrl_mask) {
        THRESHOLD_OMP(
            MQ_DO_PRAGMA(omp parallel for reduction(+:res) schedule(static)), dim, DimTh, for (omp::idx_t l = 0; l < (dim / 4); l++) {
                    index_t r0;
                    SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask, mask.obj_high_mask, mask.obj_rev_high_mask,
                                  l, r0);
                    auto r3 = r0 + mask.obj_mask;
                    auto r1 = r0 + mask.obj_min_mask;
                    auto r2 = r0 + mask.obj_max_mask;
                    for (index_t col = 0; col < dim; col++) {
                        res += IMAGE_MI
                               * (GetValue(qs, r0, col) * GetValue(ham_matrix, col, r0)
                                  - GetValue(qs, r1, col) * GetValue(ham_matrix, col, r1)
                                  - GetValue(qs, r2, col) * GetValue(ham_matrix, col, r2)
                                  + GetValue(qs, r3, col) * GetValue(ham_matrix, col, r3));
                    }
                })
    } else {
        THRESHOLD_OMP(
            MQ_DO_PRAGMA(omp parallel for reduction(+:res) schedule(static)), dim, DimTh, for (omp::idx_t l = 0; l < (dim / 4); l++) {
                    index_t r0;
                    SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask, mask.obj_high_mask, mask.obj_rev_high_mask,
                                  l, r0);
                    if ((r0 & mask.ctrl_mask) == mask.ctrl_mask) {
                        auto r3 = r0 + mask.obj_mask;
                        auto r1 = r0 + mask.obj_min_mask;
                        auto r2 = r0 + mask.obj_max_mask;
                        for (index_t col = 0; col < dim; col++) {
                            res += IMAGE_MI
                                   * (GetValue(qs, r0, col) * GetValue(ham_matrix, col, r0)
                                      - GetValue(qs, r1, col) * GetValue(ham_matrix, col, r1)
                                      - GetValue(qs, r2, col) * GetValue(ham_matrix, col, r2)
                                      + GetValue(qs, r3, col) * GetValue(ham_matrix, col, r3));
                        }
                    }
                })
    }
    return res;
};

}  // namespace mindquantum::sim::densitymatrix::detail
