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
// X like operator
// ========================================================================================================

void CPUDensityMatrixPolicyBase::ApplyXLike(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls, qs_data_t v1,
                                            qs_data_t v2, index_t dim) {
    SingleQubitGateMask mask(objs, ctrls);
    if (!mask.ctrl_mask) {
        THRESHOLD_OMP_FOR(
            dim, DimTh, for (index_t k = 0; k < (dim / 2); k++) {
                auto i = ((k & mask.obj_high_mask) << 1) + (k & mask.obj_low_mask);
                auto j = i | mask.obj_mask;
                for (index_t l = 0; l <= k; l++) {
                    auto m = ((l & mask.obj_high_mask) << 1) + (l & mask.obj_low_mask);
                    auto n = m | mask.obj_mask;
                    auto tmp = qs[IdxMap(i, m)];
                    qs[IdxMap(i, m)] = qs[IdxMap(j, n)] * v1 * std::conj(v1);
                    qs[IdxMap(j, n)] = tmp * v2 * std::conj(v2);
                    tmp = qs[IdxMap(j, m)];
                    qs[IdxMap(j, m)] = tmp * v2 * std::conj(v1);
                    if (i >= n) {
                        qs[IdxMap(i, n)] = qs[IdxMap(j, m)] * v1 * std::conj(v2);
                    } else {
                        qs[IdxMap(n, i)] = std::conj(qs[IdxMap(j, m)] * v1) * v2;
                    }
                }
            })

    } else {
        THRESHOLD_OMP_FOR(
            dim, DimTh, for (index_t k = 0; k < (dim / 2); k++) {
                auto i = ((k & mask.obj_high_mask) << 1) + (k & mask.obj_low_mask);
                auto j = i | mask.obj_mask;
                for (index_t l = 0; l <= k; l++) {
                    auto m = ((l & mask.obj_high_mask) << 1) + (l & mask.obj_low_mask);
                    auto n = m | mask.obj_mask;
                    if (((i & mask.ctrl_mask) != mask.ctrl_mask)
                        && ((m & mask.ctrl_mask) != mask.ctrl_mask)) {  // both not in control
                        continue;
                    }
                    if ((i & mask.ctrl_mask) == mask.ctrl_mask) {
                        if ((m & mask.ctrl_mask) == mask.ctrl_mask) {  // both in control
                            auto tmp = qs[IdxMap(i, m)];
                            qs[IdxMap(i, m)] = qs[IdxMap(j, n)] * v1 * std::conj(v1);
                            qs[IdxMap(j, n)] = tmp * v2 * std::conj(v2);
                            tmp = qs[IdxMap(j, m)];
                            qs[IdxMap(j, m)] = tmp * v2 * std::conj(v1);
                            if (i >= n) {  // for matrix[row, col], only in this case that (row < col) is possible
                                qs[IdxMap(i, n)] = qs[IdxMap(j, m)] * v1 * std::conj(v2);
                            } else {
                                qs[IdxMap(n, i)] = std::conj(qs[IdxMap(j, m)] * v1) * v2;
                            }
                        } else {  // row in control but not column
                            auto tmp = qs[IdxMap(i, m)];
                            qs[IdxMap(i, m)] = qs[IdxMap(j, m)] * v1;
                            qs[IdxMap(j, m)] = tmp * v2;
                            if (i >= n) {
                                tmp = qs[IdxMap(i, n)];
                                qs[IdxMap(i, n)] = qs[IdxMap(j, n)] * v1;
                                qs[IdxMap(j, n)] = tmp * v2;
                            } else {
                                tmp = std::conj(qs[IdxMap(n, i)]);
                                qs[IdxMap(n, i)] = std::conj(qs[IdxMap(j, n)] * v1);
                                qs[IdxMap(j, n)] = tmp * v2;
                            }
                        }
                    } else {  // column in control but not row
                        auto tmp = qs[IdxMap(j, m)];
                        qs[IdxMap(j, m)] = qs[IdxMap(j, n)] * std::conj(v1);
                        qs[IdxMap(j, n)] = tmp * std::conj(v2);
                        tmp = qs[IdxMap(i, m)];
                        if (i >= n) {
                            qs[IdxMap(i, m)] = qs[IdxMap(i, n)] * std::conj(v1);
                            qs[IdxMap(i, n)] = tmp * std::conj(v2);
                        } else {
                            qs[IdxMap(i, m)] = std::conj(qs[IdxMap(n, i)]) * v1;
                            qs[IdxMap(n, i)] = std::conj(tmp) * v2;
                        }
                    }
                }
            })
    }
}

void CPUDensityMatrixPolicyBase::ApplyX(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls, index_t dim) {
    ApplyXLike(qs, objs, ctrls, 1, 1, dim);
}

void CPUDensityMatrixPolicyBase::ApplyY(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls, index_t dim) {
    ApplyXLike(qs, objs, ctrls, IMAGE_MI, IMAGE_I, dim);
}
}  // namespace mindquantum::sim::densitymatrix::detail
