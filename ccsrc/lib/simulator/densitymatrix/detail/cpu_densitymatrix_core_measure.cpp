/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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

#include <array>
#include <vector>

#include "config/openmp.h"
#include "core/utils.h"
#include "simulator/densitymatrix/detail/cpu_densitymatrix_policy.h"

namespace mindquantum::sim::densitymatrix::detail {
template <typename calc_type_>
auto CPUDensityMatrixPolicy<calc_type_>::OutcomeWeights(const qs_data_p_t& qs, qbit_t qubit, index_t dim)
    -> std::array<calc_type, 2> {
    const auto mask = index_t{1} << qubit;
    calc_type zero = 0;
    calc_type one = 0;
    THRESHOLD_OMP(
        MQ_DO_PRAGMA(omp parallel for schedule(static) reduction(+ : zero, one)), Size(dim), DimTh,
                     for (omp::idx_t i = 0; i < static_cast<omp::idx_t>(dim); i++) {
                         const auto population = qs[static_cast<index_t>(i) * dim + static_cast<index_t>(i)].real();
                         if ((static_cast<index_t>(i) & mask) != 0) {
                             one += population;
                         } else {
                             zero += population;
                         }
                     })
    return {zero, one};
}

template <typename calc_type_>
void CPUDensityMatrixPolicy<calc_type_>::Collapse(qs_data_p_t qs, qbit_t qubit, bool outcome, calc_type weight,
                                                  index_t dim) {
    const auto mask = index_t{1} << qubit;
    const index_t expected = outcome ? mask : 0;
    const auto scale = 1 / weight;
    THRESHOLD_OMP_FOR(
        Size(dim), DimTh, for (omp::idx_t i = 0; i < static_cast<omp::idx_t>(dim); i++) {
            auto* row = qs + static_cast<index_t>(i) * dim;
            const bool row_matches = (static_cast<index_t>(i) & mask) == expected;
            for (index_t j = 0; j < dim; j++) {
                if (row_matches && (j & mask) == expected) {
                    row[j] *= scale;
                } else {
                    row[j] = 0;
                }
            }
        })
}

template <typename calc_type_>
auto CPUDensityMatrixPolicy<calc_type_>::DiagonalCumulativeProbabilities(const qs_data_p_t& qs, index_t dim)
    -> VT<calc_type> {
    VT<calc_type> cumulative(dim);
    calc_type total = 0;
    for (index_t i = 0; i < dim; ++i) {
        total += qs[i * dim + i].real();
        cumulative[i] = total;
    }
    // The last bucket absorbs the rounding drift of the trace.
    cumulative[dim - 1] = 1;
    return cumulative;
}

template struct CPUDensityMatrixPolicy<float>;
template struct CPUDensityMatrixPolicy<double>;
}  // namespace mindquantum::sim::densitymatrix::detail
