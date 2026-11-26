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

#include "simulator/test_circuit_tiling_sve.h"

#include <algorithm>
#include <array>
#include <complex>
#include <cstddef>
#include <vector>

#include "simulator/circuit_tiling/detail/kernels.h"
#include "simulator/circuit_tiling/detail/sve_runtime.h"
#include "simulator/state_index.h"

namespace mindquantum::sim::circuit_tiling::test {
namespace {
template <typename calc_type>
auto VerifyPrecisionWindowControls() -> SveWindowControlReport {
    using value_t = std::complex<calc_type>;
    using algebra_t = detail::SveTileAlgebra<calc_type>;
    using kernels_t = detail::Kernels<algebra_t>;

    // Use a narrow local window so window-wide controls above bit 31 can be
    // tested without allocating a multi-gigabyte state vector.
    const state_index::StateIndexWindow window(6, {0});
    constexpr index_t kHighFirst = index_t{1} << 33;
    constexpr index_t kHighSecond = index_t{1} << 45;
    constexpr index_t kLocalControl = index_t{1} << 1;
    constexpr index_t kControlMask = kHighFirst | kHighSecond | kLocalControl;
    constexpr index_t kFixedMask = kHighFirst | kHighSecond;

    const Operation<value_t> operation{
        kControlMask,
        Kernel<value_t>{std::in_place_type<DiagonalKernel<value_t>>,
                        DiagonalKernel<value_t>{{Scale<value_t>{1, value_t{0, 1}}}}},
    };
    const Derivative<value_t> derivative{
        kControlMask,
        DerivativeKernel<value_t>{std::in_place_type<DiagonalKernel<value_t>>,
                                  DiagonalKernel<value_t>{{Scale<value_t>{1, value_t{0, 1}}}}},
        {},
    };

    const auto lanes = sve::Traits<calc_type>::ScalarLanes();
    const auto step = static_cast<size_t>(sve::ComplexStep<calc_type>());
    const std::array<index_t, 3> origins{0, kHighFirst, kHighFirst | kHighSecond | (index_t{1} << 36)};
    const std::array<size_t, 2> counts{step > 1 ? step - 1 : 1, step};

    for (const auto origin : origins) {
        const bool fixed_satisfied = (origin & kFixedMask) == kFixedMask;
        const detail::SveIndexContext context{origin, window.LocalStateMask()};
        for (const index_t start : {index_t{0}, index_t{1}}) {
            for (const size_t active_lanes : counts) {
                std::vector<calc_type> tile(2 * lanes), expected_tile(2 * lanes);
                std::vector<calc_type> bra(2 * lanes), ket(2 * lanes);
                value_t expected_derivative{0, 0};

                for (size_t basis = 0; basis < 2; ++basis) {
                    for (size_t lane = 0; lane < step; ++lane) {
                        const auto real = static_cast<calc_type>(2 + (lane + 3 * basis) % 7);
                        const auto imag = static_cast<calc_type>(1 + (2 * lane + basis) % 5) - calc_type{3};
                        const auto bra_real = static_cast<calc_type>(1 + (lane + basis) % 4);
                        const auto bra_imag = static_cast<calc_type>((lane + 2 * basis) % 3) - calc_type{1};
                        const size_t offset = basis * lanes + 2 * lane;
                        tile[offset] = real;
                        tile[offset + 1] = imag;
                        expected_tile[offset] = real;
                        expected_tile[offset + 1] = imag;
                        bra[offset] = bra_real;
                        bra[offset + 1] = bra_imag;
                        ket[offset] = real;
                        ket[offset + 1] = imag;

                        const auto state_base = window.ExpandLocal(start + static_cast<index_t>(lane));
                        const bool controlled = fixed_satisfied && lane < active_lanes
                                                && (state_base & kLocalControl) == kLocalControl;
                        if (basis == 1 && controlled) {
                            expected_tile[offset] = -imag;
                            expected_tile[offset + 1] = real;
                            expected_derivative += std::conj(value_t{bra_real, bra_imag}) * value_t{-imag, real};
                        }
                    }
                }

                const auto loop_index = sve::LoopIndex<calc_type>(start);
                const auto active = sve::ActiveComplexPredicate<calc_type>(loop_index,
                                                                           start + static_cast<index_t>(active_lanes));
                const auto state_base = sve::ExpandBaseIndex<calc_type>(active, loop_index,
                                                                        window.LocalExcludedQubits());
                kernels_t::Apply(operation, active, state_base, context, tile.data(), nullptr);
                if (!std::equal(tile.begin(), tile.end(), expected_tile.begin())) {
                    return {false, "SVE Apply mismatched window/lane external control semantics."};
                }
                const auto actual_derivative = kernels_t::ExpectDerivative(derivative, active, state_base, context,
                                                                           bra.data(), ket.data());
                if (actual_derivative != expected_derivative) {
                    return {false, "SVE ExpectDerivative mismatched window/lane external control semantics."};
                }
            }
        }
    }
    return {};
}
}  // namespace

auto VerifySveExternalControlWindows() -> SveWindowControlReport {
    if (const auto float_report = VerifyPrecisionWindowControls<float>(); !float_report.ok) {
        return float_report;
    }
    return VerifyPrecisionWindowControls<double>();
}
}  // namespace mindquantum::sim::circuit_tiling::test
