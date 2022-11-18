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

#ifndef DETAILS_STD_COMPLEX_DOUBLE_COEFF_POLICY_HPP
#define DETAILS_STD_COMPLEX_DOUBLE_COEFF_POLICY_HPP

#include <algorithm>
#include <complex>

#include "config/format/std_complex.hpp"

#include "ops/gates/details/coeff_policy.hpp"
#include "ops/gates/details/floating_point_coeff_policy.hpp"

namespace mindquantum::ops::details {
template <typename float_t>
struct CoeffSubsProxy<std::complex<float_t>> {
    using coeff_t = std::complex<float_t>;
    using subs_t = coeff_t;

    explicit CoeffSubsProxy(subs_t /* params */) {
    }

    void apply(std::complex<float_t>& /* coeff */) const {
    }
};

template <typename float_t>
struct CoeffPolicy<std::complex<float_t>> : CoeffPolicyBase<std::complex<float_t>> {
    using coeff_t = std::complex<float_t>;
    using base_t = CoeffPolicyBase<coeff_t>;
    using coeff_policy_real_t = typename base_t::coeff_policy_real_t;
    using matrix_coeff_t = coeff_t;

    static constexpr auto zero = coeff_t{0, 0};
    static constexpr auto one = coeff_t{1, 0};

    // Getter
    static constexpr auto get_num(const coeff_t& coeff) {
        return coeff;
    }

    // Comparisons/Checks
    static constexpr auto is_const(const coeff_t& /* coeff */) {
        return true;
    }
    static auto equal(const coeff_t& lhs, const coeff_t& rhs) {
        return std::abs(lhs - rhs) <= std::max(EQ_TOLERANCE, EQ_TOLERANCE * std::max(std::abs(lhs), std::abs(rhs)));
    }

    // Unary operators
    static auto uminus(const coeff_t& lhs) {
        return -lhs;
    }

    // Binary operators
    static auto iadd(coeff_t& lhs, const coeff_t& rhs) {  // NOLINT(runtime/references)
        return lhs += rhs;
    }
    static auto add(const coeff_t& lhs, const coeff_t& rhs) {
        return lhs + rhs;
    }
    static auto isub(coeff_t& lhs, const coeff_t& rhs) {  // NOLINT(runtime/references)
        return lhs -= rhs;
    }
    static auto sub(const coeff_t& lhs, const coeff_t& rhs) {
        return lhs - rhs;
    }
    static auto imul(coeff_t& lhs, const coeff_t& rhs) {  // NOLINT(runtime/references)
        return lhs *= rhs;
    }
    static auto mul(const coeff_t& lhs, const coeff_t& rhs) {
        return lhs * rhs;
    }
    static auto idiv(coeff_t& lhs, const coeff_t& rhs) {  // NOLINT(runtime/references)
        return lhs /= rhs;
    }
    static auto div(const coeff_t& lhs, const coeff_t& rhs) {
        return lhs / rhs;
    }

    // Misc. functions
    static auto conjugate(const coeff_t& coeff) {
        return std::conj(coeff);
    }
    static auto is_zero(const coeff_t& coeff, double abs_tol = EQ_TOLERANCE) {
        return std::abs(coeff) <= abs_tol;
    }
    static auto discard_imag(coeff_t& coeff) {  // NOLINT(runtime/references)
        coeff.imag(0.);
    }
    static auto discard_real(coeff_t& coeff) {  // NOLINT(runtime/references)
        coeff.real(0.);
    }
    static auto compress(coeff_t& coeff, double abs_tol = EQ_TOLERANCE) {  // NOLINT(runtime/references)
        if (std::abs(coeff) <= abs_tol) {
            coeff = coeff_t{0.};
        } else if (std::abs(coeff.imag()) <= abs_tol) {
            discard_imag(coeff);
        } else if (std::abs(coeff.real()) <= abs_tol) {
            discard_real(coeff);
        }
    }
};

using CmplxFloatCoeffPolicy = CoeffPolicy<std::complex<float>>;
using CmplxDoubleCoeffPolicy = CoeffPolicy<std::complex<double>>;
}  // namespace mindquantum::ops::details

#endif /* DETAILS_STD_COMPLEX_DOUBLE_COEFF_POLICY_HPP */
