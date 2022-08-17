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

#ifndef DETAILS_COMPLEX_DOUBLE_COEFF_POLICY_HPP
#define DETAILS_COMPLEX_DOUBLE_COEFF_POLICY_HPP

#include <cmath>

#include <algorithm>
#include <complex>
#include <optional>

#include <boost/range/iterator_range.hpp>

#include <fmt/format.h>

#include "core/config.hpp"

#include "core/format/format_complex.hpp"

namespace mindquantum::ops::details {
struct CmplxDoubleCoeffPolicy {
    using coeff_t = std::complex<double>;

    static constexpr auto one = coeff_t{1, 0};
    static constexpr auto EQ_TOLERANCE = 1.e-8;

    // Comparisons
    static auto equal(const coeff_t& lhs, const coeff_t& rhs) {
        return std::abs(lhs - rhs) <= std::max(EQ_TOLERANCE, EQ_TOLERANCE * std::max(std::abs(lhs), std::abs(rhs)));
    }

    // Conversion
    static std::optional<coeff_t> coeff_from_string(
        const boost::iterator_range<std::string_view::const_iterator>& range);

    // Unary operators
    static auto uminus(const coeff_t& lhs) {
        return -lhs;
    }

    // Binary operators
    static auto iadd(coeff_t& lhs, const coeff_t& rhs) {
        return lhs += rhs;
    }
    static auto add(const coeff_t& lhs, const coeff_t& rhs) {
        return lhs + rhs;
    }
    static auto isub(coeff_t& lhs, const coeff_t& rhs) {
        return lhs -= rhs;
    }
    static auto sub(const coeff_t& lhs, const coeff_t& rhs) {
        return lhs - rhs;
    }
    static auto imul(coeff_t& lhs, const coeff_t& rhs) {
        return lhs *= rhs;
    }
    static auto mul(const coeff_t& lhs, const coeff_t& rhs) {
        return lhs * rhs;
    }
    static auto idiv(coeff_t& lhs, const coeff_t& rhs) {
        return lhs /= rhs;
    }
    static auto div(const coeff_t& lhs, const coeff_t& rhs) {
        return lhs / rhs;
    }

    // Misc. functions
    static auto is_zero(const coeff_t& coeff, double abs_tol = EQ_TOLERANCE) {
        return std::abs(coeff) <= abs_tol;
    }
    static auto cast_real(coeff_t& coeff) {
        coeff.imag(0.);
    }
    static auto cast_imag(coeff_t& coeff) {
        coeff = coeff.imag();
    }
    static auto compress(coeff_t& coeff, double abs_tol = EQ_TOLERANCE) {
        if (coeff.imag() <= abs_tol) {
            cast_real(coeff);
        } else if (coeff.real() <= abs_tol) {
            cast_imag(coeff);
        }
    }
};
}  // namespace mindquantum::ops::details

#endif /* DETAILS_COMPLEX_DOUBLE_COEFF_POLICY_HPP */
