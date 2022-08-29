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

#ifndef DETAILS_PARAMETERRESOLVER_COEFF_POLICY_HPP
#define DETAILS_PARAMETERRESOLVER_COEFF_POLICY_HPP

#include <optional>

#include <boost/range/iterator_range.hpp>

#include "core/config.hpp"

#include "core/format/parameter_resolver.hpp"
#include "core/type.h"
#include "ops/gates/traits.hpp"
#include "pr/parameter_resolver.h"

// =============================================================================

namespace mindquantum::traits {
template <typename float_t>
inline constexpr auto is_termsop_number<ParameterResolver<float_t>> = true;
}  // namespace mindquantum::traits

// -----------------------------------------------------------------------------

namespace mindquantum::ops::details {
template <typename float_t_>
struct ParameterResolverCoeffPolicyBase {
    using float_t = float_t_;
    using coeff_t = ParameterResolver<float_t>;

    static constexpr auto EQ_TOLERANCE = PRECISION;

    static constexpr auto is_complex_valued = traits::is_complex_v<float_t>;

    // Comparisons
    static auto equal(const coeff_t& lhs, const coeff_t& rhs) {
        return lhs == rhs;
    }

    // Unary operators
    static auto uminus(const coeff_t& lhs) {
        return -lhs;
    }

    // Binary operators
    template <typename rhs_t>
    static auto iadd(coeff_t& lhs, const rhs_t& rhs) {
        return lhs += rhs;
    }
    template <typename rhs_t>
    static auto add(const coeff_t& lhs, const rhs_t& rhs) {
        return lhs + rhs;
    }
    template <typename rhs_t>
    static auto isub(coeff_t& lhs, const rhs_t& rhs) {
        return lhs -= rhs;
    }
    template <typename rhs_t>
    static auto sub(const coeff_t& lhs, const rhs_t& rhs) {
        return lhs - rhs;
    }
    template <typename rhs_t>
    static auto imul(coeff_t& lhs, const rhs_t& rhs) {
        return lhs *= rhs;
    }
    template <typename rhs_t>
    static auto mul(const coeff_t& lhs, const rhs_t& rhs) {
        return lhs * rhs;
    }
    template <typename rhs_t>
    static auto idiv(coeff_t& lhs, const rhs_t& rhs) {
        return lhs /= rhs;
    }
    template <typename rhs_t>
    static auto div(const coeff_t& lhs, const rhs_t& rhs) {
        return lhs / rhs;
    }

    // Misc. math functions
    static auto is_zero(const coeff_t& coeff, double abs_tol = EQ_TOLERANCE) {
        if (coeff.IsConst()) {
            return std::abs(coeff.const_value) <= abs_tol;
        }
        return false;
    }
    static auto cast_real(coeff_t& coeff) {
        if constexpr (is_complex_valued) {
            if (coeff.IsConst()) {
                coeff.const_value.imag(0.);
            }
        }
    }
    static auto cast_imag(coeff_t& coeff) {
        if constexpr (is_complex_valued) {
            if (coeff.IsConst()) {
                coeff.const_value = coeff.const_value.real();
            }
        } else {
            coeff *= 0;
        }
    }
    static auto compress(coeff_t& coeff, double abs_tol = EQ_TOLERANCE) {
        // NB: This assumes a complex API similar to std::complex
        if constexpr (is_complex_valued) {
            if (coeff.IsConst()) {
                if (std::abs(coeff.const_value) <= abs_tol) {
                    coeff.const_value = float_t{0.};
                    // TODO(dnguyen): Should we clear the PR in this case?
                    // coeff.data_.clear();
                } else if (std::abs(coeff.const_value.imag()) <= abs_tol) {
                    cast_real(coeff);
                } else if (std::abs(coeff.const_value.real()) <= abs_tol) {
                    cast_imag(coeff);
                }
            }
        }
    }
};

struct DoublePRCoeffPolicy : public ParameterResolverCoeffPolicyBase<double> {
    static const coeff_t one;
    // Conversion
    static std::optional<coeff_t> coeff_from_string(
        const boost::iterator_range<std::string_view::const_iterator>& range);
};
inline const DoublePRCoeffPolicy::coeff_t DoublePRCoeffPolicy::one(DoublePRCoeffPolicy::float_t{1.0});
struct CmplxDoublePRCoeffPolicy : public ParameterResolverCoeffPolicyBase<std::complex<double>> {
    static const coeff_t one;
    // Conversion
    static std::optional<coeff_t> coeff_from_string(
        const boost::iterator_range<std::string_view::const_iterator>& range);
};
inline const CmplxDoublePRCoeffPolicy::coeff_t CmplxDoublePRCoeffPolicy::one(CmplxDoublePRCoeffPolicy::float_t{1.0, 0});
}  // namespace mindquantum::ops::details

#endif /* DETAILS_PARAMETERRESOLVER_COEFF_POLICY_HPP */
