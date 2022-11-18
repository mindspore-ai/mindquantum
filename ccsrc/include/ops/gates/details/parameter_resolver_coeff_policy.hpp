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
#include <utility>

#include <boost/range/iterator_range.hpp>

#include "config/config.hpp"
#include "config/format/parameter_resolver.hpp"
#include "config/type_traits.hpp"

#include "core/parameter_resolver.hpp"
#include "ops/gates/details/coeff_policy.hpp"
#include "ops/gates/details/floating_point_coeff_policy.hpp"
#include "ops/gates/details/std_complex_coeff_policy.hpp"
#include "ops/gates/traits.hpp"
#include "ops/gates/types.hpp"

// =============================================================================

namespace mindquantum::ops::details {
template <typename float_t>
struct CoeffSubsProxy<ParameterResolver<float_t>> {
    using coeff_t = ParameterResolver<float_t>;
    using subs_t = coeff_t;
    explicit CoeffSubsProxy(subs_t params_a) : params(std::move(params_a)) {
    }

    void apply(coeff_t& coeff) const {  // NOLINT(runtime/references)
        coeff = coeff.Combination(params);
    }

    subs_t params;
};

// -----------------------------------------------------------------------------

template <typename float_t>
struct CoeffPolicy<ParameterResolver<float_t>> : CoeffPolicyBase<ParameterResolver<float_t>> {
    using coeff_t = ParameterResolver<float_t>;
    using base_t = CoeffPolicyBase<coeff_t>;
    using coeff_policy_real_t = typename base_t::coeff_policy_real_t;
    using matrix_coeff_t = float_t;

    static const coeff_t zero;
    static const coeff_t one;
    static constexpr auto is_complex_valued = traits::is_std_complex_v<float_t>;

    // Getter
    static constexpr auto get_num(const coeff_t& coeff) {
        assert(coeff.IsConst());
        return coeff.const_value;
    }

    // Comparisons/Checks
    static auto is_const(const coeff_t& coeff) {
        return coeff.IsConst();
    }
    static auto equal(const coeff_t& lhs, const coeff_t& rhs) {
        return lhs == rhs;
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

    // Misc. math functions
    static auto conjugate(const coeff_t& coeff) {
        return coeff.Conjugate();
    }
    static auto is_zero(const coeff_t& coeff, double abs_tol = EQ_TOLERANCE) {
        if (coeff.IsConst()) {
            return std::abs(coeff.const_value) <= abs_tol;
        }
        return false;
    }
    static auto discard_imag(coeff_t& coeff) {  // NOLINT(runtime/references)
        if constexpr (is_complex_valued) {
            coeff.const_value.imag(0.);
            std::for_each(begin(coeff.data_), end(coeff.data_), [](auto& param) { std::get<1>(param).imag(0.); });
        }
    }
    static auto discard_real(coeff_t& coeff) {  // NOLINT(runtime/references)
        coeff.const_value.real(0.);
        std::for_each(begin(coeff.data_), end(coeff.data_), [](auto& param) { std::get<1>(param).real(0.); });
    }
    static auto compress(coeff_t& coeff, double abs_tol = EQ_TOLERANCE) {  // NOLINT(runtime/references)
        // NB: This assumes a complex API similar to std::complex
        if constexpr (is_complex_valued) {
            if (coeff.IsConst()) {
                if (std::abs(coeff.const_value) <= abs_tol) {
                    coeff.const_value = float_t{0.};
                    // TODO(dnguyen): Should we clear the PR in this case?
                    // coeff.data_.clear();
                } else if (std::abs(coeff.const_value.imag()) <= abs_tol) {
                    discard_imag(coeff);
                } else if (std::abs(coeff.const_value.real()) <= abs_tol) {
                    discard_real(coeff);
                }
            }
        }
    }
};

// =============================================================================

using FloatPRCoeffPolicy = CoeffPolicy<ParameterResolver<float>>;
using DoublePRCoeffPolicy = CoeffPolicy<ParameterResolver<double>>;
using CmplxFloatPRCoeffPolicy = CoeffPolicy<ParameterResolver<std::complex<float>>>;
using CmplxDoublePRCoeffPolicy = CoeffPolicy<ParameterResolver<std::complex<double>>>;

template <typename float_t>
inline const typename CoeffPolicy<ParameterResolver<float_t>>::coeff_t CoeffPolicy<ParameterResolver<float_t>>::zero{
    typename CoeffPolicy<ParameterResolver<float_t>>::coeff_t{0.0}};

template <typename float_t>
inline const typename CoeffPolicy<ParameterResolver<float_t>>::coeff_t CoeffPolicy<ParameterResolver<float_t>>::one{
    typename CoeffPolicy<ParameterResolver<float_t>>::coeff_t{1.0}};

// =============================================================================
}  // namespace mindquantum::ops::details

#endif /* DETAILS_PARAMETERRESOLVER_COEFF_POLICY_HPP */
