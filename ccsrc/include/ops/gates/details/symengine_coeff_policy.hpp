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

#ifndef DETAILS_SYMENGINE_COEFF_POLICY_HPP
#define DETAILS_SYMENGINE_COEFF_POLICY_HPP

#include <cmath>

#include <complex>
#include <optional>
#include <utility>

#include <boost/range/iterator_range.hpp>

#include <symengine/add.h>
#include <symengine/basic.h>
#include <symengine/complex_double.h>
#include <symengine/constants.h>
#include <symengine/dict.h>
#include <symengine/eval_double.h>
#include <symengine/functions.h>
#include <symengine/mul.h>
#include <symengine/number.h>
#include <symengine/sets.h>
#include <symengine/symengine_casts.h>

#include "config/config.hpp"
#include "config/format/symengine.hpp"
#include "config/real_cast.hpp"

#include "ops/gates/details/std_complex_coeff_policy.hpp"
#include "ops/gates/traits.hpp"

// =============================================================================

namespace mindquantum {
namespace traits {
template <>
struct is_scalar<SymEngine::RCP<const SymEngine::Basic>> : std::true_type {};
}  // namespace traits

// -----------------------------------------------------------------------------

namespace details {
template <RealCastType cast_type>
struct real_cast_impl<cast_type, SymEngine::RCP<const SymEngine::Basic>> {
    using type = SymEngine::RCP<const SymEngine::Basic>;
    static auto apply(const type& coeff) {
        if (SymEngine::is_a_Complex(*coeff)) {
            return real_cast<cast_type>(SymEngine::complex_double(SymEngine::eval_complex_double(*coeff)));
        }
        return coeff;
    }
};
}  // namespace details
}  // namespace mindquantum

// -----------------------------------------------------------------------------

namespace mindquantum::ops::details {
template <>
struct CoeffSubsProxy<SymEngine::RCP<const SymEngine::Basic>> {
    using coeff_t = SymEngine::RCP<const SymEngine::Basic>;
    using subs_t = SymEngine::map_basic_basic;

    explicit CoeffSubsProxy(subs_t params_a) : params(std::move(params_a)) {
    }

    void apply(coeff_t& coeff) const {  // NOLINT(runtime/references)
        coeff = SymEngine::expand(coeff->subs(params));
    }

    subs_t params;
};

template <>
struct CoeffPolicy<SymEngine::RCP<const SymEngine::Basic>> {
    using coeff_t = SymEngine::RCP<const SymEngine::Basic>;
    using self_t = CoeffPolicy<SymEngine::RCP<const SymEngine::Basic>>;
    using base_t = CoeffPolicyBase<coeff_t>;
    using coeff_policy_real_t = self_t;
    using matrix_coeff_t = std::complex<double>;

    static const coeff_t zero;
    static const coeff_t one;

    // Getter
    static auto get_num(const coeff_t& coeff) {
        assert(SymEngine::is_a_Number(*coeff));
        if (SymEngine::is_a_Complex(*coeff)) {
            return SymEngine::eval_complex_double(*coeff);
        }
        return static_cast<std::complex<double>>(SymEngine::eval_double(*coeff));
    }

    // Comparisons/Checks
    static auto is_const(const coeff_t& coeff) {
        return SymEngine::is_a_Number(*coeff);
    }
    static auto equal(const coeff_t& lhs, const coeff_t& rhs) {
        return SymEngine::eq(*lhs, *rhs);
    }

    // Conversion
    static std::optional<coeff_t> coeff_from_string(
        const boost::iterator_range<std::string_view::const_iterator>& range);

    // Unary operators
    static auto uminus(const coeff_t& lhs) {
        return SymEngine::mul(self_t::one, lhs);
    }

    // Binary operators
    static auto add(const coeff_t& lhs, const coeff_t& rhs) {
        return SymEngine::add(lhs, rhs);
        // TODO(dnguyen): This might be more efficient?
        // return SymEngine::Add{SymEngine::zero,
        //                       SymEngine::umap_basic_num{{lhs, SymEngine::one}, {rhs, SymEngine::one}}};
    }
    static auto iadd(coeff_t& lhs, const coeff_t& rhs) {  // NOLINT(runtime/references)
        return lhs = self_t::add(lhs, rhs);
        // return lhs = self_t::add(lhs, rhs).rcp_from_this();
    }
    static auto sub(const coeff_t& lhs, const coeff_t& rhs) {
        return SymEngine::sub(lhs, rhs);
    }
    static auto isub(coeff_t& lhs, const coeff_t& rhs) {  // NOLINT(runtime/references)
        return lhs = self_t::sub(lhs, rhs);
    }
    static auto mul(const coeff_t& lhs, const coeff_t& rhs) {
        return SymEngine::mul(lhs, rhs);
    }
    static auto imul(coeff_t& lhs, const coeff_t& rhs) {  // NOLINT(runtime/references)
        return lhs = self_t::mul(lhs, rhs);
    }
    static auto div(const coeff_t& lhs, const coeff_t& rhs) {
        return SymEngine::div(lhs, rhs);
    }
    static auto idiv(coeff_t& lhs, const coeff_t& rhs) {  // NOLINT(runtime/references)
        return lhs = self_t::div(lhs, rhs);
    }

    // Misc. math functions
    static auto conjugate(const coeff_t& coeff) {
        // TODO(xusheng): implement conjugate for symengine coefficient.
        throw std::runtime_error("not implement yet.");
    }
    static auto is_zero(const coeff_t& coeff, double abs_tol = EQ_TOLERANCE) {
        if (SymEngine::is_a_Number(*coeff)) {
            if (SymEngine::is_a_Complex(*coeff)) {
                return std::abs(SymEngine::eval_complex_double(*coeff)) < abs_tol;
            }
            return SymEngine::eval_double(*coeff) < abs_tol;
        }
        return false;
    }
    static auto discard_imag(coeff_t& coeff) {  // NOLINT(runtime/references)
        if (SymEngine::is_a_Complex(*coeff)) {
            coeff = SymEngine::complex_double(SymEngine::eval_complex_double(*coeff).real());
        }
    }
    static auto discard_real(coeff_t& coeff) {  // NOLINT(runtime/references)
        if (SymEngine::is_a_Complex(*coeff)) {
            coeff = SymEngine::complex_double(SymEngine::eval_complex_double(*coeff).imag());
        }
    }
    static auto compress(coeff_t& coeff, double abs_tol = EQ_TOLERANCE) {  // NOLINT(runtime/references)
        if (SymEngine::is_a_Complex(*coeff)) {
            auto tmp = SymEngine::eval_complex_double(*coeff);
            CoeffPolicy<std::complex<double>>::compress(tmp, abs_tol);
            coeff = SymEngine::complex_double(tmp);
        }
    }
};

inline const CoeffPolicy<SymEngine::RCP<const SymEngine::Basic>>::coeff_t
    CoeffPolicy<SymEngine::RCP<const SymEngine::Basic>>::zero
    = SymEngine::zero;

inline const CoeffPolicy<SymEngine::RCP<const SymEngine::Basic>>::coeff_t
    CoeffPolicy<SymEngine::RCP<const SymEngine::Basic>>::one
    = SymEngine::one;

using SymEngineCoeffPolicy = CoeffPolicy<SymEngine::RCP<const SymEngine::Basic>>;
}  // namespace mindquantum::ops::details

#endif /* DETAILS_SYMENGINE_COEFF_POLICY_HPP */
