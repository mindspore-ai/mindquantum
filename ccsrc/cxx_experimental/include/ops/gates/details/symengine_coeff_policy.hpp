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

#include "core/config.hpp"

#include "core/format/symengine_basic.hpp"
#include "ops/gates/details/complex_double_coeff_policy.hpp"

namespace mindquantum::ops::details {
struct SymEngineCoeffPolicy {
    using coeff_t = SymEngine::RCP<const SymEngine::Basic>;
    using self_t = SymEngineCoeffPolicy;

    static const coeff_t one;
    static constexpr auto EQ_TOLERANCE = 1.e-8;

    // Comparisons
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
    static auto iadd(coeff_t& lhs, const coeff_t& rhs) {
        return lhs = self_t::add(lhs, rhs);
        // return lhs = self_t::add(lhs, rhs).rcp_from_this();
    }
    static auto sub(const coeff_t& lhs, const coeff_t& rhs) {
        return SymEngine::sub(lhs, rhs);
    }
    static auto isub(coeff_t& lhs, const coeff_t& rhs) {
        return lhs = self_t::sub(lhs, rhs);
    }
    static auto mul(const coeff_t& lhs, const coeff_t& rhs) {
        return SymEngine::mul(lhs, rhs);
    }
    static auto imul(coeff_t& lhs, const coeff_t& rhs) {
        return lhs = self_t::mul(lhs, rhs);
    }
    static auto div(const coeff_t& lhs, const coeff_t& rhs) {
        return SymEngine::div(lhs, rhs);
    }
    static auto idiv(coeff_t& lhs, const coeff_t& rhs) {
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
    static auto cast_real(coeff_t& coeff) {
        if (SymEngine::is_a_Complex(*coeff)) {
            coeff = SymEngine::complex_double(SymEngine::eval_complex_double(*coeff).real());
        }
    }
    static auto cast_imag(coeff_t& coeff) {
        if (SymEngine::is_a_Complex(*coeff)) {
            coeff = SymEngine::complex_double(SymEngine::eval_complex_double(*coeff).imag());
        }
    }
    static auto compress(coeff_t& coeff, double abs_tol = EQ_TOLERANCE) {
        if (SymEngine::is_a_Complex(*coeff)) {
            auto tmp = SymEngine::eval_complex_double(*coeff);
            CmplxDoubleCoeffPolicy::compress(tmp, abs_tol);
            coeff = SymEngine::complex_double(tmp);
        }
    }
};
inline const SymEngineCoeffPolicy::coeff_t SymEngineCoeffPolicy::one = SymEngine::one;
}  // namespace mindquantum::ops::details

#endif /* DETAILS_SYMENGINE_COEFF_POLICY_HPP */
