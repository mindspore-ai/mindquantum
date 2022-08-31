//   Copyright 2021 <Huawei Technologies Co., Ltd>
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

#ifndef PARAM_ANGLE_BASE_HPP
#define PARAM_ANGLE_BASE_HPP

#include <cmath>

#include <utility>
#if __has_include(<numbers>) && __cplusplus > 201703L
#    include <numbers>
#endif  // __has_include(<numbers>) && C++20
#include <tuple>

#include <tweedledum/IR/Operator.h>

#include <symengine/constants.h>
#include <symengine/logic.h>
#include <symengine/mul.h>

#include "experimental/core/config.hpp"
#include "experimental/ops/parametric/gate_base.hpp"
#include "experimental/ops/parametric/param_names.hpp"

namespace mindquantum::ops::parametric {
#if MQ_HAS_CONCEPTS
template <typename derived_t, concepts::AngleGate non_param_t, std::size_t mod_pi>
#else
template <typename derived_t, typename non_param_t, std::size_t mod_pi>
#endif  // MQ_HAS_CONCEPTS
class AngleParametricBase : public ParametricBase<derived_t, non_param_t, real::theta> {
 public:
    using operator_t = tweedledum::Operator;
    using parent_t = ParametricBase<derived_t, non_param_t, real::theta>;
    using base_t = AngleParametricBase;
    using self_t = AngleParametricBase<derived_t, non_param_t, mod_pi>;

    using typename parent_t::non_param_type;
    using typename parent_t::subs_map_t;

    // NB: using typename parent_t::ParametricBase should work fine but not on older compilers...
    using ParametricBase<derived_t, non_param_t, real::theta>::ParametricBase;

    //! Evaluation helper function
    template <typename evaluated_param_t>
    MQ_NODISCARD static auto to_param_type(const self_t& /* self */, evaluated_param_t&& evaluated_param) {
        return derived_t{std::forward<evaluated_param_t>(evaluated_param)};
    }

    //! Evaluation helper function
    template <typename evaluated_param_t>
    MQ_NODISCARD static auto to_non_param_type(const self_t& /* self */, evaluated_param_t&& evaluated_param) {
        // NB: non-parametric classes simply accept the evaluated parameter
        return non_param_type{std::forward<evaluated_param_t>(evaluated_param)};
    }

    //! Test whether another operation is the same as this instance
    MQ_NODISCARD bool operator==(const AngleParametricBase& other) const noexcept {
        return eq(*this->param(0), *other.param(0));
        // return eq(*expand(sub(this->param(0), other.param(0))), SymEngine::Integer(0));
    }

    //! Get the adjoint of an \c AngleParametricBase gate instance
    MQ_NODISCARD auto adjoint() const noexcept {
        auto params = base_t::params_;
        for (auto& param : params) {
            param = expand(neg(param));
        }

        if constexpr (base_t::has_const_num_targets) {
            return derived_t{std::move(params)};
        } else {
            return derived_t{this->num_targets_, std::move(params)};
        }
    }

    //! Fully evaluate this parametric gate
    /*!
     * Attempt to fully evaluate this parametric gate (ie. evaluate all parameter numerically). The constructor
     * of the non-parametric gate type is called by passing each of the numerically evaluated parameter in the
     * order that is defined when passing the type as the template parameters to this base class.
     *
     * \return An instance of a non-parametric gate (\c non_param_type)
     * \throw SymEngine::SymEngineException if the expression cannot be fully evaluated numerically
     */
    MQ_NODISCARD non_param_type eval_full() const {
        using param_t = typename std::tuple_element_t<0, typename parent_t::params_type>::param_type;

        return non_param_type {
            std::fmod(param_t::eval(this->params_[0]),
#if __has_include(<numbers>) && __cplusplus > 201703L
                      std::numbers::pi * mod_pi
#else
                      3.1415926535897932 * mod_pi
#endif  // __has_include(<numbers>) && C++20
            )
        };
    }

    //! Fully evaluate this parametric gate
    /*!
     * Attempt to fully evaluate this parametric gate (ie. evaluate all parameter numerically). The constructor
     * of the non-parametric gate type is called by passing each of the numerically evaluated parameter in the
     * order that is defined when passing the type as the template parameters to this base class.
     *
     * This overload accepts a dictionary of substitutions to perform on the parameters.
     *
     * \param subs_map Dictionary containing all the substitution to perform
     * \return An instance of a non-parametric gate (\c non_param_type)
     * \throw SymEngine::SymEngineException if the expression cannot be fully evaluated numerically
     */
    MQ_NODISCARD non_param_type eval_full(const subs_map_t& subs_map) const {
        using param_t = typename std::tuple_element_t<0, typename parent_t::params_type>::param_type;

        return non_param_type {
            std::fmod(param_t::eval(this->params_[0]->subs(subs_map)),
#if __has_include(<numbers>) && __cplusplus > 201703L
                      std::numbers::pi * mod_pi
#else
                      3.1415926535897932 * mod_pi
#endif  // __has_include(<numbers>) && C++20
            )
        };
    }
};
}  // namespace mindquantum::ops::parametric

#endif /* PARAM_ANGLE_BASE_HPP */
