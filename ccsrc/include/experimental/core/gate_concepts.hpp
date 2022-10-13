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

#ifndef GATE_CONCEPTS_HPP
#define GATE_CONCEPTS_HPP

#include <functional>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <vector>

#include <tweedledum/IR/Operator.h>

#include <symengine/basic.h>

#include "experimental/core/operator_traits.hpp"
#include "experimental/ops/parametric/config.hpp"

#if MQ_HAS_CONCEPTS
#    include "experimental/core/concepts.hpp"
#endif  // MQ_HAS_CONCEPTS

namespace mindquantum::concepts {
#if MQ_HAS_CONCEPTS
template <typename op_t>
concept Gate = requires(op_t optor) {
    { op_t::kind() } -> std::same_as<std::string_view>;
};  // NOLINT(readability/braces)

template <typename op_t>
concept FixedNumTargetGate = requires(op_t optor) {
    requires Gate<op_t>;
    requires std::default_initializable<op_t>;
    // clang-format off
    requires std::greater<>{}(traits::num_targets<op_t>, 0);
    // clang-format on
};  // NOLINT(readability/braces)

template <typename op_t>
concept VariableNumTargetGate = requires(op_t optor) {
    requires Gate<op_t>;
    requires std::constructible_from<op_t, uint32_t>;
};  // NOLINT(readability/braces)

template <typename op_t>
concept SingleDoubleGate = requires(op_t optor) {
    requires Gate<op_t>;
    { optor.param() } -> same_decay_as<double>;
};  // NOLINT(readability/braces)

template <typename op_t>
concept MultiDoubleGate = requires(op_t optor) {
    requires Gate<op_t>;
    // TODO(damien): Perhaps store the number of parameters as a static constexpr class variable?
    { optor.params() } -> same_decay_as<std::vector<double>>;
};  // NOLINT(readability/braces)

template <typename op_t>
concept AngleGate = requires(op_t optor) {
    requires Gate<op_t>;
    requires std::constructible_from<op_t, double>;
    { optor.angle() } -> same_decay_as<double>;
};  // NOLINT(readability/braces)

template <typename op_t>
concept ParametricGate = requires(op_t optor, SymEngine::map_basic_basic subs) {
    requires Gate<op_t>;
    requires std::same_as<typename op_t::is_parametric, void>;
    requires Gate<typename op_t::non_param_type>;
    requires std::integral<decltype(op_t::num_params)>;
    requires std::greater<> {
    }
    (op_t::num_params, 0);

    { optor.param(0UL) } -> same_decay_as<ops::parametric::basic_t>;
    { optor.params() } -> same_decay_as<ops::parametric::param_list_t>;
    { optor.eval(subs) } -> same_decay_as<op_t>;
    { optor.eval_full(subs) } -> same_decay_as<typename op_t::non_param_type>;
    { optor.eval_smart(subs) } -> same_decay_as<tweedledum::Operator>;
};  // NOLINT(readability/braces)

template <typename op_t>
concept NonParametricGate = requires(op_t optor) {
    requires Gate<op_t>;
    requires !ParametricGate<op_t>;
};  // NOLINT(readability/braces)

//! Helper typedef
template <std::size_t idx, typename param_t>
using param_eval_t = typename std::tuple_element_t<0, param_t>::param_type::type;

template <typename op_t /*, typename evaluated_t */>
concept SingleParameterGate = requires(op_t optor) {
    requires ParametricGate<op_t>;
    // clang-format off
    requires std::equal_to<>{}(op_t::num_params, 1);
    // clang-format on
    // // Make sure that the parameter evaluates to what we expect
    // requires std::same_as<param_eval_t<0, typename op_t::params_type>, evaluated_t>;
};  // NOLINT(readability/braces)
#else
#endif  // MQ_HAS_CONCEPTS
}  // namespace mindquantum::concepts

#endif /* GATE_CONCEPTS_HPP */
