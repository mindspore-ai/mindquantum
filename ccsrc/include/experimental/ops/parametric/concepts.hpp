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

#ifndef PARAMETRIC_CONCEPTS_HPP
#define PARAMETRIC_CONCEPTS_HPP

#include <type_traits>

#include <symengine/complex_double.h>
#include <symengine/integer.h>
#include <symengine/real_double.h>

#include "experimental/ops/parametric/config.hpp"

#if MQ_HAS_CONCEPTS
#    include "experimental/core/concepts.hpp"
#endif  // MQ_HAS_CONCEPTS
#include "experimental/ops/parametric/param_names.hpp"

namespace mindquantum::concepts {
#if MQ_HAS_CONCEPTS
// template <typename key_t, typename value_t, std::size_t N>
// using umap_t = frozen::unordered_map<key_t, value_t, N, frozen::anna<key_t>, std::equal_to<>>;

template <typename expr_t>
concept symengine_expr = std::is_convertible_v<expr_t, SymEngine::RCP<const SymEngine::Basic>>;

template <typename param_t>
concept parameter = requires(param_t p) {
    requires std::same_as<decltype(param_t::name), const std::string_view>;
    // clang-format off
    // NOLINTNEXTLINE(whitespace/parens)
    requires (std::same_as<typename param_t::param_type, ops::parametric::details::real_tag_t>
              || std::same_as<typename param_t::param_type, ops::parametric::details::complex_tag_t>);
    // clang-format on
};

template <typename T>
concept expr_or_number = symengine_expr<T> || number<T>;
#else
template <typename expr_t, typename U = void>
struct symengine_expr : std::false_type {};

template <typename expr_t>
struct symengine_expr<
    expr_t, std::enable_if_t<std::disjunction_v<std::is_convertible<expr_t, SymEngine::RCP<const SymEngine::Basic>>,
                                                std::is_convertible<expr_t, SymEngine::Expression>>>>
    : std::true_type {};

template <typename T, typename U = void>
struct number : std::false_type {};

template <typename T>
struct number<T, std::enable_if_t<(std::is_integral_v<T> || std::is_floating_point_v<T>)
                                  || (std::is_same_v<std::complex<double>, T>)>> : std::true_type {};
#endif  // MQ_HAS_CONCEPTS
}  // namespace mindquantum::concepts
#endif  // PARAMETRIC_CONCEPTS_HPP
