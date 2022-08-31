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

#ifndef PARAMETRIC_SUBSTITUTIONS_TPP
#define PARAMETRIC_SUBSTITUTIONS_TPP

#ifndef PARAMETRIC_SUBSTITUTIONS_HPP
#    error This file must only be included by ops/parametric/substitutions.hpp!
#endif  // PARAMETRIC_SUBSTITUTIONS_HPP

// NB: This is mainly for syntax checkers and completion helpers as this file
//     is only intended to be included directly by atom_storage.hpp

#include <utility>
#include <vector>

#include "experimental/ops/parametric/config.hpp"
#include "experimental/ops/parametric/substitutions.hpp"
#include "experimental/ops/parametric/to_symengine.hpp"

namespace mindquantum::ops::parametric {

template <typename op_t, typename... args_t>
auto generate_subs(args_t&&... args) {
    return details::create_subs_from_params<operator_t>(std::index_sequence_for<args_t...>{},
                                                        std::forward<args_t>(args)...);
}

template <typename op_t>
auto generate_subs(const double_list_t& params) {
    return details::generate_subs<op_t>(params);
}

template <typename op_t>
auto generate_subs(const param_list_t& params) {
    return details::generate_subs<op_t>(params);
}

// -------------------------------------------------------------------------

namespace details {
template <typename op_t, typename T>
auto generate_subs(const std::vector<T>& params) {
    subs_map_t subs;
    auto idx = 0UL;
    for (const auto& param : params) {
        subs.emplace(op_t::create_op().param(idx), to_symengine(param));
        ++idx;
    }
    return subs;
}

// =========================================================================

#if MQ_HAS_CONCEPTS
template <typename op_t, std::size_t... indices, concepts::expr_or_number... exprs_t>
#else
template <typename op_t, std::size_t... indices, typename... exprs_t>
#endif  // MQ_HAS_CONCEPTS
auto create_subs_from_params(std::index_sequence<indices...> /* indices */, exprs_t&&... exprs) {
    static_assert(sizeof...(indices) == op_t::num_params);
    static_assert(sizeof...(indices) == sizeof...(exprs));
    return subs_map_t{std::make_pair(op_t::create_op().param(indices), to_symengine(std::forward<exprs_t>(exprs)))...};
}

}  // namespace details

}  // namespace mindquantum::ops::parametric

#endif  // PARAMETRIC_SUBSTITUTIONS_TPP
