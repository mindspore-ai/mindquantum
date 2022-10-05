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

#ifndef GATE_DECOMPOSER_TPP
#define GATE_DECOMPOSER_TPP

#ifndef GATE_DECOMPOSER_HPP
#    error This file must only be included by gate_decomposer.hpp!
#endif  // !GATE_DECOMPOSER_HPP

// clang-format off
// NB: This is mainly for syntax checkers and completion helpers as this file
//     is only intended to be included directly by gate_decomposer.hpp
#include "experimental/decompositions/gate_decomposer.hpp"
// clang-format on

#include <utility>

namespace mindquantum::decompositions {

// =========================================================================
// ::has_atom

template <typename o_atom_t>
bool GateDecomposer::has_atom() const noexcept {
    if constexpr (concepts::GateDecomposition<o_atom_t>) {
        return atom_storage_.has_atom<o_atom_t>();
    } else {
        constexpr auto name = o_atom_t::name();
        return std::find_if(begin(general_rule_storage_), end(general_rule_storage_),
                            [&name](const auto& item) { return item.name() == name; })
               != end(general_rule_storage_);
    }
}

// =========================================================================

template <typename o_atom_t, std::size_t kind_idx, typename... args_t>
auto GateDecomposer::add_or_replace_atom(args_t&&... args) -> atom_t* {
    if constexpr (concepts::GateDecomposition<o_atom_t>) {
        return atom_storage_.add_or_replace_atom<o_atom_t, kind_idx>(std::forward<args_t>(args)...);
    } else {
        auto [it, _] = general_rule_storage_.emplace(o_atom_t::create(atom_storage_, std::forward<args_t>(args)...));
        /* NB: const_cast() is needed for compilers/STL implementations where std::set elements are always
         *     immutable when accessed through iterators.
         */
        return &const_cast<DecompositionAtom&>(*it);
    }
}

}  // namespace mindquantum::decompositions

// =============================================================================

#endif /* GATE_DECOMPOSER_TPP */
