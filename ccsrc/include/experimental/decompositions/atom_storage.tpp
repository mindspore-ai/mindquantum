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

#ifndef ATOM_STORAGE_TPP
#define ATOM_STORAGE_TPP

#include <algorithm>
#include <string_view>
#include <utility>

#ifdef _MSC_VER
#    include <functional>
#endif  // _MSC_VER

#include "experimental/decompositions/config.hpp"
#include "experimental/decompositions/decomposition_atom.hpp"
#include "experimental/decompositions/details/concepts.hpp"
#ifndef ATOM_STORAGE_HPP
#    error This file must only be included by atom_storage.hpp!
#endif  // ATOM_STORAGE_HPP

// NB: This is mainly for syntax checkers and completion helpers as this file
//     is only intended to be included directly by atom_storage.hpp
#include "experimental/decompositions/atom_storage.hpp"

namespace mindquantum::decompositions {
// =========================================================================
// ::has_atom

template <typename o_atom_t>
bool AtomStorage::has_atom() const noexcept {
    return has_atom_helper_<o_atom_t, typename o_atom_t::kinds_t>::apply(*this, o_atom_t::num_controls(),
                                                                         o_atom_t::name());
}

template <typename o_atom_t, typename... operators_t>
bool AtomStorage::has_atom(num_control_t num_controls, std::string_view name) const noexcept {
    return std::find_if(begin(atoms_), end(atoms_),
                        [&num_controls, &name](const auto& item) {
                            return (item.second.is_kind(operators_t::kind()) || ...) && item.second.name() == name
                                   && item.first.second == num_controls;
                        })
           != end(atoms_);
}

// =========================================================================
// ::add_or_compatible_atom

template <typename o_atom_t, std::size_t idx, typename... args_t>
auto AtomStorage::add_or_compatible_atom(args_t&&... args) -> atom_t* {
    return add_or_non_replace_atom_<o_atom_t, std::less_equal<>, idx, args...>(std::forward<args_t>(args)...);
}

// =========================================================================
// ::add_or_return_atom

template <typename o_atom_t, std::size_t idx, typename... args_t>
auto AtomStorage::add_or_return_atom(args_t&&... args) -> atom_t* {
    return add_or_non_replace_atom_<o_atom_t, std::equal_to<>, idx, args...>(std::forward<args_t>(args)...);
}

// =========================================================================
// ::add_or_replace_atom

template <typename o_atom_t, std::size_t idx, typename... args_t>
auto AtomStorage::add_or_replace_atom(args_t&&... args) -> atom_t* {
    static constexpr auto num_controls = o_atom_t::num_controls();

    const auto kind = std::tuple_element_t<idx, typename o_atom_t::kinds_t>::kind();
    const auto atoms_end = end(atoms_);

    const auto kind_match = [kind = kind](const map_t::value_type& item) { return item.first.first == kind; };

    /* Multiple cases here:
     *  - if there is a rule that can handle any number of qubits, see if we need to insert
     *  - else
     *     + look if there is already a match based on the kind and number of constrained control qubits
     *        * if there is, replace the matching atom
     *        * else, do a simple insertion
     */

    auto it_begin = std::find_if(begin(atoms_), atoms_end, kind_match);

    map_t::iterator it_match(atoms_end);

    if (num_controls == any_control && it_begin != atoms_end && it_begin->first.second == any_control) {
        // Do replacement
        it_match = it_begin;
    } else {
#ifdef _MSC_VER
        const std::function<bool(const map_t::value_type&)>
#else
        const auto
#endif  // _MSC_VER
            atom_match = [kind_match = kind_match, num_controls = num_controls](const map_t::value_type& item) {
                return kind_match(item) && item.first.second == num_controls;
            };
        // First have a look if we know of this kind of atom
        it_match = std::find_if(it_begin, atoms_end, atom_match);

        if (it_match == atoms_end) {
            // Simple insertion
            const auto& [it, _] = atoms_.emplace(map_t::key_type{kind, num_controls},
                                                 o_atom_t::create(*this, std::forward<args_t>(args)...));
            return &it->second;
        }
    }

    // TODO(damien): Avoid replacement if name and kind match?
    it_match->second = o_atom_t::create(*this, std::forward<args_t>(args)...);
    return &it_match->second;
}

// =========================================================================
// ::add_or_non_replace_atom_

template <typename o_atom_t, typename ctrl_comp_t, std::size_t kind_idx, typename... args_t>
auto AtomStorage::add_or_non_replace_atom_(args_t&&... args) -> atom_t* {
    static constexpr auto num_controls = o_atom_t::num_controls();

    const auto kind = std::tuple_element_t<kind_idx, typename o_atom_t::kinds_t>::kind();
    const auto atoms_end = end(atoms_);

#ifdef _MSC_VER
    const std::function<bool(const map_t::value_type&)>
#else
    const auto
#endif  // _MSC_VER
        atom_match = [&kind, num_controls = num_controls](const map_t::value_type& item) {
            return item.first.first == kind && ctrl_comp_t{}(item.first.second, num_controls);
        };

    const auto it_match = std::find_if(begin(atoms_), atoms_end, atom_match);

    if (it_match == atoms_end) {
        // Simple insertion
        const auto& [it, _] = atoms_.emplace(map_t::key_type{kind, num_controls},
                                             o_atom_t::create(*this, std::forward<args_t>(args)...));
        return &it->second;
    }

    return &it_match->second;
}

}  // namespace mindquantum::decompositions

#endif /* ATOM_STORAGE_TPP */
