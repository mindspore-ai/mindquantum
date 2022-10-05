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

#ifndef ATOM_STORAGE_HPP
#define ATOM_STORAGE_HPP

#include <algorithm>
#include <map>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>

#include "experimental/core/config.hpp"
#include "experimental/decompositions/config.hpp"
#include "experimental/decompositions/decomposition_atom.hpp"

#if MQ_HAS_CONCEPTS
#    include "experimental/core/concepts.hpp"
#endif  // MQ_HAS_CONCEPTS

#ifdef UNIT_TESTS
class UnitTestAccessor;
#endif  // UNIT_TESTS

namespace mindquantum::concepts {
#if MQ_HAS_CONCEPTS
template <typename atom_t>
concept atom_compatible_t = requires(atom_t) {
    // clang-format off
    { atom_t::kind() } -> std::same_as<std::string_view>;
    { atom_t::num_controls() } -> std::same_as<decompositions::num_control_t>;
    // clang-format on
};
#else
template <typename op_t, typename = void>
struct atom_compatible_t : std::false_type {};

template <typename op_t>
struct atom_compatible_t<op_t, std::void_t<std::tuple<decltype(op_t::kind()), decltype(op_t::num_controls())>>>
    : std::true_type {};
#endif  // MQ_HAS_CONCEPTS
}  // namespace mindquantum::concepts

namespace mindquantum::decompositions {
namespace details {
template <typename T>
struct is_pair : std::false_type {};

template <typename T, typename U>
struct is_pair<std::pair<T, U>> : std::true_type {};

struct atom_less {
    using is_transparent = void;

    template <typename T, typename U,
              typename
              = std::enable_if_t<!is_pair<std::remove_cvref_t<T>>::value && !is_pair<std::remove_cvref_t<U>>::value>>
    constexpr auto operator()(T&& lhs, U&& rhs) const {
        return std::forward<T>(lhs) < std::forward<U>(rhs);
    }

    template <typename T, typename U, typename V>
    constexpr auto operator()(const std::pair<T, V>& lhs, const std::pair<U, V>& rhs) const {
        if (lhs.first < rhs.first) {
            return true;
        }
        return (lhs.first == rhs.first) && (lhs.second < rhs.second);
    }
};

template <typename T, typename U, typename V>
constexpr auto kind_lookup(const T& lhs, const std::pair<U, V>& rhs) {
    return lhs < rhs.first;
}
}  // namespace details

class AtomStorage {
 public:
    using atom_t = DecompositionAtom;
    using map_t = std::map<std::pair<std::string, num_control_t>, atom_t, details::atom_less>;

    // Read-only accessors

    //! Return the number of decomposition atoms in the storage
    /*!
     * \return Pointer to inserted element, pointer to compatible element or nullptr.
     */
    MQ_NODISCARD auto size() const noexcept {
        // TODO(dnguyen): take general decompositions into account? If not need to change name!
        return std::size(atoms_);
    }

    //! Check whether a matching (gate) atom can be found in the storage
    /*!
     * The comparison is performed based on the value of \c o_atom_t::num_controls(), \c o_atom_t::name(), as well
     * as taking \c o_atom_t::kinds_t into account. The matching for the kind is performed by calling \c
     * atom->is_kind(type::kind()) for each operator contained in \c o_atom_t::kinds_t.
     *
     * \tparam o_atom_t Type of atom to look for
     * \return True/false depending on whether the atom can be found or not
     * \sa has_atom(num_control_t num_controls, std::string_view name) const
     */
    template <typename o_atom_t>
    MQ_NODISCARD bool has_atom() const noexcept;

    //! Check whether a matching (gate) atom can be found in the storage
    /*!
     * \note This method does not take general decompositions into account.
     *
     * \param kind Kind of atom to look for
     * \param num_controls Number of control the atom must be constrained by
     * \param name Name of atom to look for
     * \return True/false depending on whether the atom can be found or not
     */
    template <typename o_atom_t, typename... op_t>
    MQ_NODISCARD bool has_atom(num_control_t num_controls, std::string_view name) const noexcept;

    //! Look for a suitable decomposition atom within the storage
    /*!
     * \param inst An instruction
     * \return Pointer to atom if any, \c nullptr otherwise
     */
    MQ_NODISCARD atom_t* get_atom_for(const instruction_t& inst) noexcept;

    // Read-write accessors

    //! Inserts a new element, constructed in-place with the given args or returns an existing compatible atom
    /*!
     * This is different from add_or_return_atom in the sense that it does not enforce an exact match for the atom.
     *
     * \tparam o_atom_t Type of atom to insert/return
     * \tparam kind_idx If the atom has multiple element in its kinds_t tuple, this is the index of the type in that
     *                  tuple to use to register the atom inside the storage.
     * \return Pointer to inserted element, pointer to compatible element.
     */
    template <typename o_atom_t, std::size_t kind_idx = 0, typename... args_t>
    MQ_NODISCARD atom_t* add_or_compatible_atom(args_t&&... args);

    //! Inserts a new element, constructed in-place with the given args or returns an existing one
    /*!
     * This is different from add_or_compatible_atom in that it looks for an exact match.
     *
     * \tparam o_atom_t Type of atom to insert/return
     * \tparam kind_idx If the atom has multiple element in its kinds_t tuple, this is the index of the type in that
     *                  tuple to use to register the atom inside the storage.
     * \return Pointer to inserted element, pointer to compatible element.
     */
    template <typename o_atom_t, std::size_t kind_idx = 0, typename... args_t>
    MQ_NODISCARD atom_t* add_or_return_atom(args_t&&... args);

    //! Inserts or replaces a new element, constructed in-place with the given args
    /*!
     * \tparam o_atom_t Type of atom to insert/replace
     * \tparam kind_idx If the atom has multiple element in its kinds_t tuple, this is the index of the type in that
     *                  tuple to use to register the atom inside the storage.
     * \return Pointer to inserted element, pointer to compatible element or nullptr.
     */
    template <typename o_atom_t, std::size_t kind_idx = 0, typename... args_t>
    MQ_NODISCARD atom_t* add_or_replace_atom(args_t&&... args);

 private:
#ifdef UNIT_TESTS
    friend class ::UnitTestAccessor;
#endif  // UNIT_TESTS

    template <typename o_atom_t, typename ctrl_comp_t, std::size_t kind_idx = 0, typename... args_t>
    MQ_NODISCARD atom_t* add_or_non_replace_atom_(args_t&&... args);

    template <typename, typename>
    struct has_atom_helper_;

    template <typename o_atom_t, typename... operators_t>
    struct has_atom_helper_<o_atom_t, std::tuple<operators_t...>> {
        static constexpr auto apply(const AtomStorage& storage, num_control_t num_controls, std::string_view name) {
            return storage.has_atom<o_atom_t, operators_t...>(num_controls, name);
        }
    };

    map_t atoms_;

    // TODO(dnguyen): add vector of atoms for non-gate decompositions
};
}  // namespace mindquantum::decompositions

#include "atom_storage.tpp"

#endif /* ATOM_STORAGE_HPP */
