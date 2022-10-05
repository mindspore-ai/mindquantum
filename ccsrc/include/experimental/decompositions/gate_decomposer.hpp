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

#ifndef GATE_DECOMPOSER_HPP
#define GATE_DECOMPOSER_HPP

#include <set>
#include <string>
#include <utility>

#include "experimental/decompositions/atom_storage.hpp"

#ifdef UNIT_TESTS
class UnitTestAccessor;
#endif  // UNIT_TESTS

namespace mindquantum::decompositions {
namespace details {
struct rules_less {  // NOLINT(altera-struct-pack-align)
    using is_transparent = void;

    template <typename T, typename U>
    constexpr auto operator()(T&& lhs, U&& rhs) const {
        return std::forward<T>(lhs).name() < std::forward<U>(rhs).name();
    }
};
}  // namespace details

class GateDecomposer {
 public:
    using atom_storage_t = AtomStorage;
    using atom_t = atom_storage_t::atom_t;
    using general_rule_storage_t = std::set<DecompositionAtom, details::rules_less>;

    //! Return the number of atoms in the internal storage
    MQ_NODISCARD auto num_atoms() const noexcept {
        return std::size(atom_storage_);
    }

    //! Return the number of general decomposition rules in the internal storage
    MQ_NODISCARD auto num_rules() const noexcept {
        return std::size(general_rule_storage_);
    }

    //! Simple getter to the internal storage
    MQ_NODISCARD const auto& storage() const noexcept {
        return atom_storage_;
    }

    //! Check whether a matching (gate) atom can be found in the storage
    /*!
     * The comparison is performed based on the value of \c o_atom_t::num_controls(), \c o_atom_t::name(), as well
     * as taking \c o_atom_t::kinds_t into account. The matching for the kind is performed by calling \c
     * atom->is_kind(type::kind()) for each operator contained in \c o_atom_t::kinds_t.
     *
     * \note This method does not take general decompositions into account.
     *
     * \tparam o_atom_t Type of atom to look for
     * \return True/false depending on whether the atom can be found or not
     */
    template <typename o_atom_t>
    MQ_NODISCARD bool has_atom() const noexcept;

    //! Look for a suitable decomposition within the storages
    /*!
     * This method will favour gate decompositions over general decompositions when looking for a match.
     *
     * \param inst An instruction
     * \return Pointer to atom if any, \c nullptr otherwise
     */
    MQ_NODISCARD atom_t* get_atom_for(const instruction_t& inst) noexcept;

    // Read-write accessors

    //! Inserts a new element, constructed in-place with the given args
    /*!
     * The type \c o_atom_t will be used to determine where the atom will be stored; ie. whether it is a gate
     * decomposition or a general decomposition.
     *
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

    atom_storage_t atom_storage_;
    general_rule_storage_t general_rule_storage_;
};
}  // namespace mindquantum::decompositions

#include "experimental/decompositions/gate_decomposer.tpp"  // NOLINT(build/include)

#endif /* GATE_DECOMPOSER_HPP */
