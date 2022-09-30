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

#ifndef DECOMPOSITION_RULE_HPP
#define DECOMPOSITION_RULE_HPP

#include <type_traits>
#include <utility>

#include "experimental/decompositions/atom_storage.hpp"
#include "experimental/decompositions/config.hpp"
#include "experimental/decompositions/decomposition_atom.hpp"
#include "experimental/decompositions/details/traits.hpp"
#include "experimental/ops/parametric/config.hpp"

namespace mindquantum::decompositions {
template <typename derived_t, typename... atoms_t>
class DecompositionRule {
 public:
    using base_t = DecompositionRule;
    using self_t = DecompositionRule<derived_t, atoms_t...>;

    using gate_param_t = ops::parametric::gate_param_t;

    static_assert(traits::is_unique<typename traits::atom_traits<atoms_t>::type...>, "Atom types must be unique");

    //! Return the name of this DecompositionRule
    static constexpr auto name() noexcept {
        return derived_t::name();
    }

    //! Helper function to create a DecompositionRule instance
    /*!
     * \param storage Atom storage within which this decomposition will live in
     */
    template <typename... args_t>
    MQ_NODISCARD static auto create(AtomStorage& storage, args_t&&... args) noexcept {
        return derived_t{storage, std::forward<args_t>(args)...};
    }

    // ---------------------------------------------------------------------

    //! Constructor
    /*!
     * \note The atoms of this decomposition rule will be insertd into the storage if they are not already
     *       present. However, existing (exactly matching) atoms will not be replaced.
     *
     * \param storage Atom storage within which this decomposition will live in
     */
    explicit DecompositionRule(AtomStorage& storage);

    // ---------------------------------------------------------------------

    //! Getter function for the individual atoms
    /*!
     * Overload using a non-type template parameter corresponding to the index of the atom in the atom list.
     *
     * \tparam idx Index of atom in atom list
     */
    template <std::size_t idx>
    constexpr auto* atom() noexcept MQ_REQUIRES((idx < sizeof...(atoms_t)));

    //! Getter function for the individual atoms
    /*!
     * Overload using a type template type parameter. This works since the list of atoms contains only unique
     * values.
     *
     * \tparam atom_t Type of atom to look for.
     */
    template <typename atom_t>
    constexpr auto* atom() noexcept
        MQ_REQUIRES((concepts::tuple_contains<typename traits::atom_traits<atom_t>::type,
                                              typename traits::atom_traits<atoms_t>::type...>) );

    //! Apply a decomposition
    /*!
     * \param circuit A quantum circuit to apply the decomposition atom to
     * \param op A quantum operation to decompose
     * \param qubits A list of qubits to apply the decomposition atom
     * \param cbits A list of classical bit the decomposition applies to
     *
     * \note Currently the \c cbits parameter is not used at all! It is here to make the API futureproof.
     */
    void apply(circuit_t& circuit, const operator_t& op, const qubits_t& qubits, const cbits_t& cbits) noexcept;

 protected:
    //! Apply an invalid operatoir to a circuit (to indicate an error in processing)
    /*!
     * \param circuit Quantum circuit
     * \param qubits List of qubits
     */
    static void invalid_op_(circuit_t& circuit, const qubits_t& qubits);
    //! Apply an invalid operatoir to a circuit (to indicate an error in processing)
    /*!
     * Overload for parametric operators.
     *
     * \param circuit Quantum circuit
     * \param qubits List of qubits
     * \param param Parameter to use during the application
     */
    static void invalid_op_(circuit_t& circuit, const qubits_t& qubits, const gate_param_t& param);

 private:
    template <typename... args_t>
    auto create_(AtomStorage& storage);

    template <typename T>
    auto create_el_(AtomStorage& storage);

    std::array<DecompositionAtom*, sizeof...(atoms_t)> atoms_ = {};
};
}  // namespace mindquantum::decompositions

#include "decomposition_rule.tpp"

#endif /* DECOMPOSITION_RULE_HPP */
