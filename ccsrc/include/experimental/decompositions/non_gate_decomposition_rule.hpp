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

#ifndef NON_GATE_DECOMPOSITION_RULE_HPP
#define NON_GATE_DECOMPOSITION_RULE_HPP

#include "experimental/decompositions/atom_storage.hpp"
#include "experimental/decompositions/config.hpp"
#include "experimental/decompositions/decomposition_atom.hpp"
#include "experimental/decompositions/decomposition_rule.hpp"

namespace mindquantum::decompositions {
template <typename derived_t, typename... atoms_t>
class NonGateDecompositionRule : public DecompositionRule<derived_t, atoms_t...> {
 public:
    using base_t = NonGateDecompositionRule;
    using self_t = NonGateDecompositionRule<derived_t, atoms_t...>;
    using parent_t = DecompositionRule<derived_t, atoms_t...>;
    using non_gate_decomposition = void;

    // ---------------------------------------------------------------------

    explicit NonGateDecompositionRule(AtomStorage& storage) : parent_t{storage}, storage_{storage} {
    }

    // ---------------------------------------------------------------------

    //! Getter function for the individual atoms
    /*!
     * Overload using a type template type parameter. This works since the list of atoms contains only unique
     * values.
     *
     * \tparam atom_t Type of atom to look for.
     */
    template <typename atom_t, typename... args_t>
    constexpr auto* atom(args_t&&... args) noexcept;

    auto& storage() noexcept {
        return storage_;
    }

    //! Apply a decomposition rule
    /*!
     * \param circuit Quantum circuit
     * \param inst Quantum instructio to decompose
     */
    void apply(circuit_t& circuit, const instruction_t& inst) noexcept;

 private:
    AtomStorage& storage_;
};
}  // namespace mindquantum::decompositions

#include "non_gate_decomposition_rule.tpp"

#endif /* NON_GATE_DECOMPOSITION_RULE_HPP */
