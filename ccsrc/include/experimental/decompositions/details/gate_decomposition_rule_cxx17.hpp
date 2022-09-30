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

#ifndef GATE_DECOMPOSITION_RULE_CXX17_HPP
#define GATE_DECOMPOSITION_RULE_CXX17_HPP

#include <tuple>
#include <type_traits>

#include "experimental/decompositions/config.hpp"
#include "experimental/decompositions/decomposition_rule.hpp"
#include "experimental/ops/parametric/config.hpp"

namespace mindquantum::decompositions {
template <typename derived_t, typename kinds_t, uint32_t, num_control_t, uint32_t, typename... atoms_t>
class GateDecompositionRuleCXX17;

template <typename derived_t, typename kinds_t, uint32_t num_targets, num_control_t num_controls, uint32_t num_params,
          typename... atoms_t>
using GateDecompositionRule
    = GateDecompositionRuleCXX17<derived_t, kinds_t, num_targets, num_controls, num_params, atoms_t...>;
}  // namespace mindquantum::decompositions

// =============================================================================

namespace mindquantum::decompositions {
template <typename derived_t, typename kinds_t_, uint32_t num_targets_, num_control_t num_controls_,
          uint32_t num_params_, typename... atoms_t>
class GateDecompositionRuleCXX17
    : public DecompositionRule<derived_t, atoms_t...>
    , public traits::controls<num_controls_> {
 public:
    using base_t = GateDecompositionRuleCXX17;
    using kinds_t = kinds_t_;
    using parent_t = DecompositionRule<derived_t, atoms_t...>;
    using self_t = GateDecompositionRuleCXX17<derived_t, kinds_t, num_targets_, num_controls_, num_params_, atoms_t...>;

    using gate_param_t = ops::parametric::gate_param_t;
    using double_list_t = ops::parametric::double_list_t;
    using param_list_t = ops::parametric::param_list_t;

    // ---------------------------------------------------------------------

    using DecompositionRule<derived_t, atoms_t...>::DecompositionRule;

    // ---------------------------------------------------------------------

    //! Return the number of target qubits this DecompositionRule is constrained on
    static constexpr auto num_targets() noexcept {
        return num_targets_;
    }

    //! Return the number of control qubits this DecompositionRule is constrained on
    static constexpr auto num_controls() noexcept {
        return num_controls_;
    }

    //! Constant boolean indicating whether the GateDecompositionRuleCXX17 is parametric or not
    static constexpr auto is_parametric = num_params_ != 0U;

    //! Return the number of parameters of this GateDecompositionRuleCXX17
    static constexpr auto num_params() noexcept {
        return num_params_;
    }

    // ---------------------------------------------------------------------

    //! Check whether a decomposition is compatible with another one.
    /*!
     * Another GateDecompositionRuleCXX17 instance is deemed compatible iff:
     *   - the number of target qubit are identical
     *   - the number of controls are compatible:
     *      - the number of control qubits in the decomposition rule is left undefined
     *      - or they have the same number of controls
     *
     * \param num_targets Number of target qubits of the operation to decompose
     * \param num_controls Number of control qubits of the operation to decompose
     */
    // TODO(dnguyen): constrain `rule_t` to decomposition atoms
    template <typename rule_t>
    MQ_NODISCARD constexpr bool is_compatible() const noexcept;

    //! Check whether a decomposition is applicable with a given instruction
    /*!
     * \param inst A quantum instruction
     */
    MQ_NODISCARD bool is_applicable(const instruction_t& inst) const noexcept;
};
}  // namespace mindquantum::decompositions

#include "gate_decomposition_rule_cxx17.tpp"

#endif /* GATE_DECOMPOSITION_RULE_CXX17_HPP */
