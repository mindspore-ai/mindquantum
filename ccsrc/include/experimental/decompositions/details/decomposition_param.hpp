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

#ifndef DECOMPOSITION_PARAM_HPP
#define DECOMPOSITION_PARAM_HPP

#include <cstdint>

#include "experimental/decompositions/config.hpp"
#include "experimental/ops/parametric/config.hpp"

namespace mindquantum::decompositions {
//! Aggregate type to store DecompositionRule template parameters
struct DecompositionRuleParam {
    num_target_t num_targets;    //!< Number of target qubits the decomposition is constrained on
    num_control_t num_controls;  //!< Number of control qubits the decomposition is constrained on
    num_param_t num_params;      //!< Number of parameters the decomposition rule possesses
} MQ_ALIGN(16);

// ---------------------------------
// Definition of special values for decomposition rule template parameter

//! Namespace containing some helper constants to use when defining DecompositionRule classes
namespace tparam {
//! Constants for decomposition rules without constraints on the number of target qubits
namespace any_tgt {
static constexpr auto any_ctrl = DecompositionRuleParam{0, any_control, 0};
static constexpr auto no_ctrl = DecompositionRuleParam{0, 0, 0};
static constexpr auto single_ctrl = DecompositionRuleParam{0, 1, 0};
static constexpr auto double_ctrl = DecompositionRuleParam{0, 2, 0};
}  // namespace any_tgt

//! Constants for decomposition rules for single target qubit gates
namespace single_tgt {
static constexpr auto any_ctrl = DecompositionRuleParam{1, any_control, 0};
static constexpr auto no_ctrl = DecompositionRuleParam{1, 0, 0};
static constexpr auto single_ctrl = DecompositionRuleParam{1, 1, 0};
static constexpr auto double_ctrl = DecompositionRuleParam{1, 2, 0};
}  // namespace single_tgt

//! Constants for decomposition rules for single target qubit gates
namespace dual_tgt {
static constexpr auto any_ctrl = DecompositionRuleParam{2, any_control, 0};
static constexpr auto no_ctrl = DecompositionRuleParam{2, 0, 0};
static constexpr auto single_ctrl = DecompositionRuleParam{2, 1, 0};
static constexpr auto double_ctrl = DecompositionRuleParam{2, 2, 0};
}  // namespace dual_tgt

//! Constants for decomposition rules for single target qubit with parameteric gates with 1 parameter
namespace single_tgt_param {
static constexpr auto any_ctrl = DecompositionRuleParam{1, any_control, 1};
static constexpr auto no_ctrl = DecompositionRuleParam{1, 0, 1};
static constexpr auto single_ctrl = DecompositionRuleParam{1, 1, 1};
static constexpr auto double_ctrl = DecompositionRuleParam{1, 2, 1};
}  // namespace single_tgt_param

static constexpr auto default_t = any_tgt::any_ctrl;
}  // namespace tparam
}  // namespace mindquantum::decompositions

#if MQ_HAS_CLASS_NON_TYPE_TEMPLATE_ARGS
#    define ANY_TGT_ANY_CTRL    mindquantum::decompositions::tparam::any_tgt::any_ctrl
#    define ANY_TGT_NO_CTRL     mindquantum::decompositions::tparam::any_tgt::no_ctrl
#    define ANY_TGT_SINGLE_CTRL mindquantum::decompositions::tparam::any_tgt::single_ctrl
#    define ANY_TGT_DOUBLE_CTRL mindquantum::decompositions::tparam::any_tgt::double_ctrl

#    define SINGLE_TGT_ANY_CTRL    mindquantum::decompositions::tparam::single_tgt::any_ctrl
#    define SINGLE_TGT_NO_CTRL     mindquantum::decompositions::tparam::single_tgt::no_ctrl
#    define SINGLE_TGT_SINGLE_CTRL mindquantum::decompositions::tparam::single_tgt::single_ctrl
#    define SINGLE_TGT_DOUBLE_CTRL mindquantum::decompositions::tparam::single_tgt::double_ctrl

#    define DUAL_TGT_ANY_CTRL    mindquantum::decompositions::tparam::dual_tgt::any_ctrl
#    define DUAL_TGT_NO_CTRL     mindquantum::decompositions::tparam::dual_tgt::no_ctrl
#    define DUAL_TGT_SINGLE_CTRL mindquantum::decompositions::tparam::dual_tgt::single_ctrl
#    define DUAL_TGT_DOUBLE_CTRL mindquantum::decompositions::tparam::dual_tgt::double_ctrl

#    define SINGLE_TGT_PARAM_ANY_CTRL    mindquantum::decompositions::tparam::single_tgt_param::any_ctrl
#    define SINGLE_TGT_PARAM_NO_CTRL     mindquantum::decompositions::tparam::single_tgt_param::no_ctrl
#    define SINGLE_TGT_PARAM_SINGLE_CTRL mindquantum::decompositions::tparam::single_tgt_param::single_ctrl
#    define SINGLE_TGT_PARAM_DOUBLE_CTRL mindquantum::decompositions::tparam::single_tgt_param::double_ctrl
#else
#    define MQ_INTERNAL_DR_EXPAND_TPARAM_(value) value.num_targets, value.num_controls, value.num_params
// NOLINTNEXTLINE(whitespace/line_length)
#    define MQ_INTERNAL_DR_EXPAND_TPARAM(value)  MQ_INTERNAL_DR_EXPAND_TPARAM_(mindquantum::decompositions::value)

#    define ANY_TGT_ANY_CTRL    MQ_INTERNAL_DR_EXPAND_TPARAM(tparam::any_tgt::any_ctrl)
#    define ANY_TGT_NO_CTRL     MQ_INTERNAL_DR_EXPAND_TPARAM(tparam::any_tgt::no_ctrl)
#    define ANY_TGT_SINGLE_CTRL MQ_INTERNAL_DR_EXPAND_TPARAM(tparam::any_tgt::single_ctrl)
#    define ANY_TGT_DOUBLE_CTRL MQ_INTERNAL_DR_EXPAND_TPARAM(tparam::any_tgt::double_ctrl)

#    define SINGLE_TGT_ANY_CTRL    MQ_INTERNAL_DR_EXPAND_TPARAM(tparam::single_tgt::any_ctrl)
#    define SINGLE_TGT_NO_CTRL     MQ_INTERNAL_DR_EXPAND_TPARAM(tparam::single_tgt::no_ctrl)
#    define SINGLE_TGT_SINGLE_CTRL MQ_INTERNAL_DR_EXPAND_TPARAM(tparam::single_tgt::single_ctrl)
#    define SINGLE_TGT_DOUBLE_CTRL MQ_INTERNAL_DR_EXPAND_TPARAM(tparam::single_tgt::double_ctrl)

#    define DUAL_TGT_ANY_CTRL    MQ_INTERNAL_DR_EXPAND_TPARAM(tparam::dual_tgt::any_ctrl)
#    define DUAL_TGT_NO_CTRL     MQ_INTERNAL_DR_EXPAND_TPARAM(tparam::dual_tgt::no_ctrl)
#    define DUAL_TGT_DUAL_CTRL   MQ_INTERNAL_DR_EXPAND_TPARAM(tparam::dual_tgt::dual_ctrl)
#    define DUAL_TGT_DOUBLE_CTRL MQ_INTERNAL_DR_EXPAND_TPARAM(tparam::dual_tgt::double_ctrl)

#    define SINGLE_TGT_PARAM_ANY_CTRL    MQ_INTERNAL_DR_EXPAND_TPARAM(tparam::single_tgt_param::any_ctrl)
#    define SINGLE_TGT_PARAM_NO_CTRL     MQ_INTERNAL_DR_EXPAND_TPARAM(tparam::single_tgt_param::no_ctrl)
#    define SINGLE_TGT_PARAM_SINGLE_CTRL MQ_INTERNAL_DR_EXPAND_TPARAM(tparam::single_tgt_param::single_ctrl)
#    define SINGLE_TGT_PARAM_DOUBLE_CTRL MQ_INTERNAL_DR_EXPAND_TPARAM(tparam::single_tgt_param::double_ctrl)
#endif  // MQ_HAS_CLASS_NON_TYPE_TEMPLATE_ARGS

#endif /* DECOMPOSITION_PARAM_HPP */
