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

#ifndef PARAM_ANGLE_GATES_HPP
#define PARAM_ANGLE_GATES_HPP

#include <tweedledum/Operators/Ising/Rxx.h>
#include <tweedledum/Operators/Ising/Ryy.h>
#include <tweedledum/Operators/Ising/Rzz.h>
#include <tweedledum/Operators/Standard/P.h>
#include <tweedledum/Operators/Standard/Rx.h>
#include <tweedledum/Operators/Standard/Ry.h>
#include <tweedledum/Operators/Standard/Rz.h>

#include "experimental/core/operator_traits.hpp"
#include "experimental/ops/gates/ph.hpp"
#include "experimental/ops/parametric/angle_base.hpp"

namespace mindquantum::ops::parametric {
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define DEFINE_PARAMETRIC_ANGLE_GATE_CLASS(klass, non_param_type, gate_kind, mod_pi)                                   \
    /* NOLINTNEXTLINE(bugprone-macro-parentheses) */                                                                   \
    class klass : public AngleParametricBase<klass, non_param_type, mod_pi> {                                          \
     public:                                                                                                           \
        using base_t::base_t;                                                                                          \
        static constexpr auto num_targets = traits::num_targets<non_param_type>;                                       \
                                                                                                                       \
        static constexpr std::string_view kind() {                                                                     \
            return gate_kind;                                                                                          \
        }                                                                                                              \
    }

DEFINE_PARAMETRIC_ANGLE_GATE_CLASS(Ph, ops::Ph, "projectq.param.ph", 2);
DEFINE_PARAMETRIC_ANGLE_GATE_CLASS(P, tweedledum::Op::P, "projectq.param.p", 2);
DEFINE_PARAMETRIC_ANGLE_GATE_CLASS(Rx, tweedledum::Op::Rx, "projectq.param.rx", 4);
DEFINE_PARAMETRIC_ANGLE_GATE_CLASS(Rxx, tweedledum::Op::Rxx, "projectq.param.rxx", 4);
DEFINE_PARAMETRIC_ANGLE_GATE_CLASS(Ry, tweedledum::Op::Ry, "projectq.param.ry", 4);
DEFINE_PARAMETRIC_ANGLE_GATE_CLASS(Ryy, tweedledum::Op::Ryy, "projectq.param.ryy", 4);
DEFINE_PARAMETRIC_ANGLE_GATE_CLASS(Rz, tweedledum::Op::Rz, "projectq.param.rz", 4);
DEFINE_PARAMETRIC_ANGLE_GATE_CLASS(Rzz, tweedledum::Op::Rzz, "projectq.param.rzz", 4);

static_assert(Ph::num_targets == 1);
static_assert(P::num_targets == 1);
static_assert(Rx::num_targets == 1);
static_assert(Ry::num_targets == 1);
static_assert(Rz::num_targets == 1);
static_assert(Rxx::num_targets == 2);
static_assert(Ryy::num_targets == 2);
static_assert(Rzz::num_targets == 2);

static_assert(Ph::has_const_num_targets);
static_assert(P::has_const_num_targets);
static_assert(Rx::has_const_num_targets);
static_assert(Ry::has_const_num_targets);
static_assert(Rz::has_const_num_targets);
static_assert(Rxx::has_const_num_targets);
static_assert(Ryy::has_const_num_targets);
static_assert(Rzz::has_const_num_targets);

#undef DEFINE_PARAMETRIC_ANGLE_GATE_CLASS
}  // namespace mindquantum::ops::parametric

#endif /* PARAM_ANGLE_GATES_HPP */
