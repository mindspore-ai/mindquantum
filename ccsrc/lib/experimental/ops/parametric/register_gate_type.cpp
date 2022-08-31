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

#include "experimental/ops/parametric/register_gate_type.hpp"

#include <unordered_map>

#include <tweedledum/Operators/Ising.h>
#include <tweedledum/Operators/Standard.h>

#include "experimental/ops/gates.hpp"
#include "experimental/ops/gates/time_evolution.hpp"
#include "experimental/ops/parametric/angle_gates.hpp"
#include "experimental/ops/parametric/time_evolution.hpp"

// =============================================================================

namespace {
using mindquantum::ops::parametric::double_func_t;
using mindquantum::ops::parametric::params_func_t;
using mindquantum::ops::parametric::vec_double_func_t;

#define ADD_GATE(klass)                                                                                                \
    { std::string(klass::kind()), mindquantum::traits::gate_traits<klass>::param }

static std::unordered_map<std::string, double_func_t> double_gate_map = {
    ADD_GATE(tweedledum::Op::Rx),  ADD_GATE(tweedledum::Op::Rxx),  ADD_GATE(tweedledum::Op::Ry),
    ADD_GATE(tweedledum::Op::Ryy), ADD_GATE(tweedledum::Op::Rz),   ADD_GATE(tweedledum::Op::Rzz),
    ADD_GATE(tweedledum::Op::P),   ADD_GATE(mindquantum::ops::Ph), ADD_GATE(mindquantum::ops::TimeEvolution)};

static std::unordered_map<std::string, vec_double_func_t> vec_double_gate_map;

static std::unordered_map<std::string, params_func_t> param_gate_map = {
    ADD_GATE(mindquantum::ops::parametric::Rx),
    ADD_GATE(mindquantum::ops::parametric::Rxx),
    ADD_GATE(mindquantum::ops::parametric::Ry),
    ADD_GATE(mindquantum::ops::parametric::Ryy),
    ADD_GATE(mindquantum::ops::parametric::Rz),
    ADD_GATE(mindquantum::ops::parametric::Rzz),
    ADD_GATE(mindquantum::ops::parametric::P),
    ADD_GATE(mindquantum::ops::parametric::Ph),
    ADD_GATE(mindquantum::ops::parametric::TimeEvolution)};

#undef ADD_GATE

template <typename map_t>
void register_gate(map_t& map, std::string_view kind, typename map_t::mapped_type&& func) {
    map.emplace(std::string(kind), std::move(func));
}
}  // namespace

// -----------------------------------------------------------------------------

namespace mindquantum::ops::parametric::details {
void register_gate(std::string_view kind, double_func_t angle_func) {
    ::register_gate(::double_gate_map, kind, std::move(angle_func));
}

void register_gate(std::string_view kind, vec_double_func_t vec_double_func) {
    ::register_gate(::vec_double_gate_map, kind, std::move(vec_double_func));
}

void register_gate(std::string_view kind, params_func_t params_func) {
    ::register_gate(::param_gate_map, kind, std::move(params_func));
}
}  // namespace mindquantum::ops::parametric::details

// =============================================================================

namespace mindquantum::ops::parametric {
gate_param_t get_param(const operator_t& optor) noexcept {
    std::string kind(optor.kind());
    if (auto it = ::double_gate_map.find(kind); it != std::end(::double_gate_map)) {
        return it->second(optor);
    }
    if (auto it = ::vec_double_gate_map.find(kind); it != std::end(::vec_double_gate_map)) {
        return it->second(optor);
    }
    if (auto it = ::param_gate_map.find(kind); it != std::end(::param_gate_map)) {
        return it->second(optor);
    }
    return std::monostate{};
}
}  // namespace mindquantum::ops::parametric

// =============================================================================
