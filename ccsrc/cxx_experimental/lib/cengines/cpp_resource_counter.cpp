//   Copyright 2020 <Huawei Technologies Co., Ltd>
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

#include "cengines/cpp_resource_counter.hpp"

#include <tweedledum/Operators/Ising.h>
#include <tweedledum/Operators/Standard.h>

#include "ops/gates.hpp"

// =============================================================================

namespace td = tweedledum;

#define add_angle_gate(type, angle_factor)                                                                             \
    ++gate_counts_[std::make_tuple(kind, instruction.cast<type>().angle() * angle_factor, n_controls)]

void mindquantum::cengines::ResourceCounter::add_gate_counts(const td::Circuit& circuit) {
    circuit.foreach_instruction([&](const td::Instruction& instruction) {
        const auto kind = instruction.kind();
        const auto n_controls = instruction.num_controls();
        ++gate_class_counts_[std::make_pair(kind, n_controls)];

        if (kind == ops::Rx::kind()) {
            add_angle_gate(ops::Rx, 1);
        } else if (kind == ops::Ry::kind()) {
            add_angle_gate(ops::Ry, 1);
        } else if (kind == ops::Rz::kind()) {
            add_angle_gate(ops::Rz, 2);
        } else if (kind == ops::P::kind()) {
            add_angle_gate(ops::P, 1);
        } else if (kind == ops::Ph::kind()) {
            add_angle_gate(ops::Ph, 1);
        } else if (kind == ops::Rxx::kind()) {
            add_angle_gate(ops::Rxx, 2);
        } else if (kind == ops::Ryy::kind()) {
            add_angle_gate(ops::Ryy, 2);
        } else if (kind == ops::Rzz::kind()) {
            add_angle_gate(ops::Rzz, 1);
        }
        // TODO(dnguyen): missing QubitOperator, TimeEvolution
        else {
            ++gate_counts_[std::make_tuple(kind, param_t{}, n_controls)];
        }
    });
}

#undef add_angle_gate

// =============================================================================

void mindquantum::cengines::ResourceCounter::add_gate_count(std::string_view kind, param_t param,
                                                            std::size_t n_controls, std::size_t count) {
    gate_counts_[std::make_tuple(kind, param, n_controls)] += count;
    gate_class_counts_[std::make_pair(kind, n_controls)] += count;
}

// =============================================================================
