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

#ifndef PYTHON_COMMAND_HPP
#define PYTHON_COMMAND_HPP

#include <memory>
#include <utility>

#include <tweedledum/IR/Operator.h>

#include "core/config.hpp"

#include "core/details/macros.hpp"
#include "ops/cpp_command.hpp"

CLANG_DIAG_OFF("-Wdeprecated-declarations")
#include <pybind11/cast.h>
#include <pybind11/pytypes.h>
CLANG_DIAG_ON("-Wdeprecated-declarations");

#include <string>
#include <vector>

// clang-format: off
#include "details/macros_conv_begin.hpp"
// clang-format: on

namespace mindquantum::python {
namespace td = tweedledum;

class Command : public ops::Command {
 public:
    void set_qubits(const qureg_t& qubits) {
        qubits_ = qubits;
    }

    void set_control_qubits(const qureg_t& qubits) {
        control_qubits_ = qubits;
    }

    void set_gate(const gate_t& operation) {
        operation_ = std::make_unique<td::Operator>(operation);
    }
};
}  // namespace mindquantum::python

// ==============================================================================

namespace mindquantum::details {
//! Helper function to extract attributes from a Python Command object
bool load_command(pybind11::handle src, python::Command& cmd);

//! Helper function to extract attributes from a Python gate object
tweedledum::Operator load_gate(PyObject* src, std::size_t n_targets, std::size_t n_controls, bool is_dagger = false);
}  // namespace mindquantum::details

// ==============================================================================

namespace pybind11::detail {
template <>
struct type_caster<mindquantum::ops::Command> {
 public:
    using value_type = mindquantum::ops::Command;

    PYBIND11_TYPE_CASTER(value_type, _("Command"));

    bool load(handle src, bool) {
        mindquantum::python::Command cmd;
        auto ok = mindquantum::details::load_command(src, cmd);
        if (ok) {
            value = std::move(cmd);
        }
        return ok;
    }
};
}  // namespace pybind11::detail

#include "details/macros_conv_end.hpp"

#endif /* PYTHON_COMMAND_HPP */
