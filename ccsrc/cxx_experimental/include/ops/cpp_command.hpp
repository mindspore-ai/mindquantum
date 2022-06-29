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

#ifndef CPP_COMMAND_HPP
#define CPP_COMMAND_HPP

#include <memory>
#include <string>
#include <vector>

#include <tweedledum/IR/Operator.h>

#include "core/config.hpp"

namespace mindquantum::ops {
namespace td = tweedledum;

class Command {
 public:
    using gate_t = td::Operator;

    const auto& get_qubits() const {
        return qubits_;
    }
    const auto& get_control_qubits() const {
        return control_qubits_;
    }
    const gate_t& get_gate() const {
        assert(static_cast<bool>(operation_));
        return *operation_;
    }

 protected:
    qubit_ids_t qubits_;
    qubit_ids_t control_qubits_;

    // TODO(dnguyen): Can we do better than this?
    std::unique_ptr<td::Operator> operation_;
};
}  // namespace mindquantum::ops

#endif /* CPP_COMMAND_HPP */
