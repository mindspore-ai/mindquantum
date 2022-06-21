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

#ifndef DAGGERCIRCUIT_HPP
#define DAGGERCIRCUIT_HPP

#include <algorithm>
#include <memory>
#include <type_traits>

#include <tweedledum/IR/Circuit.h>
#include <tweedledum/Passes/Utility/shallow_duplicate.h>

namespace mindquantum {
namespace td = tweedledum;

class DaggeredCircuit {
 public:
    using instruction_t = td::Instruction;
    using instruction_ref_t = td::InstRef;
    using qubit_t = tweedledum::Qubit;
    using cbit_t = tweedledum::Cbit;
    using circuit_t = tweedledum::Circuit;

    explicit DaggeredCircuit(circuit_t& original)
        : processed_(false), original_(original), circuit_(tweedledum::shallow_duplicate(original)) {
    }

    ~DaggeredCircuit() {
        apply();
    }

    void apply() {
        if (!processed_) {
            processed_ = true;

            circuit_.foreach_qubit([&original = original_](const qubit_t& qubit, std::string_view name) {
                if (qubit >= original.num_qubits()) {
#ifndef NDEBUG
                    const auto new_qubit =
#endif  // NDEBUG
                        original.create_qubit(name);

#ifndef NDEBUG
                    assert(qubit == new_qubit);
#endif  // NDEBUG
                }
            });

            circuit_.foreach_cbit([&original = original_](const cbit_t& cbit, std::string_view name) {
                if (cbit > original.num_cbits()) {
#ifndef NDEBUG
                    const auto new_cbit =
#endif  // NDEBUG
                        original.create_cbit(name);
#ifndef NDEBUG
                    assert(cbit == new_cbit);
#endif  // NDEBUG
                }
            });

            // Execute commands in reverse order while inverting all commands
            circuit_.foreach_r_instruction([&original = original_](const instruction_t& inst) {
                auto op = inst.adjoint();
                if (op) {
                    original.apply_operator(*op, inst.qubits(), inst.cbits());
                } else {
                    // TODO(dnguyen): Think of error handling!
                }
            });
        }
    }

    operator circuit_t&() {
        return circuit_;
    }

 private:
    bool processed_;
    circuit_t& original_;
    circuit_t circuit_;
};
}  // namespace mindquantum

#endif /* DAGGERCIRCUIT_HPP */
