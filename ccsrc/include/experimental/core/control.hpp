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

#ifndef CONTROLCIRCUIT_HPP
#define CONTROLCIRCUIT_HPP

#include <algorithm>
#include <memory>

#include <tweedledum/Passes/Utility/shallow_duplicate.h>

#include "experimental/core/config.hpp"
#include "experimental/core/types.hpp"

namespace mindquantum::cengines {
class ControlledCircuit {
 public:
    ControlledCircuit(circuit_t& original, const qubits_t& controls)  // NOLINT(runtime/references)
        : processed_(empty(controls))
        , original_(original)
        , circuit_(empty(controls) ? circuit_t{} : tweedledum::shallow_duplicate(original))
        , controls_(controls) {
    }

    ~ControlledCircuit() {
        apply();
    }

    ControlledCircuit(const ControlledCircuit&) = delete;
    ControlledCircuit(ControlledCircuit&&) = default;
    ControlledCircuit& operator=(const ControlledCircuit&) = delete;
    ControlledCircuit& operator=(ControlledCircuit&&) = delete;

    void apply() {
        if (!processed_) {
            assert(!empty(controls_));
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

            circuit_.foreach_instruction([&original = original_, &controls = controls_](const instruction_t& inst) {
                auto qubits = controls;
                for (const auto& qubit : inst.qubits()) {
                    qubits.emplace_back(qubit);
                }
                original.apply_operator(inst, qubits, inst.cbits());
            });
        }
    }

    explicit operator circuit_t&() {
        if (empty(controls_)) {
            return original_;
        }
        return circuit_;
    }

 private:
    bool processed_;
    circuit_t& original_;
    circuit_t circuit_;
    const qubits_t& controls_;
};
}  // namespace mindquantum::cengines

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define MQ_WITH_CONTROL_IMPL(original, name, unique_name, controls)                                                    \
    mindquantum::cengines::ControlledCircuit unique_name{original, controls};                                          \
    if (mindquantum::circuit_t & (name) = static_cast<mindquantum::circuit_t&>((unique_name)); true)
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define MQ_WITH_CONTROL(original, name, controls) MQ_WITH_CONTROL_IMPL(original, name, MQ_UNIQUE_NAME(name), controls)
#endif /* CONTROLCIRCUIT_HPP */
