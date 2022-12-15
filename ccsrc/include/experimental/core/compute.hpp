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

#ifndef COMPUTE_HPP
#define COMPUTE_HPP

#include <algorithm>
#include <memory>

#include <tweedledum/IR/Circuit.h>
#include <tweedledum/Passes/Utility/shallow_duplicate.h>

#include "experimental/core/types.hpp"
#include "experimental/ops/gates/invalid.hpp"

namespace mindquantum::cengines {
//! Circuit wrapper class that implements the compute-uncompute pattern
class ComputeCircuit {
 public:
    //! Constructor
    /*!
     * \param original A quantum circuit
     */
    explicit ComputeCircuit(circuit_t& original)  // NOLINT(runtime/references)
        : original_(original)
        , do_compute_(true)
        , no_bits_added_(true)
        , compute_(tweedledum::shallow_duplicate(original)) {
    }

    //! Destructor
    /*!
     * Only when the ComputeCircuit object is being destroyed are the instructions added to it so far will be
     * transferred to the original quantum circuit object.
     */
    ~ComputeCircuit() {
        no_bits_added_ &= (original_.num_qubits() == non_compute_.num_qubits()
                           && original_.num_cbits() == non_compute_.num_cbits());
        if (!no_bits_added_) {
            add_missing_qubits_cbits_(non_compute_, original_);
        }
        compute_.foreach_instruction(
            [&original = original_](const instruction_t& inst) { original.apply_operator(inst); });
        non_compute_.foreach_instruction(
            [&original = original_](const instruction_t& inst) { original.apply_operator(inst); });
        compute_.foreach_r_instruction([&original = original_](const instruction_t& inst) {
            if (const auto& op = inst.adjoint(); op) {
                original.apply_operator(op.value(), inst.qubits(), inst.cbits());
            } else {
                original.apply_operator(ops::Invalid(inst.num_qubits()), inst.qubits(), inst.cbits());
            }
        });
    }

    ComputeCircuit(const ComputeCircuit&) = delete;
    ComputeCircuit(ComputeCircuit&&) = default;
    ComputeCircuit& operator=(const ComputeCircuit&) = delete;
    ComputeCircuit& operator=(ComputeCircuit&&) = delete;

    //! For internal-use only
    void done_compute() {
        do_compute_ = false;
        no_bits_added_ = (original_.num_qubits() == compute_.num_qubits()
                          && original_.num_cbits() == compute_.num_cbits());
        non_compute_ = tweedledum::shallow_duplicate(compute_);
    }

    //! Read-write getter to the circuit storing the \e computed instructions
    circuit_t& compute() {
        return compute_;
    }

    //! Read-write getter to the circuit storing the instructions between the compute and uncompute regions
    circuit_t& non_compute() {
        return non_compute_;
    }

 private:
    static void add_missing_qubits_cbits_(const circuit_t& reference,
                                          circuit_t& circuit) {  // NOLINT(runtime/references)
        reference.foreach_qubit([&circuit](const qubit_t& qubit, std::string_view name) {
            if (qubit >= circuit.num_qubits()) {
#ifndef NDEBUG
                const auto new_qubit =
#endif  // NDEBUG
                    circuit.create_qubit(name);

#ifndef NDEBUG
                assert(qubit == new_qubit);
#endif  // NDEBUG
            }
        });

        reference.foreach_cbit([&circuit](const cbit_t& cbit, std::string_view name) {
            if (cbit > circuit.num_cbits()) {
#ifndef NDEBUG
                const auto new_cbit =
#endif  // NDEBUG
                    circuit.create_cbit(name);
#ifndef NDEBUG
                assert(cbit == new_cbit);
#endif  // NDEBUG
            }
        });
    }

    circuit_t& original_;
    bool do_compute_;
    bool no_bits_added_;
    circuit_t compute_;
    circuit_t non_compute_;
};

namespace details {
//! Helper class to implement WITH_COMPUTE statements in C++
class ComputeCircuitProxy {
 public:
    //! Constructor
    /*!
     * \param compute ComputCircuit object to wrap.
     */
    explicit ComputeCircuitProxy(ComputeCircuit& compute) : compute_(compute) {
    }

    //! Destructor
    ~ComputeCircuitProxy() {
        compute_.done_compute();
    }

    ComputeCircuitProxy(const ComputeCircuitProxy&) = delete;
    ComputeCircuitProxy(ComputeCircuitProxy&&) = delete;
    ComputeCircuitProxy& operator=(const ComputeCircuitProxy&) = delete;
    ComputeCircuitProxy& operator=(ComputeCircuitProxy&&) = delete;

    //! Read-write getter to the circuit storing the \e computed instructions
    // NOLINTNEXTLINE(google-explicit-constructor,hicpp-explicit-conversions)
    operator circuit_t&() & {
        return compute_.compute();
    }

 private:
    ComputeCircuit& compute_;
};
}  // namespace details
}  // namespace mindquantum::cengines

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage,readability/braces)
#define MQ_WITH_COMPUTE_IMPL(original, name, unique_name)                                                              \
    mindquantum::cengines::ComputeCircuit unique_name{original};                                                       \
    mindquantum::circuit_t&(name) ((unique_name).non_compute());                                                       \
    if (mindquantum::cengines::details::ComputeCircuitProxy proxy{(unique_name)}; true) {                              \
        auto&(unique_name) = static_cast<mindquantum::circuit_t&>(proxy);
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define MQ_WITH_COMPUTE(original, name) MQ_WITH_COMPUTE_IMPL(original, name, MQ_UNIQUE_NAME(name))
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define MQ_WITH_COMPUTE_END }
#endif /* COMPUTE_HPP */
