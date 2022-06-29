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

#include "core/cpp_core.hpp"

#include <iostream>
#include <type_traits>

#include <tweedledum/Operators/Ising.h>
#include <tweedledum/Operators/Standard.h>
#include <tweedledum/Passes/Optimization/gate_cancellation.h>
#include <tweedledum/Passes/Optimization/phase_folding.h>

#include "cengines/write_projectq.hpp"
#include "mapping/types.hpp"
#include "ops/gates/allocate.hpp"
#include "ops/gates/deallocate.hpp"
#include "ops/gates/measure.hpp"
#include "ops/gates/ph.hpp"
#include "ops/gates/sqrtswap.hpp"

// #define MEASURE_TIMINGS
#ifdef MEASURE_TIMINGS
#    include <chrono>
#endif  // MEASURE_TIMINGS

namespace mindquantum::core {
CppCore::CppCore() : simulator_backend_(false), has_new_operations_(false), sim_(nullptr), output_stream(&(std::cout)) {
}

void CppCore::allocate_qubit(unsigned id) {
    has_new_operations_ = true;
    if (!std::empty(deallocations_)) {
        flush();  // Make sure deallocations happen before new
                  // allocations
    }

    const QubitID qubit_id(id);

    if (!circuit_manager_.has_qubit(qubit_id)) {
        auto added = circuit_manager_.add_qubit(qubit_id);
        if (!added) {
            throw std::runtime_error("AllocateQubit: Failed to add qubit! Did you reach the device qubit limit?");
        }

        assert(circuit_manager_.has_qubit(qubit_id));

        if (sim_backend()) {
            sim_->allocate_qubit(id);
        }
    } else {
        throw std::runtime_error(
            "AllocateQubit: ID already exists. Qubit IDs should be "
            "unique.");
    }
}

void CppCore::set_engine_list(const engine_list_t& engine_list) {
    engine_list_ = engine_list;
}

void CppCore::set_simulator_backend(::projectq::Simulator& sim) {
    sim_ = &sim;
    simulator_backend_ = true;
}

void CppCore::apply_command(const ops::Command& cmd) {
    has_new_operations_ = true;
    apply_operation_(cmd.get_gate(), cmd.get_control_qubits(), cmd.get_qubits());
}

void CppCore::flush() {
    if (!has_new_operations_) {
        return;
    }

    traverse_engine_list();

    if (sim_backend()) {  // Only simulate if there is a simulator backend
        std::vector<qubit_id_t> target_ids;
        std::vector<qubit_id_t> control_ids;
        MatrixType gate_matrix;

#ifdef MEASURE_TIMINGS
        std::vector<std::pair<std::string, unsigned>> kinds;
        std::vector<decltype(std::chrono::steady_clock::now())> starts;
        std::vector<decltype(std::chrono::steady_clock::now())> qubits;
        std::vector<decltype(std::chrono::steady_clock::now())> matrix;
        std::vector<decltype(std::chrono::steady_clock::now())> sims;
#endif  // MEASURE_TIMINGS

        circuit_manager_.foreach_instruction(
            [&](const instruction_t& inst) {
#ifdef MEASURE_TIMINGS
                starts.emplace_back(std::chrono::steady_clock::now());
#endif  // MEASURE_TIMINGS

                gate_matrix.clear();
                target_ids.clear();
                control_ids.clear();

                inst.foreach_control([&](const auto& control) {
                    control_ids.emplace_back(qubit_id_t(circuit_manager_.translate_id(control)));
                });
                inst.foreach_target([&](const auto& target) {
                    target_ids.emplace_back(qubit_id_t(circuit_manager_.translate_id(target)));
                });

#ifdef MEASURE_TIMINGS
                kinds.emplace_back(inst.kind(), std::size(control_ids));
                qubits.emplace_back(std::chrono::steady_clock::now());
#endif  // MEASURE_TIMINGS

                if (inst.is_one<ops::Measure>()) {
                    const auto measure_results = sim_->measure_qubits_return(target_ids);
                    for (auto i(0UL); i < std::size(target_ids); ++i) {
                        measure_info_.insert({target_ids.at(i), measure_results.at(i)});
                    }
                    return;
                } else if (inst.is_one<td::Op::X, td::Op::Y, td::Op::Z, td::Op::S, td::Op::Sdg, td::Op::T, td::Op::Tdg,
                                       td::Op::P, td::Op::H, td::Op::Rx, td::Op::Ry, td::Op::Rz, td::Op::Sx,
                                       ops::Ph>()) {
                    const auto matrix = inst.matrix().value();
                    assert(std::size(matrix) == 4);
                    gate_matrix.emplace_back(ArrayType{matrix(0, 0), matrix(0, 1)});
                    gate_matrix.emplace_back(ArrayType{matrix(1, 0), matrix(1, 1)});
                } else if (inst.is_one<td::Op::Swap, td::Op::Rxx, td::Op::Ryy, td::Op::Rzz, ops::SqrtSwap>()) {
                    const auto matrix = inst.matrix().value();
                    assert(std::size(matrix) == 16);
                    gate_matrix.emplace_back(ArrayType{matrix(0, 0), matrix(0, 1), matrix(0, 2), matrix(0, 3)});
                    gate_matrix.emplace_back(ArrayType{matrix(1, 0), matrix(1, 1), matrix(1, 2), matrix(1, 3)});
                    gate_matrix.emplace_back(ArrayType{matrix(2, 0), matrix(2, 1), matrix(2, 2), matrix(2, 3)});
                    gate_matrix.emplace_back(ArrayType{matrix(3, 0), matrix(3, 1), matrix(3, 2), matrix(3, 3)});
                } else if (inst.is_one<ops::QubitOperator>()) {
                    const auto& qubit_op = inst.cast<ops::QubitOperator>();
                    assert(std::empty(control_ids));
                    std::vector<ops::QubitOperator::ComplexTerm> terms;
                    std::copy(std::begin(qubit_op.get_terms()), std::end(qubit_op.get_terms()),
                              std::back_inserter(terms));
                    sim_->apply_qubit_operator(terms, target_ids);
                } else if (inst.is_one<ops::TimeEvolution>()) {
                    const auto& time_evol = inst.cast<ops::TimeEvolution>();
                    std::vector<ops::QubitOperator::ComplexTerm> terms;
                    std::copy(std::begin(time_evol.get_hamiltonian().get_terms()),
                              std::end(time_evol.get_hamiltonian().get_terms()), std::back_inserter(terms));
                    sim_->emulate_time_evolution(terms, time_evol.get_time(), target_ids, control_ids);
                } else {
                    std::cerr << "Simulator doesn't support gate type:\n";
                    std::cerr << inst.kind() << std::endl;
                    std::cerr << "If applicable: Consider adding a "
                              << "CppDecomposer Engine\n";
                    assert(0);
                    return;
                }

#ifdef MEASURE_TIMINGS
                matrix.emplace_back(std::chrono::steady_clock::now());
#endif  // MEASURE_TIMINGS

                if (std::empty(gate_matrix)) {
                    std::cerr << "Error: Empty gate used in simulator" << std::endl;
                }
                sim_->apply_controlled_gate(gate_matrix, target_ids, control_ids);
                sim_->run();

#ifdef MEASURE_TIMINGS
                sims.emplace_back(std::chrono::steady_clock::now());
#endif  // MEASURE_TIMINGS
            },
            uncommitted);

#ifdef MEASURE_TIMINGS
        for (auto i(0UL); i < std::size(kinds); ++i) {
            using dur_t = std::chrono::duration<double>;
            const auto& s = starts[i];
            const auto& q = qubits[i];
            const auto& m = matrix[i];
            const auto& sim = sims[i];

            std::cout << kinds[i].first << "/" << kinds[i].second << ": " << dur_t(q - s).count() << ", "
                      << dur_t(m - q).count() << ", " << dur_t(sim - m).count() << ", " << dur_t(sim - s).count()
                      << std::endl;
        }
#endif  // MEASURE_TIMINGS

        // Force flush
        sim_->run();
    }

    circuit_manager_.commit_changes();

    // Deallocate
    for (const auto& id : deallocations_) {
        if (sim_backend()) {
            sim_->deallocate_qubit(qubit_id_t(id));
        }
    }
    circuit_manager_.delete_qubits(deallocations_);
    deallocations_.clear();

    has_new_operations_ = false;
}

void CppCore::traverse_engine_list() {
    for (auto& veng : engine_list_) {
        if (std::holds_alternative<cengines::cpp::LocalOptimizer>(veng)) {
            circuit_manager_.transform(tweedledum::gate_cancellation);
            circuit_manager_.transform(tweedledum::phase_folding);
        } else if (std::holds_alternative<cengines::CppPrinter>(veng)) {
            auto& cpp_printer = std::get<cengines::CppPrinter>(veng);
            cpp_printer.print_output(circuit_manager_.as_projectq(uncommitted), *output_stream);
        } else if (std::holds_alternative<cengines::cpp::TagRemover>(veng)) {
            // Do nothing
        } else if (std::holds_alternative<cengines::cpp::InstructionFilter>(veng)) {
            // Not intended in C++
            std::cerr << "In C++ using CppDecomposer "
                      << "instead of InstructionFilter is recommended\n";
        } else if (std::holds_alternative<cengines::ResourceCounter>(veng)) {
            auto& resource_counter = std::get<cengines::ResourceCounter>(veng);
            circuit_manager_.transform(
                [&engine = resource_counter](const circuit_t& circuit) { engine.add_gate_counts(circuit); });

            /* NB: updating of the original python resource counter object is done within
             *     python/lib/core/core.cpp
             */
        } else if (std::holds_alternative<cengines::CppGraphMapper>(veng)) {
            using circuit_t = CircuitManager::circuit_t;
            using mapping_t = CircuitManager::mapping_t;

            auto& graph_mapper = std::get<cengines::CppGraphMapper>(veng);

            circuit_manager_.apply_mapping(graph_mapper);
        } else if (std::holds_alternative<cengines::cpp::LinearMapper>(veng)) {
            using circuit_t = CircuitManager::circuit_t;
            using mapping_t = CircuitManager::mapping_t;

            const auto linear_mapper = std::get<cengines::cpp::LinearMapper>(veng);

            cengines::CppGraphMapper mapper(mapping::sabre_config{});
            mapper.path_device(linear_mapper.get_num_qubits(), linear_mapper.get_cyclic());
            veng = std::move(mapper);
            assert(std::holds_alternative<cengines::CppGraphMapper>(veng));

            auto& graph_mapper = std::get<cengines::CppGraphMapper>(veng);

            circuit_manager_.apply_mapping(graph_mapper);
        } else if (std::holds_alternative<cengines::cpp::GridMapper>(veng)) {
            using circuit_t = CircuitManager::circuit_t;
            using mapping_t = CircuitManager::mapping_t;

            const auto grid_mapper = std::get<cengines::cpp::GridMapper>(veng);

            cengines::CppGraphMapper mapper(mapping::sabre_config{});
            mapper.grid_device(grid_mapper.get_num_columns(), grid_mapper.get_num_rows());
            veng = std::move(mapper);
            assert(std::holds_alternative<cengines::CppGraphMapper>(veng));

            auto& graph_mapper = std::get<cengines::CppGraphMapper>(veng);

            circuit_manager_.apply_mapping(graph_mapper);
        }
    }
}

std::map<unsigned, bool> CppCore::get_measure_info() {
    const auto tmp_info_ = measure_info_;
    measure_info_.clear();
    return tmp_info_;
}

void CppCore::set_output_stream(std::string_view file_name) {
    if (filestream.is_open()) {
        filestream.close();
    }
    if (file_name == "cout" || file_name == "std::cout" || file_name == "stdout") {
        output_stream = &(std::cout);
    } else {
        filestream.open(file_name.data());
        output_stream = &filestream;
    }
}

void CppCore::write(std::string_view format) {
    if (format == "projectq") {
        write_projectq(circuit_manager_.as_projectq(committed), *output_stream);
    } else {
        *output_stream << "Unrecognized format: " << format << "\n";
    }
}

void CppCore::apply_operation_(const gate_t& gate, const qubit_ids_t& control_qubit_ids, const qubit_ids_t& qubit_ids) {
    using ext_id_t = decltype(circuit_manager_)::ext_id_t;

    if (gate.is_one<ops::Measure>()) {
        circuit_manager_.apply_measurement(qubit_ids[0]);
    } else if (gate.is_one<ops::Deallocate>()) {
        if (circuit_manager_.has_qubit(QubitID{qubit_ids[0]})) {
            deallocations_.emplace_back(qubit_ids[0]);
        } else {
            std::cerr << "Qubit with id: " << qubit_ids[0] << " "
                      << " already deallocated\n";
        }
    } else {
        circuit_manager_.apply_operator(gate, control_qubit_ids, qubit_ids);
    }
}
}  // namespace mindquantum::core
