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

#ifndef CPP_CORE_HPP
#define CPP_CORE_HPP

#include <complex>
#include <fstream>
#include <map>
#include <string_view>
#include <vector>

#include "core/config.hpp"

#include "cengines/cpp_engine_list.hpp"
#include "core/circuit_manager.hpp"
#include "ops/cpp_command.hpp"
#include "projectq/backends/_sim/_cppkernels/simulator.hpp"

namespace td = tweedledum;

namespace mindquantum::core {
class CppCore {
    /*!
     * \brief Provide ProjectQ's functionality in C++
     */
 public:
    using gate_t = ops::Command::gate_t;
    using engine_list_t = std::vector<cengines::engine_t>;
    using circuit_t = td::Circuit;
    using instruction_t = CircuitManager::instruction_t;

    using Complex = std::complex<double>;
    using c_type = std::complex<double>;
    using ArrayType = std::vector<c_type, ::projectq::aligned_allocator<c_type, 64>>;
    using MatrixType = std::vector<ArrayType>;

    CppCore();

    CppCore(CppCore const&) = delete;

    CppCore operator=(CppCore const&) = delete;

    bool sim_backend() const {
        return simulator_backend_;
    }

    void set_engine_list(const engine_list_t& engine_list);

    // WARN: this function does not take ownership of sim!
    void set_simulator_backend(::projectq::Simulator& sim);

    /*!
     * \brief Allocate a single qubit
     */
    void allocate_qubit(unsigned id);

    //! Apply a ProjectQ command
    void apply_command(const ops::Command& cmd);

    /*!
     * \brief Execute stored gates
     */
    void flush();

    //! Go through all engines in engine_list_
    /*!
     * Gets called in beginning of flush()
     */
    void traverse_engine_list();

    /*!
     * \brief Return measurement outcomes
     * \return A map where the key values are the measured qubits' IDs
     *     and the mapped values are the measurement outcomes
     */
    std::map<unsigned, bool> get_measure_info();

    //! Set output file name (stdout for printing to standard output)
    void set_output_stream(std::string_view file_name);

    /*!
     * \brief Output Quantum circuit to standard output
     * \param format in which to output circuit
     */
    void write(std::string_view format);

    /*!
     * \brief Return state vector
     * \return A map where the key values are the allocated qubits' ids
     *     and the mapped values are their positions in the ket:
     *     |q_n, q_(n-1), ..., q_1, q_0>
     * \return The current state vector of the allocated qubits
     */
    auto cheat();

 protected:
    //! Insert an operation into circuit
    void apply_operation_(const gate_t& gate, const qureg_t& control_qubit_ids, const qureg_t& qubit_ids);

    bool simulator_backend_;
    bool has_new_operations_;

    ::projectq::Simulator* sim_;

    CircuitManager circuit_manager_;

    std::map<qubit_id_t, bool> measure_info_;
    std::map<std::string_view, MatrixType> custom_map_;
    std::vector<QubitID> deallocations_;

    engine_list_t engine_list_;

    std::ofstream filestream;     // Stream to file for set_output_stream()
    std::ostream* output_stream;  // Where to print output
};

inline auto CppCore::cheat() {
    return sim_->cheat();
}
}  // namespace mindquantum::core

#endif /* CPP_CORE_HPP */
