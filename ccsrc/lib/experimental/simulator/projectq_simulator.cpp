//   Copyright 2022 <Huawei Technologies Co., Ltd>
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

#include "experimental/simulator/projectq_simulator.hpp"

#include <algorithm>
#include <cstdint>

#include "ops/gates/qubit_operator.hpp"
#include "projectq/backends/_sim/_cppkernels/simulator.hpp"

#include "experimental/core/types.hpp"
#include "experimental/ops/gates.hpp"

namespace mindquantum::simulation::projectq {
Simulator::Simulator(uint32_t seed) : base_t{seed} {
}

// =============================================================================

bool Simulator::has_qubit(const qubit_t& qubit) const {
    return sim_.has_qubit(qubit_id_t{qubit});
}

bool Simulator::allocate_qubits(const qubits_t& qubits) {
    for (const auto& qubit : qubits) {
        if (!has_qubit(qubit)) {
            sim_.allocate_qubit(qubit_id_t{qubit});
        }
    }
    return true;
}
// TODO(explain): Do we have better solution that do not do data type conversion for matrix every time we run
// instruction
bool Simulator::run_instruction(const instruction_t& inst) {
    using Complex = std::complex<double>;
    using c_type = std::complex<double>;
    using ArrayType = std::vector<c_type, ::projectq::aligned_allocator<c_type, 64>>;
    using MatrixType = std::vector<ArrayType>;

    std::vector<qubit_id_t> target_ids;
    inst.foreach_target([&target_ids](const qubit_t& qubit) { target_ids.push_back(qubit); });
    std::vector<qubit_id_t> control_ids;
    inst.foreach_control([&control_ids](const qubit_t& qubit) { control_ids.push_back(qubit); });

    MatrixType gate_matrix;

    if (inst.is_one<ops::Measure>()) {
        auto qubits = std::vector<qubit_id_t>{};
        for (const auto& qubit : inst.qubits()) {
            qubits.emplace_back(qubit_id_t{qubit});
        }
        std::vector<bool> res;
        sim_.measure_qubits(qubits, res);
        return true;
    } else if (inst.is_one<ops::X, ops::Y, ops::Z, ops::S, ops::Sdg, ops::T, ops::Tdg, ops::P, ops::H, ops::Rx, ops::Ry,
                           ops::Rz, ops::Sx, ops::Sxdg, ops::Ph>()) {
        const auto matrix = inst.matrix().value();
        assert(std::size(matrix) == 4);
        gate_matrix.emplace_back(ArrayType{matrix(0, 0), matrix(0, 1)});
        gate_matrix.emplace_back(ArrayType{matrix(1, 0), matrix(1, 1)});
    } else if (inst.is_one<ops::Swap, ops::Rxx, ops::Ryy, ops::Rzz, ops::SqrtSwap>()) {
        const auto matrix = inst.matrix().value();
        assert(std::size(matrix) == 16);
        gate_matrix.emplace_back(ArrayType{matrix(0, 0), matrix(0, 1), matrix(0, 2), matrix(0, 3)});
        gate_matrix.emplace_back(ArrayType{matrix(1, 0), matrix(1, 1), matrix(1, 2), matrix(1, 3)});
        gate_matrix.emplace_back(ArrayType{matrix(2, 0), matrix(2, 1), matrix(2, 2), matrix(2, 3)});
        gate_matrix.emplace_back(ArrayType{matrix(3, 0), matrix(3, 1), matrix(3, 2), matrix(3, 3)});
    } else if (inst.is_one<ops::Barrier>()) {
        // Silently ignore gates
        return true;
    } else {
        std::cerr << "Simulator doesn't support gate type:\n";
        std::cerr << inst.kind() << std::endl;
        std::cerr << "If applicable: Consider adding a "
                  << "CppDecomposer Engine\n";
        return false;
    }

    sim_.apply_controlled_gate(gate_matrix, target_ids, control_ids);
    sim_.run();

    return true;
}

// =============================================================================
}  // namespace mindquantum::simulation::projectq
