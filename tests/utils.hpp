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

#ifndef TESTS_UTILS_HPP
#define TESTS_UTILS_HPP

#include <sstream>
#include <string>

#include <catch2/catch.hpp>
#include <tweedledum/IR/Circuit.h>
#include <tweedledum/IR/Instruction.h>

#include "ops/gates.hpp"
#include "ops/meta/dagger.hpp"

namespace Catch {
template <>
struct StringMaker<tweedledum::Circuit> {
    static std::string convert(const tweedledum::Circuit& circuit) {
        namespace ops = mindquantum::ops;

        std::ostringstream ssout;
        ssout << "Circuit (";
        circuit.foreach_qubit(
            [&ssout, &circuit](const tweedledum::Qubit& qubit) { ssout << "Q[" << circuit.name(qubit) << "], "; });
        circuit.foreach_cbit(
            [&ssout, &circuit](const tweedledum::Cbit& cbit) { ssout << "C[" << circuit.name(cbit) << "], "; });
        ssout << "):\n";

        circuit.foreach_instruction([&ssout](const tweedledum::Instruction& inst) {
            if (inst.kind() == mindquantum::ops::DaggerOperation::kind()) {
                ssout << "  Dagger(" << inst.adjoint().value().kind() << ")";
            } else {
                ssout << "  " << inst.kind();
            }

            if (inst.is_a<ops::P>()) {
                ssout << "(" << inst.cast<ops::P>().angle() << ")";
            } else if (inst.is_a<ops::Ph>()) {
                ssout << "(" << inst.cast<ops::Ph>().angle() << ")";
            } else if (inst.is_a<ops::Rx>()) {
                ssout << "(" << inst.cast<ops::Rx>().angle() << ")";
            } else if (inst.is_a<ops::Rxx>()) {
                ssout << "(" << inst.cast<ops::Rxx>().angle() << ")";
            } else if (inst.is_a<ops::Ry>()) {
                ssout << "(" << inst.cast<ops::Ry>().angle() << ")";
            } else if (inst.is_a<ops::Ryy>()) {
                ssout << "(" << inst.cast<ops::Ryy>().angle() << ")";
            } else if (inst.is_a<ops::Rz>()) {
                ssout << "(" << inst.cast<ops::Rz>().angle() << ")";
            } else if (inst.is_a<ops::Rzz>()) {
                ssout << "(" << inst.cast<ops::Rzz>().angle() << ")";
            }

            ssout << " [";
            inst.foreach_qubit([&ssout](const tweedledum::Qubit& wire) { ssout << 'Q' << int(wire) << ','; });
            inst.foreach_cbit([&ssout](const tweedledum::Cbit& wire) { ssout << 'C' << int(wire) << ','; });

            ssout << "]\n";
        });

        return ssout.str();
    }
};
}  // namespace Catch

struct CircuitEquals : Catch::MatcherBase<tweedledum::Circuit> {
    explicit CircuitEquals(const tweedledum::Circuit& circuit) : ref_circuit{circuit} {
    }

    bool match(const tweedledum::Circuit& circuit) const override {
        using namespace std::literals::string_literals;

        if (ref_circuit.num_wires() != circuit.num_wires()) {
            return false;
        }

        for (auto i(0UL); i < ref_circuit.num_qubits(); ++i) {
            const auto ref_wire = ref_circuit.qubit(i);
            const auto wire = circuit.qubit(i);
            if (ref_wire != wire) {
                return false;
            }
        }
        for (auto i(0UL); i < ref_circuit.num_cbits(); ++i) {
            const auto ref_wire = ref_circuit.cbit(i);
            const auto wire = circuit.cbit(i);
            if (ref_wire != wire) {
                return false;
            }
        }

        if (std::size(ref_circuit) != std::size(circuit)) {
            return false;
        }
        for (auto i(0UL); i < std::size(ref_circuit); ++i) {
            const auto ref_inst = ref_circuit.instruction(tweedledum::InstRef(i));
            const auto inst = circuit.instruction(tweedledum::InstRef(i));

            std::string ref_kind(ref_inst.kind());
            if (ref_kind == mindquantum::ops::DaggerOperation::kind()) {
                ref_kind = "Dagger("s + std::string(ref_inst.adjoint().value().kind()) + ")"s;
            }
            std::string kind(inst.kind());
            if (kind == mindquantum::ops::DaggerOperation::kind()) {
                kind = "Dagger("s + std::string(inst.adjoint().value().kind()) + ")"s;
            }

            if (ref_kind != kind || ref_inst.num_wires() != inst.num_wires() || ref_inst.qubits() != inst.qubits()
                || ref_inst.cbits() != inst.cbits()) {
                return false;
            }
        }

        return true;
    }
    std::string describe() const override {
        return "Equals: " + ::Catch::Detail::stringify(ref_circuit);
    }

    const tweedledum::Circuit& ref_circuit;
};

inline auto Equals(const tweedledum::Circuit& circuit) {
    return CircuitEquals(circuit);
}

#endif /* TESTS_UTILS_HPP */
