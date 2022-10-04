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

#ifndef MQ_CATCH2_TWEEDLEDUM_HPP
#define MQ_CATCH2_TWEEDLEDUM_HPP

#include <string>
#include <vector>

#include <tweedledum/IR/Circuit.h>
#include <tweedledum/IR/Instruction.h>

#include <fmt/format.h>

#include "mindquantum/catch2/catch2_fmt_formatter.hpp"

#include "experimental/format/tweedledum.hpp"

#include <catch2/catch_tostring.hpp>
#include <catch2/matchers/catch_matchers_templated.hpp>

// =============================================================================

namespace Catch {
template <>
struct StringMaker<tweedledum::Qubit::Polarity> {
    static std::string convert(tweedledum::Qubit::Polarity value) {
        static const auto& enumInfo = ::Catch::getMutableRegistryHub().getMutableEnumValuesRegistry().registerEnum(
            "tweedledum::Qubit::Polarity", "pos, neg", {tweedledum::Qubit::positive, tweedledum::Qubit::negative});
        return static_cast<std::string>(enumInfo.lookup(static_cast<int>(value)));
    }
};

// -----------------------------------------------------------------------------

template <>
struct StringMaker<tweedledum::Cbit::Polarity> {
    static std::string convert(tweedledum::Cbit::Polarity value) {
        static const auto& enumInfo = ::Catch::getMutableRegistryHub().getMutableEnumValuesRegistry().registerEnum(
            "tweedledum::Cbit::Polarity", "pos, neg", {tweedledum::Cbit::positive, tweedledum::Cbit::negative});
        return static_cast<std::string>(enumInfo.lookup(static_cast<int>(value)));
    }
};

// -----------------------------------------------------------------------------

template <>
struct StringMaker<tweedledum::Cbit> : mindquantum::details::FmtStringMakerBase<tweedledum::Cbit> {};

template <>
struct StringMaker<tweedledum::Qubit> : mindquantum::details::FmtStringMakerBase<tweedledum::Qubit> {};

template <>
struct StringMaker<tweedledum::Instruction> : mindquantum::details::FmtStringMakerBase<tweedledum::Instruction> {};

template <>
struct StringMaker<tweedledum::Circuit> : mindquantum::details::FmtStringMakerBase<tweedledum::Circuit> {};
}  // namespace Catch

// =============================================================================

namespace mindquantum::catch2 {
struct TweedledumCircuitMatcher : Catch::Matchers::MatcherGenericBase {
    explicit TweedledumCircuitMatcher(const tweedledum::Circuit& circuit) : ref_circuit{circuit} {
    }

    bool match(const tweedledum::Circuit& circuit) const {
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

        if (ref_circuit.num_instructions() != circuit.num_instructions()) {
            return false;
        }
        for (auto i(0UL); i < ref_circuit.num_instructions(); ++i) {
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
        return fmt::format("Equals: {}", ref_circuit);
    }

 private:
    const tweedledum::Circuit& ref_circuit;
};

// -----------------------------------------------------------------------------

inline auto Equals(const tweedledum::Circuit& circuit) {
    return TweedledumCircuitMatcher(circuit);
}

// =============================================================================

template <typename Range>
class InstructionPtrRange : public Catch::Matchers::MatcherGenericBase {
 public:
    explicit InstructionPtrRange(const Range& range) : range_{range} {
    }

    bool match(const Range& other) const {
        if (std::size(range_) != std::size(other)) {
            std::cout << "Range size not equal!" << std::endl;
            return false;
        }

        return std::equal(std::begin(range_), std::end(range_), std::begin(other), std::end(other),
                          [](const auto* lhs, const auto* rhs) {
                              return lhs->kind() == rhs->kind() && lhs->qubits() == rhs->qubits();
                          });
    }

    std::string describe() const override {
        std::ostringstream ss;
        ss << "Instructions equal. Reference value: ";
        for (const auto& inst : range_) {
            ss << "(" << inst->kind() << ": ";
            for (const auto& qubit : inst->qubits()) {
                ss << unsigned(qubit) << ", ";
            }
            ss << "), ";
        }
        return ss.str();
    }

 private:
    const Range& range_;
};

// -----------------------------------------------------------------------------

template <typename T>
auto Equals(const std::vector<T*>& range) {
    return InstructionPtrRange(range);
}

// =============================================================================
}  // namespace mindquantum::catch2

#endif /* MQ_CATCH2_TWEEDLEDUM_HPP */
