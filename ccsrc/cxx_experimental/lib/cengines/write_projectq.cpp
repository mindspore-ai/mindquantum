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

#include "cengines/write_projectq.hpp"

#include <fstream>
#include <map>
#include <string>

#include <tweedledum/IR/Qubit.h>
#include <tweedledum/Operators/Ising.h>
#include <tweedledum/Operators/Standard.h>

#include "ops/gates.hpp"

namespace mindquantum {
std::string to_string(std::size_t qubit_id) {
    return fmt::format("Qureg[{}]", qubit_id);
}
std::string to_string(const std::vector<std::size_t>& qubit_ids) {
    std::string result("Qureg[");

    auto start_id = qubit_ids.front();
    auto count(0UL);
    // print successive qubit_ids n, n + 1, ... n + k as "n-(n + k)"
    for (const auto& qubit_id : qubit_ids) {
        if (qubit_id == start_id + count) {
            ++count;
        } else {
            if (count > 1) {
                result += fmt::format("{}-{}, ", start_id, start_id + count - 1);
            } else {
                result += fmt::format("{}, ", start_id);
            }
            start_id = qubit_id;
            count = 1;
        }
    }
    if (count > 1) {
        result += fmt::format("{}-{}", start_id, start_id + count - 1);
    } else {
        result += fmt::format("{}", start_id);
    }

    result += "]";
    return result;
}
std::string to_string(ops::QubitOperator const& qb_op, std::vector<std::size_t> const& targets) {
    std::string result = "";
    using ComplexTermsDict = std::map<std::vector<std::pair<uint32_t, char>>, std::complex<double>>;
    ComplexTermsDict terms = qb_op.get_terms();
    bool first_term = true;
    for (auto& term : terms) {
        if (!first_term) {
            result += " +\n";
        }
        if (abs(term.second.imag()) < 1e-9) {
            result += fmt::format("{}", term.second.real());
        } else if (abs(term.second.real()) < 1e-9) {
            result += fmt::format("{}j", term.second.imag());
        } else {
            result += fmt::format("({}{}j)", term.second.real(), term.second.imag());
        }
        for (auto& pauli : term.first) {
            result += fmt::format(" {}{}", pauli.second, targets.at(pauli.first));
        }
        first_term = false;
    }
    return result;
}

namespace details {
static std::map<std::string, std::string, std::less<>> kind_map = {
    {std::string(ops::X::kind()), "X"},
    {std::string(ops::Y::kind()), "Y"},
    {std::string(ops::Z::kind()), "Z"},
    {std::string(ops::S::kind()), "S"},
    {std::string(ops::Sdg::kind()), "Sdg"},
    {std::string(ops::T::kind()), "T"},
    {std::string(ops::Tdg::kind()), "Tdg"},
    {std::string(ops::H::kind()), "H"},
    {std::string(ops::Swap::kind()), "Swap"},
    {std::string(ops::P::kind()), "P"},
    {std::string(ops::Sx::kind()), "Sx"},
    {std::string(ops::Rx::kind()), "Rx"},
    {std::string(ops::Ry::kind()), "Ry"},
    {std::string(ops::Rz::kind()), "Rz"},
    {std::string(ops::Rxx::kind()), "Rxx"},
    {std::string(ops::Ryy::kind()), "Ryy"},
    {std::string(ops::Rzz::kind()), "Rzz"},
    {std::string(ops::Entangle::kind()), "Entangle"},
    {std::string(ops::Measure::kind()), "Measure"},
    {std::string(ops::Ph::kind()), "Ph"},
};
}  // namespace details

std::string to_string(std::string_view kind) {
    if (auto it = details::kind_map.find(kind); it != std::end(details::kind_map)) {
        return it->second;
    } else {
        return std::string(kind);
    }
}

std::string to_string(const td::Instruction& inst) {
    const auto& kind = inst.kind();
    if (auto it = details::kind_map.find(kind); it != std::end(details::kind_map)) {
        if (it->first == td::Op::P::kind()) {
            return fmt::format("{}({})", it->second, inst.cast<td::Op::P>().angle());
        } else if (it->first == td::Op::Rx::kind()) {
            return fmt::format("{}({})", it->second, inst.cast<td::Op::Rx>().angle());
        } else if (it->first == td::Op::Ry::kind()) {
            return fmt::format("{}({})", it->second, inst.cast<td::Op::Ry>().angle());
        } else if (it->first == td::Op::Rz::kind()) {
            // NB: see definition of td::Op::Rz for 2 factor
            return fmt::format("{}({})", it->second, 2 * inst.cast<td::Op::Rz>().angle());
        } else if (it->first == td::Op::Rxx::kind()) {
            // NB: see definition of td::Op::Rxx for 2 factor
            return fmt::format("{}({})", it->second, 2 * inst.cast<td::Op::Rxx>().angle());
        } else if (it->first == td::Op::Ryy::kind()) {
            // NB: see definition of td::Op::Ryy for 2 factor
            return fmt::format("{}({})", it->second, 2 * inst.cast<td::Op::Ryy>().angle());
        } else if (it->first == td::Op::Rzz::kind()) {
            return fmt::format("{}({})", it->second, inst.cast<td::Op::Rzz>().angle());
        } else if (it->first == ops::Ph::kind()) {
            return fmt::format("{}({})", it->second, inst.cast<ops::Ph>().angle());
        } else {
            return it->second;
        }
    } else {
        return std::string(kind);
    }
}

void write_projectq(const td::Instruction& inst, std::ostream& os) {
    using qubit_t = tweedledum::Qubit;

    if (inst.is_a<ops::Measure>()) {
        assert(inst.num_controls() == 0);
        os << "Measure | " << to_string(inst.qubit(0)) << std::endl;
        return;
    }

    std::vector<std::size_t> controls;
    std::vector<std::size_t> complemented_controls;

    inst.foreach_control([&](const qubit_t& control) {
        controls.emplace_back(control);
        if (control.polarity() == qubit_t::negative) {
            os << fmt::format("X | {}\n", to_string(control)) << std::endl;
            complemented_controls.emplace_back(control);
        }
    });

    std::vector<std::size_t> targets;
    inst.foreach_target([&](auto const& target) { targets.emplace_back(target); });

    os << fmt::format("{}", std::string(inst.num_controls(), 'C'));

    if (inst.is_a<ops::QubitOperator>()) {
        os << to_string(inst.cast<ops::QubitOperator>(), targets);
    } else if (inst.is_a<ops::TimeEvolution>()) {
        os << fmt::format("exp({}j * ({}))", -inst.cast<ops::TimeEvolution>().get_time(),
                          to_string(inst.cast<ops::TimeEvolution>().get_hamiltonian(), targets));
    } else {
        os << to_string(inst);
    }

    os << " | ";

    if (std::empty(controls)) {
        os << to_string(targets) << std::endl;
    } else {
        os << "( " << to_string(controls) << ", " << to_string(targets) << " )" << std::endl;
    }

    for (const auto& control : complemented_controls) {
        os << fmt::format("X | {}\n", to_string(control)) << std::endl;
    }
}

void write_projectq(const td::Circuit& circuit, std::string_view filename) {
    std::ofstream os(std::string(filename), std::ofstream::out);
    write_projectq(circuit, os);
}
}  // namespace mindquantum
