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

#ifndef TEST_CORE_UTILS_HPP
#define TEST_CORE_UTILS_HPP

#include <sstream>
#include <string>
#include <vector>

#include <catch2/catch.hpp>

#include "tweedledum/IR/Instruction.h"

#include "../utils.hpp"

template <typename Range>
class InstructionPtrRange : public Catch::MatcherBase<Range> {
 public:
    explicit InstructionPtrRange(const Range& range) : range_{range} {
    }

    bool match(const Range& other) const override {
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

template <typename T>
auto Equals(const std::vector<T*>& range) {
    return InstructionPtrRange(range);
}

#endif /* TEST_CORE_UTILS_HPP */
