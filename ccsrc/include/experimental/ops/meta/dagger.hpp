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

#ifndef DAGGER_OP_HPP
#define DAGGER_OP_HPP

#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

#include <tweedledum/IR/Operator.h>
#include <tweedledum/IR/OperatorTraits.h>

namespace mindquantum::ops {
namespace td = tweedledum;

class DaggerOperation {
 public:
    template <typename OpT>
    explicit DaggerOperation(OpT&& op) : op_(std::forward<OpT>(op)) {
    }

    static constexpr std::string_view kind() {
        return "projectq.daggeroperation";
    }

    uint32_t num_targets() const {
        return op_.num_targets();
    }

    td::Operator adjoint() const {
        return op_;
    }

    std::optional<td::UMatrix> matrix() const {
        const auto m = op_.matrix();
        if (m) {
            return m.value().inverse();
        } else {
            return std::nullopt;
        }
    }

    bool operator==(const DaggerOperation& other) const {
        return op_ == other.op_;
    }

 private:
    td::Operator op_;
};
}  // namespace mindquantum::ops

#endif /* DAGGER_OP_HPP */
