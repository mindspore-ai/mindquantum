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

#ifndef INVALID_OP_HPP
#define INVALID_OP_HPP

#include <cstdint>
#include <string_view>

namespace mindquantum::ops {
class Invalid {
 public:
    static constexpr std::string_view kind() {
        return "projectq.invalid";
    }

    Invalid() = default;
    explicit Invalid(uint64_t num_targets) : num_targets_(num_targets) {
    }

    auto num_targets() const noexcept {
        return num_targets_;
    }

 private:
    uint32_t num_targets_ = 1;
};
}  // namespace mindquantum::ops

#endif /* INVALID_OP_HPP */
