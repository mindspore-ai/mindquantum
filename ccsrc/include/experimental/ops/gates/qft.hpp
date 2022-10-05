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

#ifndef QFT_HPP
#define QFT_HPP

#include <string_view>

#include "experimental/ops/meta/dagger.hpp"

namespace mindquantum::ops {
class QFT {
 public:
    using non_const_num_targets = void;

    static constexpr std::string_view kind() {
        return "projectq.qft";
    }

    explicit QFT(uint32_t num_targets) : num_targets_(num_targets) {
    }

    td::Operator adjoint() const {
        return DaggerOperation(*this);
    }

    uint32_t num_targets() const {
        return num_targets_;
    }

    bool operator==(const QFT& other) const {
        return num_targets_ == other.num_targets_;
    }

 private:
    uint32_t num_targets_;
};
}  // namespace mindquantum::ops

#endif /* QFT_HPP */
