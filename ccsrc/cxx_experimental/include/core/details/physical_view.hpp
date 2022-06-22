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

#ifndef PHYSICAL_VIEW_HPP
#define PHYSICAL_VIEW_HPP

#include "core/config.hpp"

#include "core/circuit_block.hpp"
#include "core/circuit_manager.hpp"
#include "core/details/external_view.hpp"

namespace mindquantum::details {
class PhysicalView {
 public:
    using manager_t = CircuitManager;
    using block_t = CircuitBlock;

    explicit PhysicalView(const manager_t& manager) : manager_{manager} {
    }

    template <typename Fn>
    void foreach_instruction(Fn&& fn) const {
        if (manager_.has_mapping()) {
            manager_.foreach_block([fn](const block_t& block) { block.foreach_instruction(fn); });
        } else {
            ExternalView(manager_).foreach_instruction(fn);
        }
    }

    template <typename Fn>
    void foreach_r_instruction(Fn&& fn) const {
        if (manager_.has_mapping()) {
            manager_.foreach_r_block([fn](const block_t& block) { block.foreach_r_instruction(fn); });
        } else {
            ExternalView(manager_).foreach_r_instruction(fn);
        }
    }

 private:
    const manager_t& manager_;
};
}  // namespace mindquantum::details

#endif /* PHYSICAL_VIEW_HPP */
