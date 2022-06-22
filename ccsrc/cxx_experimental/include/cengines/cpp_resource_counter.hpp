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

#ifndef CPP_RESOURCE_COUNTER_HPP
#define CPP_RESOURCE_COUNTER_HPP

#include <map>
#include <optional>
#include <string_view>
#include <tuple>
#include <utility>

#include <tweedledum/IR/Circuit.h>

namespace mindquantum::cengines {
//! C++ equivalent to projectq.backends.ResourceCounter
/*!
 * Prints all gate classes and specific gates it encountered
 * (cumulative over several flushes)
 */
struct ResourceCounter {
    using param_t = std::optional<double>;
    using ctrl_count_t = std::size_t;
    using class_desc_t = std::pair<std::string_view, ctrl_count_t>;
    using gate_desc_t = std::tuple<std::string_view, param_t, ctrl_count_t>;

    void add_gate_count(std::string_view kind, param_t param, std::size_t n_controls, std::size_t count);

    //! Add gates in Network to gate (class) counts
    void add_gate_counts(const tweedledum::Circuit& network);

    // TODO(dnguyen): calculate max_width properly!
    std::size_t max_width_;
    std::map<class_desc_t, std::size_t> gate_class_counts_;
    std::map<gate_desc_t, std::size_t> gate_counts_;

    // Used for Python interactions
    void* origin_;
};
}  // namespace mindquantum::cengines

#endif /* CPP_RESOURCE_COUNTER_HPP */
