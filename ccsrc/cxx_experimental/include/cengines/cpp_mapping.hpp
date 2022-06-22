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

#ifndef CPP_MAPPING_HPP
#define CPP_MAPPING_HPP

#include <map>

// clang-format off
#include "details/macros_conv_begin.hpp"
// clang-format on

namespace mindquantum::cengines::cpp {
using mapping_t = std::map<unsigned int, unsigned int>;

class LinearMapper {
 public:
    DECLARE_ATTRIBUTE(mapping_t, _current_mapping);
    DECLARE_ATTRIBUTE(unsigned int, num_qubits);
    DECLARE_ATTRIBUTE(unsigned int, num_mappings);
    DECLARE_ATTRIBUTE(unsigned int, storage);

    DECLARE_ATTRIBUTE(bool, cyclic);
};

class GridMapper {
 public:
    DECLARE_ATTRIBUTE(mapping_t, _current_mapping);
    DECLARE_ATTRIBUTE(unsigned int, num_qubits);
    DECLARE_ATTRIBUTE(unsigned int, num_mappings);
    DECLARE_ATTRIBUTE(unsigned int, storage);

    DECLARE_ATTRIBUTE(unsigned int, num_rows);
    DECLARE_ATTRIBUTE(unsigned int, num_columns);
};
}  // namespace mindquantum::cengines::cpp

#include "details/macros_conv_end.hpp"

#endif /* CPP_MAPPING_HPP */
