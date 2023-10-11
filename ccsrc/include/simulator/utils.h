/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef INCLUDE_QUANTUMSTATE_UTILS_HPP
#define INCLUDE_QUANTUMSTATE_UTILS_HPP

#include <cassert>
#include <vector>

#include "core/mq_base_types.h"

namespace mindquantum::sim {
index_t QIndexToMask(qbits_t objs);
PauliMask GenPauliMask(const std::vector<PauliWord>& pws);
struct SingleQubitGateMask {
    qbit_t q0 = 0;
    qbits_t ctrl_qubits{};
    index_t obj_mask = static_cast<uint64_t>(0);
    index_t ctrl_mask = static_cast<uint64_t>(0);
    index_t obj_high_mask = static_cast<uint64_t>(0);
    index_t obj_low_mask = static_cast<uint64_t>(0);

    SingleQubitGateMask(const qbits_t& obj_qubits, const qbits_t& ctrl_qubits);
};
struct DoubleQubitGateMask {
    qbit_t q_min = 0;
    qbit_t q_max = 0;
    qbits_t ctrl_qubits{};
    index_t obj_min_mask = static_cast<uint64_t>(0);
    index_t obj_max_mask = static_cast<uint64_t>(0);
    index_t obj_mask = static_cast<uint64_t>(0);
    index_t ctrl_mask = static_cast<uint64_t>(0);
    index_t obj_high_mask = static_cast<uint64_t>(0);
    index_t obj_rev_high_mask = static_cast<uint64_t>(0);
    index_t obj_low_mask = static_cast<uint64_t>(0);
    index_t obj_rev_low_mask = static_cast<uint64_t>(0);

    DoubleQubitGateMask(const qbits_t& obj_qubits, const qbits_t& ctrl_qubits);
};

#define SHIFT_BIT_TWO(obj_low_mask, obj_rev_low_mask, obj_high_mask, obj_rev_high_mask, ori, des)                      \
    do {                                                                                                               \
        (des) = (((ori) & (obj_rev_low_mask)) << 1) + ((ori) & (obj_low_mask));                                        \
        (des) = (((des) & (obj_rev_high_mask)) << 1) + ((des) & (obj_high_mask));                                      \
    } while (0)
}  // namespace mindquantum::sim

#endif
