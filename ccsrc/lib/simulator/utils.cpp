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

#include "simulator/utils.hpp"

#include <cassert>
#include <numeric>

#include "simulator/types.hpp"

namespace mindquantum::sim {
index_t QIndexToMask(qbits_t objs) {
    return std::accumulate(objs.begin(), objs.end(), index_t(0), [](index_t a, qbit_t b) { return a + (1UL << b); });
}

PauliMask GenPauliMask(const std::vector<PauliWord> &pws) {
    VT<Index> out = {0, 0, 0, 0, 0, 0};
    for (auto &pw : pws) {
        for (Index i = 0; i < 3; i++) {
            if (static_cast<Index>(pw.second - 'X') == i) {
                out[i] += (1UL << pw.first);
                out[3 + i] += 1;
            }
        }
    }
    return {out[0], out[1], out[2], out[3], out[4], out[5]};
}

SingleQubitGateMask::SingleQubitGateMask(const qbits_t &obj_qubits, const qbits_t &ctrl_qubits) {
    assert(obj_qubits.size() == 1);
    q0 = obj_qubits[0];
    this->ctrl_qubits = ctrl_qubits;
    obj_mask = (1UL << q0);
    ctrl_mask = QIndexToMask(ctrl_qubits);
    for (qbit_t i = 0; i < q0; i++) {
        obj_low_mask = (obj_low_mask << 1) + 1;
    }
    obj_high_mask = ~obj_low_mask;
}

DoubleQubitGateMask::DoubleQubitGateMask(const qbits_t &obj_qubits, const qbits_t &ctrl_qubits) {
    assert(obj_qubits.size() == 2);
    q_min = obj_qubits[0];
    q_max = obj_qubits[1];
    if (q_min > q_max) {
        q_min = obj_qubits[1];
        q_max = obj_qubits[0];
    }
    this->ctrl_qubits = ctrl_qubits;
    obj_min_mask = (1UL << q_min);
    obj_max_mask = (1UL << q_max);
    obj_mask = obj_min_mask + obj_max_mask;
    ctrl_mask = QIndexToMask(ctrl_qubits);
    for (qbit_t i = 0; i < q_min; i++) {
        obj_low_mask = (obj_low_mask << 1) + 1;
    }
    for (qbit_t i = 0; i < q_max; i++) {
        obj_high_mask = (obj_high_mask << 1) + 1;
    }
    obj_rev_low_mask = ~obj_low_mask;
    obj_rev_high_mask = ~obj_high_mask;
}
}  // namespace mindquantum::sim
