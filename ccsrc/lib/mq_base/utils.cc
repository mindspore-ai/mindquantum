/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "core/utils.hpp"

namespace mindquantum {
const VT<CT<double>> POLAR = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}};
TimePoint NOW() {
    return std::chrono::steady_clock::now();
}

int TimeDuration(TimePoint start, TimePoint end) {
    auto d = end - start;
    return std::chrono::duration_cast<std::chrono::milliseconds>(d).count();
}

Index GetControlMask(const VT<Index> &ctrls) {
    Index ctrlmask = std::accumulate(ctrls.begin(), ctrls.end(), 0, [&](Index a, Index b) { return a | (1UL << b); });
    return ctrlmask;
}

PauliMask GetPauliMask(const VT<PauliWord> &pws) {
    VT<Index> out = {0, 0, 0, 0, 0, 0};
    for (auto &pw : pws) {
        for (Index i = 0; i < 3; i++) {
            if (static_cast<Index>(pw.second - 'X') == i) {
                out[i] += (1UL << pw.first);
                out[3 + i] += 1;
            }
        }
    }
    PauliMask res = {out[0], out[1], out[2], out[3], out[4], out[5]};
    return res;
}
}  // namespace mindquantum
