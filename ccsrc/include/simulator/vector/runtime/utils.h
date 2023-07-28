/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
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
#ifndef MQ_SIM_VECTOR_RT
#define MQ_SIM_VECTOR_RT
#include <string>
#include <tuple>

#include "simulator/vector/runtime/rt_gate.h"
#define MAX_QUBIT 64
#define MAX_SEED  65536

namespace mindquantum::sim::rt {
enum class State {
    null,
    W_GATE,
    W_OBJ,
    W_CTRL,
    W_ANG,
    W_M_KEY,
    W_SHOTS,
};

std::tuple<bool, Index> convert_int(const std::string &s, int64_t limit, bool raise_error = true);

std::tuple<bool, double> convert_double(const std::string &s, bool raise_error = true);
}  // namespace mindquantum::sim::rt
#endif
