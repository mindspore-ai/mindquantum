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
#ifndef MQ_SIM_VECTOR_CMD
#define MQ_SIM_VECTOR_CMD
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "ops/basic_gate.h"
#include "ops/gate_id.h"
#include "ops/gates.h"
#include "simulator/vector/detail/cpu_vector_avx_double_policy.h"
#include "simulator/vector/runtime/rt_gate.h"
#include "simulator/vector/runtime/utils.h"

namespace mindquantum::sim::rt {
int cmd(const std::vector<std::string>& args);
int cmd_file(const char* filename);
}  // namespace mindquantum::sim::rt
#endif
