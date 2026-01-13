/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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

#ifndef INCLUDE_SIMULATOR_MPS_COMPILER_MPS_PROGRAM_HPP
#define INCLUDE_SIMULATOR_MPS_COMPILER_MPS_PROGRAM_HPP

#include <array>
#include <string>
#include <utility>
#include <vector>

#include "core/mq_base_types.h"

namespace mindquantum::sim::mps::compiler {
template <typename complex_t>
struct Op1Q {
    qbit_t pos = 0;
    std::array<complex_t, 4> mat{};  // row-major 2x2
};

template <typename complex_t>
struct Op2Q {
    qbit_t bond = 0;                  // left site index (bond between bond and bond+1)
    std::array<complex_t, 16> mat{};  // row-major 4x4, basis |00>,|01>,|10>,|11> (left,right)
};

struct OpMeasure {
    std::string key;
    qbit_t pos = 0;  // physical position to measure
    bool reset = false;
    index_t reset_to = 0;
};

template <typename complex_t>
struct Stage {
    std::vector<Op1Q<complex_t>> op1;
    std::vector<Op2Q<complex_t>> op2;
    std::vector<OpMeasure> meas;
};

template <typename complex_t>
struct MPSProgram {
    std::vector<Stage<complex_t>> stages;
    qbits_t pos_of_logical_final;
    qbits_t logical_at_pos_final;
};
}  // namespace mindquantum::sim::mps::compiler

#endif  // INCLUDE_SIMULATOR_MPS_COMPILER_MPS_PROGRAM_HPP
