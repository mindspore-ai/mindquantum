/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
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
#include "simulator/stabilizer/random_benchmarking.h"

#include <random>

#include "simulator/stabilizer/stabilizer.h"
namespace mindquantum::stabilizer {
VT<StabilizerTableau> SingleQubitRBCircuit(size_t len, int seed) {
    VT<StabilizerTableau> out{};
    if (len < 2) {
        return out;
    }
    std::mt19937 gen(seed);
    std::uniform_int_distribution<> dis(0, 23);
    for (size_t i = 1; i < len; i++) {
        out.push_back(QuerySingleQubitCliffordElem(dis(gen)));
    }
    auto first_stab = out[0];
    for (size_t i = 1; i < len - 1; i++) {
        first_stab.ApplyCircuit(out[i].Decompose());
    }
    auto lhs = first_stab.Decompose();
    auto rhs = CliffordCircDagger(lhs);
    auto last_stab = StabilizerTableau(1);
    last_stab.ApplyCircuit(rhs);
    out.push_back(last_stab);
    return out;
}

VT<StabilizerTableau> DoubleQubitsRBCircuit(size_t len, int seed) {
    VT<StabilizerTableau> out;
    if (len < 2) {
        return out;
    }
    std::mt19937 gen(seed);
    std::uniform_int_distribution<> dis(0, 11519);
    for (size_t i = 1; i < len; i++) {
        out.push_back(QueryDoubleQubitsCliffordElem(dis(gen)));
    }
    auto first_stab = out[0];
    for (size_t i = 1; i < len - 1; i++) {
        first_stab.ApplyCircuit(out[i].Decompose());
    }
    auto lhs = first_stab.Decompose();
    auto rhs = CliffordCircDagger(lhs);
    auto last_stab = StabilizerTableau(2);
    last_stab.ApplyCircuit(rhs);
    out.push_back(last_stab);
    return out;
}
}  // namespace mindquantum::stabilizer
