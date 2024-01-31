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
#ifndef SIMULATOR_STABILIZER_STABILIZER_H_
#define SIMULATOR_STABILIZER_STABILIZER_H_
#include <cstddef>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "math/longbits/longbits.h"
#include "ops/basic_gate.h"
#include "ops/gate_id.h"
namespace mindquantum::stabilizer {
using stab_circ_t = VT<std::shared_ptr<BasicGate>>;

class StabilizerTableau {
    using RndEngine = std::mt19937;

 public:
    StabilizerTableau() = default;
    explicit StabilizerTableau(size_t n_qubits, unsigned seed = 42);

    // -----------------------------------------------------------------------------
    void SetSeed(unsigned new_seed);

    std::string TableauToString() const;

    std::string StabilizerToString() const;

    size_t GetElement(size_t row, size_t col) const;
    void SetElement(size_t row, size_t col, size_t val);
    void Reset();
    void ApplyX(size_t idx);
    void ApplyY(size_t idx);
    void ApplyZ(size_t idx);
    void ApplySGate(size_t idx);
    void ApplySdag(size_t idx);
    void ApplyV(size_t idx);
    void ApplyW(size_t idx);
    void ApplyH(size_t idx);
    void ApplyCNOT(size_t obj, size_t ctrl);
    size_t ApplyMeasurement(size_t idx);
    size_t ApplyGate(GateID g_id, size_t obj, size_t ctrl = 0);
    std::map<std::string, int> ApplyCircuit(const stab_circ_t& circ);
    VT<unsigned> Sampling(const stab_circ_t& circ, size_t shots, const MST<size_t>& key_map, unsigned int seed) const;
    VT<unsigned> SamplingMeasurementEndingWithoutNoise(const stab_circ_t& circ, size_t shots,
                                                       const MST<size_t>& key_map, unsigned int seed) const;
    void RowSum(size_t h, size_t i);
    stab_circ_t Decompose() const;
    VVT<size_t> TableauToVector() const;
    bool operator==(const StabilizerTableau& other) const;

 private:
    size_t n_qubits = 1;
    std::vector<LongBits> table;
    LongBits phase;
    unsigned seed = 0;
    RndEngine rnd_eng_;
    std::function<double()> rng_;
};

stab_circ_t CliffordCircDagger(const stab_circ_t& circ);
}  // namespace mindquantum::stabilizer
#endif
