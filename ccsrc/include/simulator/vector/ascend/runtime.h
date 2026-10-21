/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef INCLUDE_SIMULATOR_VECTOR_ASCEND_RUNTIME_H_
#define INCLUDE_SIMULATOR_VECTOR_ASCEND_RUNTIME_H_

#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "simulator/vector/ascend/program.h"

namespace mindquantum::sim::ascend {
class Executable {
 public:
    ~Executable();
    size_t TileCount() const;
    size_t CubeTileCount() const;
    size_t InstructionCount() const;
    size_t MetadataBytes() const;

 private:
    friend class State;
    friend class GradientPlan;
    struct Impl;
    explicit Executable(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

// Shared, immutable device metadata. Its context outlives the preparing state.
class Observable {
 public:
    ~Observable();

 private:
    friend class State;
    friend class GradientPlan;
    struct Impl;
    explicit Observable(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

class State {
 public:
    State(qbit_t n_qubits, int device);
    ~State();
    State(const State&) = delete;
    State& operator=(const State&) = delete;

    std::shared_ptr<Executable> Prepare(const Program& program) const;
    void Apply(const Executable& executable, unsigned repetitions = 1);
    void Reset();
    void Set(const std::complex<float>* data, size_t size);
    std::vector<std::complex<float>> Get() const;
    std::unique_ptr<State> Copy() const;
    void CopyFrom(const State& other);
    std::shared_ptr<Observable> PrepareObservable(const Hamiltonian<float>& hamiltonian) const;
    std::complex<double> Expectation(const Observable& observable, const State* bra = nullptr) const;
    void ApplyHamiltonian(const Observable& observable);
    std::complex<double> Dot(const State& ket) const;
    double Norm(uint64_t mask = 0, uint64_t value = 0) const;
    std::vector<std::vector<std::complex<double>>> ReducedDensity(const qbits_t& kept_qubits) const;
    void Scale(double factor);
    int Measure(qbit_t qubit, double draw, int reset_to = -1);
    std::vector<uint64_t> Sample(uint64_t shots, uint64_t seed) const;
    std::string SocName() const;
    unsigned VectorCores() const;

 private:
    friend class GradientPlan;
    struct Impl;
    explicit State(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

class GradientPlan {
 public:
    using Result = std::vector<std::vector<std::vector<std::complex<double>>>>;
    GradientPlan(const State& owner, const std::vector<Hamiltonian<float>>& hams,
                 const std::vector<std::shared_ptr<BasicGate>>& right_gates,
                 const std::vector<std::shared_ptr<BasicGate>>& left_gates,
                 const std::vector<std::string>& parameter_names, bool hermitian, unsigned tile_qubits = 12,
                 unsigned fusion_qubits = 6, unsigned cube_min_gates = 4);
    ~GradientPlan();
    Result Evaluate(const State& right_state, const State& left_state,
                    const std::vector<parameter::ParameterResolver>& parameters) const;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}  // namespace mindquantum::sim::ascend
#endif  // INCLUDE_SIMULATOR_VECTOR_ASCEND_RUNTIME_H_
