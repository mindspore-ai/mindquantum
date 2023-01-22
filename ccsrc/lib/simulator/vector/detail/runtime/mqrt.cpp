//   Copyright 2023 <Huawei Technologies Co., Ltd>
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
#include <functional>

#include "simulator/types.hpp"
#include "simulator/vector/detail/cpu_vector_avx_double_policy.hpp"

namespace mindquantum::sim::vector::detail {

class MQVector {
    using policy = CPUVectorPolicyAvxDouble;
    using qs_data_p_t = policy::qs_data_p_t;
    using qs_data_t = policy::qs_data_t;
    using RndEngine = std::mt19937;

 public:
    explicit MQVector(qbit_t n_qubits, unsigned seed = 42) : n_qubits(n_qubits), dim(1UL << n_qubits), seed(seed) {
        qs = policy::InitState(dim);
        std::uniform_real_distribution<double> dist(0., 1.);
        rng_ = std::bind(dist, std::ref(rnd_eng_));
    }

    void Display(qbit_t q_limit = 10) {
        policy::Display(this->qs, this->n_qubits, q_limit);
    }

    void ApplyX(qbit_t obj_qubit, const qbits_t& ctrl_qubits = qbits_t()) {
        policy::ApplyX(this->qs, {obj_qubit}, ctrl_qubits, dim);
    }
    void ApplyY(qbit_t obj_qubit, const qbits_t& ctrl_qubits = qbits_t()) {
        policy::ApplyY(this->qs, {obj_qubit}, ctrl_qubits, dim);
    }
    void ApplyZ(qbit_t obj_qubit, const qbits_t& ctrl_qubits = qbits_t()) {
        policy::ApplyZ(this->qs, {obj_qubit}, ctrl_qubits, dim);
    }
    void ApplyH(qbit_t obj_qubit, const qbits_t& ctrl_qubits = qbits_t()) {
        policy::ApplyH(this->qs, {obj_qubit}, ctrl_qubits, dim);
    }
    void ApplySGate(qbit_t obj_qubit, const qbits_t& ctrl_qubits = qbits_t()) {
        policy::ApplySGate(this->qs, {obj_qubit}, ctrl_qubits, dim);
    }
    void ApplySdag(qbit_t obj_qubit, const qbits_t& ctrl_qubits = qbits_t()) {
        policy::ApplySdag(this->qs, {obj_qubit}, ctrl_qubits, dim);
    }
    void ApplyT(qbit_t obj_qubit, const qbits_t& ctrl_qubits = qbits_t()) {
        policy::ApplyT(this->qs, {obj_qubit}, ctrl_qubits, dim);
    }
    void ApplyTdag(qbit_t obj_qubit, const qbits_t& ctrl_qubits = qbits_t()) {
        policy::ApplyTdag(this->qs, {obj_qubit}, ctrl_qubits, dim);
    }
    void ApplySWAP(const qbits_t& obj_qubits, const qbits_t& ctrl_qubits = qbits_t()) {
        policy::ApplySWAP(this->qs, obj_qubits, ctrl_qubits, dim);
    }
    void ApplyISWAP(const qbits_t& obj_qubits, const qbits_t& ctrl_qubits = qbits_t(), bool dagger = false) {
        policy::ApplyISWAP(this->qs, obj_qubits, ctrl_qubits, dagger, dim);
    }
    void ApplyRX(double val, qbit_t obj_qubit, const qbits_t& ctrl_qubits = {}) {
        policy::ApplyRX(this->qs, {obj_qubit}, ctrl_qubits, val, dim, false);
    }
    void ApplyRY(double val, qbit_t obj_qubit, const qbits_t& ctrl_qubits = {}) {
        policy::ApplyRX(this->qs, {obj_qubit}, ctrl_qubits, val, dim, false);
    }
    void ApplyRZ(double val, qbit_t obj_qubit, const qbits_t& ctrl_qubits = {}) {
        policy::ApplyRX(this->qs, {obj_qubit}, ctrl_qubits, val, dim, false);
    }
    void ApplyRxx(double val, const qbits_t& obj_qubits, const qbits_t& ctrl_qubits = {}) {
        policy::ApplyRxx(this->qs, obj_qubits, ctrl_qubits, val, dim, false);
    }
    void ApplyRyy(double val, const qbits_t& obj_qubits, const qbits_t& ctrl_qubits = {}) {
        policy::ApplyRyy(this->qs, obj_qubits, ctrl_qubits, val, dim, false);
    }
    void ApplyRzz(double val, const qbits_t& obj_qubits, const qbits_t& ctrl_qubits = {}) {
        policy::ApplyRzz(this->qs, obj_qubits, ctrl_qubits, val, dim, false);
    }
    index_t ApplyMeasure(qbit_t obj_qubit) {
        index_t one_mask = (1UL << obj_qubit);
        auto one_amp = policy::ConditionalCollect(this->qs, one_mask, one_mask, true, dim).real();
        index_t collapse_mask = (static_cast<index_t>(rng_() < one_amp) << obj_qubit);
        qs_data_t norm_fact = (collapse_mask == 0) ? 1 / std::sqrt(1 - one_amp) : 1 / std::sqrt(one_amp);
        policy::ConditionalMul(qs, qs, one_mask, collapse_mask, norm_fact, 0.0, dim);
        return static_cast<index_t>(collapse_mask != 0);
    }

 private:
    qbit_t n_qubits = 0;
    index_t dim = 0;
    unsigned seed = 0;
    qs_data_p_t qs = nullptr;
    RndEngine rnd_eng_;
    std::function<double()> rng_;
};
}  // namespace mindquantum::sim::vector::detail

int main() {
    auto sim = mindquantum::sim::vector::detail::MQVector(2);
    sim.ApplyX(0);
    sim.Display();
    return 0;
}
