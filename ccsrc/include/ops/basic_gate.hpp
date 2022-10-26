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

#ifndef MINDQUANTUM_GATE_basic_gate_H_
#define MINDQUANTUM_GATE_basic_gate_H_

#include <functional>
#include <initializer_list>
#include <string>
#include <utility>

#include "core/numba_wrapper.hpp"
#include "core/parameter_resolver.hpp"
#include "core/two_dim_matrix.hpp"
#include "core/utils.hpp"

namespace mindquantum {
template <typename T>
struct BasicGate {
    using matrix_t = Dim2Matrix<T>;
    bool parameterized_ = false;
    std::string name_;
    VT<Index> obj_qubits_;
    VT<Index> ctrl_qubits_;
    ParameterResolver<T> params_;
    int64_t hermitian_prop_ = SELFHERMITIAN;
    bool daggered_ = false;
    T applied_value_ = 0;
    bool is_measure_ = false;
    Dim2Matrix<T> base_matrix_;
    std::function<Dim2Matrix<T>(T)> param_matrix_;
    std::function<Dim2Matrix<T>(T)> param_diff_matrix_;
    NumbaMatFunWrapper numba_param_matrix_;
    NumbaMatFunWrapper numba_param_diff_matrix_;
    // Dim2Matrix<T> (*param_matrix_)(T para);
    // Dim2Matrix<T> (*param_diff_matrix_)(T para);
    bool is_channel_ = false;
    bool is_custom_ = false;
    VT<BasicGate<T>> gate_list_;
    VT<T> probs_;
    VT<T> cumulative_probs_;
    T damping_coeff_;
    VT<VVT<CT<T>>> kraus_operator_set_;

    void PrintInfo() {
        if (!daggered_) {
            std::cout << "Gate name: " << name_ << std::endl;
        } else {
            std::cout << "Gate name: " << name_ << " (daggered version)" << std::endl;
        }
        std::cout << "Parameterized: " << parameterized_ << std::endl;
        if (!parameterized_) {
            base_matrix_.PrintInfo();
        }
        if (!obj_qubits_.empty()) {
            std::cout << "Obj qubits: ";
            for (auto qubit_id : obj_qubits_) {
                std::cout << qubit_id << " ";
            }
            std::cout << std::endl;
        }
        if (!ctrl_qubits_.empty()) {
            std::cout << "Control qubits: ";
            for (auto qubit_id : ctrl_qubits_) {
                std::cout << qubit_id << " ";
            }
            std::cout << std::endl;
        }
    }
    void ApplyValue(T theta) {
        if (parameterized_) {
            parameterized_ = false;
            applied_value_ = theta;
            base_matrix_ = param_matrix_(theta);
        }
    }

    BasicGate() = default;
    BasicGate(bool parameterized, std::string name, int64_t hermitian_prop, Dim2Matrix<T> base_matrix)
        : parameterized_(parameterized)
        , name_(std::move(name))
        , hermitian_prop_(hermitian_prop)
        , base_matrix_(base_matrix) {
    }
    BasicGate(std::string name, bool is_channel, T px, T py, T pz)  // for pauli channel
        : name_(std::move(name)), is_channel_(is_channel), probs_{px, py, pz, 1 - px - py - pz} {
        T sum = 0.;
        cumulative_probs_.push_back(sum);
        for (auto it = probs_.begin(); it != probs_.end(); it++) {
            sum += *it;
            cumulative_probs_.push_back(sum);
        }
    }
    BasicGate(std::string name, bool is_channel, T damping_coeff)  // for damping channel
        : name_(std::move(name)), is_channel_(is_channel), damping_coeff_(damping_coeff) {
    }
    BasicGate(std::string name, bool is_channel, VT<VVT<CT<T>>> kraus_operator_set)  // for kraus channel
        : name_(std::move(name)), is_channel_(is_channel), kraus_operator_set_(kraus_operator_set) {
    }
    BasicGate(bool parameterized, std::string name, int64_t hermitian_prop,
              std::function<Dim2Matrix<T>(T)> param_matrix, std::function<Dim2Matrix<T>(T)> param_diff_matrix)
        : parameterized_(parameterized)
        , name_(std::move(name))
        , hermitian_prop_(hermitian_prop)
        , param_matrix_(std::move(param_matrix))
        , param_diff_matrix_(std::move(param_diff_matrix)) {
    }
    BasicGate(std::string name, int64_t hermitian_prop, uint64_t m_addr, uint64_t dm_addr, int dim)
        : parameterized_(true)
        , name_(std::move(name))
        , hermitian_prop_(hermitian_prop)
        , numba_param_matrix_(m_addr, dim)
        , numba_param_diff_matrix_(dm_addr, dim)
        , is_custom_(true) {
    }
};
}  // namespace mindquantum
#endif  // MINDQUANTUM_GATE_basic_gate_H_
