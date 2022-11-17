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

#ifndef INCLUDE_DENSITYMATRIX_DENSITYMATRIX_STATE_HPP
#define INCLUDE_DENSITYMATRIX_DENSITYMATRIX_STATE_HPP
#include <cmath>

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <functional>
#include <iterator>
#include <memory>
#include <mutex>
#include <random>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <vector>

// #include "core/mq_base_types.hpp"
// #include "core/parameter_resolver.hpp"
// #include "ops/basic_gate.hpp"
// #include "ops/gates.hpp"
// #include "ops/hamiltonian.hpp"
// #include "core/type.h"
#include "gate/basic_gate.h"
#include "gate/gates.h"
#include "hamiltonian/hamiltonian.h"
#include "pr/parameter_resolver.h"
#include "simulator/timer.h"
#include "simulator/types.hpp"
#include "simulator/utils.hpp"

namespace mindquantum::sim::densitymatrix::detail {
template <typename qs_policy_t_>
struct BLAS;

template <typename qs_policy_t_>
class DensityMatrixState {
    friend struct BLAS<qs_policy_t_>;

 public:
    using qs_policy_t = qs_policy_t_;
    using derived_t = DensityMatrixState<qs_policy_t>;
    using circuit_t = std::vector<std::shared_ptr<BasicGate<calc_type>>>;
    using qs_data_t = typename qs_policy_t::qs_data_t;
    using qs_data_p_t = typename qs_policy_t::qs_data_p_t;
    using py_qs_data_t = typename qs_policy_t::py_qs_data_t;
    using py_qs_datas_t = typename qs_policy_t::py_qs_datas_t;
    using matrix_t = typename qs_policy_t::matrix_t;
    using RndEngine = std::mt19937;

    //! ctor
    DensityMatrixState() = default;
    explicit DensityMatrixState(qbit_t n_qubits, unsigned seed = 42);
    DensityMatrixState(qbit_t n_qubits, unsigned seed, qs_data_p_t vec);
    DensityMatrixState(qs_data_p_t qs, qbit_t n_qubits, unsigned seed = 42);

    DensityMatrixState(const DensityMatrixState<qs_policy_t>& sim);
    derived_t& operator=(const DensityMatrixState<qs_policy_t>& sim);
    DensityMatrixState(DensityMatrixState<qs_policy_t>&& sim);
    derived_t& operator=(DensityMatrixState<qs_policy_t>&& sim);

    //! dtor
    ~DensityMatrixState() {
        qs_policy_t::FreeState(qs);
    }

    //! Reset the quantum state to quantum zero state
    void Reset();

    //! Display basic information of this quantum state
    void Display(qbit_t qubits_limit = 10) const;

    void DisplayQS() const;

    //! Get the quantum state value
    py_qs_datas_t GetQS() const;

    //! Set the quantum state value
    void SetQS(const py_qs_datas_t& qs_out);

    //! Apply a quantum gate on this quantum state, quantum gate can be normal quantum gate, measurement gate and noise
    //! channel
    index_t ApplyGate(const std::shared_ptr<BasicGate<calc_type>>& gate,
                      const ParameterResolver<calc_type>& pr = ParameterResolver<calc_type>(), bool diff = false);

    auto ApplyChannel(const std::shared_ptr<BasicGate<calc_type>>& gate);

    //! Apply a quantum circuit on this quantum state
    auto ApplyCircuit(const circuit_t& circ, const ParameterResolver<calc_type>& pr = ParameterResolver<calc_type>());

    auto ApplyMeasure(const std::shared_ptr<BasicGate<calc_type>>& gate);

    py_qs_data_t ExpectDiffGate(qs_data_p_t qs, qs_data_p_t ham_matrix,
                                const std::shared_ptr<BasicGate<calc_type>>& gate,
                                const ParameterResolver<calc_type>& pr, index_t dim);

    py_qs_datas_t GetExpectationReversibleWithGrad(const Hamiltonian<calc_type>& ham, const circuit_t& circ,
                                                   const circuit_t& herm_circ, const ParameterResolver<calc_type>& pr,
                                                   const MST<size_t>& p_map);

    py_qs_datas_t GetExpectationNonReversibleWithGrad(const Hamiltonian<calc_type>& ham, const circuit_t& circ,
                                                      const circuit_t& herm_circ,
                                                      const ParameterResolver<calc_type>& pr, const MST<size_t>& p_map);

 private:
    qs_data_p_t qs = nullptr;
    qbit_t n_qubits = 0;
    index_t dim = 0;
    unsigned seed = 0;
    RndEngine rnd_eng_;
    std::function<double()> rng_;
};
}  // namespace mindquantum::sim::densitymatrix::detail

#include "simulator/densitymatrix/densitymatrix_state.tpp"  // NOLINT

#endif
