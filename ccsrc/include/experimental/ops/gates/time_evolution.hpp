//   Copyright 2020 <Huawei Technologies Co., Ltd>
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

#ifndef TIMEEVOLUTION_OP_HPP
#define TIMEEVOLUTION_OP_HPP

#include <utility>

#include "ops/gates/details/std_complex_coeff_policy.hpp"
#include "ops/gates/qubit_operator.hpp"

#include "experimental/core/config.hpp"

namespace mindquantum::ops {
class TimeEvolution {
 public:
    using non_const_num_targets = void;

    static constexpr std::string_view kind() {
        return "projectq.timeevolution";
    }

    using QubitOperatorCD = QubitOperator<std::complex<double>>;

    //! Constructor
    /*!
     * Overload required in some cases for metaprogramming with operators.
     */
    TimeEvolution(uint32_t num_targets, QubitOperatorCD hamiltonian, double time)
        : TimeEvolution(std::move(hamiltonian), time) {
        assert(num_targets == hamiltonian.num_targets());
    }

    //! Constructor
    TimeEvolution(QubitOperatorCD hamiltonian, double time) : hamiltonian_(std::move(hamiltonian)), time_(time) {
    }

    MQ_NODISCARD TimeEvolution adjoint() const {
        return {hamiltonian_.num_targets(), hamiltonian_, -time_};
    }

    MQ_NODISCARD uint32_t num_targets() const {
        return hamiltonian_.num_targets();
    }

    bool operator==(const TimeEvolution& other) const {
        return hamiltonian_ == other.hamiltonian_ && time_ == other.time_;
    }

    // -------------------------------------------------------------------

    MQ_NODISCARD const QubitOperatorCD& get_hamiltonian() const {
        return hamiltonian_;
    }

    MQ_NODISCARD auto get_time() const {
        return time_;
    }

    MQ_NODISCARD auto param() const {
        return get_time();
    }

 private:
    QubitOperatorCD hamiltonian_;
    double time_;
};
}  // namespace mindquantum::ops

namespace tweedledum {
template <>
inline constexpr uint8_t num_param_v<mindquantum::ops::TimeEvolution> = 1;
}  // namespace tweedledum

#endif /* TIMEEVOLUTION_OP_HPP */
