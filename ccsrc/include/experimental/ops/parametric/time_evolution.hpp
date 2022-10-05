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

#ifndef PARAM_TIME_EVOLUTION_HPP
#define PARAM_TIME_EVOLUTION_HPP

#include <utility>

#include "experimental/core/config.hpp"
#include "experimental/ops/gates/time_evolution.hpp"
#include "experimental/ops/parametric/gate_base.hpp"
#include "experimental/ops/parametric/param_names.hpp"

namespace mindquantum::ops::parametric {
class TimeEvolution : public ParametricBase<TimeEvolution, ops::TimeEvolution, real::alpha> {
 public:
    using operator_t = tweedledum::Operator;
    using parent_t = ParametricBase<TimeEvolution, ops::TimeEvolution, real::alpha>;
    using self_t = TimeEvolution;

    using typename parent_t::non_param_type;
    using typename parent_t::subs_map_t;

    using QubitOperatorCD = QubitOperator<std::complex<double>>;

    using non_const_num_targets = void;

    static constexpr std::string_view kind() {
        return "projectq.param.timeevolution";
    }

    //! Constructor
    /*!
     * Overload required in some cases for metaprogramming with operators.
     */
    template <typename param_t>
    TimeEvolution(uint32_t num_targets, QubitOperatorCD hamiltonian, param_t&& param)
        : TimeEvolution(hamiltonian, std::forward<param_t>(param)) {
        assert(num_targets == hamiltonian.num_targets());
    }

    //! Constructor
    template <typename param_t>
    TimeEvolution(QubitOperatorCD hamiltonian, param_t&& param)
        : ParametricBase<TimeEvolution, ops::TimeEvolution, real::alpha>(hamiltonian.num_targets(),
                                                                         std::forward<param_t>(param))
        , hamiltonian_(std::move(hamiltonian)) {
    }

    //! Get the adjoint of an \c TimeEvolution gate instance
    MQ_NODISCARD auto adjoint() const noexcept {
        auto params = base_t::params_;
        for (auto& param : params) {
            param = expand(SymEngine::neg(param));
        }

        return self_t{hamiltonian_.num_targets(), hamiltonian_, std::move(params)};
    }

    bool operator==(const self_t& other) const {
        return hamiltonian_ == other.hamiltonian_ && this->parent_t::operator==(other);
    }

    template <typename evaluated_param_t>
    MQ_NODISCARD static auto to_param_type(const self_t& self, evaluated_param_t&& evaluated_param) {
        return self_t{self.hamiltonian_, std::forward<evaluated_param_t>(evaluated_param)};
    }

    template <typename evaluated_param_t>
    MQ_NODISCARD static auto to_non_param_type(const self_t& self, evaluated_param_t&& evaluated_param) {
        return non_param_type{self.hamiltonian_, std::forward<evaluated_param_t>(evaluated_param)};
    }

    // -------------------------------------------------------------------

    MQ_NODISCARD const QubitOperatorCD& get_hamiltonian() const {
        return hamiltonian_;
    }

    MQ_NODISCARD const auto& get_time() const {
        return param(0);
    }

 private:
    QubitOperatorCD hamiltonian_;
};
}  // namespace mindquantum::ops::parametric

#endif /* PARAM_TIME_EVOLUTION_HPP */
