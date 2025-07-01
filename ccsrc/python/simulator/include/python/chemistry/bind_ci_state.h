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

#ifndef PYTHON_LIB_CHEM_BIND_CI_STATE_HPP
#define PYTHON_LIB_CHEM_BIND_CI_STATE_HPP

#include <string_view>

#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "math/pr/parameter_resolver.h"
#include "ops/hamiltonian.h"
#include "simulator/chemistry/detail/cpp_excitation_operator.h"
#include "simulator/chemistry/ci_state.h"

namespace py = pybind11;

// Bind CIState simulator
template <typename sim_t>
auto BindSim(py::module& module, const std::string_view& name) {  // NOLINT
    using namespace py::literals;                                 // NOLINT
    using qbit_t = mindquantum::qbit_t;
    using calc_type = typename sim_t::calc_type;
    using circuit_t = typename sim_t::circuit_t;

    return py::class_<sim_t>(module, name.data())
        .def(py::init<qbit_t, int, unsigned>(), "n_qubits"_a, "n_electrons"_a, "seed"_a = 42)
        .def("dtype", &sim_t::DType)
        .def("reset", &sim_t::Reset)
        .def("get_qs", &sim_t::GetQS)
        .def("set_qs", &sim_t::SetQS)
        .def("apply_single_ucc_gate", &sim_t::ApplySingleUCCGate, "gate"_a, "pr"_a = parameter::ParameterResolver())
        .def("get_expectation", &sim_t::GetExpectationValue, "ham"_a)
        .def("get_expectation_with_grad_multi_multi", &sim_t::GetExpectationWithGradMultiMulti, "ham"_a, "circ"_a,
             "enc_data"_a, "ans_data"_a, "enc_name"_a, "ans_name"_a, "batch_threads"_a, "mea_threads"_a)
        .def("apply_circuit", &sim_t::ApplyCircuit, "circuit"_a, "pr"_a = parameter::ParameterResolver())
        .def("copy", [](const sim_t& sim) { return sim; })
        .def(
            "get_state_vector",
            [](const sim_t& sim, qbit_t n_qubits) {
                size_t dim = static_cast<size_t>(1) << n_qubits;
                py::array_t<std::complex<calc_type>> arr(dim);
                auto buf = arr.mutable_data();
                for (size_t i = 0; i < dim; ++i) {
                    buf[i] = std::complex<calc_type>(0);
                }
                for (auto const& kv : sim.GetQS()) {
                    buf[kv.first] = std::complex<calc_type>(kv.second);
                }
                return arr;
            },
            "n_qubits"_a);
}

#endif  // PYTHON_LIB_CHEM_BIND_CI_STATE_HPP
