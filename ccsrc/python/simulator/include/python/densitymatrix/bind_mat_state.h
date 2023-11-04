/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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
#ifndef PYTHON_LIB_QUANTUMSTATE_BIND_MAT_STATE_HPP
#define PYTHON_LIB_QUANTUMSTATE_BIND_MAT_STATE_HPP
#include <memory>
#include <string_view>

#include <pybind11/complex.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "math/pr/parameter_resolver.h"

#ifdef __CUDACC__
#    include "simulator/densitymatrix/detail/gpu_densitymatrix_double_policy.cuh"
#    include "simulator/densitymatrix/detail/gpu_densitymatrix_float_policy.cuh"
#    include "simulator/densitymatrix/detail/gpu_densitymatrix_policy.cuh"
#elif defined(__x86_64__)
#    include "simulator/densitymatrix/detail/cpu_densitymatrix_avx_double_policy.h"
#    include "simulator/densitymatrix/detail/cpu_densitymatrix_avx_float_policy.h"
#    include "simulator/densitymatrix/detail/cpu_densitymatrix_policy.h"
#elif defined(__amd64)
#    include "simulator/densitymatrix/detail/cpu_densitymatrix_arm_double_policy.h"
#    include "simulator/densitymatrix/detail/cpu_densitymatrix_arm_float_policy.h"
#    include "simulator/densitymatrix/detail/cpu_densitymatrix_policy.h"
#endif  // __CUDACC__

#include "simulator/densitymatrix/densitymatrix_state.h"

template <typename sim_t>
auto BindSim(pybind11::module& module, const std::string_view& name) {  // NOLINT
    using namespace pybind11::literals;                                 // NOLINT
    using qbit_t = mindquantum::qbit_t;

    return pybind11::class_<sim_t>(module, name.data())
        .def(pybind11::init<qbit_t, unsigned>(), "n_qubits"_a, "seed"_a = 42)
        .def("dtype", &sim_t::DType)
        .def("display", &sim_t::Display, "qubits_limit"_a = 10)
        .def("apply_gate", &sim_t::ApplyGate, "gate"_a, "pr"_a = parameter::ParameterResolver(), "diff"_a = false)
        .def("apply_circuit", &sim_t::ApplyCircuit, "gate"_a, "pr"_a = parameter::ParameterResolver())
        .def("reset", &sim_t::Reset)
        .def("get_qs", &sim_t::GetQS)
        .def("set_qs", &sim_t::SetQS)
        .def("set_dm", &sim_t::SetDM)
        .def("purity", &sim_t::Purity)
        .def("get_partial_trace", &sim_t::GetPartialTrace)
        .def("pure_state_vector", &sim_t::PureStateVector)
        .def("apply_hamiltonian", &sim_t::ApplyHamiltonian)
        .def("copy", [](const sim_t& sim) { return sim; })
        .def("sampling", &sim_t::Sampling)
        .def("sampling_measure_ending_without_nosise", &sim_t::SamplingMeasurementEndingWithoutNoise)
        .def("get_expectation", &sim_t::GetExpectation)
        .def("get_expectation_with_grad_multi_multi", &sim_t::GetExpectationWithReversibleGradMultiMulti)
        .def("get_expectation_with_noise_grad_multi_multi", &sim_t::GetExpectationWithNoiseGradMultiMulti);
}

#endif
