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
#ifndef PYTHON_LIB_QUANTUM_STATE_BIND_VEC_STATE_HPP
#define PYTHON_LIB_QUANTUM_STATE_BIND_VEC_STATE_HPP
#include <memory>
#include <string_view>

#include <pybind11/complex.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "math/pr/parameter_resolver.h"

#ifdef __CUDACC__
#    include "simulator/vector/detail/gpu_vector_double_policy.cuh"
#    include "simulator/vector/detail/gpu_vector_float_policy.cuh"
#    include "simulator/vector/detail/gpu_vector_policy.cuh"
#elif defined(__x86_64__)
#    include "simulator/vector/detail/cpu_vector_avx_double_policy.h"
#    include "simulator/vector/detail/cpu_vector_avx_float_policy.h"
#    include "simulator/vector/detail/cpu_vector_policy.h"
#elif defined(__amd64)
#    include "simulator/vector/detail/cpu_vector_arm_double_policy.h"
#    include "simulator/vector/detail/cpu_vector_arm_float_policy.h"
#    include "simulator/vector/detail/cpu_vector_policy.h"
#endif  // __CUDACC__

#include "ops/hamiltonian.h"
#include "simulator/vector/blas.h"
#include "simulator/vector/vector_state.h"

template <typename sim_t>
auto BindSim(pybind11::module& module, const std::string_view& name) {  // NOLINT
    using namespace pybind11::literals;                                 // NOLINT
    using qbit_t = mindquantum::qbit_t;
    using calc_type = typename sim_t::calc_type;
    using circuit_t = typename sim_t::circuit_t;

    return pybind11::class_<sim_t>(module, name.data())
        .def(pybind11::init<qbit_t, unsigned>(), "n_qubits"_a, "seed"_a = 42)
        .def("dtype", &sim_t::DType)
        .def("display", &sim_t::Display, "qubits_limit"_a = 10)
        .def("apply_gate", &sim_t::ApplyGate, "gate"_a, "pr"_a = parameter::ParameterResolver(), "diff"_a = false)
        .def("apply_circuit", &sim_t::ApplyCircuit, "gate"_a, "pr"_a = parameter::ParameterResolver())
        .def("reset", &sim_t::Reset)
        .def("get_qs", &sim_t::GetQS)
        .def("set_qs", &sim_t::SetQS)
        .def("apply_hamiltonian", &sim_t::ApplyHamiltonian)
        .def("copy", [](const sim_t& sim) { return sim; })
        .def("sampling", &sim_t::Sampling)
        .def("sampling_measure_ending_without_nosise", &sim_t::SamplingMeasurementEndingWithoutNoise)
        .def("get_circuit_matrix", &sim_t::GetCircuitMatrix)
        .def("get_expectation",
             pybind11::overload_cast<const mindquantum::Hamiltonian<calc_type>&, const circuit_t&, const circuit_t&,
                                     const typename sim_t::derived_t&, const parameter::ParameterResolver&>(
                 &sim_t::GetExpectation, pybind11::const_))
        .def("get_expectation",
             pybind11::overload_cast<const mindquantum::Hamiltonian<calc_type>&, const circuit_t&, const circuit_t&,
                                     const parameter::ParameterResolver&>(&sim_t::GetExpectation, pybind11::const_))
        .def("get_expectation",
             pybind11::overload_cast<const mindquantum::Hamiltonian<calc_type>&, const circuit_t&,
                                     const parameter::ParameterResolver&>(&sim_t::GetExpectation, pybind11::const_))
        .def("qram_expectation_with_grad", &sim_t::QramExpectationWithGrad)
        .def("get_expectation_with_grad_multi_multi", &sim_t::GetExpectationWithGradMultiMulti)
        .def("get_expectation_with_grad_non_hermitian_multi_multi",
             &sim_t::GetExpectationNonHermitianWithGradMultiMulti)
        .def("get_expectation_with_grad_parameter_shift_multi_multi",
             &sim_t::GetExpectationWithGradParameterShiftMultiMulti);
}

template <typename sim_t>
auto BindBlas(pybind11::module& module) {  // NOLINT
    using qs_policy_t = typename sim_t::qs_policy_t;
    module.def("inner_product", mindquantum::sim::vector::detail::BLAS<qs_policy_t>::InnerProduct);
}

#endif
