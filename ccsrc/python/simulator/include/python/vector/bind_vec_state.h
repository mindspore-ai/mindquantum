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
#ifndef PYTHON_LIB_QUANTUMSTATE_BIND_VEC_STATE_HPP
#define PYTHON_LIB_QUANTUMSTATE_BIND_VEC_STATE_HPP
#include <memory>
#include <string_view>

#include <pybind11/complex.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "core/parameter_resolver.hpp"
#include "simulator/types.hpp"

#ifdef ENABLE_GPU
#    include "simulator/vector/detail/gpu_vector_policy.cuh"
#else
#    include "simulator/vector/detail/cpu_vector_policy.hpp"
#endif

#include "simulator/vector/blas.hpp"
#include "simulator/vector/vector_state.hpp"
namespace py = pybind11;

namespace mindquantum::sim::bind {
using namespace pybind11::literals;  // NOLINT

using namespace mindquantum::sim::vector::detail;  // NOLINT

#ifdef ENABLE_GPU
using vec_sim = VectorState<GPUVectorPolicyBase>;
#else
using vec_sim = VectorState<CPUVectorPolicyBase>;
#endif

template <typename sim_t>
auto BindSim(py::module& module, const std::string_view& name) {  // NOLINT
    py::class_<sim_t>(module, name.data())
        .def(py::init<qbit_t, unsigned>(), "n_qubits"_a, "seed"_a = 42)
        .def("display", &sim_t::Display, "qubits_limit"_a = 10)
        .def("apply_gate", &sim_t::ApplyGate, "gate"_a, "pr"_a = ParameterResolver<calc_type>(), "diff"_a = false)
        .def("apply_circuit", &sim_t::ApplyCircuit, "gate"_a, "pr"_a = ParameterResolver<calc_type>())
        .def("reset", &sim_t::Reset)
        .def("get_qs", &sim_t::GetQS)
        .def("set_qs", &sim_t::SetQS)
        .def("apply_hamiltonian", &sim_t::ApplyHamiltonian)
        .def("copy", [](const sim_t& sim) { return sim; })
        .def("sampling", &sim_t::Sampling)
        .def("get_expectation_with_grad_one_one", &sim_t::GetExpectationWithGradOneOne)
        .def("get_expectation_with_grad_one_multi", &sim_t::GetExpectationWithGradOneMulti)
        .def("get_expectation_with_grad_multi_multi", &sim_t::GetExpectationWithGradMultiMulti)
        .def("get_expectation_with_grad_non_hermitian_multi_multi",
             &sim_t::GetExpectationNonHermitianWithGradMultiMulti);
}

template <typename sim_t>
auto BindBlas(py::module& module) {  // NOLINT
    using qs_policy_t = typename sim_t::qs_policy_t;
    module.def("inner_product", BLAS<qs_policy_t>::InnerProduct);
}
}  // namespace mindquantum::sim::bind

#endif
