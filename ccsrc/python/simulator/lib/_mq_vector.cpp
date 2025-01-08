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

#include <pybind11/pybind11.h>

#include "simulator/stabilizer/stabilizer.h"
#include "simulator/utils.h"

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
#endif

#include "simulator/stabilizer/random_benchmarking.h"

#include "python/vector/bind_vec_state.h"

PYBIND11_MODULE(_mq_vector, module) {
    using namespace pybind11::literals;  // NOLINT
    std::string sim_name = "mqvector";
#ifdef __CUDACC__
    using float_policy_t = mindquantum::sim::vector::detail::GPUVectorPolicyFloat;
    using double_policy_t = mindquantum::sim::vector::detail::GPUVectorPolicyDouble;
    sim_name = "mqvector_gpu";
#elif defined(__x86_64__)
    using float_policy_t = mindquantum::sim::vector::detail::CPUVectorPolicyAvxFloat;
    using double_policy_t = mindquantum::sim::vector::detail::CPUVectorPolicyAvxDouble;
#elif defined(__amd64)
    using float_policy_t = mindquantum::sim::vector::detail::CPUVectorPolicyArmFloat;
    using double_policy_t = mindquantum::sim::vector::detail::CPUVectorPolicyArmDouble;
#endif  // __CUDACC__

    using float_vec_sim = mindquantum::sim::vector::detail::VectorState<float_policy_t>;
    using double_vec_sim = mindquantum::sim::vector::detail::VectorState<double_policy_t>;

    module.doc() = "MindQuantum c++ vector state simulator.";
    pybind11::module float_sim = module.def_submodule("float", "float simulator");
    pybind11::module double_sim = module.def_submodule("double", "double simulator");

    BindSim<float_vec_sim>(float_sim, sim_name)
        .def("complex128", &float_vec_sim::astype<double_policy_t, mindquantum::sim::vector::detail::CastTo>)
        .def("complex64", &float_vec_sim::astype<float_policy_t, mindquantum::sim::vector::detail::CastTo>)
        .def("sim_name", [](const float_vec_sim& sim) {
#ifdef __CUDACC__
            return "mqvector_gpu";
#else
            return "mqvector";
#endif
        });

    BindSim<double_vec_sim>(double_sim, sim_name)
        .def("complex128", &double_vec_sim::astype<double_policy_t, mindquantum::sim::vector::detail::CastTo>)
        .def("complex64", &double_vec_sim::astype<float_policy_t, mindquantum::sim::vector::detail::CastTo>)
        .def("sim_name", [](const double_vec_sim& sim) {
#ifdef __CUDACC__
            return "mqvector_gpu";
#else
            return "mqvector";
#endif
        });

    pybind11::module float_blas = float_sim.def_submodule("blas", "MindQuantum simulator algebra module.");
    pybind11::module double_blas = double_sim.def_submodule("blas", "MindQuantum simulator algebra module.");
    BindBlas<float_vec_sim>(float_blas);
    BindBlas<double_vec_sim>(double_blas);

    module.def("ground_state_of_zs", &double_policy_t::GroundStateOfZZs, "masks_value"_a, "n_qubits"_a);
    pybind11::module stabilizer = module.def_submodule("stabilizer", "MindQuantum stabilizer simulator.");
#ifndef __CUDACC__
    using namespace mindquantum::stabilizer;  // NOLINT
    pybind11::class_<StabilizerTableau>(stabilizer, "StabilizerTableau")
        .def(pybind11::init<size_t, unsigned>(), "n_qubits"_a, "seed"_a = 42)
        .def("copy", [](const StabilizerTableau& s) { return s; })
        .def("tableau_to_string", &StabilizerTableau::TableauToString)
        .def("stabilizer_to_string", &StabilizerTableau::StabilizerToString)
        .def("apply_circuit", &StabilizerTableau::ApplyCircuit)
        .def("decompose", &StabilizerTableau::Decompose)
        .def("apply_gate", &StabilizerTableau::ApplyGate)
        .def("tableau_to_vector", &StabilizerTableau::TableauToVector)
        .def("reset", &StabilizerTableau::Reset)
        .def("sampling", &StabilizerTableau::Sampling)
        .def("sampling_measure_ending_without_noise", &StabilizerTableau::SamplingMeasurementEndingWithoutNoise)
        .def("__eq__", [](const StabilizerTableau& lhs, const StabilizerTableau& rhs) { return lhs == rhs; })
        .def("get_expectation", &StabilizerTableau::GetExpectation, "ham_termlist"_a, "circuit"_a);
    stabilizer.def("query_single_qubit_clifford_elem", &QuerySingleQubitCliffordElem, "idx"_a);
    stabilizer.def("query_double_qubits_clifford_elem", &QueryDoubleQubitsCliffordElem, "idx"_a);
    stabilizer.def("generate_single_qubit_rb_circ", &SingleQubitRBCircuit, "len"_a, "seed"_a);
    stabilizer.def("generate_double_qubits_rb_circ", &DoubleQubitsRBCircuit, "len"_a, "seed"_a);
    stabilizer.def("verify", &Verification);
#endif
}
