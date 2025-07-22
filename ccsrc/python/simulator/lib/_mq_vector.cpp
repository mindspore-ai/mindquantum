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

#include "simulator/utils.h"

#if defined(__x86_64__)
#    include "simulator/vector/detail/cpu_vector_avx_double_policy.h"
#    include "simulator/vector/detail/cpu_vector_avx_float_policy.h"
#    include "simulator/vector/detail/cpu_vector_policy.h"
#elif defined(__amd64)
#    include "simulator/vector/detail/cpu_vector_arm_double_policy.h"
#    include "simulator/vector/detail/cpu_vector_arm_float_policy.h"
#    include "simulator/vector/detail/cpu_vector_policy.h"
#endif

#include "python/vector/bind_vec_state.h"

PYBIND11_MODULE(_mq_vector, module) {
    using namespace pybind11::literals;  // NOLINT
    std::string sim_name = "mqvector";
#if defined(__x86_64__)
    using float_policy_t = mindquantum::sim::vector::detail::CPUVectorPolicyAvxFloat;
    using double_policy_t = mindquantum::sim::vector::detail::CPUVectorPolicyAvxDouble;
#elif defined(__amd64)
    using float_policy_t = mindquantum::sim::vector::detail::CPUVectorPolicyArmFloat;
    using double_policy_t = mindquantum::sim::vector::detail::CPUVectorPolicyArmDouble;
#else
#    error "Unsupported CPU architecture!"
#endif

    using float_vec_sim = mindquantum::sim::vector::detail::VectorState<float_policy_t>;
    using double_vec_sim = mindquantum::sim::vector::detail::VectorState<double_policy_t>;

    module.doc() = "MindQuantum c++ vector state simulator (CPU backend).";
    pybind11::module float_sim = module.def_submodule("float", "float simulator");
    pybind11::module double_sim = module.def_submodule("double", "double simulator");

    BindSim<float_vec_sim>(float_sim, sim_name)
        .def("complex128", &float_vec_sim::astype<double_policy_t, mindquantum::sim::vector::detail::CastTo>)
        .def("complex64", &float_vec_sim::astype<float_policy_t, mindquantum::sim::vector::detail::CastTo>)
        .def("sim_name", [](const float_vec_sim& sim) { return "mqvector"; });

    BindSim<double_vec_sim>(double_sim, sim_name)
        .def("complex128", &double_vec_sim::astype<double_policy_t, mindquantum::sim::vector::detail::CastTo>)
        .def("complex64", &double_vec_sim::astype<float_policy_t, mindquantum::sim::vector::detail::CastTo>)
        .def("sim_name", [](const double_vec_sim& sim) { return "mqvector"; });

    pybind11::module float_blas = float_sim.def_submodule("blas", "MindQuantum simulator algebra module.");
    pybind11::module double_blas = double_sim.def_submodule("blas", "MindQuantum simulator algebra module.");
    BindBlas<float_vec_sim>(float_blas);
    BindBlas<double_vec_sim>(double_blas);

    module.def("ground_state_of_zs", &double_policy_t::GroundStateOfZZs, "masks_value"_a, "n_qubits"_a);
}
