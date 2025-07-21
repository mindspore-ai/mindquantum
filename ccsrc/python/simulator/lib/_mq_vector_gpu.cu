/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2025. All rights reserved.
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

#include "simulator/vector/detail/cuquantum_loader.cuh"
#include "simulator/vector/detail/cuquantum_vector_double_policy.cuh"
#include "simulator/vector/detail/cuquantum_vector_float_policy.cuh"
#include "simulator/vector/detail/gpu_vector_double_policy.cuh"
#include "simulator/vector/detail/gpu_vector_float_policy.cuh"
#include "simulator/vector/detail/gpu_vector_policy.cuh"

#include "python/vector/bind_vec_state.h"

PYBIND11_MODULE(_mq_vector_gpu, module) {
    using namespace pybind11::literals;  // NOLINT

    module.doc() = "MindQuantum c++ vector state simulator (GPU backend).";

    using float_policy_t = mindquantum::sim::vector::detail::GPUVectorPolicyFloat;
    using double_policy_t = mindquantum::sim::vector::detail::GPUVectorPolicyDouble;
    using float_vec_sim = mindquantum::sim::vector::detail::VectorState<float_policy_t>;
    using double_vec_sim = mindquantum::sim::vector::detail::VectorState<double_policy_t>;

    pybind11::module float_sim = module.def_submodule("float", "float simulator for mqvector_gpu");
    pybind11::module double_sim = module.def_submodule("double", "double simulator for mqvector_gpu");

    BindSim<float_vec_sim>(float_sim, "mqvector_gpu")
        .def("complex128", &float_vec_sim::astype<double_policy_t, mindquantum::sim::vector::detail::CastTo>)
        .def("complex64", &float_vec_sim::astype<float_policy_t, mindquantum::sim::vector::detail::CastTo>)
        .def("sim_name", [](const float_vec_sim& sim) { return "mqvector_gpu"; });

    BindSim<double_vec_sim>(double_sim, "mqvector_gpu")
        .def("complex128", &double_vec_sim::astype<double_policy_t, mindquantum::sim::vector::detail::CastTo>)
        .def("complex64", &double_vec_sim::astype<float_policy_t, mindquantum::sim::vector::detail::CastTo>)
        .def("sim_name", [](const double_vec_sim& sim) { return "mqvector_gpu"; });

    pybind11::module float_blas = float_sim.def_submodule("blas", "MindQuantum simulator algebra module.");
    pybind11::module double_blas = double_sim.def_submodule("blas", "MindQuantum simulator algebra module.");
    BindBlas<float_vec_sim>(float_blas);
    BindBlas<double_vec_sim>(double_blas);
    module.def("ground_state_of_zs", &double_policy_t::GroundStateOfZZs, "masks_value"_a, "n_qubits"_a);

    if (mindquantum::sim::vector::detail::CuQuantumLoader::GetInstance().IsAvailable()) {
        using float_policy_t_cq = mindquantum::sim::vector::detail::CuQuantumVectorPolicyFloat;
        using double_policy_t_cq = mindquantum::sim::vector::detail::CuQuantumVectorPolicyDouble;
        using float_vec_sim_cq = mindquantum::sim::vector::detail::VectorState<float_policy_t_cq>;
        using double_vec_sim_cq = mindquantum::sim::vector::detail::VectorState<double_policy_t_cq>;

        // Create new, separate submodules for the cuQuantum backend
        pybind11::module cuquantum_float_sim = module.def_submodule("cuquantum_float",
                                                                    "float simulator for mqvector_cq");
        pybind11::module cuquantum_double_sim = module.def_submodule("cuquantum_double",
                                                                     "double simulator for mqvector_cq");

        BindSim<float_vec_sim_cq>(cuquantum_float_sim, "mqvector_cq")
            .def("complex128", &float_vec_sim_cq::astype<double_policy_t_cq, mindquantum::sim::vector::detail::CastTo>)
            .def("complex64", &float_vec_sim_cq::astype<float_policy_t_cq, mindquantum::sim::vector::detail::CastTo>)
            .def("sim_name", [](const float_vec_sim_cq& sim) { return "mqvector_cq"; });

        BindSim<double_vec_sim_cq>(cuquantum_double_sim, "mqvector_cq")
            .def("complex128", &double_vec_sim_cq::astype<double_policy_t_cq, mindquantum::sim::vector::detail::CastTo>)
            .def("complex64", &double_vec_sim_cq::astype<float_policy_t_cq, mindquantum::sim::vector::detail::CastTo>)
            .def("sim_name", [](const double_vec_sim_cq& sim) { return "mqvector_cq"; });

        pybind11::module float_blas_cq = cuquantum_float_sim.def_submodule("blas",
                                                                           "MindQuantum simulator algebra module.");
        pybind11::module double_blas_cq = cuquantum_double_sim.def_submodule("blas",
                                                                             "MindQuantum simulator algebra module.");
        BindBlas<float_vec_sim_cq>(float_blas_cq);
        BindBlas<double_vec_sim_cq>(double_blas_cq);
    }
}
