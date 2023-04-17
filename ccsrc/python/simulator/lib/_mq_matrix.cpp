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

#include <pybind11/pybind11.h>

#ifdef __CUDACC__
#    include "simulator/densitymatrix/detail/gpu_densitymatrix_double_policy.cuh"
#    include "simulator/densitymatrix/detail/gpu_densitymatrix_float_policy.cuh"
#    include "simulator/densitymatrix/detail/gpu_densitymatrix_policy.cuh"
#elif defined(__x86_64__)
#    include "simulator/densitymatrix/detail/cpu_densitymatrix_avx_double_policy.hpp"
#    include "simulator/densitymatrix/detail/cpu_densitymatrix_avx_float_policy.hpp"
#    include "simulator/densitymatrix/detail/cpu_densitymatrix_policy.hpp"
#elif defined(__amd64)
#    include "simulator/densitymatrix/detail/cpu_densitymatrix_arm_double_policy.hpp"
#    include "simulator/densitymatrix/detail/cpu_densitymatrix_arm_float_policy.hpp"
#    include "simulator/densitymatrix/detail/cpu_densitymatrix_policy.hpp"
#endif

#include "python/densitymatrix/bind_mat_state.hpp"

PYBIND11_MODULE(_mq_matrix, module) {
#ifdef __CUDACC__
    using float_policy_t = mindquantum::sim::densitymatrix::detail::GPUDensityMatrixPolicyFloat;
    using double_policy_t = mindquantum::sim::densitymatrix::detail::GPUDensityMatrixPolicyDouble;
#elif defined(__x86_64__)
    using float_policy_t = mindquantum::sim::densitymatrix::detail::CPUDensityMatrixPolicyAvxFloat;
    using double_policy_t = mindquantum::sim::densitymatrix::detail::CPUDensityMatrixPolicyAvxDouble;
#elif defined(__amd64)
    using float_policy_t = mindquantum::sim::densitymatrix::detail::CPUDensityMatrixPolicyArmFloat;
    using double_policy_t = mindquantum::sim::densitymatrix::detail::CPUDensityMatrixPolicyArmDouble;
#endif  // __CUDACC__

    using float_mat_sim = mindquantum::sim::densitymatrix::detail::DensityMatrixState<float_policy_t>;
    using double_mat_sim = mindquantum::sim::densitymatrix::detail::DensityMatrixState<double_policy_t>;

    module.doc() = "MindQuantum c++ density matrix state simulator.";
    pybind11::module float_sim = module.def_submodule("float", "float simulator");
    pybind11::module double_sim = module.def_submodule("double", "double simulator");

    BindSim<float_mat_sim>(float_sim, "mqmatrix")
        .def("complex128", &float_mat_sim::astype<double_policy_t, mindquantum::sim::densitymatrix::detail::CastTo>)
        .def("complex64", &float_mat_sim::astype<float_policy_t, mindquantum::sim::densitymatrix::detail::CastTo>)
        .def("sim_name", [](const float_mat_sim& sim) { return "mqmatrix"; });
    BindSim<double_mat_sim>(double_sim, "mqmatrix")
        .def("complex128", &double_mat_sim::astype<double_policy_t, mindquantum::sim::densitymatrix::detail::CastTo>)
        .def("complex64", &double_mat_sim::astype<float_policy_t, mindquantum::sim::densitymatrix::detail::CastTo>)
        .def("sim_name", [](const double_mat_sim& sim) { return "mqmatrix"; });
}
