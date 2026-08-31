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

#include <string>

#include <pybind11/pybind11.h>

#include "simulator/circuit_tiling/executor.h"

#include "python/densitymatrix/bind_mat_state.h"

PYBIND11_MODULE(_mq_matrix, module) {
    using namespace pybind11::literals;  // NOLINT
    using float_policy_t = mindquantum::sim::densitymatrix::detail::CPUDensityMatrixPolicyFloat;
    using double_policy_t = mindquantum::sim::densitymatrix::detail::CPUDensityMatrixPolicyDouble;
    using float_mat_sim = mindquantum::sim::densitymatrix::detail::DensityMatrixState<float_policy_t>;
    using double_mat_sim = mindquantum::sim::densitymatrix::detail::DensityMatrixState<double_policy_t>;

    module.doc() = "MindQuantum c++ density matrix state simulator.";
    using mindquantum::sim::circuit_tiling::ExecutionBackendName;
    module.attr("_cpu_backends") = pybind11::dict(
        "complex64"_a = std::string(ExecutionBackendName<float_policy_t::py_qs_data_t>()),
        "complex128"_a = std::string(ExecutionBackendName<double_policy_t::py_qs_data_t>()));
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
