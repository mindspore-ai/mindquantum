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

#include <pybind11/pybind11.h>

#include "simulator/mps/detail/cpu_mps_policy.h"
#include "simulator/utils.h"

#include "python/mps/bind_mps_state.h"

PYBIND11_MODULE(_mq_mps, module) {
    std::string sim_name = "mqmps";

    using float_policy_t = mindquantum::sim::mps::detail::CPUMPSPolicy<float>;
    using double_policy_t = mindquantum::sim::mps::detail::CPUMPSPolicy<double>;

    using float_mps_sim = mindquantum::sim::mps::detail::MPSState<float_policy_t>;
    using double_mps_sim = mindquantum::sim::mps::detail::MPSState<double_policy_t>;

    module.doc() = "MindQuantum c++ matrix product state simulator (CPU backend).";
    pybind11::module float_sim = module.def_submodule("float", "float simulator");
    pybind11::module double_sim = module.def_submodule("double", "double simulator");

    BindSim<float_mps_sim>(float_sim, sim_name)
        .def("complex128", &float_mps_sim::template astype<double_policy_t>)
        .def("complex64", &float_mps_sim::template astype<float_policy_t>)
        .def("sim_name", [](const float_mps_sim&) { return "mqmps"; });

    BindSim<double_mps_sim>(double_sim, sim_name)
        .def("complex128", &double_mps_sim::template astype<double_policy_t>)
        .def("complex64", &double_mps_sim::template astype<float_policy_t>)
        .def("sim_name", [](const double_mps_sim&) { return "mqmps"; });
}

