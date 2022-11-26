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

#include "python/densitymatrix/bind_dens_state.h"

PYBIND11_MODULE(_mq_densitymatrix, module) {

using dens_sim = mindquantum::sim::densitymatrix::detail::DensityMatrixState<mindquantum::sim::densitymatrix::detail::CPUDensityMatrixPolicyBase>;


    module.doc() = "MindQuantum c++ density matrix state simulator.";
    BindSim<dens_sim>(module, "mqmatrix");
}
