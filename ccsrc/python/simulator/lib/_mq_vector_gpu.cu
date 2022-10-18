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

#include "python/vector/bind_vec_state.h"

#define _mq_vector _mq_vector_gpu

namespace mindquantum::sim::bind {
PYBIND11_MODULE(_mq_vector, module) {
#ifdef __CUDACC__
    using vec_sim
        = mindquantum::sim::vector::detail::VectorState<mindquantum::sim::vector::detail::GPUVectorPolicyBase>;
#else
    using vec_sim
        = mindquantum::sim::vector::detail::VectorState<mindquantum::sim::vector::detail::CPUVectorPolicyBase>;
#endif

    module.doc() = "MindQuantum c++ vector state simulator.";
    BindSim<vec_sim>(module, "mqvector");

    pybind11::module blas = module.def_submodule("blas", "MindQuantum simulator algebra module.");
    BindBlas<vec_sim>(blas);
}
}  // namespace mindquantum::sim::bind
