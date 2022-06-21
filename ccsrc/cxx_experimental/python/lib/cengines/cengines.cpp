//   Copyright 2020 <Huawei Technologies Co., Ltd>
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

#include "python/cengines/mapping.hpp"
#include "python/cengines/printer.hpp"
#include "python/cengines/resource_counter.hpp"

PYBIND11_MODULE(_mindquantum_cxx_cengines, module) {
    namespace py = pybind11;
    module.doc() = "MindQuantum cengines module";

    init_mapping(module);
    init_printer(module);
    init_resource_counter(module);
}
