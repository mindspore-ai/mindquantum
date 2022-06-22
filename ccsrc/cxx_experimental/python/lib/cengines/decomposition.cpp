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

#include "cengines/decomposition.hpp"

#include <pybind11/stl.h>

void init_decomposition(pybind11::module& m) {
    namespace py = pybind11;
    namespace python = mindquantum::python;

    py::class_<python::CppDecomposer>(m, "CppDecomposer")
        .def(py::init<const std::vector<std::string>&>())
        .def("receive", &python::CppDecomposer::receive)
        .def("send", &python::CppDecomposer::send);
}
