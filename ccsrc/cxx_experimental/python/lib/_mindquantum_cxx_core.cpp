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

#include <complex>
#include <string>
#include <vector>

#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

#include "python/cengines/engine_list.hpp"
#include "python/core/core.hpp"
#include "python/ops/command.hpp"  // required to get the pybind11 type casters

namespace py = pybind11;

PYBIND11_MODULE(_mindquantum_cxx_core, m) {
    m.doc() = "C++ core module for ProjectQ";

    using mindquantum::python::CppCore;

    py::class_<CppCore>(m, "CppCore")
        .def(py::init<>())
        .def("set_engine_list", &CppCore::set_engine_list)
        .def("set_simulator_backend", &CppCore::set_simulator_backend)
        .def("allocate_qubit", &CppCore::allocate_qubit)
        .def("apply_command", &CppCore::apply_command)
        .def("flush", &CppCore::flush)
        .def("get_measure_info", &CppCore::get_measure_info)
        .def("set_output_stream", &CppCore::set_output_stream)
        .def("write", &CppCore::write)
        .def("cheat", &CppCore::cheat);
}
