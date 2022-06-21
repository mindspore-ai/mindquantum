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

#include "python/cengines/printer.hpp"

void init_printer(pybind11::module& m) {
    namespace py = pybind11;
    namespace python = mindquantum::python;
    using language_t = python::CommandPrinter::language_t;

    py::class_<python::CommandPrinter> printer(m, "CppPrinter");

    printer.def(py::init<language_t>())
        .def(py::init<>())
        .def(py::init<std::string_view>())
        .def(py::init<python::CommandPrinter::language_t>())
        .def("receive", &python::CommandPrinter::receive)
        .def("send", &python::CommandPrinter::send);

    py::enum_<python::CommandPrinter::language_t>(printer, "language_t")
        .value("projectq", language_t::projectq)
        .value("openqasm", language_t::openqasm)
        .value("qasm", language_t::qasm)
        .value("qiskit", language_t::qiskit);
}
