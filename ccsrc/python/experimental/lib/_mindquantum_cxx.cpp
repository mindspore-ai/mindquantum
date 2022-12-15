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

#include "config/logging.hpp"

#include "python/bindings.hpp"

PYBIND11_MODULE(_mindquantum_cxx, module) {
    namespace py = pybind11;
    namespace python = mindquantum::python;

    module.doc() = "Python-C++ bindings for MindQuantum";

    py::module circuit = module.def_submodule("circuit", "MindQuantum-C++ circuit interface");
    python::init_circuit(circuit);

    py::module ops = module.def_submodule("ops", "MindQuantum-C++ operators interface");
    python::init_ops(ops);

    py::module simulator = module.def_submodule("simulator", "MindQuantum-C++ C++ simulators");
    python::init_simulator(simulator);

    py::module symengine = module.def_submodule("symengine", "MindQuantum light wrapper for SymEngine");
    python::init_symengine(symengine);

    py::module logging = module.def_submodule("logging", "MindQuantum-C++ logging module");
    python::init_logging(logging);
}
