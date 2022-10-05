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

#include "python/simulator/simulator.hpp"

#include <pybind11/complex.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "experimental/core/circuit_block.hpp"
#include "experimental/core/types.hpp"
#include "experimental/simulator/projectq_simulator.hpp"

#include "python/bindings.hpp"

namespace py = pybind11;

// =============================================================================

void mindquantum::python::init_simulator(pybind11::module& module) {
    using pq_simulator = simulation::projectq::Simulator;

    py::module projectq = module.def_submodule("projectq", "MindQuantum-C++ ProjectQ C++ simulator");
    py::class_<pq_simulator>(projectq, "Simulator")
        .def(py::init<uint32_t>())
        .def("run_circuit",
             static_cast<bool (pq_simulator::*)(const circuit_t&)>(&pq_simulator::run_circuit<mindquantum::circuit_t>),
             py::arg("Circuit"))
        .def("run_circuit",
             static_cast<bool (pq_simulator::*)(const CircuitBlock&)>(
                 &pq_simulator::run_circuit<mindquantum::CircuitBlock>),
             py::arg("CircuitBlock"))
        .def("measure_qubits", &pq_simulator::measure_qubits_return)
        .def("cheat", &pq_simulator::cheat);
}
