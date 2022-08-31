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

#include "python/core/circuit.hpp"

#include <cstdint>
#include <vector>

#include <tweedledum/IR/Circuit.h>
#include <tweedledum/IR/Qubit.h>
#include <tweedledum/Parser/qasm.h>
#include <tweedledum/Parser/tfc.h>
#include <tweedledum/Utils/Visualization/string_utf8.h>

#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "experimental/core/types.hpp"
#include "experimental/ops/gates.hpp"
#include "experimental/ops/parametric/angle_gates.hpp"

#include "python/bindings.hpp"

namespace py = pybind11;
namespace td = tweedledum;
namespace mq = mindquantum;

template <typename operator_t>
auto apply_operator_t() {
    return static_cast<mq::inst_ref_t (mq::circuit_t::*)(const operator_t&, const mq::qubits_t&, const mq::cbits_t&)>(
        &mq::circuit_t::apply_operator);
}

#define DEF_APPLY_OPERATOR_OVERLOAD(operator_t)                                                                        \
    def("apply_operator", apply_operator_t<operator_t>(), py::arg("optor"), py::arg("qubits"),                         \
        py::arg("cbits") = cbits_t{})

void mindquantum::python::init_circuit(pybind11::module& module) {
    py::class_<instruction_t>(module, "Instruction");
    py::class_<inst_ref_t>(module, "InstRef");
    py::class_<qubit_t>(module, "Qubit");

    py::class_<circuit_t>(module, "Circuit")
        .def_static("from_qasm_file", &td::qasm::parse_source_file)
        .def_static("from_qasm_string", &td::qasm::parse_source_buffer)
        .def_static("from_tfc_file", &td::tfc::parse_source_file)
        .def(py::init<>())
        // Properties
        .def("num_ancillae", &circuit_t::num_ancillae)
        .def("num_cbits", &circuit_t::num_cbits)
        .def("num_qubits", &circuit_t::num_qubits)
        .def("num_wires", &circuit_t::num_wires)
        // Wires
        .def("create_cbit", py::overload_cast<>(&circuit_t::create_cbit))
        .def("create_qubit", py::overload_cast<>(&circuit_t::create_qubit))
        .def("request_ancilla", &circuit_t::request_ancilla)
        .def("release_ancilla", &circuit_t::release_ancilla)
        // Ising Operators
        .DEF_APPLY_OPERATOR_OVERLOAD(ops::Rxx)
        .DEF_APPLY_OPERATOR_OVERLOAD(ops::parametric::Rxx)
        .DEF_APPLY_OPERATOR_OVERLOAD(ops::Ryy)
        .DEF_APPLY_OPERATOR_OVERLOAD(ops::parametric::Ryy)
        .DEF_APPLY_OPERATOR_OVERLOAD(ops::Rzz)
        .DEF_APPLY_OPERATOR_OVERLOAD(ops::parametric::Rzz)
        // Meta Operators
        .DEF_APPLY_OPERATOR_OVERLOAD(ops::Barrier)
        // Standard Operators
        .DEF_APPLY_OPERATOR_OVERLOAD(ops::H)
        .DEF_APPLY_OPERATOR_OVERLOAD(ops::Measure)
        .DEF_APPLY_OPERATOR_OVERLOAD(ops::P)
        .DEF_APPLY_OPERATOR_OVERLOAD(ops::Ph)
        .DEF_APPLY_OPERATOR_OVERLOAD(ops::parametric::Ph)
        .DEF_APPLY_OPERATOR_OVERLOAD(ops::Rx)
        .DEF_APPLY_OPERATOR_OVERLOAD(ops::parametric::Rx)
        .DEF_APPLY_OPERATOR_OVERLOAD(ops::Ry)
        .DEF_APPLY_OPERATOR_OVERLOAD(ops::parametric::Ry)
        .DEF_APPLY_OPERATOR_OVERLOAD(ops::Rz)
        .DEF_APPLY_OPERATOR_OVERLOAD(ops::parametric::Rz)
        .DEF_APPLY_OPERATOR_OVERLOAD(ops::Sx)
        .DEF_APPLY_OPERATOR_OVERLOAD(ops::Sxdg)
        .DEF_APPLY_OPERATOR_OVERLOAD(ops::S)
        .DEF_APPLY_OPERATOR_OVERLOAD(ops::Sdg)
        .DEF_APPLY_OPERATOR_OVERLOAD(ops::Swap)
        .DEF_APPLY_OPERATOR_OVERLOAD(ops::SqrtSwap)
        .DEF_APPLY_OPERATOR_OVERLOAD(ops::T)
        .DEF_APPLY_OPERATOR_OVERLOAD(ops::Tdg)
        .DEF_APPLY_OPERATOR_OVERLOAD(ops::X)
        .DEF_APPLY_OPERATOR_OVERLOAD(ops::Y)
        .DEF_APPLY_OPERATOR_OVERLOAD(ops::Z)
        // TODO(dnguyen): Add TimeEvolution, etc.
        .DEF_APPLY_OPERATOR_OVERLOAD(instruction_t)
        .def("apply_operator", static_cast<inst_ref_t (circuit_t::*)(const instruction_t&)>(&circuit_t::apply_operator))
        .def("append", &circuit_t::append, py::arg("other"), py::arg("qubits"), py::arg("cbits") = cbits_t{})
        // Python stuff
        .def(
            "__iter__", [](const circuit_t& c) { return py::make_iterator(c.py_begin(), c.py_end()); },
            py::keep_alive<0, 1>())
        .def("__len__", &circuit_t::num_instructions)
        .def("__str__", [](const circuit_t& c) { return td::to_string_utf8(c); });
}
