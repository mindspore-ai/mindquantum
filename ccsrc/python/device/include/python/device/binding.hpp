//   Copyright 2023 <Huawei Technologies Co., Ltd>
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

#ifndef PYTHON_DEVICE_BINDING_HPP_
#define PYTHON_DEVICE_BINDING_HPP_
#include <map>
#include <memory>
#include <string>
#include <utility>

#include <pybind11/attr.h>
#include <pybind11/cast.h>
#include <pybind11/complex.h>
#include <pybind11/detail/common.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "device/mapping.hpp"
namespace py = pybind11;
using namespace pybind11::literals;  // NOLINT(build/namespaces_literals)

void BindTopology(py::module &module) {  // NOLINT(runtime/references)
    namespace mm = mindquantum::mapping;
    py::class_<mm::QubitNode, std::shared_ptr<mm::QubitNode>>(module, "QubitNode")
        .def(py::init<mm::qbit_t, std::string, double, double>(), "id"_a, "color"_a = "#000000", "poi_x"_a = 0.0,
             "poi_y"_a = 0.0)
        .def("set_poi", &mm::QubitNode::SetPoi, "poi_x"_a, "poi_y"_a)
        .def("__lt__", &mm::QubitNode::operator<, "other"_a)
        .def("__gt__", &mm::QubitNode::operator>, "other"_a)
        .def("__lshift__", &mm::QubitNode::operator<<, "other"_a)
        .def("__rshift__", &mm::QubitNode::operator>>, "other"_a);
    py::class_<mm::QubitsTopology, std::shared_ptr<mm::QubitsTopology>>(module, "QubitsTopology")
        .def(py::init<const mm::VT<mm::QNodePtr> &>())
        .def("size", &mm::QubitsTopology::size)
        .def("n_edges", &mm::QubitsTopology::NEdges)
        .def("all_qubit_id", &mm::QubitsTopology::AllQubitID)
        .def("set_position", py::overload_cast<mm::qbit_t, double, double>(&mm::QubitsTopology::SetPosition))
        .def("set_position",
             py::overload_cast<std::map<mm::qbit_t, std::pair<double, double>>>(&mm::QubitsTopology::SetPosition))
        .def("set_color", py::overload_cast<mm::qbit_t, std::string>(&mm::QubitsTopology::SetColor))
        .def("set_color", py::overload_cast<std::map<mm::qbit_t, std::string>>(&mm::QubitsTopology::SetColor))
        .def("edges_with_id", &mm::QubitsTopology::EdgesWithID)
        .def("edges_with_poi", &mm::QubitsTopology::EdgesWithPoi)
        .def("remove_qubit_node", &mm::QubitsTopology::RemoveQubitNode)
        .def("add_qubit_node", &mm::QubitsTopology::AddQubitNode)
        .def("has_qubit_node", &mm::QubitsTopology::HasQubitNode)
        .def("__getitem__", &mm::QubitsTopology::operator[])
        .def("remove_isoloate_node", &mm::QubitsTopology::RemoveIsolateNode);
    py::class_<mm::LinearQubits, mm::QubitsTopology, std::shared_ptr<mm::LinearQubits>>(module, "LinearQubits")
        .def(py::init<mm::qbit_t>(), "n_qubits"_a);
    py::class_<mm::GridQubits, mm::QubitsTopology, std::shared_ptr<mm::GridQubits>>(module, "GridQubits")
        .def(py::init<mm::qbit_t, mm::qbit_t>(), "n_row"_a, "n_col"_a)
        .def("n_row", &mm::GridQubits::NRow)
        .def("n_col", &mm::GridQubits::NCol);
}  // namespace mindquantum::mapping
#endif
