/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef PYTHON_DEVICE_BINDING_HPP_
#define PYTHON_DEVICE_BINDING_HPP_
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <pybind11/attr.h>
#include <pybind11/cast.h>
#include <pybind11/complex.h>
#include <pybind11/detail/common.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "device/mapping.h"
#include "device/topology.h"
#include "ops/basic_gate.h"
namespace py = pybind11;
using namespace pybind11::literals;  // NOLINT(build/namespaces_literals)

namespace mm = mindquantum::mapping;
namespace mindquantum::python {
void BindTopology(py::module &module) {  // NOLINT(runtime/references)
    auto qnode_module
        = py::class_<mm::QubitNode, std::shared_ptr<mm::QubitNode>>(module, "QubitNode")
              .def(py::init<mm::qbit_t, std::string, double, double, const std::set<mm::qbit_t> &>(), "id"_a,
                   "color"_a = "#000000", "poi_x"_a = 0.0, "poi_y"_a = 0.0, "neighbour"_a = std::set<mm::qbit_t>(),
                   "Initialize a qubit node.")
              .def(
                  "get_id", [](const mm::QubitNode &node) { return node.id; }, "Index of this qubit.")
              .def(
                  "get_color", [](const mm::QubitNode &node) { return node.color; }, "Color of this qubit.")
              .def(
                  "get_poi_x", [](const mm::QubitNode &node) { return node.poi_x; }, "X position of this qubit.")
              .def(
                  "get_poi_y", [](const mm::QubitNode &node) { return node.poi_y; }, "Y position of this qubit.")
              .def("set_poi", &mm::QubitNode::SetPoi, "poi_x"_a, "poi_y"_a, "Set the position of this qubit.")
              .def("__lt__", &mm::QubitNode::operator<, "other"_a, "Disconnect with other qubit node and return lhs.")
              .def("__gt__", &mm::QubitNode::operator>, "other"_a, "Disconnect with other qubit node and return rhs.")
              .def("__lshift__", &mm::QubitNode::operator<<, "other"_a, "Connect with other qubit node and return lhs.")
              .def("__rshift__", &mm::QubitNode::operator>>, "other"_a, "Connect with other qubit node and return rhs.")
              .def("__int__", [](const mm::QubitNode &node) { return node.id; });
    qnode_module.doc() = R"(
        Qubit node.
    )";

    // -----------------------------------------------------------------------------

    auto qubits_topo_m
        = py::class_<mm::QubitsTopology, std::shared_ptr<mm::QubitsTopology>>(module, "QubitsTopology")
              .def(py::init<const mm::VT<mm::QNodePtr> &>(), "Initialize a physical qubit topology.")
              .def("size", &mm::QubitsTopology::size, "Get total qubit number.")
              .def("n_edges", &mm::QubitsTopology::NEdges, "Get total connected edge number.")
              .def("all_qubit_id", &mm::QubitsTopology::AllQubitID, "Get total qubit id.")
              .def("set_position", py::overload_cast<mm::qbit_t, double, double>(&mm::QubitsTopology::SetPosition),
                   "Set position of a certain qubit.")
              .def("set_position",
                   py::overload_cast<std::map<mm::qbit_t, std::pair<double, double>>>(&mm::QubitsTopology::SetPosition),
                   "Set position of many qubits.")
              .def("set_color", py::overload_cast<mm::qbit_t, std::string>(&mm::QubitsTopology::SetColor),
                   "Set color of certain qubit.")
              .def("set_color", py::overload_cast<std::map<mm::qbit_t, std::string>>(&mm::QubitsTopology::SetColor),
                   "Set color of many qubits.")
              .def("edges_with_id", &mm::QubitsTopology::EdgesWithID, "Get edges with id of two connected qubits.")
              .def("edges_with_poi", &mm::QubitsTopology::EdgesWithPoi,
                   "Get edges with position of two connected qubits.")
              .def("remove_qubit_node", &mm::QubitsTopology::RemoveQubitNode,
                   "Remove a qubit node out of this topology.")
              .def("choose", &mm::QubitsTopology::Choose, "Choose qubit nodes based on given id.")
              .def("add_qubit_node", &mm::QubitsTopology::AddQubitNode, "Add a qubit node into this topology.")
              .def("has_qubit_node", &mm::QubitsTopology::HasQubitNode, "Check whether a qubit is in this topology.")
              .def("isolate_with_near", &mm::QubitsTopology::IsolateWithNear, "id"_a,
                   "Disconnect with all coupling qubits.")
              .def("is_coupled_with", &mm::QubitsTopology::IsCoupledWith, "id1"_a, "id2"_a,
                   "Check whether two qubit nodes are coupled.")
              .def("__getitem__", &mm::QubitsTopology::operator[], "Get qubit node base on qubit id.")
              .def("dict", &mm::QubitsTopology::Dict,
                   "Get the map of qubits with key as qubit id and value as qubit itself.")
              .def("remove_isolate_node", &mm::QubitsTopology::RemoveIsolateNode,
                   "Remove qubit node that do not connect with any other qubits.");
    qubits_topo_m.doc() = "Topology of qubit in physical device.";

    // -----------------------------------------------------------------------------

    auto linear_qubits_m = py::class_<mm::LinearQubits, mm::QubitsTopology, std::shared_ptr<mm::LinearQubits>>(
                               module, "LinearQubits")
                               .def(py::init<mm::qbit_t>(), "n_qubits"_a, "Initialize a linear qubit topology.");
    linear_qubits_m.doc() = "Linear qubit topology.";

    // -----------------------------------------------------------------------------

    auto grid_qubits_m = py::class_<mm::GridQubits, mm::QubitsTopology, std::shared_ptr<mm::GridQubits>>(module,
                                                                                                         "GridQubits")
                             .def(py::init<mm::qbit_t, mm::qbit_t>(), "n_row"_a, "n_col"_a,
                                  "Initialize a grid topology with row and col number.")
                             .def("n_row", &mm::GridQubits::NRow, "Get row number.")
                             .def("n_col", &mm::GridQubits::NCol, "Get column number.");
    grid_qubits_m.doc() = "Grid qubit topology.";
}  // namespace mindquantum::mapping

void BindQubitMapping(py::module &module) {  // NOLINT(runtime/references)
    auto saber_m = py::class_<mm::SABRE, std::shared_ptr<mm::SABRE>>(module, "SABRE")
                       .def(py::init<const mindquantum::VT<std::shared_ptr<mindquantum::BasicGate>> &,
                                     const std::shared_ptr<mm::QubitsTopology> &>(),
                            "Initialize saber method.")
                       .def("solve", &mm::SABRE::Solve, "iter_num"_a, "W"_a, "delta1"_a, "delta2"_a,
                            "Solve qubit mapping problem with saber method.");
    saber_m.doc() = "SABER method to implement qubit mapping task.";
    //------------------------------------------------------------------------------
    auto ha_saber_m = py::class_<mm::MQ_SABRE, std::shared_ptr<mm::MQ_SABRE>>(module, "MQ_SABRE")
                          .def(py::init<const mindquantum::VT<std::shared_ptr<mindquantum::BasicGate>> &,
                                        const std::shared_ptr<mm::QubitsTopology> &,
                                        const std::vector<std::pair<std::pair<int, int>, std::vector<double>>> &>(),
                               "Initialize mq_saber method.")
                          .def("solve", &mm::MQ_SABRE::Solve, "W"_a, "alpha1"_a, "alpha2"_a, "alpha3"_a,
                               "Solve qubit mapping problem with ha_saber method.");
    ha_saber_m.doc() = "MQ_SABER method to implement qubit mapping task.";
}
}  // namespace mindquantum::python
#endif
