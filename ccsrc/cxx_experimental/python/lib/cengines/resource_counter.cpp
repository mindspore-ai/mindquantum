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

#include "python/cengines/resource_counter.hpp"

#include <fmt/format.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <tweedledum/Operators/Ising.h>
#include <tweedledum/Operators/Standard.h>

#include "cengines/write_projectq.hpp"
#include "ops/gates/ph.hpp"
#include "ops/gates/qubit_operator.hpp"
#include "ops/gates/time_evolution.hpp"
#include "python/details/python2cpp_conv.hpp"
#include "python/details/python_api.hpp"
#include "python/ops/command.hpp"

// -----------------------------------------------------------------------------

namespace py = pybind11;
namespace td = tweedledum;

using PSG = mindquantum::details::PythonScopeGuard;

// =============================================================================

void init_resource_counter(pybind11::module& m) {
    namespace py = pybind11;
    namespace details = mindquantum::details;

    py::class_<details::RCPseudoGate>(m, "RCPseudoGate")
        .def(py::init<>())
        .def("__str__", [](const details::RCPseudoGate& rc) { return rc.to_string(); })
        .def("__repr__", [](const details::RCPseudoGate& rc) { return rc.to_string(); });
}

// =============================================================================

std::string mindquantum::details::RCPseudoGate::to_string() const {
    if (param_) {
        return fmt::format("{}({})", kind_, *param_);
    } else {
        return std::string(kind_);
    }
}

// =============================================================================

void mindquantum::python::ResourceCounter::write_data_to_python() const {
    auto gate_class_counts = py::dict();
    for (const auto& [gate_class_desc, count] : gate_class_counts_) {
        const auto& [kind, n_controls] = gate_class_desc;
        gate_class_counts[py::make_tuple(to_string(kind), n_controls)] = py::int_(count);
    }

    auto gate_counts = py::dict();
    for (const auto& [gate_desc, count] : gate_counts_) {
        const auto& [kind, param, n_controls] = gate_desc;
        gate_counts[py::make_tuple(details::RCPseudoGate(to_string(kind), param), n_controls)] = py::int_(count);
    }

    py::handle origin(static_cast<PyObject*>(origin_));

    origin.attr("max_width") = py::cast(0);
    origin.attr("gate_class_counts") = gate_class_counts;
    origin.attr("gate_counts") = gate_counts;

    // NB: not sure why calling .dec_ref() leads to some segmentation faults
    // origin.dec_ref();
}

// =============================================================================

bool mindquantum::details::load_resource_counter(pybind11::handle src, python::ResourceCounter& value) {
    if (auto dict = PSG(PyObject_GetAttrString(src.ptr(), "gate_counts")); dict) {
        auto* dict_keys = PyDict_Keys(dict);
        const auto size(PyList_Size(dict_keys));
        for (auto i(0UL); i < size; ++i) {
            auto* key = PyList_GET_ITEM(dict_keys, i);

            assert(PyTuple_Size(key) == 2);
            const auto n_targets(0UL);
            const auto n_controls = PyLong_AsLong(PyTuple_GET_ITEM(key, 1));

            auto* gate = PyTuple_GET_ITEM(key, 0);
            auto kind = load_gate(gate, n_targets, n_controls).kind();

            std::optional<double> angle;
            const auto is_rotation = PyObject_HasAttrString(gate, "angle");
            if (is_rotation) {
                if (auto attr = PSG(PyObject_GetAttrString(gate, "angle")); attr) {
                    using caster_t = py::detail::make_caster<double>;
                    if (caster_t caster; caster.load(py::handle(attr), true)) {
                        angle = caster;
                    } else {
                        return false;
                    }
                }
            }

            const auto count = PyLong_AsLong(PyDict_GetItem(dict, key));

            value.add_gate_count(kind, angle, n_controls, count);
        }
    } else {
        return false;
    }

    py::handle origin(src);
    origin.inc_ref();

    value.set_origin(origin);

    return true;
}
