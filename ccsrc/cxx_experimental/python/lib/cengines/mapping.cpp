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

#include "python/cengines/mapping.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "python/details/python2cpp_conv.hpp"

// =============================================================================

void init_mapping(pybind11::module& m) {
    namespace python = mindquantum::python;
    namespace py = pybind11;

    py::class_<python::CppGraphMapper>(m, "CppGraphMapper")
        .def(py::init<uint32_t, const python::CppGraphMapper::edge_list_t&,
                      const python::CppGraphMapper::mapping_param_t&>())
        .def("receive", &python::CppGraphMapper::receive)
        .def("send", &python::CppGraphMapper::send);

    py::class_<mindquantum::mapping::sabre_config>(m, "SabreConfig").def(py::init<>());
    py::class_<mindquantum::mapping::jit_config>(m, "JitConfig").def(py::init<>());
}

// =============================================================================

#define GET_ATTR_FROM_PYTHON(name) mindquantum::details::get_attr_from_python(src, value, #name, &mapper_t::set_##name)

bool mindquantum::details::load_mapper(pybind11::handle src, python::cpp::LinearMapper& value) {
    using mapper_t = python::cpp::LinearMapper;

    if (!GET_ATTR_FROM_PYTHON(_current_mapping)) {
        return false;
    }
    if (!GET_ATTR_FROM_PYTHON(num_qubits)) {
        return false;
    }
    if (!GET_ATTR_FROM_PYTHON(num_mappings)) {
        return false;
    }
    if (!GET_ATTR_FROM_PYTHON(storage)) {
        return false;
    }
    if (!GET_ATTR_FROM_PYTHON(cyclic)) {
        return false;
    }

    return true;
}

bool mindquantum::details::load_mapper(pybind11::handle src, python::cpp::GridMapper& value) {
    using mapper_t = python::cpp::GridMapper;

    if (!GET_ATTR_FROM_PYTHON(_current_mapping)) {
        return false;
    }
    if (!GET_ATTR_FROM_PYTHON(num_qubits)) {
        return false;
    }
    if (!GET_ATTR_FROM_PYTHON(num_mappings)) {
        return false;
    }
    if (!GET_ATTR_FROM_PYTHON(storage)) {
        return false;
    }
    if (!GET_ATTR_FROM_PYTHON(num_rows)) {
        return false;
    }
    if (!GET_ATTR_FROM_PYTHON(num_columns)) {
        return false;
    }
    return true;
}

#undef GET_ATTR_FROM_PYTHON
