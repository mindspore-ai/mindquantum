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

#ifndef PYTHON_MAPPING_HPP
#define PYTHON_MAPPING_HPP

#include <map>
#include <string>
#include <string_view>

#include <pybind11/cast.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

#include "cengines/cpp_graph_mapper.hpp"
#include "cengines/cpp_mapping.hpp"
#include "python/cengines/base.hpp"

void init_mapping(pybind11::module& module);

namespace mindquantum::python {
class CppGraphMapper
    : public cengines::CppGraphMapper
    , public cpp::BasicEngine {
 public:
    using cengines::CppGraphMapper::CppGraphMapper;
};

namespace cpp {
//! C++ equivalent to projectq.cengines.LinearMapper
/*!
 * This class is mainly intended as storage for the parameter of the
 * qubit mapping/routing pass for a linear arrangement of qubits
 */
class LinearMapper
    : public cengines::cpp::LinearMapper
    , public cpp::BasicEngine {};

//! C++ equivalent to projectq.cengines.GridMapper
/*!
 * This class is mainly intended as storage for the parameter of the
 * qubit mapping/routing pass for a linear arrangement of qubits
 */
class GridMapper
    : public cengines::cpp::GridMapper
    , public cpp::BasicEngine {};
}  // namespace cpp
}  // namespace mindquantum::python

// ==============================================================================

namespace mindquantum::details {
//! Helper function to extract attributes from a CppOnlyLinearMapper
bool load_mapper(pybind11::handle src, python::cpp::LinearMapper& value);

//! Helper function to extract attributes from a CppOnlyGridMapper
bool load_mapper(pybind11::handle src, python::cpp::GridMapper& value);
}  // namespace mindquantum::details

// ==============================================================================

namespace pybind11::detail {
template <>
struct type_caster<mindquantum::python::cpp::LinearMapper> {
 public:
    using value_type = mindquantum::python::cpp::LinearMapper;

    PYBIND11_TYPE_CASTER(value_type, _("LinearMapper_cpp"));

    bool load(handle src, bool) {
        return mindquantum::details::load_mapper(src, value);
    }
};

template <>
struct type_caster<mindquantum::python::cpp::GridMapper> {
 public:
    using value_type = mindquantum::python::cpp::GridMapper;

    PYBIND11_TYPE_CASTER(value_type, _("GridMapper_cpp"));

    bool load(handle src, bool) {
        return mindquantum::details::load_mapper(src, value);
    }
};
}  // namespace pybind11::detail

#endif /* PYTHON_MAPPING_HPP */
