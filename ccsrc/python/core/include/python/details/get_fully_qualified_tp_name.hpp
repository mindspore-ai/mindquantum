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

#ifndef GET_FULLY_QUALIFIED_TP_NAME_HPP
#define GET_FULLY_QUALIFIED_TP_NAME_HPP

#include <string>
#include <utility>

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

// =============================================================================

namespace mindquantum::python::pybind11_details {
#ifdef MQ_MINDSPORE_CI
inline std::string get_fully_qualified_tp_name(PyTypeObject* type) {
#    if !defined(PYPY_VERSION)
    return type->tp_name;
#    else
    auto module_name = pybind11::handle(reinterpret_cast<PyObject*>(type)).attr("__module__").cast<std::string>();
    if (module_name == "builtins") {
        return type->tp_name;
    } else {
        return std::move(module_name) + "." + type->tp_name;
    }
#    endif
}
#else
using pybind11::detail::get_fully_qualified_tp_name;
#endif  // MQ_MINDSPORE_CI
}  // namespace mindquantum::python::pybind11_details

// =============================================================================

namespace mindquantum::python {
inline auto get_fully_qualified_tp_name(const pybind11::handle& src) {
    return pybind11_details::get_fully_qualified_tp_name(Py_TYPE(src.ptr()));
}
}  // namespace mindquantum::python

#endif /* GET_FULLY_QUALIFIED_TP_NAME_HPP */
