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

#ifndef PYTHON2CPP_CONV_HPP
#define PYTHON2CPP_CONV_HPP

#include <string_view>
#include <type_traits>

#include <pybind11/cast.h>

namespace mindquantum::details {
namespace py = pybind11;

//! Type traits class to extract function information (default impl.)
template <typename T>
struct function_traits : public std::false_type {};

//! Type traits  class to extract function information
/*!
 * Specialisation for pointer-to-member-functions that take exactly one
 * argument.
 */
template <typename R, typename C, typename A>
struct function_traits<R (C::*)(A)> {
    using class_type = C;
    using return_type = R;
    using arg_type = A;
};

//! Helper function to extract and set an attribute from a Python class
/*!
 * The goal of this function is to retrieve a single attribute from a
 * Python class, convert it to a C++ type and then call the appropriate
 * setter method on the C++ instance.
 *
 * Uses the Pybind11 type caster to do the actual conversion
 *
 * \tparam T Type of C++ object to manage
 * \tparam method_t Pointer-to-member-function for setting the attribute
 *
 * \param py_obj Pointer to Python object
 * \param cpp_obj C++ object instance to modify
 * \param attr_name Name of the attribute to retrieve
 * \param method Pointer to member function to call as setter for the
 *               attribute
 *
 * \return True if successful, false otherwise
 */
template <typename T, typename method_t>
auto get_attr_from_python(py::handle py_obj, T& cpp_obj, std::string_view attr_name, method_t method) {
    using arg_type = typename function_traits<method_t>::arg_type;
    using caster_t = py::detail::make_caster<arg_type>;

    if (auto* attr = PyObject_GetAttrString(py_obj.ptr(), attr_name.data()); attr != NULL) {
        if (attr == Py_None) {
            using type = std::decay_t<arg_type>;
            (cpp_obj.*method)(type());
            return true;
        } else if (caster_t caster; caster.load(py::handle(attr), true)) {
            (cpp_obj.*method)(caster);
            Py_DECREF(attr);
            return true;
        }
        Py_DECREF(attr);
    }
    return false;
}
}  // namespace mindquantum::details

#endif /* PYTHON2CPP_CONV_HPP */
