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

#ifndef PYBIND11_COMPAT_HPP
#define PYBIND11_COMPAT_HPP

/* In some rare instances, throwing C++ exceptions can lead to segmentation
 * faults. In those cases, setting the exception directly using the Python C
 * API seems to help.
 */

#ifdef PYBIND11_NO_ERROR_THROW
#    define THROW_RUNTIME_ERROR(msg)                                                                                   \
        do {                                                                                                           \
            PyErr_SetString(PyExc_RuntimeError, msg);                                                                  \
            throw pybind11::error_already_set();                                                                       \
        } while (0)
#else
#    define THROW_RUNTIME_ERROR(msg) throw std::runtime_error(msg)
#endif /* PYBIND11_NO_ERROR_THROW */

#endif /* PYBIND11_COMPAT_HPP */
