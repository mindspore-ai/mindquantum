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

#ifndef PYTHON_BASE_HPP
#define PYTHON_BASE_HPP

#include "core/details/macros.hpp"

CLANG_DIAG_OFF("-Wdeprecated-declarations")
#include <pybind11/pybind11.h>
CLANG_DIAG_ON("-Wdeprecated-declarations")

namespace mindquantum::python::cpp {
//! Base class for C++-only compiler engines
class BasicEngine {
 public:
    //! Receive ProjectQ commands
    /*!
     * This ProjectQ compiler engine is not intended to be used to do any
     * processing from Python. Instead, it is meant as a way to
     * print the commands inside the C++ network in the chosen format.
     * This method therefore always throws a NotImplementedError.
     *
     * \throw Always throws NotImplementedError
     */
    void receive(pybind11::handle source);

    //! Send ProjectQ commands
    /*!
     * This ProjectQ compiler engine is not intended to be used to do any
     * processing from Python. Instead, it is meant as a way to
     * print the commands inside the C++ network in the chosen format.
     * This method therefore always throws a NotImplementedError.
     *
     * \throw Always throws NotImplementedError
     */
    void send(pybind11::handle source);
};
}  // namespace mindquantum::python::cpp

#endif /* PYTHON_BASE_HPP */
