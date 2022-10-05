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

#include "python/cengines/base.hpp"

void mindquantum::python::cpp::BasicEngine::receive(pybind11::handle) {
    PyErr_SetString(PyExc_NotImplementedError,
                    "BasicEngine engine only designed to work with C++ code. "
                    "Cannot receive ProjectQ commands from Python!");
    throw pybind11::error_already_set();
}

void mindquantum::python::cpp::BasicEngine::send(pybind11::handle) {
    PyErr_SetString(PyExc_NotImplementedError,
                    "BasicEngine engine only designed to work with C++ code. "
                    "Cannot send ProjectQ commands to other compiler engines!");
    throw pybind11::error_already_set();
}
