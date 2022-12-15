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

#ifndef PYTHON_BINDINGS_HPP
#define PYTHON_BINDINGS_HPP

#include <pybind11/pybind11.h>

namespace mindquantum::python {
void init_circuit(pybind11::module& module);    // NOLINT(runtime/references)
void init_ops(pybind11::module& module);        // NOLINT(runtime/references)
void init_symengine(pybind11::module& module);  // NOLINT(runtime/references)
void init_simulator(pybind11::module& module);  // NOLINT(runtime/references)
void init_logging(pybind11::module& module);    // NOLINT(runtime/references)
}  // namespace mindquantum::python

#endif /* PYTHON_BINDINGS_HPP */
