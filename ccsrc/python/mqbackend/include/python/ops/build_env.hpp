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

#ifndef MQ_PYTHON_BUILD_ENV_HPP_
#define MQ_PYTHON_BUILD_ENV_HPP_
#include <algorithm>
#include <string>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

namespace mindquantum {
std::string GetBuildABI();
std::string GetCompilerType();
std::string GetStdLib();
void BindPybind11Env(py::module &module);  // NOLINT(runtime/references)
}  // namespace mindquantum
#endif
