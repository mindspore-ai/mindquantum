//   Copyright 2023 <Huawei Technologies Co., Ltd>
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

#ifndef MQ_PYTHON_BUILD_ENV_HPP_
#define MQ_PYTHON_BUILD_ENV_HPP_
#include "python/ops/build_env.hpp"

#include <algorithm>
#include <string>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#define macro_to_string(x)       std::string(#x)
#define macro_value_to_string(x) macro_to_string(x)
namespace py = pybind11;

namespace mindquantum {
std::string GetBuildABI() {
    auto abi = macro_value_to_string(PYBIND11_BUILD_ABI);
    abi.erase(std::remove(abi.begin(), abi.end(), '\"'), abi.end());
    abi.erase(std::remove(abi.begin(), abi.end(), ' '), abi.end());
    return abi;
}

std::string GetCompilerType() {
    auto abi = macro_value_to_string(PYBIND11_COMPILER_TYPE);
    return abi;
}

std::string GetStdLib() {
    auto abi = macro_value_to_string(PYBIND11_STDLIB);
    return abi;
}
void BindPybind11Env(py::module &module) {
    module.def("build_abi", &GetBuildABI);
    module.def("compiler_type", &GetCompilerType);
    module.def("std_lib", &GetStdLib);
}
}  // namespace mindquantum
#endif
