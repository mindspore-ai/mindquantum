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

#ifndef PYTHON_ENGINE_LIST_HPP
#define PYTHON_ENGINE_LIST_HPP

#include "cengines/cpp_engine_list.hpp"

CLANG_DIAG_OFF("-Wdeprecated-declarations")
#include <pybind11/cast.h>
#include <pybind11/functional.h>
#include <pybind11/pytypes.h>
CLANG_DIAG_ON("-Wdeprecated-declarations")

#include <variant>

namespace mindquantum::details {
//! Helper functions to extract an engine from Python
bool load_cengine(pybind11::handle src, cengines::engine_t& engine);
}  // namespace mindquantum::details

// ==============================================================================

namespace pybind11::detail {
template <>
struct type_caster<mindquantum::cengines::engine_t> {
 public:
    using value_type = mindquantum::cengines::engine_t;

    PYBIND11_TYPE_CASTER(value_type, _("CppEngineType"));

    bool load(handle src, bool) {
        return mindquantum::details::load_cengine(src, value);
    }
};
}  // namespace pybind11::detail

#endif /* PYTHON_ENGINE_LIST_HPP */
