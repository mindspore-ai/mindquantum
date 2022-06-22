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

#ifndef TAG_REMOVER_HPP
#define TAG_REMOVER_HPP

#include "cengines/cpp_engine_list.hpp"
#include "python/cengines/base.hpp"

namespace mindquantum::python::cpp {
//! Class that only exists for compatibility with ProjectQ
/*!
 * Behaviour: Does nothing
 */
class TagRemover
    : public BasicEngine
    , public cengines::cpp::TagRemover {};
}  // namespace mindquantum::python::cpp

namespace pybind11::detail {
template <>
struct type_caster<mindquantum::python::cpp::TagRemover> {
 public:
    using value_type = mindquantum::python::cpp::TagRemover;

    PYBIND11_TYPE_CASTER(value_type, _("TagRemover_cpp"));

    bool load(handle src, bool) {
        return true;
    }
};
}  // namespace pybind11::detail
#endif /* TAG_REMOVER_HPP */
