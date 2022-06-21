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

#ifndef PYTHON_OPTIMISATION_HPP
#define PYTHON_OPTIMISATION_HPP

#include <map>
#include <string_view>

#include <pybind11/cast.h>
#include <pybind11/pytypes.h>

#include "cengines/cpp_optimisation.hpp"
#include "details/macros_conv_begin.hpp"
#include "python/cengines/base.hpp"

namespace mindquantum::python::cpp {
//! C++ equivalent to projectq.cengines.LocalOptimizer
/*!
 * This class is mainly intended as storage for the parameter of the
 * optimization pass
 */
class LocalOptimizer
    : public cengines::cpp::LocalOptimizer
    , public BasicEngine {
 public:
    DECLARE_GETTER_SETTER(unsigned int, _m);
};
}  // namespace mindquantum::python::cpp

namespace mindquantum::details {
//! Helper function to extract attribute from a local optimiser
bool load_optimiser(pybind11::handle src, python::cpp::LocalOptimizer& value);
}  // namespace mindquantum::details

namespace pybind11::detail {
template <>
struct type_caster<mindquantum::python::cpp::LocalOptimizer> {
 public:
    using value_type = mindquantum::python::cpp::LocalOptimizer;

    PYBIND11_TYPE_CASTER(value_type, _("LocalOptimizer_cpp"));

    bool load(handle src, bool) {
        return mindquantum::details::load_optimiser(src, value);
    }
};
}  // namespace pybind11::detail

#include "details/macros_conv_end.hpp"

#endif /* PYTHON_OPTIMISATION_HPP */
