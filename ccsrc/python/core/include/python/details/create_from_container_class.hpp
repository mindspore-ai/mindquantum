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

#ifndef MQ_PYTHON_CORE_CREATE_FROM_CONTAINER_CLASS_HPP
#define MQ_PYTHON_CORE_CREATE_FROM_CONTAINER_CLASS_HPP

#include <fmt/format.h>
#include <pybind11/cast.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

#include "config/constexpr_type_name.hpp"
#include "config/logging.hpp"

namespace mindquantum::python {
template <typename type_t>
auto create_from_python_container_class(const pybind11::object& src) {
    namespace mq = mindquantum;
    if (!pybind11::hasattr(src, "_cpp_obj")) {
        pybind11::detail::make_caster<type_t> caster;
        if (!caster.load(src, true)) {
            MQ_INFO("Cannot initialize {} with {}", mq::get_type_name<type_t>(), src.ptr()->ob_type->tp_name);
            throw pybind11::type_error(
                fmt::format("Cannot initialize {} with {}", mq::get_type_name<type_t>(), src.ptr()->ob_type->tp_name));
        }
        return static_cast<type_t>(caster);
    }

    auto cpp_obj = src.attr("_cpp_obj");
    if (!pybind11::isinstance<type_t>(cpp_obj)) {
        MQ_INFO("{} is not a {}", cpp_obj.ptr()->ob_type->tp_name, mq::get_type_name<type_t>());
        throw pybind11::type_error(
            fmt::format("{} is not a {}", cpp_obj.ptr()->ob_type->tp_name, mq::get_type_name<type_t>()));
    }

    return cpp_obj.cast<type_t>();
}

}  // namespace mindquantum::python

#endif /* MQ_PYTHON_CORE_CREATE_FROM_CONTAINER_CLASS_HPP */
