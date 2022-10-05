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

#include <string>

#include <fmt/format.h>
#include <pybind11/cast.h>
#include <pybind11/detail/common.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

#include "config/constexpr_type_name.hpp"
#include "config/logging.hpp"

#include "python/details/get_fully_qualified_tp_name.hpp"

namespace mindquantum::python {
template <typename type_t>
auto create_from_python_container_class(const pybind11::object& src) {
    MQ_TRACE("Called create_from_python_container_class<{}>({})", get_type_name<type_t>(),
             get_fully_qualified_tp_name(src));
#ifndef NDEBUG
    MQ_TRACE("Python value is: {}", static_cast<std::string>(pybind11::str(src)));
#endif  // !NDEBUG

    if (!pybind11::hasattr(src, "_cpp_obj")) {
        pybind11::detail::make_caster<type_t> caster;
        if (!caster.load(src, true)) {
            MQ_DEBUG("Cannot initialize C++ {} with {}", get_type_name<type_t>(), get_fully_qualified_tp_name(src));
            throw pybind11::type_error(fmt::format("Cannot initialize C++ {} with {}", get_type_name<type_t>(),
                                                   get_fully_qualified_tp_name(src)));
        }
        return static_cast<type_t>(caster);
    }

    auto cpp_obj = src.attr("_cpp_obj");
    if (!pybind11::isinstance<type_t>(cpp_obj)) {
        MQ_DEBUG("{} is not a C++ {}", get_fully_qualified_tp_name(cpp_obj), get_type_name<type_t>());
        throw pybind11::type_error(
            fmt::format("{} is not a C++ {}", get_fully_qualified_tp_name(cpp_obj), get_type_name<type_t>()));
    }

    MQ_TRACE("Casting {} to C++ {}", get_fully_qualified_tp_name(cpp_obj), get_type_name<type_t>());
    return cpp_obj.cast<type_t>();
}

namespace details {
template <typename type_t, typename trampoline_type_t>
constexpr auto try_cast_from_impl(const pybind11::object& src) -> type_t {
    MQ_TRACE("Try casting {} to C++ {}", get_fully_qualified_tp_name(src), get_type_name<trampoline_type_t>());
#ifndef NDEBUG
    MQ_TRACE("Python value is: {}", static_cast<std::string>(pybind11::repr(src)));
#endif  // !NDEBUG
    return type_t{pybind11::cast<trampoline_type_t>(src)};
}

// TODO(dnguyen): Make this constexpr if the compiler supports it
template <typename type_t, typename trampoline_type_t, typename... trampoline_types_t>
auto try_cast_from(const pybind11::object& src) -> type_t {
    try {
        return try_cast_from_impl<type_t, trampoline_type_t>(src);
    } catch (pybind11::type_error&) {
    }
    if constexpr (sizeof...(trampoline_types_t) > 0) {
        return try_cast_from<type_t, trampoline_types_t...>(src);
    } else {
        throw pybind11::type_error(fmt::format("Unable to convert {} to C++", get_fully_qualified_tp_name(src)));
    }
}
}  // namespace details

template <typename type_t, typename... cpp_trampoline_types_t>
auto create_from_python_container_class_with_trampoline(const pybind11::object& src) {
    MQ_TRACE("Called create_from_python_container_class<{}, ...>({})", get_type_name<type_t>(),
             get_fully_qualified_tp_name(src));
#ifndef NDEBUG
    MQ_TRACE("Python value is: {}", static_cast<std::string>(pybind11::repr(src)));
#endif  // !NDEBUG

    if (!pybind11::hasattr(src, "_cpp_obj")) {
        return details::try_cast_from<type_t, cpp_trampoline_types_t...>(src);
    }

    return create_from_python_container_class<type_t>(src);
}
}  // namespace mindquantum::python

#endif /* MQ_PYTHON_CORE_CREATE_FROM_CONTAINER_CLASS_HPP */
