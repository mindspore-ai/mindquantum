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

#ifndef PYTHON_TSL_ORDERED_MAP_HPP
#define PYTHON_TSL_ORDERED_MAP_HPP

#include <utility>

#include <fmt/ranges.h>
#include <pybind11/cast.h>
#include <pybind11/detail/typeid.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

#include "config/logging.hpp"

#include "ops/gates/terms_coeff_dict.hpp"

#include "python/details/create_from_container_class.hpp"

// =============================================================================

namespace pybind11::detail {
template <typename coeff_t>
struct type_caster<mindquantum::ops::term_dict_t<coeff_t>> {
    using value_type = mindquantum::ops::term_dict_t<coeff_t>;
    using key_t = typename value_type::value_type::first_type;
    using value_t = typename value_type::value_type::second_type;

    PYBIND11_TYPE_CASTER(value_type, const_name("mindquantum::ops::term_dict_t"));

    //! Python to C++ conversion
    /*!
     * Load a \c tsl::ordered_map from a Python tuple/list of 2 elements:
     *   - the first one needs to be a tuple/list of keys
     *   - the second one needs to be a tuple/list of values
     *
     * \param src Python object to convert
     */
    bool load(handle src, bool) {
        MQ_DEBUG("type_caster<mindquantum::ops::term_dict_t<{}>>::load({})", pybind11::type_id<coeff_t>(),
                 mindquantum::python::get_fully_qualified_tp_name(src));
        if (!(isinstance<pybind11::tuple>(src) || isinstance<pybind11::list>(src))) {
            MQ_DEBUG("mindquantum::ops::term_dict_t<> requires Tuple[List[], List[]] but received: {}",
                     src.ptr()->ob_type->tp_name);
            return false;
        }
        const auto args = src.cast<pybind11::tuple>();
        if (args.size() != 2) {
            MQ_DEBUG("mindquantum::ops::term_dict_t<> requires Tuple[List[], List[]] but size of tuple/list is {}",
                     args.size());
            return false;
        }

        if (!(isinstance<pybind11::list>(args[0]) || isinstance<pybind11::tuple>(args[0]))) {
            MQ_DEBUG("mindquantum::ops::term_dict_t<>: first element of tuple needs to be a tuple/list but got {}",
                     args[0].ptr()->ob_type->tp_name);
            return false;
        }
        if (!(isinstance<pybind11::list>(args[1]) || isinstance<pybind11::tuple>(args[1]))) {
            MQ_DEBUG("mindquantum::ops::term_dict_t<>: second element of tuple needs to be a tuple/list but got {}",
                     args[1].ptr()->ob_type->tp_name);
            return false;
        }

        const auto keys_python = args[0].cast<pybind11::list>();
        const auto mapped_values_python = args[1].cast<pybind11::list>();

        if (keys_python.size() != mapped_values_python.size()) {
            MQ_DEBUG("Size of keys ({}) and values ({}) must match!", keys_python.size(), mapped_values_python.size());
            return false;
        }

        for (auto i(0ULL); i < keys_python.size(); ++i) {
            const auto key = keys_python[i].cast<key_t>();
            const auto mapped_value = mapped_values_python[i].cast<value_t>();
            MQ_DEBUG("{}: {}", key, mapped_value);
            value.emplace_back(std::make_pair(key, mapped_value));
        }
        return true;
    }

    //! C++ to Python conversion
    /*!
     * \param src C++ object to convert
     * \param policy Return value policy
     * \param parent Parent object (for \c return_value_policy::reference_internal)
     */
    static handle cast(value_type src, return_value_policy /* policy */, handle /* parent */) {
        return pybind11::make_tuple();
    }
};
}  // namespace pybind11::detail

// =============================================================================

#endif /* PYTHON_TSL_ORDERED_MAP_HPP */
