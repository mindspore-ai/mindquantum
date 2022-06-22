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

#ifndef RESOURCE_COUNTER_HPP
#define RESOURCE_COUNTER_HPP

#include <tweedledum/IR/Circuit.h>

#include "cengines/cpp_resource_counter.hpp"
#include "core/details/macros.hpp"
#include "details/macros_conv_begin.hpp"

CLANG_DIAG_OFF("-Wdeprecated-declarations")
#include <pybind11/pybind11.h>
CLANG_DIAG_ON("-Wdeprecated-declarations")

#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>

// =============================================================================

void init_resource_counter(pybind11::module& m);

// =============================================================================

namespace mindquantum::details {
//! Interface with Python ResourceCounter (internal use only)
class RCPseudoGate {
    using param_t = std::optional<double>;

 public:
    RCPseudoGate() : param_{}, kind_{} {
    }
    explicit RCPseudoGate(std::string_view kind) : param_{}, kind_(kind) {
    }
    RCPseudoGate(std::string_view kind, param_t param) : param_(param), kind_(kind) {
    }

    std::string to_string() const;

 private:
    param_t param_;
    std::string kind_;
};
}  // namespace mindquantum::details

namespace mindquantum::python {
//! C++ equivalent to projectq.backends.ResourceCounter
/*!
 * Prints all gate classes and specific gates it encountered
 * (cumulative over several flushes)
 */
class ResourceCounter : public cengines::ResourceCounter {
 public:
    //! Write statistics data back to Python
    void write_data_to_python() const;

    void set_origin(const pybind11::handle& origin) {
        origin_ = origin.ptr();
    }
};
}  // namespace mindquantum::python

// ==============================================================================

namespace mindquantum::details {
//! Helper function to extract attributes common to all mappers
bool load_resource_counter(pybind11::handle src, python::ResourceCounter& value);
}  // namespace mindquantum::details

// ==============================================================================

namespace pybind11::detail {
template <>
struct type_caster<mindquantum::python::ResourceCounter> {
 public:
    using value_type = mindquantum::python::ResourceCounter;

    PYBIND11_TYPE_CASTER(value_type, _("ResourceCounter"));

    bool load(handle src, bool) {
        return mindquantum::details::load_resource_counter(src, value);
    }
};
}  // namespace pybind11::detail

#include "details/macros_conv_end.hpp"

#endif /* RESOURCE_COUNTER_HPP */
