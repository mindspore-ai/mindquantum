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

#include "experimental/core/logging.hpp"

#include "python/bindings.hpp"

namespace py = pybind11;

// -----------------------------------------------------------------------------

#ifndef ENABLE_LOGGING
namespace spdlog::level {
enum level_enum : int {
    trace = 0,
    debug,
    info,
    warn,
    err,
    critical,
    off,
};
}  // namespace spdlog::level
#endif  //! ENABLE_LOGGING

// =============================================================================

void enable_logging(spdlog::level::level_enum level) {
#ifdef ENABLE_LOGGING
    spdlog::default_logger()->set_level(level);
#endif  // ENABLE_LOGGING
}

void disable_logging() {
#ifdef ENABLE_LOGGING
    spdlog::default_logger()->set_level(spdlog::level::off);
#endif  // ENABLE_LOGGING
}

void mindquantum::python::init_logging(pybind11::module& module) {
    using namespace pybind11::literals;

    py::enum_<spdlog::level::level_enum>(module, "LogLevel")
        .value("OFF", spdlog::level::off)
        .value("TRACE", spdlog::level::trace)
        .value("DEBUG", spdlog::level::debug)
        .value("INFO", spdlog::level::info)
        .value("WARN", spdlog::level::warn)
        .value("ERROR", spdlog::level::err)
        .value("CRITICAL", spdlog::level::critical);

    module.def("enable", enable_logging, "level"_a = spdlog::level::info);
    module.def("disable", disable_logging);

    // NB: disable logging by default on init
    disable_logging();
}
