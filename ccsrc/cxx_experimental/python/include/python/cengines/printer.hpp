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

#ifndef PYTHON_PRINTER_HPP
#define PYTHON_PRINTER_HPP

#include <pybind11/pybind11.h>

#include "cengines/cpp_printer.hpp"
#include "python/cengines/base.hpp"

void init_printer(pybind11::module& module);

namespace mindquantum::python {
class CommandPrinter
    : public cengines::CppPrinter
    , public cpp::BasicEngine {
 public:
    CommandPrinter() : cengines::CppPrinter(language_t::projectq) {
    }
    explicit CommandPrinter(language_t language) : cengines::CppPrinter(language) {
    }
    explicit CommandPrinter(std::string_view language) : cengines::CppPrinter(language) {
    }
};
}  // namespace mindquantum::python

#endif /* PYTHON_PRINTER_HPP */
