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

#ifndef CPP_PRINTER_TPP
#define CPP_PRINTER_TPP

#ifndef CPP_PRINTER_HPP
#    error This file must only be included by cengines/cpp_printer.hpp!
#endif  // CPP_PRINTER_HPP

// NB: This is mainly for syntax checkers and completion helpers as this file
//     is only intended to be included directly by cpp_printer.hpp
#include <iostream>

#include "cengines/cpp_printer.hpp"
#include "write_projectq.hpp"

// ==============================================================================

constexpr std::string_view mindquantum::cengines::CppPrinter::lang_to_str(language_t language) {
    switch (language) {
        case language_t::openqasm:
            return "openqasm";
        case language_t::projectq:
            return "projectq";
        default:
            return std::string_view{};
    }
}

// -----------------------------------------------------------------------------

constexpr auto mindquantum::cengines::CppPrinter::str_to_lang(std::string_view language) -> std::optional<language_t> {
    if (language == "projectq") {
        return language_t::projectq;
    } else if (language == "openqasm" || language == "qasm" || language == "qiskit") {
        return language_t::openqasm;
    }
    return {};
}

// =============================================================================

template <typename circuit_t>
void mindquantum::cengines::CppPrinter::print_output(const circuit_t& circuit, std::ostream& output_stream) {
    switch (language_) {
        case language_t::openqasm:
            // td::write_qasm(network, output_stream);
            break;
        case language_t::projectq:
            write_projectq(circuit, output_stream);
            break;
            // td::write_dotqc(network_, output_stream);
            // td::write_quil(network, output_stream);
            // td::write_dot(network);
            // td::write_unicode(network);
            // td::write_quirk(network);
        default:
            output_stream << "Unrecognized language format: " << lang_to_str(language_) << std::endl;
    }
}
#endif /* CPP_PRINTER_TPP */
