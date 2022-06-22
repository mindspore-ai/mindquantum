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

#ifndef CPP_PRINTER_HPP
#define CPP_PRINTER_HPP

#include <optional>
#include <string_view>

#include "core/config.hpp"

namespace tweedledum {
class Circuit;
}  // namespace tweedledum

namespace mindquantum::cengines {
//! C++ class to represent a command printer with seleprojectqctable language
/*!
 * This is intended to be instantiated in Python by users in order to
 * define the output language they want to use
 */
class CppPrinter {
 public:
    enum class language_t : uint8_t {
        projectq,
        openqasm,
        qasm = openqasm,
        qiskit = openqasm,
    };

    //! Convert language_t to string
    static constexpr std::string_view lang_to_str(language_t language);
    //! Convert string to language_t
    static constexpr std::optional<language_t> str_to_lang(std::string_view language);

    //! Constructor
    /*!
     * This is intended to be instantiated in Python by users in order to
     * print all the gates stored in the C++ network.
     *
     * \param language The format in which the network should be printed.g
     */
    explicit CppPrinter(language_t language) noexcept;

    //! Constructor
    /*!
     * This is intended to be instantiated in Python by users in order to
     * print all the gates stored in the C++ network.
     *
     * \param language The format in which the network should be printed.
     * \throws std::runtime_error if format is invalid
     */
    explicit CppPrinter(std::string_view language);

    //! Print circuit to using specified format
    /*!
     * \param circuit The network to be printed
     * \param output_stream Where to print the network
     * Tweedledum ids
     */
    template <typename circuit_t>
    void print_output(const circuit_t& circuit, std::ostream& output_stream);

 private:
    language_t language_;
};
}  // namespace mindquantum::cengines

#include "cengines/cpp_printer.tpp"

#endif /* CPP_PRINTER_HPP */
