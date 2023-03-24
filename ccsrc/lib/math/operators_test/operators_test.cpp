//   Copyright 2023 <Huawei Technologies Co., Ltd>
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

#include <iostream>

#include "math/operators/qubit_operator_view.hpp"
std::tuple<operators::qubit::TermValue, size_t> parse_token(const std::string& token) {
    if (token.size() <= 1) {
        throw std::runtime_error("Wrong token: '" + token + "'. Need a pauli word following a int, for example 'X0'.");
    }
    std::string pauli_word = token.substr(0, 1);
    operators::qubit::TermValue pauli;
    if (pauli_word == "X") {
        pauli = operators::qubit::TermValue::X;
    } else if (pauli_word == "Y") {
        pauli = operators::qubit::TermValue::Y;
    } else if (pauli_word == "Z") {
        pauli = operators::qubit::TermValue::Z;
    } else {
        throw std::runtime_error("Wrong token: '" + token + "'; Can not convert '" + pauli_word
                                 + "' to pauli operator.");
    }
    std::string idx_str = token.substr(1);
    int idx;
    try {
        size_t pos;
        idx = std::stoi(idx_str, &pos);
        std::cout << idx << std::endl;
        std::cout << idx_str << std::endl;
        if (pos != idx_str.length()) {
            throw std::runtime_error("");
        }
    } catch (const std::exception& e) {
        throw std::runtime_error("Wrong token: '" + token + "'; Can not convert '" + idx_str + "' to int.");
    }
    if (idx < 0) {
        throw std::runtime_error("Wrong token: '" + token + "'; Qubit index should not less than zero, but get "
                                 + idx_str + ".");
    }
    return {pauli, idx};
}

int main() {
    operators::qubit::SinglePauliStr qv(
        {
            {operators::qubit::TermValue::X, 0},
            {operators::qubit::TermValue::Y, 1},
            {operators::qubit::TermValue::Z, 1},
        },
        tensor::ops::ones(1));
    std::cout << qv.pauli_string.size() << std::endl;
    std::cout << qv.pauli_string[0] << std::endl;
    std::cout << qv.coeff << std::endl;

    std::string str = "X0 Y1 Y1-9";
    std::istringstream iss(str);

    for (std::string s; iss >> s;) {
        parse_token(s);
    }
    return 0;
}
