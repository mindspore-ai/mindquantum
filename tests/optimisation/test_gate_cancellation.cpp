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

/*-------------------------------------------------------------------------------------------------
| Part of the tests are taken from Tweedledum which is distributed under the MIT License.
|
| MIT License
|
| Copyright (c) 2018, EPFL
| Copyright (c) 2018, Bruno Schmitt <bruno [dot] schmitt [at] epfl [dot] ch>
|
| Permission is hereby granted, free of charge, to any person obtaining a copy
| of this software and associated documentation files (the "Software"), to deal
| in the Software without restriction, including without limitation the rights
| to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
| copies of the Software, and to permit persons to whom the Software is
| furnished to do so, subject to the following conditions:
|
| The above copyright notice and this permission notice shall be included in all
| copies or substantial portions of the Software.
|
| THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
| IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
| FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
| AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
| LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
| OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
| SOFTWARE.
*------------------------------------------------------------------------------------------------*/

#include <vector>

#include <catch2/catch.hpp>
#include <tweedledum/IR/Circuit.h>
#include <tweedledum/Operators/Ising.h>
#include <tweedledum/Operators/Standard.h>

#include "optimisation/gate_cancellation.hpp"

namespace td = tweedledum;

TEST_CASE("Simple gate cancellations", "[gate_cancellation][optimization]") {
    td::Circuit circuit;
    SECTION("Single qubit gates") {
        auto q0 = circuit.create_qubit("q0");
        auto q1 = circuit.create_qubit();
        circuit.apply_operator(td::Op::H(), {q0});
        circuit.apply_operator(td::Op::H(), {q0});
        circuit.apply_operator(td::Op::H(), {q1});
        circuit.apply_operator(td::Op::T(), {q1});
        circuit.apply_operator(td::Op::Tdg(), {q1});
        auto optimized = mindquantum::optim::gate_cancellation(circuit);
        CHECK(std::size(optimized) == 1);
    }
    SECTION("CNOT 2-qubits") {
        auto q0 = circuit.create_qubit("q0");
        auto q1 = circuit.create_qubit();
        circuit.apply_operator(td::Op::X(), {q0, q1});
        circuit.apply_operator(td::Op::X(), {q0, q1});
        circuit.apply_operator(td::Op::X(), {q1, q0});
        auto optimized = mindquantum::optim::gate_cancellation(circuit);
        CHECK(std::size(optimized) == 1);
    }
    SECTION("CNOT 2-qubits with T") {
        auto q0 = circuit.create_qubit("q0");
        auto q1 = circuit.create_qubit();
        circuit.apply_operator(td::Op::X(), {q0, q1});
        circuit.apply_operator(td::Op::T(), {q0});
        circuit.apply_operator(td::Op::X(), {q0, q1});
        circuit.apply_operator(td::Op::X(), {q1, q0});
        auto optimized = mindquantum::optim::gate_cancellation(circuit);
        CHECK(std::size(optimized) == 2);
    }
    SECTION("CNOT 3-qubits sandwich") {
        auto q0 = circuit.create_qubit();
        auto q1 = circuit.create_qubit();
        auto q2 = circuit.create_qubit();
        circuit.apply_operator(td::Op::X(), {q0, q2});
        circuit.apply_operator(td::Op::X(), {q1, q0});
        circuit.apply_operator(td::Op::X(), {q1, q0});
        circuit.apply_operator(td::Op::X(), {q0, q2});
        auto optimized = mindquantum::optim::gate_cancellation(circuit);
        CHECK(std::size(optimized) == 0);
    }
    SECTION("CNOT 3-qubits same control") {
        auto q0 = circuit.create_qubit();
        auto q1 = circuit.create_qubit();
        auto q2 = circuit.create_qubit();
        circuit.apply_operator(td::Op::X(), {q0, q1});
        circuit.apply_operator(td::Op::X(), {q0, q2});
        circuit.apply_operator(td::Op::X(), {q0, q1});
        circuit.apply_operator(td::Op::X(), {q0, q2});
        auto optimized = mindquantum::optim::gate_cancellation(circuit);
        CHECK(std::size(optimized) == 0);
    }
    SECTION("CNOT 3-qubits same target") {
        auto q0 = circuit.create_qubit();
        auto q1 = circuit.create_qubit();
        auto q2 = circuit.create_qubit();
        circuit.apply_operator(td::Op::X(), {q2, q1});
        circuit.apply_operator(td::Op::X(), {q0, q1});
        circuit.apply_operator(td::Op::X(), {q2, q1});
        circuit.apply_operator(td::Op::X(), {q0, q1});
        auto optimized = mindquantum::optim::gate_cancellation(circuit);
        CHECK(std::size(optimized) == 0);
    }
    SECTION("CNOT 3-qubits") {
        auto q0 = circuit.create_qubit();
        auto q1 = circuit.create_qubit();
        auto q2 = circuit.create_qubit();
        circuit.apply_operator(td::Op::X(), {q2, q1});
        circuit.apply_operator(td::Op::X(), {q0, q2});
        circuit.apply_operator(td::Op::X(), {q0, q1});
        circuit.apply_operator(td::Op::X(), {q2, q1});
        circuit.apply_operator(td::Op::X(), {q0, q1});
        circuit.apply_operator(td::Op::X(), {q0, q2});
        auto optimized = mindquantum::optim::gate_cancellation(circuit);
        CHECK(std::size(optimized) == 4);
    }
    SECTION("Multiple qubit gates") {
        auto q0 = circuit.create_qubit();
        auto q1 = circuit.create_qubit();
        auto q2 = circuit.create_qubit();
        circuit.apply_operator(td::Op::Z(), {q0, q1, q2});
        circuit.apply_operator(td::Op::X(), {q0, q1, q2});
        circuit.apply_operator(td::Op::X(), {q0, q1, q2});
        circuit.apply_operator(td::Op::Z(), {q0, q1, q2});
        auto optimized = mindquantum::optim::gate_cancellation(circuit);
        CHECK(std::size(optimized) == 0);
    }
    SECTION("Multiple qubit gates") {
        auto q0 = circuit.create_qubit();
        auto q1 = circuit.create_qubit();
        auto q2 = circuit.create_qubit();
        circuit.apply_operator(td::Op::X(), {q0, q1});
        circuit.apply_operator(td::Op::X(), {q0, q1, q2});
        circuit.apply_operator(td::Op::X(), {q0, q1});
        auto optimized = mindquantum::optim::gate_cancellation(circuit);
        CHECK(std::size(optimized) == 3);
    }
    SECTION("Multiple qubit gates") {
        auto q0 = circuit.create_qubit();
        auto q1 = circuit.create_qubit();
        auto q2 = circuit.create_qubit();
        circuit.apply_operator(td::Op::X(), {q0, q2});
        circuit.apply_operator(td::Op::X(), {q0, q1, q2});
        circuit.apply_operator(td::Op::X(), {q0, q2});
        auto optimized = mindquantum::optim::gate_cancellation(circuit);
        CHECK(std::size(optimized) == 1);
    }
    SECTION("Multiple qubit gates") {
        auto q0 = circuit.create_qubit();
        auto q1 = circuit.create_qubit();
        auto q2 = circuit.create_qubit();
        auto q3 = circuit.create_qubit();
        circuit.apply_operator(td::Op::X(), {q0, q1});
        circuit.apply_operator(td::Op::X(), {q1, q2, q3});
        circuit.apply_operator(td::Op::X(), {q0, q1});
        auto optimized = mindquantum::optim::gate_cancellation(circuit);
        CHECK(std::size(optimized) == 3);
    }
    SECTION("Multiple qubit gates") {
        auto q0 = circuit.create_qubit();
        auto q1 = circuit.create_qubit();
        auto q2 = circuit.create_qubit();
        auto q3 = circuit.create_qubit();
        auto q4 = circuit.create_qubit();
        circuit.apply_operator(td::Op::X(), {q0, q2});
        circuit.apply_operator(td::Op::X(), {q1, q2});
        circuit.apply_operator(td::Op::X(), {q2, q3, q4});
        circuit.apply_operator(td::Op::X(), {q1, q2});
        circuit.apply_operator(td::Op::X(), {q0, q2});
        circuit.apply_operator(td::Op::X(), {q0, q2});
        circuit.apply_operator(td::Op::X(), {q2, q3, q4});
        auto optimized = mindquantum::optim::gate_cancellation(circuit);
        CHECK(std::size(optimized) == 5);
    }
}

TEST_CASE("Even Sequences", "[gate_cancellation][optimization]") {
    td::Circuit circuit;
    SECTION("Even sequence of hadamards") {
        auto q0 = circuit.create_qubit();
        for (uint32_t i = 0; i < 1024; ++i) {
            circuit.apply_operator(td::Op::H(), {q0});
        }
        auto optimized = mindquantum::optim::gate_cancellation(circuit);
        CHECK(std::size(optimized) == 0);
    }
    SECTION("Even sequence of pauli-x") {
        auto q0 = circuit.create_qubit();
        for (uint32_t i = 0; i < 1024; ++i) {
            circuit.apply_operator(td::Op::X(), {q0});
        }
        auto optimized = mindquantum::optim::gate_cancellation(circuit);
        CHECK(std::size(optimized) == 0);
    }
    SECTION("Even sequence of pauli-z") {
        auto q0 = circuit.create_qubit();
        for (uint32_t i = 0; i < 1024; ++i) {
            circuit.apply_operator(td::Op::Z(), {q0});
        }
        auto optimized = mindquantum::optim::gate_cancellation(circuit);
        CHECK(std::size(optimized) == 0);
    }
    SECTION("Even sequence of pauli-y") {
        auto q0 = circuit.create_qubit();
        for (uint32_t i = 0; i < 1024; ++i) {
            circuit.apply_operator(td::Op::Y(), {q0});
        }
        auto optimized = mindquantum::optim::gate_cancellation(circuit);
        CHECK(std::size(optimized) == 0);
    }
    SECTION("Even sequence of cx") {
        auto q0 = circuit.create_qubit();
        auto q1 = circuit.create_qubit();
        for (uint32_t i = 0; i < 1024; ++i) {
            circuit.apply_operator(td::Op::X(), {q0, q1});
        }
        auto optimized = mindquantum::optim::gate_cancellation(circuit);
        CHECK(std::size(optimized) == 0);
    }
    SECTION("Even sequence of cx") {
        auto q0 = circuit.create_qubit();
        auto q1 = circuit.create_qubit();
        auto q2 = circuit.create_qubit();
        for (uint32_t i = 0; i < 1024; ++i) {
            circuit.apply_operator(td::Op::X(), {q0, q1});
            circuit.apply_operator(td::Op::X(), {q2, q1});
        }
        auto optimized = mindquantum::optim::gate_cancellation(circuit);
        CHECK(std::size(optimized) == 0);
    }
}

TEST_CASE("Odd Sequences", "[gate_cancellation][optimization]") {
    td::Circuit circuit;
    SECTION("Odd sequence of hadamards") {
        auto q0 = circuit.create_qubit();
        for (uint32_t i = 0; i < 1023; ++i) {
            circuit.apply_operator(td::Op::H(), {q0});
        }
        auto optimized = mindquantum::optim::gate_cancellation(circuit);
        CHECK(std::size(optimized) == 1);
    }
    SECTION("Odd sequence of pauli-x") {
        auto q0 = circuit.create_qubit();
        for (uint32_t i = 0; i < 1023; ++i) {
            circuit.apply_operator(td::Op::X(), {q0});
        }
        auto optimized = mindquantum::optim::gate_cancellation(circuit);
        CHECK(std::size(optimized) == 1);
    }
    SECTION("Odd sequence of pauli-z") {
        auto q0 = circuit.create_qubit();
        for (uint32_t i = 0; i < 1023; ++i) {
            circuit.apply_operator(td::Op::Z(), {q0});
        }
        auto optimized = mindquantum::optim::gate_cancellation(circuit);
        CHECK(std::size(optimized) == 1);
    }
    SECTION("Odd sequence of pauli-y") {
        auto q0 = circuit.create_qubit();
        for (uint32_t i = 0; i < 1023; ++i) {
            circuit.apply_operator(td::Op::Y(), {q0});
        }
        auto optimized = mindquantum::optim::gate_cancellation(circuit);
        CHECK(std::size(optimized) == 1);
    }
    SECTION("Odd sequence of cx") {
        auto q0 = circuit.create_qubit();
        auto q1 = circuit.create_qubit();
        for (uint32_t i = 0; i < 1023; ++i) {
            circuit.apply_operator(td::Op::X(), {q0, q1});
        }
        auto optimized = mindquantum::optim::gate_cancellation(circuit);
        CHECK(std::size(optimized) == 1);
    }
    SECTION("Odd sequence of cx") {
        auto q0 = circuit.create_qubit();
        auto q1 = circuit.create_qubit();
        auto q2 = circuit.create_qubit();
        for (uint32_t i = 0; i < 1023; ++i) {
            circuit.apply_operator(td::Op::X(), {q0, q1});
            circuit.apply_operator(td::Op::X(), {q2, q1});
        }
        auto optimized = mindquantum::optim::gate_cancellation(circuit);
        CHECK(std::size(optimized) == 2);
    }
}
