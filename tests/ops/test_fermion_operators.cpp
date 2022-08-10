//   Copyright 2021 <Huawei Technologies Co., Ltd>
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

#include <complex>
#include <sstream>
#include <string>

#include <boost/range/combine.hpp>

#include "core/circuit_block.hpp"
#include "core/logging.hpp"
#include "ops/gates/fermion_operator.hpp"
#include "ops/gates/qubit_operator.hpp"
#include "ops/utils.hpp"

// clang-format off
#include <catch2/catch.hpp>
// clang-format on

// =============================================================================

namespace ops = mindquantum::ops;

using namespace std::literals::complex_literals;
using FermionOperator = ops::FermionOperator;
using TermValue = mindquantum::ops::TermValue;
using coefficient_t = FermionOperator::coefficient_t;
using term_t = FermionOperator::term_t;
using terms_t = FermionOperator::terms_t;
using complex_term_dict_t = FermionOperator::complex_term_dict_t;

// -----------------------------------------------------------------------------

class UnitTestAccessor {
 public:
    static auto parse_string(std::string_view terms_string) {
        return FermionOperator::parse_string_(terms_string);
    }
};

// =============================================================================

TEST_CASE("FermionOperator parse_string", "[terms_op][ops]") {
    std::string terms_string;
    terms_t ref_terms;

    SECTION("Empty string") {
        terms_string.clear();
        ref_terms.clear();
    }
    SECTION("Whitespace only string") {
        terms_string = "    \t   ";
        ref_terms.clear();
    }
    SECTION("Invalid string ('X')") {
        terms_string = "X";
        ref_terms.clear();
    }
    SECTION("Invalid string ('2 ^')") {
        terms_string = "2 ^";
        ref_terms.clear();
    }
    SECTION("Invalid string ('1 +')") {
        terms_string = "1 +";
        ref_terms.clear();
    }
    SECTION("Invalid string ('1 2^^')") {
        terms_string = "1 2^^";
        ref_terms.clear();
    }
    SECTION("1") {
        terms_string = "1";
        ref_terms.emplace_back(1, TermValue::a);
    }
    SECTION("2^") {
        terms_string = "2^";
        ref_terms.emplace_back(2, TermValue::adg);
    }
    SECTION("  2^    ") {
        terms_string = "  2^    ";
        ref_terms.emplace_back(2, TermValue::adg);
    }
    SECTION("1 2 3 4") {
        terms_string = "1 2 3 4";
        ref_terms.emplace_back(1, TermValue::a);
        ref_terms.emplace_back(2, TermValue::a);
        ref_terms.emplace_back(3, TermValue::a);
        ref_terms.emplace_back(4, TermValue::a);
    }
    SECTION("  4^ 1 2  ") {
        terms_string = "  4^ 1 2  ";
        ref_terms.emplace_back(4, TermValue::adg);
        ref_terms.emplace_back(1, TermValue::a);
        ref_terms.emplace_back(2, TermValue::a);
    }

    const auto terms = UnitTestAccessor::parse_string(terms_string);

    REQUIRE(std::size(ref_terms) == std::size(terms));
    CHECK(ref_terms == terms);
}

// =============================================================================
