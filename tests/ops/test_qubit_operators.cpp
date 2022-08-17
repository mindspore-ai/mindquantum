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

#include <complex>
#include <string>
#include <string_view>

#include "core/logging.hpp"
#include "ops/gates/qubit_operator.hpp"
#include "ops/utils.hpp"

#include <catch2/catch.hpp>

// =============================================================================

namespace ops = mindquantum::ops;
using namespace std::literals::complex_literals;
using namespace std::literals::string_literals;

using QubitOperator = ops::QubitOperator;
using TermValue = ops::TermValue;
using coefficient_t = QubitOperator::coefficient_t;
using term_t = QubitOperator::term_t;
using terms_t = QubitOperator::terms_t;
using coeff_term_dict_t = QubitOperator::coeff_term_dict_t;

// =============================================================================

TEST_CASE("QubitOperator parse_string", "[terms_op][ops]") {
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
    SECTION("Invalid string ('X 1')") {
        terms_string = "X 1";
        ref_terms.clear();
    }
    SECTION("Invalid string ('1X')") {
        terms_string = "1X";
        ref_terms.clear();
    }
    SECTION("Invalid string ('X1 Y')") {
        terms_string = "X1 Y";
        ref_terms.clear();
    }
    SECTION("Invalid string ('X1 YY')") {
        terms_string = "X1 YY";
        ref_terms.clear();
    }
    SECTION("X1") {
        terms_string = "X1";
        ref_terms.emplace_back(1, TermValue::X);
    }
    SECTION("Y2") {
        terms_string = "Y2";
        ref_terms.emplace_back(2, TermValue::Y);
    }
    SECTION("Z10") {
        terms_string = "Z10";
        ref_terms.emplace_back(10, TermValue::Z);
    }
    SECTION("X2 Y1 Z3 X1") {
        terms_string = "X2 Y1 Z3 X1";
        ref_terms.emplace_back(2, TermValue::X);
        ref_terms.emplace_back(1, TermValue::Y);
        ref_terms.emplace_back(3, TermValue::Z);
        ref_terms.emplace_back(1, TermValue::X);
    }
    SECTION("    Y10 Z3    X1    ") {
        terms_string = "    Y10 Z3    X1    ";
        ref_terms.emplace_back(10, TermValue::Y);
        ref_terms.emplace_back(3, TermValue::Z);
        ref_terms.emplace_back(1, TermValue::X);
    }

    const auto terms = QubitOperator::term_policy_t::parse_terms_string(terms_string);

    INFO("terms_string = " << terms_string);
    REQUIRE(std::size(ref_terms) == std::size(terms));

    INFO("terms_string = " << terms_string);
    CHECK(ref_terms == terms);
}

TEST_CASE("QubitOperator constructor", "[qubit_op][ops]") {
    auto ref_terms = coeff_term_dict_t{{{{1, TermValue::X}, {2, TermValue::Y}, {4, TermValue::Z}}, 1.}};

    QubitOperator op("X1 Y2 Z4");
    CHECK(!std::empty(op));
    CHECK(std::size(op) == 1);
    CHECK(op.get_terms() == ref_terms);

    const auto [it, inserted] = ref_terms.emplace(terms_t{{1, TermValue::Y}}, 3.2);
    op += QubitOperator("Y1", it->second);

    CHECK(std::size(op) == 2);
    CHECK(op.get_terms() == ref_terms);

    // NB: failure to parse will result in an empty list of terms... -> identity()
    CHECK(QubitOperator("XX").is_identity());
    CHECK(QubitOperator("1X").is_identity());
    CHECK(QubitOperator("Y1 Z2 1X Y3").is_identity());
}

TEST_CASE("QubitOperator split", "[terms_op][ops]") {
    const auto lhs = QubitOperator("X1", 1.2i);
    const auto rhs = QubitOperator("Z3", 1.2);
    const auto qubit_op = lhs + rhs;

    const auto splitted = qubit_op.split();
    REQUIRE(std::size(splitted) == 2);
    if (splitted[0] == lhs) {
        CHECK(splitted[0] == lhs);
        CHECK(splitted[1] == rhs);
    } else {
        CHECK(splitted[0] == rhs);
        CHECK(splitted[1] == lhs);
    }
}

TEST_CASE("QubitOperator comparison operators", "[terms_op][ops]") {
    coeff_term_dict_t ref_terms;

    auto [it1, inserted1] = ref_terms.emplace(terms_t{{3, TermValue::X}}, 2.3);
    auto [it2, inserted2] = ref_terms.emplace(terms_t{{1, TermValue::Y}}, 1.);
    REQUIRE(inserted1);
    REQUIRE(inserted2);

    // op = {'3': 2.3,  '1^': 1.}
    const QubitOperator qubit_op(ref_terms);
    QubitOperator other(ref_terms);

    CHECK(qubit_op == qubit_op);
    CHECK(qubit_op == other);

    SECTION("Add identity term") {
        other += QubitOperator::identity();
    }
    SECTION("Add other term") {
        other += QubitOperator{terms_t{{2, TermValue::Z}}, 2.34i};
    }
    SECTION("No common terms") {
        other = QubitOperator{terms_t{{2, TermValue::Z}}, 2.34i};
    }
    CHECK(!(qubit_op == other));
    CHECK(qubit_op != other);
}

// =============================================================================
