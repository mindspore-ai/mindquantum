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

#include <unsupported/Eigen/KroneckerProduct>

#include <tweedledum/Operators/Standard.h>

#include "ops/test_utils.hpp"

#include "experimental/core/logging.hpp"
#include "experimental/ops/gates/details/eigen_sparse_identity.hpp"
#include "experimental/ops/gates/qubit_operator.hpp"

#include <catch2/catch.hpp>

// =============================================================================

namespace ops = mindquantum::ops;
using namespace std::literals::complex_literals;
using namespace std::literals::string_literals;

using QubitOperatorCD = ops::QubitOperator<std::complex<double>>;
using TermValue = ops::TermValue;
using coefficient_t = QubitOperatorCD::coefficient_t;
using term_t = QubitOperatorCD::term_t;
using terms_t = QubitOperatorCD::terms_t;
using coeff_term_dict_t = QubitOperatorCD::coeff_term_dict_t;

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

    const auto terms = QubitOperatorCD::term_policy_t::parse_terms_string(terms_string);

    INFO("terms_string = " << terms_string);
    REQUIRE(std::size(ref_terms) == std::size(terms));

    INFO("terms_string = " << terms_string);
    CHECK(ref_terms == terms);
}

TEST_CASE("QubitOperator constructor", "[qubit_op][ops]") {
    auto ref_terms = coeff_term_dict_t{{{{1, TermValue::X}, {2, TermValue::Y}, {4, TermValue::Z}}, 1.}};

    QubitOperatorCD op("X1 Y2 Z4");
    CHECK(!std::empty(op));
    CHECK(std::size(op) == 1);
    CHECK(op.get_terms() == ref_terms);

    const auto [it, inserted] = ref_terms.emplace(terms_t{{1, TermValue::Y}}, 3.2);
    op += QubitOperatorCD("Y1", it->second);

    CHECK(std::size(op) == 2);
    CHECK(op.get_terms() == ref_terms);

    // NB: failure to parse will result in an empty list of terms... -> identity()
    CHECK(QubitOperatorCD("XX").is_identity());
    CHECK(QubitOperatorCD("1X").is_identity());
    CHECK(QubitOperatorCD("Y1 Z2 1X Y3").is_identity());
}

// =============================================================================

TEST_CASE("QubitOperator count_gates", "[qubit_op][ops]") {
    using matrix_t = QubitOperatorCD::matrix_t;

    uint32_t num_gates{0UL};
    uint32_t num_gates_ref{0UL};

    SECTION("Invalid empty") {
        num_gates = QubitOperatorCD().count_gates();
        UNSCOPED_INFO("Invalid empty");
    }
    SECTION("I") {
        num_gates = QubitOperatorCD::identity().count_gates();
        num_gates_ref = 0;
        UNSCOPED_INFO("I");
    }
    SECTION("X0") {
        num_gates = QubitOperatorCD("X0").count_gates();
        num_gates_ref = 1;
        UNSCOPED_INFO("X0");
    }
    SECTION("Y1") {
        num_gates = QubitOperatorCD("Y1").count_gates();
        num_gates_ref = 1;
        UNSCOPED_INFO("Y1");
    }
    SECTION("Z2") {
        num_gates = QubitOperatorCD("Z2").count_gates();
        num_gates_ref = 1;
        UNSCOPED_INFO("Z2");
    }
    SECTION("X0 Z2") {
        num_gates = QubitOperatorCD("X0 Z2").count_gates();
        num_gates_ref = 2;
        UNSCOPED_INFO("X0 Z2");
    }
    SECTION("X0 Z2 X1") {
        num_gates = QubitOperatorCD("X0 Z2 X1").count_gates();
        num_gates_ref = 3;
        UNSCOPED_INFO("X0 Z2 X1");
    }

    CHECK(num_gates == num_gates_ref);
}

// =============================================================================

TEST_CASE("QubitOperator matrix", "[qubit_op][ops]") {
    namespace Op = tweedledum::Op;
    using matrix_t = QubitOperatorCD::sparse_matrix_t;

    std::optional<matrix_t> ref_mat;
    std::optional<matrix_t> actual_mat;

    SECTION("Invalid empty") {
        actual_mat = QubitOperatorCD().sparse_matrix();
    }

    SECTION("I") {
        ref_mat = matrix_t{2, 2};
        ref_mat.value().setIdentity();

        actual_mat = QubitOperatorCD::identity().sparse_matrix(1);
    }
    SECTION("X0") {
        ref_mat = matrix_t{Op::X::matrix().sparseView()};
        actual_mat = QubitOperatorCD("X0").sparse_matrix();
    }
    SECTION("Y0") {
        ref_mat = matrix_t{Op::Y::matrix().sparseView()};
        actual_mat = QubitOperatorCD("Y0").sparse_matrix();
    }
    SECTION("Z0") {
        ref_mat = matrix_t{Op::Z::matrix().sparseView()};
        actual_mat = QubitOperatorCD("Z0").sparse_matrix();
    }

    if (ref_mat.has_value()) {
        const auto& ref = ref_mat.value();
        REQUIRE(actual_mat.has_value());
        const auto& mat = actual_mat.value();

        REQUIRE(mat.rows() == ref.rows());
        REQUIRE(mat.cols() == ref.cols());
        CHECK(mat == ref);
    } else {
        CHECK(!actual_mat.has_value());
    }
}

// =============================================================================
