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
#include <string>
#include <string_view>

#include "config/logging.hpp"

#include "mindquantum/catch2/eigen.hpp"
#include "mindquantum/catch2/mindquantum.hpp"
#include "ops/gates/details/floating_point_coeff_policy.hpp"
#include "ops/gates/details/std_complex_coeff_policy.hpp"
#include "ops/gates/fermion_operator.hpp"

#include <catch2/catch_test_macros.hpp>

// =============================================================================

using namespace std::literals::complex_literals;
using namespace std::literals::string_literals;

namespace ops = mindquantum::ops;
using mindquantum::catch2::Equals;

using FermionOperatorCD = ops::FermionOperator<std::complex<double>>;
using TermValue = mindquantum::ops::TermValue;
using coefficient_t = FermionOperatorCD::coefficient_t;
using term_t = ops::term_t;
using terms_t = ops::terms_t;
using coeff_term_dict_t = FermionOperatorCD::coeff_term_dict_t;

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

    const auto terms = FermionOperatorCD::term_policy_t::parse_terms_string(terms_string);

    INFO("terms_string = " << terms_string);
    REQUIRE(std::size(ref_terms) == std::size(terms));

    INFO("terms_string = " << terms_string);
    CHECK(ref_terms == terms);
}

// -----------------------------------------------------------------------------

TEST_CASE("FermionOperator constructor", "[terms_op][ops]") {
    const auto coeff = 2.34i;
    auto ref_terms = coeff_term_dict_t{{{{1, TermValue::a}, {2, TermValue::adg}, {4, TermValue::a}}, coeff}};

    FermionOperatorCD fermion_op("1 2^ 4", coeff);
    CHECK(!std::empty(fermion_op));
    CHECK(std::size(fermion_op) == 1);
    CHECK(fermion_op.get_terms() == ref_terms);

    const auto [it, inserted] = ref_terms.emplace_back(terms_t{{1, TermValue::adg}}, 3.2);
    fermion_op += FermionOperatorCD("1^", it->second);

    CHECK(std::size(fermion_op) == 2);
    CHECK(fermion_op.get_terms() == ref_terms);

    // NB: failure to parse will result in an empty list of terms... -> identity()
    CHECK(FermionOperatorCD("1^^").is_identity());
    CHECK(FermionOperatorCD("^").is_identity());
    CHECK(FermionOperatorCD("2 1^ 11a 3").is_identity());
}

TEST_CASE("FermionOperator normal_ordered", "[terms_op][ops]") {
    const auto coeff = 2.34i;
    auto ref_coeff = coeff;

    std::string fermion_op_str;
    FermionOperatorCD ref_op;

    SECTION("1 0") {
        ref_op = FermionOperatorCD("1 0", ref_coeff);
        fermion_op_str = "1 0";
    }
    SECTION("0 1") {
        ref_coeff *= -1;
        ref_op = FermionOperatorCD("1 0", ref_coeff);
        fermion_op_str = "0 1";
    }
    SECTION("0 1^") {
        ref_coeff *= -1;
        ref_op = FermionOperatorCD("1^ 0", ref_coeff);
        fermion_op_str = "0 1^";
    }
    SECTION("1 0 2") {
        ref_op = FermionOperatorCD("2 1 0", ref_coeff);
        fermion_op_str = "1 0 2";
    }
    SECTION("1 4 1^") {
        ref_coeff *= -1;
        ref_op = FermionOperatorCD("4", ref_coeff) + FermionOperatorCD("1^ 4 1", ref_coeff);
        fermion_op_str = "1 4 1^";
    }

    const auto normal_ordered = FermionOperatorCD(fermion_op_str, coeff).normal_ordered();
    INFO("normal_ordered = FermionOperator(\"" << fermion_op_str << "\").normal_ordered()");
    CHECK(normal_ordered == ref_op);
}

// =============================================================================

TEST_CASE("FermionOperator matrix", "[fermion_op][ops]") {
    using matrix_t = FermionOperatorCD::sparse_matrix_t;

    std::optional<matrix_t> ref_mat;
    std::optional<matrix_t> actual_mat;

    static constexpr auto do_test = [](const auto& ref_mat, const auto& actual_mat) {
        if (ref_mat.has_value()) {
            const auto& ref = ref_mat.value();
            REQUIRE(actual_mat.has_value());
            const auto& mat = actual_mat.value();

            REQUIRE(mat.rows() == ref.rows());
            REQUIRE(mat.cols() == ref.cols());
            CHECK_THAT(mat, Equals(ref));
        } else {
            CHECK(!actual_mat.has_value());
        }
    };

    SECTION("Invalid empty") {
        actual_mat = FermionOperatorCD().sparse_matrix();
        do_test(ref_mat, actual_mat);
    }

    SECTION("I") {
        ref_mat = matrix_t{2, 2};
        ref_mat.value().setIdentity();

        actual_mat = FermionOperatorCD::identity().sparse_matrix(1);
        do_test(ref_mat, actual_mat);
    }
    SECTION("0") {
        ref_mat = matrix_t{2, 2};
        ref_mat->insert(0, 1) = 1.;
        actual_mat = FermionOperatorCD("0").sparse_matrix();
        do_test(ref_mat, actual_mat);
    }
    SECTION("0^") {
        ref_mat = matrix_t{2, 2};
        ref_mat->insert(1, 0) = 1.;
        actual_mat = FermionOperatorCD("0^").sparse_matrix();
        do_test(ref_mat, actual_mat);
    }
    SECTION("1") {
        ref_mat = matrix_t{4, 4};
        ref_mat->insert(0, 2) = 1.;
        ref_mat->insert(1, 3) = -1.;
        actual_mat = FermionOperatorCD("1").sparse_matrix();
        do_test(ref_mat, actual_mat);
    }
    SECTION("1^") {
        ref_mat = matrix_t{4, 4};
        ref_mat->insert(2, 0) = 1.;
        ref_mat->insert(3, 1) = -1.;
        actual_mat = FermionOperatorCD("1^").sparse_matrix();
        do_test(ref_mat, actual_mat);
    }

    // ---------------------------------

#define ADD_SECTION_FOR(terms_str, x_idx, y_idx, value)                                                                \
    SECTION(terms_str) {                                                                                               \
        ref_mat = matrix_t{4, 4};                                                                                      \
        ref_mat->insert(x_idx, y_idx) = value;                                                                         \
        actual_mat = FermionOperatorCD(terms_str).sparse_matrix();                                                     \
        do_test(ref_mat, actual_mat);                                                                                  \
    }

    ADD_SECTION_FOR("0 1", 0, 3, -1.)
    ADD_SECTION_FOR("0 1^", 2, 1, -1.)
    ADD_SECTION_FOR("0^ 1", 1, 2, 1.)
    ADD_SECTION_FOR("0^ 1^", 3, 0, 1.)

    // ---------------------------------

    ADD_SECTION_FOR("1 0", 0, 3, 1.)
    ADD_SECTION_FOR("1 0^", 1, 2, -1.)
    ADD_SECTION_FOR("1^ 0", 2, 1, 1.)
    ADD_SECTION_FOR("1^ 0^", 3, 0, -1.)
}

// =============================================================================
