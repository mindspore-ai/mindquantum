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
#include <iostream>

#include <boost/range/combine.hpp>
#include <catch2/catch.hpp>
#include <spdlog/common.h>
#include <spdlog/spdlog.h>

#include "ops/gates/terms_operator.hpp"
#include "ops/utils.hpp"

// =============================================================================

namespace {
struct DummyOperator : mindquantum::ops::TermsOperator<DummyOperator> {
    using TermsOperator::TermsOperator;

    static std::tuple<std::vector<term_t>, coefficient_t> simplify_(const std::vector<term_t>& terms,
                                                                    coefficient_t coeff) {
        return {terms, coeff};
    }
};
}  // namespace

using namespace std::literals::complex_literals;
using TermValue = mindquantum::ops::TermValue;
using term_t = DummyOperator::term_t;
using terms_t = DummyOperator::terms_t;
using complex_term_dict_t = DummyOperator::complex_term_dict_t;

TEST_CASE("TermsOperator constructor", "[terms_op][ops]") {
    MQ_DISABLE_LOGGING;
    complex_term_dict_t ref_terms;

    SECTION("Default constructor") {
        DummyOperator op;
        CHECK(std::empty(op));
        CHECK(std::size(op) == 0);
        CHECK(std::empty(op.get_terms()));
        CHECK(std::size(op.get_terms()) == 0);
        CHECK(op.num_targets() == 0);
        CHECK(op.count_qubits() == 0);
        CHECK(op.is_identity());
        CHECK(!op.is_singlet());
        CHECK(std::empty(op.singlet()));
    }
    SECTION("Identity") {
        DummyOperator identity{terms_t{}};
        CHECK(!std::empty(identity));
        CHECK(std::size(identity) == 1);
        CHECK(!std::empty(identity.get_terms()));
        CHECK(std::size(identity.get_terms()) == 1);
        CHECK(identity.num_targets() == 1);
        CHECK(identity.count_qubits() == 1);
        CHECK(identity.is_identity());
        CHECK(identity.is_singlet());
    }
    SECTION("Single term constructor") {
        const auto ref_term = term_t{0, TermValue::X};
        ref_terms.emplace(terms_t{ref_term}, 1);

        DummyOperator op{ref_term};
        CHECK(!std::empty(op));
        CHECK(std::size(op) == 1);
        CHECK(!std::empty(op.get_terms()));
        REQUIRE(std::size(op.get_terms()) == 1);
        CHECK(op.get_terms() == ref_terms);

        CHECK(op.num_targets() == 1);
        CHECK(op.count_qubits() == 1);
        CHECK(!op.is_identity());
        CHECK(op.is_singlet());
    }
    SECTION("Single term multiple local_ops constructor") {
        const auto ref_term = terms_t{{0, TermValue::X}, {1, TermValue::X}};
        const auto coeff = 0.5;
        ref_terms.emplace(ref_term, coeff);

        DummyOperator op{ref_term, coeff};
        CHECK(!std::empty(op));
        CHECK(std::size(op) == 1);
        CHECK(!std::empty(op.get_terms()));
        REQUIRE(std::size(op.get_terms()) == 1);
        CHECK(op.get_terms() == ref_terms);

        CHECK(op.num_targets() == 2);
        CHECK(op.count_qubits() == 2);
        CHECK(!op.is_identity());
        CHECK(op.is_singlet());

        const auto singlet_terms = op.singlet();
        REQUIRE(std::size(ref_term) == std::size(singlet_terms));
        CHECK(coeff == op.singlet_coeff());

        for (const auto& [ref, test_op] : boost::combine(ref_term, singlet_terms)) {
            const auto& test_terms = test_op.get_terms();
            REQUIRE(std::size(test_terms) == 1);
            CHECK(begin(test_terms)->second == 1.);
            const auto& test = begin(test_terms)->first;
            REQUIRE(std::size(test) == 1);
            CHECK(ref == test.front());
        }
    }
    SECTION("Multiple terms constructor") {
        ref_terms.emplace(terms_t{{0, TermValue::Y}, {3, TermValue::Z}}, 0.75i);
        ref_terms.emplace(terms_t{{1, TermValue::X}}, 2.);

        DummyOperator op{ref_terms};
        CHECK(!std::empty(op));
        CHECK(std::size(op) == 2);
        CHECK(!std::empty(op.get_terms()));
        REQUIRE(std::size(op.get_terms()) == 2);
        CHECK(op.get_terms() == ref_terms);

        CHECK(op.num_targets() == 4);
        CHECK(op.count_qubits() == 4);
        CHECK(!op.is_identity());
        CHECK(!op.is_singlet());
    }
}

// =============================================================================

TEST_CASE("TermsOperator identity", "[terms_op][ops]") {
    SECTION("Empty term operator") {
        DummyOperator op;
        CHECK(op.is_identity());

        CHECK(op.constant() == 0.);
        CHECK(std::empty(op));

        const auto coeff = 2. + 3.i;
        op.constant(coeff);  // Will insert an empty term {}
        CHECK(!std::empty(op));
        CHECK(op.constant() == 2. + 3.i);
        CHECK(op.is_singlet());
    }
    SECTION("Empty term operator") {
        auto const coeff = 1. + 2.i;
        DummyOperator op{terms_t{}, coeff};
        CHECK(op.is_identity());
        CHECK(op.constant() == coeff);
        CHECK(op.is_singlet());
    }
    SECTION("Single term operator") {
        DummyOperator op{term_t{0, TermValue::X}, 1.e-5};
        CHECK(!op.is_identity());
        CHECK(op.is_identity(1.e-5));
        CHECK(op.is_singlet());
    }
    SECTION("Single term multiple local_ops operator") {
        const auto terms = terms_t{{0, TermValue::X}, {2, TermValue::Z}};
        DummyOperator op{terms, 1.e-4};
        CHECK(!op.is_identity());
        CHECK(op.is_identity(1.e-4));
        CHECK(op.is_singlet());

        const auto singlet_terms = op.singlet();
        REQUIRE(std::size(singlet_terms) == std::size(terms));
    }
    SECTION("Single term multiple local_ops operator") {
        auto terms = complex_term_dict_t{};
        terms.emplace(terms_t{{0, TermValue::Y}}, 1.e-5);
        terms.emplace(terms_t{{2, TermValue::Z}, {4, TermValue::X}}, 1.e-6);
        DummyOperator op{terms};
        CHECK(!op.is_identity());
        CHECK(!op.is_identity(1.e-6));
        CHECK(op.is_identity(1.e-5));
    }
    SECTION("Single term multiple local_ops including I operator") {
        auto terms = complex_term_dict_t{};
        terms.emplace(terms_t{{0, TermValue::I}, {1, TermValue::I}}, 2i);
        terms.emplace(terms_t{{3, TermValue::X}}, DummyOperator::EQ_TOLERANCE / 10.);
        terms.emplace(terms_t{{1, TermValue::I}}, 1);
        DummyOperator op{terms};
        CHECK(op.is_identity());
        CHECK(!op.is_singlet());
    }
}

// =============================================================================

TEST_CASE("TermsOperator real/imag", "[terms_op][ops]") {
    auto terms = complex_term_dict_t{};
    terms.emplace(terms_t{{0, TermValue::Y}}, 1. + 2.i);
    terms.emplace(terms_t{{2, TermValue::Z}, {4, TermValue::X}}, 3. + 4.i);
    DummyOperator op{terms};

    auto real = op.real();
    REQUIRE(std::size(real) == std::size(op));
    for (const auto& [orig_terms, real_terms] : boost::combine(op.get_terms(), real.get_terms())) {
        CHECK(orig_terms.first == real_terms.first);
        CHECK(std::real(orig_terms.second) == real_terms.second);
    }

    auto imag = op.imag();
    REQUIRE(std::size(imag) == std::size(op));
    for (const auto& [orig_terms, imag_terms] : boost::combine(op.get_terms(), imag.get_terms())) {
        CHECK(orig_terms.first == imag_terms.first);
        CHECK(std::imag(orig_terms.second) == imag_terms.second);
    }
}

// =============================================================================

TEST_CASE("TermsOperator compression", "[terms_op][ops]") {
    const auto coeff1 = 1.e-5;
    const auto terms1 = terms_t{{0, TermValue::Y}};
    const auto coeff2 = 1.e-6;
    const auto terms2 = terms_t{{2, TermValue::Z}, {4, TermValue::X}};

    auto terms = complex_term_dict_t{};
    terms.emplace(terms1, coeff1);
    terms.emplace(terms2, coeff2);
    DummyOperator op{terms};

    REQUIRE(std::size(op) == 2);
    op.compress(1.e-6);

    REQUIRE(std::size(op) == 1);
    REQUIRE(begin(op.get_terms())->first == terms1);
    REQUIRE(begin(op.get_terms())->second == coeff1);

    op.compress(1.e-4);
    REQUIRE(std::empty(op));
}

// =============================================================================

// TEST_CASE("TermsOperator constructor", "[terms_op][ops]") {
//     using namespace std::literals::complex_literals;

//     ::A op, other;
//     op += other;
//     op += 2.0;
//     op += 2.i;
//     op + other;

//     op -= other;
//     op -= 2.0;
//     op -= 2.i;
//     op - other;

//     op *= other;

//     op.pow(10);

//     CHECK(op == other);
// }

// =============================================================================
