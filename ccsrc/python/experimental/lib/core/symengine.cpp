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
#include <cstdint>
#include <optional>

#include <symengine/add.h>
#include <symengine/basic.h>
#include <symengine/complex.h>
#include <symengine/complex_double.h>
#include <symengine/constants.h>
#include <symengine/dict.h>
#include <symengine/functions.h>
#include <symengine/infinity.h>
#include <symengine/integer.h>
#include <symengine/mp_class.h>
#include <symengine/mul.h>
#include <symengine/number.h>
#include <symengine/pow.h>
#include <symengine/printers.h>
#include <symengine/rational.h>
#include <symengine/real_double.h>
#include <symengine/symbol.h>

#include <pybind11/complex.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sys/types.h>

#include "python/bindings.hpp"

using rcp_const_basic = SymEngine::RCP<const SymEngine::Basic>;

template <typename T>
using rcp_const = SymEngine::RCP<const T>;

PYBIND11_DECLARE_HOLDER_TYPE(T, rcp_const<T>);

namespace py = pybind11;

#define PYBIND11_DECLARE_SYMENGINE_TYPE(klass) py::class_<SymEngine::klass, rcp_const<SymEngine::klass>>(module, #klass)
#define PYBIND11_DECLARE_SYMENGINE_TYPE2(klass, base)                                                                  \
    py::class_<SymEngine::klass, SymEngine::base, rcp_const<SymEngine::klass>>(module, #klass)
#define DEF_SYMENGINE_METHOD(klass, name)              def(#name, &SymEngine::klass::name)
#define DEF_SYMENGINE_METHOD_LAMBDA(klass, name, body) def(name, [](const SymEngine::klass& k) { return body; })

namespace mindquantum::python {
void init_symengine_basic_types(pybind11::module& module) {
    PYBIND11_DECLARE_SYMENGINE_TYPE(Basic)
        .DEF_SYMENGINE_METHOD(Basic, hash)
        .DEF_SYMENGINE_METHOD(Basic, __eq__)
        .DEF_SYMENGINE_METHOD(Basic, __cmp__)
        .DEF_SYMENGINE_METHOD(Basic, __hash__)
        .DEF_SYMENGINE_METHOD(Basic, __str__)
        .DEF_SYMENGINE_METHOD(Basic, subs)
        .DEF_SYMENGINE_METHOD(Basic, xreplace)
        .def(
            "__add__", [](const rcp_const_basic& a, const rcp_const_basic& b) { return SymEngine::add(a, b); },
            py::is_operator())
        .def(
            "__sub__", [](const rcp_const_basic& a, const rcp_const_basic& b) { return SymEngine::sub(a, b); },
            py::is_operator())
        .def(
            "__mul__", [](const rcp_const_basic& a, const rcp_const_basic& b) { return SymEngine::mul(a, b); },
            py::is_operator())
        .def(
            "__truediv__", [](const rcp_const_basic& a, const rcp_const_basic& b) { return SymEngine::div(a, b); },
            py::is_operator())
        .def(
            "__pow__", [](const rcp_const_basic& a, const rcp_const_basic& b) { return SymEngine::pow(a, b); },
            py::is_operator())
        .def(
            "__neg__", [](const rcp_const_basic& a) { return SymEngine::neg(a); }, py::is_operator())
        .def(
            "__abs__", [](const rcp_const_basic& a) { return SymEngine::abs(a); }, py::is_operator())
        .DEF_SYMENGINE_METHOD_LAMBDA(Basic, "is_Atom", false)
        .DEF_SYMENGINE_METHOD_LAMBDA(Basic, "is_Symbol", false)
        .DEF_SYMENGINE_METHOD_LAMBDA(Basic, "is_symbol", false)
        .DEF_SYMENGINE_METHOD_LAMBDA(Basic, "is_Dummy", false)
        .DEF_SYMENGINE_METHOD_LAMBDA(Basic, "is_Function", false)
        .DEF_SYMENGINE_METHOD_LAMBDA(Basic, "is_Add", false)
        .DEF_SYMENGINE_METHOD_LAMBDA(Basic, "is_Mul", false)
        .DEF_SYMENGINE_METHOD_LAMBDA(Basic, "is_Pow", false)
        .DEF_SYMENGINE_METHOD_LAMBDA(Basic, "is_Number", false)
        .DEF_SYMENGINE_METHOD_LAMBDA(Basic, "is_number", std::nullopt)
        .DEF_SYMENGINE_METHOD_LAMBDA(Basic, "is_Float", false)
        .DEF_SYMENGINE_METHOD_LAMBDA(Basic, "is_Rational", false)
        .DEF_SYMENGINE_METHOD_LAMBDA(Basic, "is_Integer", false)
        .DEF_SYMENGINE_METHOD_LAMBDA(Basic, "is_integer", false)
        .DEF_SYMENGINE_METHOD_LAMBDA(Basic, "is_finite", std::nullopt)
        .DEF_SYMENGINE_METHOD_LAMBDA(Basic, "is_Derivative", false)
        .DEF_SYMENGINE_METHOD_LAMBDA(Basic, "is_Relational", false)
        .DEF_SYMENGINE_METHOD_LAMBDA(Basic, "is_Equality", false)
        .DEF_SYMENGINE_METHOD_LAMBDA(Basic, "is_Boolean", false)
        .DEF_SYMENGINE_METHOD_LAMBDA(Basic, "is_Not", false)
        .DEF_SYMENGINE_METHOD_LAMBDA(Basic, "is_Matrix", false)
        .DEF_SYMENGINE_METHOD_LAMBDA(Basic, "is_zero", false)
        .DEF_SYMENGINE_METHOD_LAMBDA(Basic, "is_positive", false)
        .DEF_SYMENGINE_METHOD_LAMBDA(Basic, "is_negative", false)
        .DEF_SYMENGINE_METHOD_LAMBDA(Basic, "is_nonpositive", false)
        .DEF_SYMENGINE_METHOD_LAMBDA(Basic, "is_nonnegative", false)
        .DEF_SYMENGINE_METHOD_LAMBDA(Basic, "is_real", false);

    // -------------------------------------------------------------------------

    PYBIND11_DECLARE_SYMENGINE_TYPE2(Constant, Basic)
        .def(py::init<const std::string&>())
        .DEF_SYMENGINE_METHOD_LAMBDA(Basic, "is_number", true);

    // -------------------------------------------------------------------------

    PYBIND11_DECLARE_SYMENGINE_TYPE2(Number, Basic)
        .DEF_SYMENGINE_METHOD_LAMBDA(Number, "is_Atom", true)
        .DEF_SYMENGINE_METHOD_LAMBDA(Number, "is_Number", true)
        .DEF_SYMENGINE_METHOD_LAMBDA(Number, "is_number", true)
        .DEF_SYMENGINE_METHOD_LAMBDA(Number, "is_commutative", true)
        .DEF_SYMENGINE_METHOD_LAMBDA(
            Basic, "is_positive", SymEngine::rcp_static_cast<const SymEngine::Number>(k.rcp_from_this())->is_positive())
        .DEF_SYMENGINE_METHOD_LAMBDA(
            Basic, "is_negative", SymEngine::rcp_static_cast<const SymEngine::Number>(k.rcp_from_this())->is_negative())
        .DEF_SYMENGINE_METHOD_LAMBDA(
            Basic, "is_complex", SymEngine::rcp_static_cast<const SymEngine::Number>(k.rcp_from_this())->is_complex())
        .DEF_SYMENGINE_METHOD_LAMBDA(Number, "is_nonzero", !(k.is_complex() || k.is_zero()))
        .DEF_SYMENGINE_METHOD_LAMBDA(Number, "is_nonnegative", !(k.is_complex() || k.is_negative()))
        .DEF_SYMENGINE_METHOD_LAMBDA(Number, "is_nonpositive", !(k.is_complex() || k.is_positive()))
        .DEF_SYMENGINE_METHOD_LAMBDA(Number, "real", k.rcp_from_this())
        .DEF_SYMENGINE_METHOD_LAMBDA(Number, "imag", SymEngine::integer(0));

    // -------------------------------------------------------------------------

    PYBIND11_DECLARE_SYMENGINE_TYPE2(Rational, Number)
        .DEF_SYMENGINE_METHOD_LAMBDA(Rational, "is_Rational", true)
        .DEF_SYMENGINE_METHOD_LAMBDA(Rational, "is_rational", true)
        .DEF_SYMENGINE_METHOD_LAMBDA(Rational, "is_real", true)
        .DEF_SYMENGINE_METHOD_LAMBDA(Rational, "is_finite", true)
        .DEF_SYMENGINE_METHOD_LAMBDA(Rational, "is_integer", false);

    // -------------------------------------------------------------------------

    PYBIND11_DECLARE_SYMENGINE_TYPE2(Integer, Number)
        .DEF_SYMENGINE_METHOD_LAMBDA(Rational, "is_Rational", true)
        .DEF_SYMENGINE_METHOD_LAMBDA(Rational, "is_rational", true)
        .DEF_SYMENGINE_METHOD_LAMBDA(Rational, "is_real", true)
        .DEF_SYMENGINE_METHOD_LAMBDA(Rational, "is_finite", true)
        .DEF_SYMENGINE_METHOD_LAMBDA(Integer, "is_Integer", true)
        .DEF_SYMENGINE_METHOD_LAMBDA(Integer, "is_integer", true)
        .DEF_SYMENGINE_METHOD_LAMBDA(Integer, "is_integer", false);

    module.def("integer", SymEngine::integer<int32_t>)
        .def("integer", SymEngine::integer<int64_t>)
        .def("integer", SymEngine::integer<uint32_t>)
        .def("integer", SymEngine::integer<uint64_t>);

    // -------------------------------------------------------------------------

    PYBIND11_DECLARE_SYMENGINE_TYPE2(RealDouble, Number)
        .DEF_SYMENGINE_METHOD_LAMBDA(RealDouble, "is_Rational", std::nullopt)
        .DEF_SYMENGINE_METHOD_LAMBDA(RealDouble, "is_irrational", std::nullopt)
        .DEF_SYMENGINE_METHOD_LAMBDA(RealDouble, "is_real", true)
        .DEF_SYMENGINE_METHOD_LAMBDA(RealDouble, "is_Float", true)
        .DEF_SYMENGINE_METHOD_LAMBDA(
            Basic, "__float__", SymEngine::rcp_static_cast<const SymEngine::RealDouble>(k.rcp_from_this())->as_double())
        .DEF_SYMENGINE_METHOD_LAMBDA(
            Basic, "__complex__",
            std::complex<double>{
                SymEngine::rcp_static_cast<const SymEngine::RealDouble>(k.rcp_from_this())->as_double()});

    module.def("real_double", SymEngine::real_double);

    // -------------------------------------------------------------------------

    PYBIND11_DECLARE_SYMENGINE_TYPE2(ComplexBase, Number)
        .DEF_SYMENGINE_METHOD_LAMBDA(
            Basic, "real_part",
            SymEngine::rcp_static_cast<const SymEngine::ComplexBase>(k.rcp_from_this())->real_part())
        .DEF_SYMENGINE_METHOD_LAMBDA(
            Basic, "imaginary_part",
            SymEngine::rcp_static_cast<const SymEngine::ComplexBase>(k.rcp_from_this())->imaginary_part())
        .DEF_SYMENGINE_METHOD_LAMBDA(ComplexBase, "real", k.real_part())
        .DEF_SYMENGINE_METHOD_LAMBDA(ComplexBase, "imag", k.imaginary_part());

    // -------------------------------------------------------------------------

    PYBIND11_DECLARE_SYMENGINE_TYPE2(ComplexDouble, ComplexBase)
        .DEF_SYMENGINE_METHOD_LAMBDA(
            Basic, "__complex__",
            SymEngine::rcp_static_cast<const SymEngine::ComplexDouble>(k.rcp_from_this())->as_complex_double());

    module.def("complex_double", SymEngine::complex_double);

    // -------------------------------------------------------------------------

    PYBIND11_DECLARE_SYMENGINE_TYPE2(Symbol, Basic)
        .DEF_SYMENGINE_METHOD_LAMBDA(Symbol, "is_Atom", true)
        .DEF_SYMENGINE_METHOD_LAMBDA(Symbol, "is_Symbol", true)
        .DEF_SYMENGINE_METHOD_LAMBDA(Symbol, "is_symbol", true)
        .DEF_SYMENGINE_METHOD_LAMBDA(Symbol, "is_commutative", true);

    module.def("symbol", SymEngine::symbol);

    // -------------------------------------------------------------------------

    PYBIND11_DECLARE_SYMENGINE_TYPE2(Dummy, Symbol).DEF_SYMENGINE_METHOD_LAMBDA(Dummy, "is_Dummy", true);

    module.def("dummy", static_cast<rcp_const<SymEngine::Dummy> (*)()>(SymEngine::dummy));
    module.def("dummy", static_cast<rcp_const<SymEngine::Dummy> (*)(const std::string&)>(SymEngine::dummy));

    // -------------------------------------------------------------------------

    PYBIND11_DECLARE_SYMENGINE_TYPE2(Infty, Number).DEF_SYMENGINE_METHOD_LAMBDA(Symbol, "is_infinite", true);

    PYBIND11_DECLARE_SYMENGINE_TYPE2(NaN, Number)
        .DEF_SYMENGINE_METHOD_LAMBDA(Symbol, "is_rational", std::nullopt)
        .DEF_SYMENGINE_METHOD_LAMBDA(Symbol, "is_integer", std::nullopt)
        .DEF_SYMENGINE_METHOD_LAMBDA(Symbol, "is_real", std::nullopt)
        .DEF_SYMENGINE_METHOD_LAMBDA(Symbol, "is_finite", std::nullopt);
}

void init_symengine_constants(pybind11::module& module) {
    module.attr("zero") = SymEngine::zero;
    module.attr("one") = SymEngine::one;
    module.attr("two") = SymEngine::two;
    module.attr("minus_one") = SymEngine::minus_one;
    module.attr("pi") = SymEngine::pi;
    module.attr("I") = SymEngine::I;
    module.attr("E") = SymEngine::E;
    module.attr("EulerGamma") = SymEngine::EulerGamma;
    module.attr("Catalan") = SymEngine::Catalan;
    module.attr("GoldenRatio") = SymEngine::GoldenRatio;
    module.attr("Inf") = SymEngine::Inf;
    module.attr("NegInf") = SymEngine::NegInf;
    module.attr("ComplexInf") = SymEngine::ComplexInf;
    module.attr("Nan") = SymEngine::Nan;
}

void init_symengine_functions(pybind11::module& module) {
    module.def("acos", SymEngine::acos);
    module.def("acosh", SymEngine::acosh);
    module.def("acot", SymEngine::acot);
    module.def("acoth", SymEngine::acoth);
    module.def("acsc", SymEngine::acsc);
    module.def("acsch", SymEngine::acsch);
    module.def("asec", SymEngine::asec);
    module.def("asech", SymEngine::asech);
    module.def("asin", SymEngine::asin);
    module.def("asinh", SymEngine::asinh);
    module.def("atan", SymEngine::atan);
    module.def("atan2", SymEngine::atan2);
    module.def("atanh", SymEngine::atanh);
    module.def("beta", SymEngine::beta);
    module.def("ceiling", SymEngine::ceiling);
    module.def("conjugate", SymEngine::conjugate);
    module.def("cos", SymEngine::cos);
    module.def("cosh", SymEngine::cosh);
    module.def("cot", SymEngine::cot);
    module.def("coth", SymEngine::coth);
    module.def("csc", SymEngine::csc);
    module.def("csch", SymEngine::csch);
    module.def("erf", SymEngine::erf);
    module.def("erfc", SymEngine::erfc);
    module.def("floor", SymEngine::floor);
    module.def("gamma", SymEngine::gamma);
    module.def("kronecker_delta", SymEngine::kronecker_delta);
    module.def("log", static_cast<rcp_const_basic (*)(const rcp_const_basic&)>(SymEngine::log));
    module.def("log", static_cast<rcp_const_basic (*)(const rcp_const_basic&, const rcp_const_basic&)>(SymEngine::log));
    module.def("loggamma", SymEngine::loggamma);
    module.def("max", SymEngine::max);
    module.def("min", SymEngine::min);
    module.def("sec", SymEngine::sec);
    module.def("sech", SymEngine::sech);
    module.def("sign", SymEngine::sign);
    module.def("sin", SymEngine::sin);
    module.def("sinh", SymEngine::sinh);
    module.def("tan", SymEngine::tan);
    module.def("tanh", SymEngine::tanh);
    module.def("zeta", static_cast<rcp_const_basic (*)(const rcp_const_basic&)>(SymEngine::zeta));
    module.def("zeta",
               static_cast<rcp_const_basic (*)(const rcp_const_basic&, const rcp_const_basic&)>(SymEngine::zeta));
}
}  // namespace mindquantum::python

#undef PYBIND11_DECLARE_SYMENGINE_TYPE
#undef PYBIND11_DECLARE_SYMENGINE_TYPE2
#undef DEF_SYMENGINE_METHOD
#undef DEF_SYMENGINE_METHOD_LAMBDA

void mindquantum::python::init_symengine(pybind11::module& module) {
    init_symengine_basic_types(module);
    init_symengine_constants(module);
    init_symengine_functions(module);
}
