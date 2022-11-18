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

#include <cstdint>

#include <fmt/format.h>
#include <pybind11/cast.h>
#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "details/define_terms_ops.hpp"
#include "ops/gates/details/coeff_policy.hpp"
#include "ops/gates/details/parameter_resolver_coeff_policy.hpp"
#include "ops/gates/fermion_operator.hpp"
#include "ops/gates/qubit_operator.hpp"
#include "ops/gates/terms_operator_base.hpp"
#include "ops/transform/jordan_wigner.hpp"
#include "ops/transform/parity.hpp"

#include "python/core/boost_multi_index.hpp"

namespace ops = mindquantum::ops;
namespace py = pybind11;

void init_transform(py::module& module);          // NOLINT(runtime/references)
void init_fermion_operators(py::module& module);  // NOLINT(runtime/references)
void init_qubit_operators(py::module& module);    // NOLINT(runtime/references)

void init_terms_operators(pybind11::module& module) {  // NOLINT(runtime/references)
    namespace mq = mindquantum;

    auto term_value = py::enum_<mindquantum::ops::TermValue>(module, "TermValue")
                          .value("I", mindquantum::ops::TermValue::I)
                          .value("X", mindquantum::ops::TermValue::X)
                          .value("Y", mindquantum::ops::TermValue::Y)
                          .value("Z", mindquantum::ops::TermValue::Z)
                          .value("a", mindquantum::ops::TermValue::a)
                          .value("adg", mindquantum::ops::TermValue::adg)
                          .def(
                              "__lt__",
                              [](const mindquantum::ops::TermValue& lhs, const mindquantum::ops::TermValue& rhs)
                                  -> bool { return static_cast<uint8_t>(lhs) < static_cast<uint8_t>(rhs); },
                              pybind11::is_operator());

    term_value.attr("__repr__") = pybind11::cpp_function(
        [](const mindquantum::ops::TermValue& value) -> pybind11::str { return fmt::format("TermValue.{}", value); },
        pybind11::name("name"), pybind11::is_method(term_value));
    term_value.attr("__str__") = pybind11::cpp_function(
        [](const mindquantum::ops::TermValue& value) -> pybind11::str { return fmt::format("{}", value); },
        pybind11::name("name"), pybind11::is_method(term_value));

    using pr_t = mq::ParameterResolver<double>;
    using pr_cmplx_t = mq::ParameterResolver<std::complex<double>>;

    /* These types are used when one wants to replace some parameters inside a FermionOperator or QubitOperator.
     * The two types for double and std::complex<double> do not do anything in practice but are defined anyway in order
     * to have a consistent API.
     */
    py::class_<ops::details::CoeffSubsProxy<double>>(module, "DoubleSubsProxy",
                                                     "Substitution proxy class for floating point numbers")
        .def(py::init<double>());
    py::class_<ops::details::CoeffSubsProxy<std::complex<double>>>(module, "CmplxDoubleSubsProxy",
                                                                   "Substitution proxy class for complex numbers")
        .def(py::init<std::complex<double>>());
    py::class_<ops::details::CoeffSubsProxy<pr_t>>(module, "DoublePRSubsProxy",
                                                   "Substitution proxy class for mqbackend.real_pr")
        .def(py::init<pr_t>());
    py::class_<ops::details::CoeffSubsProxy<pr_cmplx_t>>(module, "CmplxPRSubsProxy",
                                                         "Substitution proxy class for mqbackend.complex_pr")
        .def(py::init<pr_cmplx_t>());

    module.attr("EQ_TOLERANCE") = py::float_(ops::details::EQ_TOLERANCE);

    // -----------------------------------------------------------------------------

    init_fermion_operators(module);
    init_qubit_operators(module);

    py::module trans = module.def_submodule("transform", "MindQuantum-C++ operators transform");
    init_transform(trans);
}

void init_transform(py::module& module) {  // NOLINT(runtime/references)
    using namespace pybind11::literals;    // NOLINT(build/namespaces_literals)

    namespace transform = mindquantum::ops::transform;

    using bindops::fop_t;
    using bindops::qop_t;
    using pr_t = mindquantum::ParameterResolver<double>;
    using pr_cmplx_t = mindquantum::ParameterResolver<std::complex<double>>;

    module.def("parity", &transform::parity<fop_t<double>>, "ops"_a, "n_qubits"_a = -1);
    module.def("parity", &transform::parity<fop_t<std::complex<double>>>, "ops"_a, "n_qubits"_a = -1);
    module.def("parity", &transform::parity<fop_t<pr_t>>, "ops"_a, "n_qubits"_a = -1);
    module.def("parity", &transform::parity<fop_t<pr_cmplx_t>>, "ops"_a, "n_qubits"_a = -1);

    module.def("reverse_jordan_wigner", &transform::reverse_jordan_wigner<qop_t<double>>);
    module.def("reverse_jordan_wigner", &transform::reverse_jordan_wigner<qop_t<std::complex<double>>>);
    module.def("reverse_jordan_wigner", &transform::reverse_jordan_wigner<qop_t<pr_t>>);
    module.def("reverse_jordan_wigner", &transform::reverse_jordan_wigner<qop_t<pr_cmplx_t>>);

    module.def("jordan_wigner", &transform::jordan_wigner<fop_t<double>>);
    module.def("jordan_wigner", &transform::jordan_wigner<fop_t<std::complex<double>>>);
    module.def("jordan_wigner", &transform::jordan_wigner<fop_t<pr_t>>);
    module.def("jordan_wigner", &transform::jordan_wigner<fop_t<pr_cmplx_t>>);
}
