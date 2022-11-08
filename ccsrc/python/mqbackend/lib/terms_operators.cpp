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

#include "python/core/tsl_ordered_map.hpp"

namespace ops = mindquantum::ops;
namespace py = pybind11;

void init_transform(py::module& module);  // NOLINT(runtime/references)

void init_terms_operators(pybind11::module& module) {  // NOLINT(runtime/references)
    namespace mq = mindquantum;
    namespace op = bindops::details;

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

    using all_scalar_types_t = std::tuple<double, std::complex<double>, pr_t, pr_cmplx_t>;

    // -----------------------------------------------------------------------------
    // FermionOperator

    // Register empty base class (for instance(X, FermionOperatorBase) purposes
    py::class_<ops::FermionOperatorBase, std::shared_ptr<ops::FermionOperatorBase>>(
        module, "FermionOperatorBase",
        "Base class for all C++ fermion operators. Use only for isinstance(obj, FermionOperatorBase) or use "
        "is_fermion_operator(obj)");
    module.def("is_fermion_operator", &pybind11::isinstance<ops::FermionOperatorBase>);

    // NB: pybind11 maps both float and double to Python float
    auto [fop_double, fop_cmplx_double, fop_pr_double, fop_pr_cmplx_double]
        = bindops::define_fermion_ops<double, std::complex<double>, pr_t, pr_cmplx_t>::apply(
            module, "FermionOperatorD", "FermionOperatorCD", "FermionOperatorPRD", "FermionOperatorPRCD");

    // ---------------------------------

    using FermionOperatorD = decltype(fop_double)::type;
    using FermionOperatorCD = decltype(fop_cmplx_double)::type;
    using FermionOperatorPRD = decltype(fop_pr_double)::type;
    using FermionOperatorPRCD = decltype(fop_pr_cmplx_double)::type;

    using all_fop_types_t = std::tuple<double, std::complex<double>, pr_t, pr_cmplx_t, FermionOperatorD,
                                       FermionOperatorCD, FermionOperatorPRD, FermionOperatorPRCD>;

    fop_double.def("cast",
                   bindops::cast<FermionOperatorD, double, std::complex<double>, pr_t, pr_cmplx_t, FermionOperatorD,
                                 FermionOperatorCD, FermionOperatorPRD, FermionOperatorPRCD>,
                   "Supported types: float, complex, ParameterResolver<double>, ParameterResolver<complex>, "
                   "FermionOperatorD, FermionOperatorCD, FermionOperatorPRD, FermionOperatorPRCD");
    fop_cmplx_double.def(
        "cast",
        bindops::cast<FermionOperatorCD, std::complex<double>, pr_cmplx_t, FermionOperatorCD, FermionOperatorPRCD>,
        "Supported types: complex, ParameterResolver<complex>, FermionOperatorCD, FermionOperatorPRCD");

    fop_pr_double.def("cast",
                      bindops::cast<FermionOperatorPRD, double, std::complex<double>, pr_t, pr_cmplx_t,
                                    FermionOperatorD, FermionOperatorCD, FermionOperatorPRD, FermionOperatorPRCD>,
                      "Supported types: float, complex, ParameterResolver<double>, ParameterResolver<complex>, "
                      "FermionOperatorD, FermionOperatorCD, FermionOperatorPRD, FermionOperatorPRCD");
    fop_pr_cmplx_double.def(
        "cast",
        bindops::cast<FermionOperatorPRCD, std::complex<double>, pr_cmplx_t, FermionOperatorCD, FermionOperatorPRCD>,
        "Supported types: complex, ParameterResolver<complex>, FermionOperatorCD, FermionOperatorPRCD");

    // ---------------------------------

    using fop_t = decltype(fop_double);
    bindops::binop_definition<op::plus, fop_t>::inplace<double>(fop_double);
    bindops::binop_definition<op::plus, fop_t>::external<all_scalar_types_t>(fop_double);
    bindops::binop_definition<op::plus, fop_t>::reverse<all_fop_types_t>(fop_double);
    bindops::binop_definition<op::minus, fop_t>::inplace<double>(fop_double);
    bindops::binop_definition<op::minus, fop_t>::external<all_scalar_types_t>(fop_double);
    bindops::binop_definition<op::minus, fop_t>::reverse<all_fop_types_t>(fop_double);
    bindops::binop_definition<op::times, fop_t>::inplace<double>(fop_double);
    bindops::binop_definition<op::times, fop_t>::external<all_scalar_types_t>(fop_double);
    bindops::binop_definition<op::times, fop_t>::reverse<all_fop_types_t>(fop_double);
    bindops::binop_definition<op::divides, fop_t>::inplace<double>(fop_double);
    bindops::binop_definition<op::divides, fop_t>::external<all_scalar_types_t>(fop_double);

    using fop_cmplx_t = decltype(fop_cmplx_double);
    bindops::binop_definition<op::plus, fop_cmplx_t>::inplace<double, std::complex<double>>(fop_cmplx_double);
    bindops::binop_definition<op::plus, fop_cmplx_t>::external<all_scalar_types_t>(fop_cmplx_double);
    bindops::binop_definition<op::plus, fop_cmplx_t>::reverse<all_fop_types_t>(fop_cmplx_double);
    bindops::binop_definition<op::minus, fop_cmplx_t>::inplace<double, std::complex<double>>(fop_cmplx_double);
    bindops::binop_definition<op::minus, fop_cmplx_t>::external<all_scalar_types_t>(fop_cmplx_double);
    bindops::binop_definition<op::minus, fop_cmplx_t>::reverse<all_fop_types_t>(fop_cmplx_double);
    bindops::binop_definition<op::times, fop_cmplx_t>::inplace<double, std::complex<double>>(fop_cmplx_double);
    bindops::binop_definition<op::times, fop_cmplx_t>::external<all_scalar_types_t>(fop_cmplx_double);
    bindops::binop_definition<op::times, fop_cmplx_t>::reverse<all_fop_types_t>(fop_cmplx_double);
    bindops::binop_definition<op::divides, fop_cmplx_t>::inplace<double, std::complex<double>>(fop_cmplx_double);
    bindops::binop_definition<op::divides, fop_cmplx_t>::external<all_scalar_types_t>(fop_cmplx_double);

    using fop_pr_t = decltype(fop_pr_double);
    bindops::binop_definition<op::plus, fop_pr_t>::inplace<double, pr_t>(fop_pr_double);
    bindops::binop_definition<op::plus, fop_pr_t>::external<all_scalar_types_t>(fop_pr_double);
    bindops::binop_definition<op::plus, fop_pr_t>::reverse<all_fop_types_t>(fop_pr_double);
    bindops::binop_definition<op::minus, fop_pr_t>::inplace<double, pr_t>(fop_pr_double);
    bindops::binop_definition<op::minus, fop_pr_t>::external<all_scalar_types_t>(fop_pr_double);
    bindops::binop_definition<op::minus, fop_pr_t>::reverse<all_fop_types_t>(fop_pr_double);
    bindops::binop_definition<op::times, fop_pr_t>::inplace<double, pr_t>(fop_pr_double);
    bindops::binop_definition<op::times, fop_pr_t>::external<all_scalar_types_t>(fop_pr_double);
    bindops::binop_definition<op::times, fop_pr_t>::reverse<all_fop_types_t>(fop_pr_double);
    bindops::binop_definition<op::divides, fop_pr_t>::inplace<double, pr_t>(fop_pr_double);
    bindops::binop_definition<op::divides, fop_pr_t>::external<all_scalar_types_t>(fop_pr_double);

    using fop_pr_cmplx_t = decltype(fop_pr_cmplx_double);
    bindops::binop_definition<op::plus, fop_pr_cmplx_t>::inplace<all_scalar_types_t>(fop_pr_cmplx_double);
    bindops::binop_definition<op::plus, fop_pr_cmplx_t>::external<all_scalar_types_t>(fop_pr_cmplx_double);
    bindops::binop_definition<op::plus, fop_pr_cmplx_t>::reverse<all_fop_types_t>(fop_pr_cmplx_double);
    bindops::binop_definition<op::minus, fop_pr_cmplx_t>::inplace<all_scalar_types_t>(fop_pr_cmplx_double);
    bindops::binop_definition<op::minus, fop_pr_cmplx_t>::external<all_scalar_types_t>(fop_pr_cmplx_double);
    bindops::binop_definition<op::minus, fop_pr_cmplx_t>::reverse<all_fop_types_t>(fop_pr_cmplx_double);
    bindops::binop_definition<op::times, fop_pr_cmplx_t>::inplace<all_scalar_types_t>(fop_pr_cmplx_double);
    bindops::binop_definition<op::times, fop_pr_cmplx_t>::external<all_scalar_types_t>(fop_pr_cmplx_double);
    bindops::binop_definition<op::times, fop_pr_cmplx_t>::reverse<all_fop_types_t>(fop_pr_cmplx_double);
    bindops::binop_definition<op::divides, fop_pr_cmplx_t>::inplace<all_scalar_types_t>(fop_pr_cmplx_double);
    bindops::binop_definition<op::divides, fop_pr_cmplx_t>::external<all_scalar_types_t>(fop_pr_cmplx_double);

    // -----------------------------------------------------------------------------
    // QubitOperator

    // Register empty base class (for instance(X, QubitOperatorBase) purposes
    py::class_<ops::QubitOperatorBase, std::shared_ptr<ops::QubitOperatorBase>>(
        module, "QubitOperatorBase",
        "Base class for all C++ qubit operators. Use only for isinstance(obj, QubitOperatorBase) or use "
        "is_qubit_operator(obj)");
    module.def("is_qubit_operator", &pybind11::isinstance<ops::QubitOperatorBase>);

    // NB: pybind11 maps both float and double to Python float
    auto [qop_double, qop_cmplx_double, qop_pr_double, qop_pr_cmplx_double]
        = bindops::define_qubit_ops<double, std::complex<double>, pr_t, pr_cmplx_t>::apply(
            module, "QubitOperatorD", "QubitOperatorCD", "QubitOperatorPRD", "QubitOperatorPRCD");

    // ---------------------------------

    using QubitOperatorD = decltype(qop_double)::type;
    using QubitOperatorCD = decltype(qop_cmplx_double)::type;
    using QubitOperatorPRD = decltype(qop_pr_double)::type;
    using QubitOperatorPRCD = decltype(qop_pr_cmplx_double)::type;

    using all_qop_types_t = std::tuple<double, std::complex<double>, pr_t, pr_cmplx_t, QubitOperatorD, QubitOperatorCD,
                                       QubitOperatorPRD, QubitOperatorPRCD>;

    qop_double.def("cast",
                   bindops::cast<QubitOperatorD, double, std::complex<double>, pr_t, pr_cmplx_t, QubitOperatorD,
                                 QubitOperatorCD, QubitOperatorPRD, QubitOperatorPRCD>,
                   "Supported types: float, complex, ParameterResolver<double>, ParameterResolver<complex>, "
                   "QubitOperatorD, QubitOperatorCD, QubitOperatorPRD, QubitOperatorPRCD");
    qop_cmplx_double.def(
        "cast", bindops::cast<QubitOperatorCD, std::complex<double>, pr_cmplx_t, QubitOperatorCD, QubitOperatorPRCD>,
        "Supported types: complex, ParameterResolver<complex>, QubitOperatorCD, QubitOperatorPRCD");

    qop_pr_double.def("cast",
                      bindops::cast<QubitOperatorPRD, double, std::complex<double>, pr_t, pr_cmplx_t, QubitOperatorD,
                                    QubitOperatorCD, QubitOperatorPRD, QubitOperatorPRCD>,
                      "Supported types: float, complex, ParameterResolver<double>, ParameterResolver<complex>, "
                      "QubitOperatorD, QubitOperatorCD, QubitOperatorPRD, QubitOperatorPRCD");
    qop_pr_cmplx_double.def(
        "cast", bindops::cast<QubitOperatorPRCD, std::complex<double>, pr_cmplx_t, QubitOperatorCD, QubitOperatorPRCD>,
        "Supported types: complex, ParameterResolver<complex>, QubitOperatorCD, QubitOperatorPRCD");

    // ---------------------------------

    qop_double.def_static("simplify", QubitOperatorD::simplify);
    qop_cmplx_double.def_static("simplify", QubitOperatorCD::simplify);
    qop_pr_double.def_static("simplify", QubitOperatorPRD::simplify);
    qop_pr_cmplx_double.def_static("simplify", QubitOperatorPRCD::simplify);

    // ---------------------------------

    using qop_t = decltype(qop_double);
    bindops::binop_definition<op::plus, qop_t>::inplace<double>(qop_double);
    bindops::binop_definition<op::plus, qop_t>::external<all_qop_types_t>(qop_double);
    bindops::binop_definition<op::plus, qop_t>::reverse<all_qop_types_t>(qop_double);
    bindops::binop_definition<op::minus, qop_t>::inplace<double>(qop_double);
    bindops::binop_definition<op::minus, qop_t>::external<all_qop_types_t>(qop_double);
    bindops::binop_definition<op::minus, qop_t>::reverse<all_qop_types_t>(qop_double);
    bindops::binop_definition<op::times, qop_t>::inplace<double>(qop_double);
    bindops::binop_definition<op::times, qop_t>::external<all_qop_types_t>(qop_double);
    bindops::binop_definition<op::times, qop_t>::reverse<all_qop_types_t>(qop_double);
    bindops::binop_definition<op::divides, qop_t>::inplace<double>(qop_double);
    bindops::binop_definition<op::divides, qop_t>::external<all_scalar_types_t>(qop_double);

    using qop_cmplx_t = decltype(qop_cmplx_double);
    bindops::binop_definition<op::plus, qop_cmplx_t>::inplace<double, std::complex<double>>(qop_cmplx_double);
    bindops::binop_definition<op::plus, qop_cmplx_t>::external<all_qop_types_t>(qop_cmplx_double);
    bindops::binop_definition<op::plus, qop_cmplx_t>::reverse<all_qop_types_t>(qop_cmplx_double);
    bindops::binop_definition<op::minus, qop_cmplx_t>::inplace<double, std::complex<double>>(qop_cmplx_double);
    bindops::binop_definition<op::minus, qop_cmplx_t>::external<all_qop_types_t>(qop_cmplx_double);
    bindops::binop_definition<op::minus, qop_cmplx_t>::reverse<all_qop_types_t>(qop_cmplx_double);
    bindops::binop_definition<op::times, qop_cmplx_t>::inplace<double, std::complex<double>>(qop_cmplx_double);
    bindops::binop_definition<op::times, qop_cmplx_t>::external<all_qop_types_t>(qop_cmplx_double);
    bindops::binop_definition<op::times, qop_cmplx_t>::reverse<all_qop_types_t>(qop_cmplx_double);
    bindops::binop_definition<op::divides, qop_cmplx_t>::inplace<double, std::complex<double>>(qop_cmplx_double);
    bindops::binop_definition<op::divides, qop_cmplx_t>::external<all_scalar_types_t>(qop_cmplx_double);

    using qop_pr_t = decltype(qop_pr_double);
    bindops::binop_definition<op::plus, qop_pr_t>::inplace<double, pr_t>(qop_pr_double);
    bindops::binop_definition<op::plus, qop_pr_t>::external<all_qop_types_t>(qop_pr_double);
    bindops::binop_definition<op::plus, qop_pr_t>::reverse<all_qop_types_t>(qop_pr_double);
    bindops::binop_definition<op::minus, qop_pr_t>::inplace<double, pr_t>(qop_pr_double);
    bindops::binop_definition<op::minus, qop_pr_t>::external<all_qop_types_t>(qop_pr_double);
    bindops::binop_definition<op::minus, qop_pr_t>::reverse<all_qop_types_t>(qop_pr_double);
    bindops::binop_definition<op::times, qop_pr_t>::inplace<double, pr_t>(qop_pr_double);
    bindops::binop_definition<op::times, qop_pr_t>::external<all_qop_types_t>(qop_pr_double);
    bindops::binop_definition<op::times, qop_pr_t>::reverse<all_qop_types_t>(qop_pr_double);
    bindops::binop_definition<op::divides, qop_pr_t>::inplace<double, pr_t>(qop_pr_double);
    bindops::binop_definition<op::divides, qop_pr_t>::external<all_scalar_types_t>(qop_pr_double);

    using qop_pr_cmplx_t = decltype(qop_pr_cmplx_double);
    bindops::binop_definition<op::plus, qop_pr_cmplx_t>::inplace<all_scalar_types_t>(qop_pr_cmplx_double);
    bindops::binop_definition<op::plus, qop_pr_cmplx_t>::external<all_qop_types_t>(qop_pr_cmplx_double);
    bindops::binop_definition<op::plus, qop_pr_cmplx_t>::reverse<all_qop_types_t>(qop_pr_cmplx_double);
    bindops::binop_definition<op::minus, qop_pr_cmplx_t>::inplace<all_scalar_types_t>(qop_pr_cmplx_double);
    bindops::binop_definition<op::minus, qop_pr_cmplx_t>::external<all_qop_types_t>(qop_pr_cmplx_double);
    bindops::binop_definition<op::minus, qop_pr_cmplx_t>::reverse<all_qop_types_t>(qop_pr_cmplx_double);
    bindops::binop_definition<op::times, qop_pr_cmplx_t>::inplace<all_scalar_types_t>(qop_pr_cmplx_double);
    bindops::binop_definition<op::times, qop_pr_cmplx_t>::external<all_qop_types_t>(qop_pr_cmplx_double);
    bindops::binop_definition<op::times, qop_pr_cmplx_t>::reverse<all_qop_types_t>(qop_pr_cmplx_double);
    bindops::binop_definition<op::divides, qop_pr_cmplx_t>::inplace<all_scalar_types_t>(qop_pr_cmplx_double);
    bindops::binop_definition<op::divides, qop_pr_cmplx_t>::external<all_scalar_types_t>(qop_pr_cmplx_double);

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
