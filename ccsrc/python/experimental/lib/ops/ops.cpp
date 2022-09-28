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

#include <memory>
#include <tuple>

#include <tweedledum/IR/Circuit.h>
#include <tweedledum/IR/Qubit.h>

#include <fmt/format.h>
#include <pybind11/cast.h>
#include <pybind11/complex.h>
#include <pybind11/detail/common.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

#include "config/constexpr_type_name.hpp"

#include "core/parameter_resolver.hpp"
#include "details/define_terms_ops.hpp"

#include "experimental/cengines/write_projectq.hpp"
#include "experimental/ops/gates.hpp"
#include "experimental/ops/gates/details/coeff_policy.hpp"
#include "experimental/ops/gates/details/parameter_resolver_coeff_policy.hpp"
#include "experimental/ops/gates/fermion_operator.hpp"
#include "experimental/ops/gates/qubit_operator.hpp"
#include "experimental/ops/gates/terms_operator_base.hpp"
#include "experimental/ops/parametric/angle_gates.hpp"
#include "experimental/ops/transform/jordan_wigner.hpp"
#include "experimental/ops/transform/parity.hpp"

#include "python/bindings.hpp"
#include "python/core/tsl_ordered_map.hpp"
#include "python/ops/gate_adapter.hpp"

namespace ops = mindquantum::ops;
namespace py = pybind11;

namespace {
// NB: These two should have auto return types but GCC 7 & 8 don't play well if we do :-(
template <typename operator_t>
std::string_view to_string(const operator_t& op) {
    return operator_t::kind();
}
template <typename operator_t>
std::string to_string_angle(const operator_t& op) {
    return fmt::format("{}", operator_t::kind(), op.angle());
}
}  // namespace

void init_tweedledum_ops(pybind11::module& module) {
    py::class_<ops::Barrier>(module, "Barrier").def(py::init<>()).def("__str__", &::to_string<ops::Barrier>);
    py::class_<ops::H>(module, "H").def(py::init<>()).def("__str__", &::to_string<ops::H>);
    py::class_<ops::Measure>(module, "Measure").def(py::init<>()).def("__str__", &::to_string<ops::Measure>);
    py::class_<ops::S>(module, "S").def(py::init<>()).def("__str__", &::to_string<ops::S>);
    py::class_<ops::Sdg>(module, "Sdg").def(py::init<>()).def("__str__", &::to_string<ops::Sdg>);
    py::class_<ops::Swap>(module, "Swap").def(py::init<>()).def("__str__", &::to_string<ops::Swap>);
    py::class_<ops::Sx>(module, "Sx").def(py::init<>()).def("__str__", &::to_string<ops::Sx>);
    py::class_<ops::Sxdg>(module, "Sxdg").def(py::init<>()).def("__str__", &::to_string<ops::Sxdg>);
    py::class_<ops::T>(module, "T").def(py::init<>()).def("__str__", &::to_string<ops::T>);
    py::class_<ops::Tdg>(module, "Tdg").def(py::init<>()).def("__str__", &::to_string<ops::Tdg>);
    py::class_<ops::X>(module, "X").def(py::init<>()).def("__str__", &::to_string<ops::X>);
    py::class_<ops::Y>(module, "Y").def(py::init<>()).def("__str__", &::to_string<ops::Y>);
    py::class_<ops::Z>(module, "Z").def(py::init<>()).def("__str__", &::to_string<ops::Z>);

    py::class_<ops::P>(module, "P").def(py::init<const double>()).def("__str__", &::to_string_angle<ops::P>);
    py::class_<ops::Rx>(module, "Rx").def(py::init<const double>()).def("__str__", &::to_string_angle<ops::Rx>);
    py::class_<ops::Rxx>(module, "Rxx").def(py::init<const double>()).def("__str__", &::to_string_angle<ops::Rxx>);
    py::class_<ops::Ry>(module, "Ry").def(py::init<const double>()).def("__str__", &::to_string_angle<ops::Ry>);
    py::class_<ops::Ryy>(module, "Ryy").def(py::init<const double>()).def("__str__", &::to_string_angle<ops::Ryy>);
    py::class_<ops::Rz>(module, "Rz").def(py::init<const double>()).def("__str__", &::to_string_angle<ops::Rz>);
    py::class_<ops::Rzz>(module, "Rzz").def(py::init<const double>()).def("__str__", &::to_string_angle<ops::Rzz>);
}

// =============================================================================

void init_mindquantum_ops(pybind11::module& module) {
    using namespace pybind11::literals;

    py::class_<ops::SqrtSwap>(module, "SqrtSwap").def(py::init<>()).def("__str__", &::to_string<ops::SqrtSwap>);

    py::class_<ops::Entangle>(module, "Entangle")
        .def(py::init<const uint32_t>())
        .def("__str__", &::to_string<ops::Entangle>);
    py::class_<ops::Ph>(module, "Ph").def(py::init<const double>());
    py::class_<ops::QFT>(module, "QFT").def(py::init<const uint32_t>()).def("__str__", &::to_string<ops::QFT>);

    py::enum_<ops::TermValue>(module, "TermValue")
        .value("I", ops::TermValue::I)
        .value("X", ops::TermValue::X)
        .value("Y", ops::TermValue::Y)
        .value("Z", ops::TermValue::Z)
        .value("a", ops::TermValue::a)
        .value("adg", ops::TermValue::adg);

    // =========================================================================

    namespace mq = mindquantum;
    namespace op = bindops::details;

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

    // =========================================================================

    // py::class_<ops::parametric::P>(module, "P").def(py::init<const double>());
    // py::class_<ops::parametric::Ph>(module, "Ph"). def(py::init<SymEngine::number>());
    // py::class_<ops::parametric::Rx>(module, "Rx").def(py::init<const double>());
    // py::class_<ops::parametric::Rxx>(module, "Rxx").def(py::init<const double>());
    // py::class_<ops::parametric::Ry>(module, "Ry").def(py::init<const double>());
    // py::class_<ops::parametric::Ryy>(module, "Ryy").def(py::init<const double>());
    // py::class_<ops::parametric::Rz>(module, "Rz").def(py::init<const double>());
    // py::class_<ops::parametric::Rzz>(module, "Rzz").def(py::init<const double>());
}

void init_transform(py::module& module) {
    using namespace pybind11::literals;

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

void mindquantum::python::init_ops(pybind11::module& module) {
    init_tweedledum_ops(module);
    init_mindquantum_ops(module);

    py::module trans = module.def_submodule("transform", "MindQuantum-C++ operators transform");
    init_transform(trans);
}
