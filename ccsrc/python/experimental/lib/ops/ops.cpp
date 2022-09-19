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

#include "core/parameter_resolver.hpp"
#include "details/define_binary_operator_helpers.hpp"

#include "experimental/ops/gates.hpp"
#include "experimental/ops/gates/details/coeff_policy.hpp"
#include "experimental/ops/gates/fermion_operator.hpp"
#include "experimental/ops/gates/qubit_operator.hpp"
#include "experimental/ops/parametric/angle_gates.hpp"
#include "experimental/ops/transform/jordan_wigner.hpp"
#include "experimental/ops/transform/parity.hpp"
#include "experimental/ops/transform/reverse_jordan_wigner.hpp"

#include "python/bindings.hpp"
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
namespace bindops {
using namespace pybind11::literals;

// -----------------------------------------------------------------------------

template <class op_t, class base_t>
auto bind_ops(pybind11::module& module, const std::string_view& name) {
    using coeff_t = typename op_t::coefficient_t;
    return py::class_<op_t, base_t, std::shared_ptr<op_t>>(module, name.data())
        .def(py::init<>())
        .def(py::init([](op_t& op, bool copy) {
            if (copy) {
                return op;
            }
            return std::move(op);
        }))
        .def(py::init<const ops::term_t&, coeff_t>(), "term"_a, "coeff"_a = op_t::coeff_policy_t::one)
        .def(py::init<const ops::terms_t&, coeff_t>(), "terms"_a, "coeff"_a = op_t::coeff_policy_t::one)
        .def(py::init<const ops::py_terms_t&, coeff_t>(), "terms"_a, "coeff"_a = op_t::coeff_policy_t::one)
        .def(py::init<const typename op_t::coeff_term_dict_t&>(), "coeff_terms"_a)
        .def(py::init<std::string_view, coeff_t>(), "terms_string"_a, "coeff"_a = op_t::coeff_policy_t::one)
        .def("num_targets", &op_t::num_targets)
        .def("count_qubits", &op_t::count_qubits)
        .def("is_identity", &op_t::is_identity, "abs_tol"_a = op_t::EQ_TOLERANCE)
        .def_static("identity", &op_t::identity)
        .def("subs", &op_t::subs, "subs_proxy"_a)
        .def_property("constant", static_cast<void (op_t::*)(const coeff_t&)>(&op_t::constant),
                      static_cast<coeff_t (op_t::*)() const>(&op_t::constant))
        .def_property_readonly("is_singlet", &op_t::is_singlet)
        .def("singlet", &op_t::singlet)
        .def("singlet_coeff", &op_t::singlet_coeff)
        .def("split", &op_t::split)
        .def("terms", &op_t::get_terms_pair)
        .def_property_readonly("real", &op_t::real)
        .def_property_readonly("imag", &op_t::imag)
        .def("hermitian", &op_t::hermitian)
        .def_property_readonly("size", &op_t::size)
        .def("compress", &op_t::compress, "abs_tol"_a = op_t::EQ_TOLERANCE)
        .def("dumps", &op_t::dumps, "indent"_a = 4)
        .def_static("loads", op_t::loads, "string_data"_a)
        .def(
            "__str__", [](const op_t& base) { return base.to_string(); }, py::is_operator())
        .def(
            "__repr__", [](const op_t& base) { return base.to_string(); }, py::is_operator())
        .def("get_coeff", &op_t::get_coeff)
        .PYBIND11_DEFINE_BINOP_INPLACE(add, op_t&, const op_t&, +)
        .PYBIND11_DEFINE_BINOP_EXT(add, const op_t&, const op_t&, +)
        .PYBIND11_DEFINE_BINOP_INPLACE(sub, op_t&, const op_t&, -)
        .PYBIND11_DEFINE_BINOP_EXT(sub, const op_t&, const op_t&, -)
        .PYBIND11_DEFINE_BINOP_INPLACE(mul, op_t&, const op_t&, *)
        .PYBIND11_DEFINE_BINOP_EXT(mul, const op_t&, const op_t&, *)
        .PYBIND11_DEFINE_UNOP(__neg__, const op_t&, -)
        .PYBIND11_DEFINE_BINOP(__eq__, const op_t&, const op_t&, ==)
        .def(
            "__pow__", [](const op_t& base, unsigned int exponent) { return base.pow(exponent); }, py::is_operator())
        .def("matrix", &op_t::matrix, "n_qubits"_a);
}

template <typename T>
using fop_t = ops::FermionOperator<T>;

template <typename T>
using qop_t = ops::QubitOperator<T>;

template <typename T>
using py_fop_t = py::class_<fop_t<T>, ops::FermionOperatorBase, std::shared_ptr<fop_t<T>>>;

template <typename T>
using py_qop_t = py::class_<qop_t<T>, ops::QubitOperatorBase, std::shared_ptr<qop_t<T>>>;

template <typename T, typename... args_t>
struct define_fermion_ops {
    template <typename... strings_t>
    static auto apply(pybind11::module& module, std::string_view name, strings_t&&... names) {
        static_assert(sizeof...(args_t) == sizeof...(strings_t));
        return std::tuple_cat(define_fermion_ops<T>::apply(module, name),
                              define_fermion_ops<args_t...>::apply(module, std::forward<strings_t>(names)...));
    }
};
template <typename T>
struct define_fermion_ops<T> {
    static auto apply(pybind11::module& module, std::string_view name) {
        return std::make_tuple(bind_ops<fop_t<T>, ops::FermionOperatorBase>(module, name)
                                   .def("normal_ordered", &fop_t<T>::normal_ordered));
    }
};

template <typename T, typename... args_t>
struct define_qubit_ops {
    template <typename... strings_t>
    static auto apply(pybind11::module& module, std::string_view name, strings_t&&... names) {
        static_assert(sizeof...(args_t) == sizeof...(strings_t));
        return std::tuple_cat(define_qubit_ops<T>::apply(module, name),
                              define_qubit_ops<args_t...>::apply(module, std::forward<strings_t>(names)...));
    }
};
template <typename T>
struct define_qubit_ops<T> {
    static auto apply(pybind11::module& module, std::string_view name) {
        return std::make_tuple(
            bind_ops<qop_t<T>, ops::QubitOperatorBase>(module, name).def("count_gates", &qop_t<T>::count_gates));
    }
};

template <typename self_t, typename... types_t>
struct cast_helper_impl;

template <typename self_t>
struct cast_helper_impl<self_t> {
    static py::object try_cast(const self_t& /* self */, const pybind11::object& /* type */) {
        throw std::runtime_error("Invalid type passed to cast() member function!");
        return py::none();
    }
};

template <typename self_t, typename T, typename... types_t>
struct cast_helper_impl<self_t, T, types_t...> {
    static py::object try_cast(const self_t& self, const pybind11::object& type) {
        if (type.is(py::type::of<T>())) {
            const auto value = self.template cast<T>();
            return py::cast(value);
        }
        return cast_helper_impl<self_t, types_t...>::try_cast(self, type);
    }
};

template <typename self_t, typename... types_t>
py::object cast(const self_t& self, const pybind11::object& type) {
    if (!PyType_Check(type.ptr())) {
        throw pybind11::type_error("Expect type as argument!");
    }
    return cast_helper_impl<self_t, types_t...>::try_cast(self, type);
}

}  // namespace bindops

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

    using all_types_t = std::tuple<double, std::complex<double>, pr_t, pr_cmplx_t>;

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

    static_assert(std::is_same_v<decltype(fop_double), bindops::py_fop_t<double>>);
    static_assert(std::is_same_v<decltype(fop_cmplx_double), bindops::py_fop_t<std::complex<double>>>);

    // ---------------------------------

    using FermionOperatorD = decltype(fop_double)::type;
    using FermionOperatorCD = decltype(fop_cmplx_double)::type;
    using FermionOperatorPRD = decltype(fop_pr_double)::type;
    using FermionOperatorPRCD = decltype(fop_pr_cmplx_double)::type;

    fop_double.def("cast", bindops::cast<FermionOperatorD, FermionOperatorCD, FermionOperatorPRD, FermionOperatorPRCD>,
                   "Supported types: FermionOperatorD, FermionOperatorCD, FermionOperatorPRD, FermionOperatorPRCD");
    fop_cmplx_double.def("cast", bindops::cast<FermionOperatorCD, FermionOperatorPRCD>,
                         "Supported types: FermionOperatorCD, FermionOperatorPRCD");

    fop_pr_double.def("cast",
                      bindops::cast<FermionOperatorD, FermionOperatorCD, FermionOperatorPRD, FermionOperatorPRCD>,
                      "Supported types: FermionOperatorD, FermionOperatorCD, FermionOperatorPRD, FermionOperatorPRCD");
    fop_pr_cmplx_double.def("cast", bindops::cast<FermionOperatorCD, FermionOperatorPRCD>,
                            "Supported types: FermionOperatorCD, FermionOperatorPRCD");

    // ---------------------------------

    using fop_t = decltype(fop_double);
    bindops::binop_definition<op::plus, fop_t>::inplace<double>(fop_double);
    bindops::binop_definition<op::plus, fop_t>::external<all_types_t>(fop_double);
    bindops::binop_definition<op::minus, fop_t>::inplace<double>(fop_double);
    bindops::binop_definition<op::minus, fop_t>::external<all_types_t>(fop_double);
    bindops::binop_definition<op::times, fop_t>::inplace<double>(fop_double);
    bindops::binop_definition<op::times, fop_t>::external<all_types_t>(fop_double);
    bindops::binop_definition<op::divides, fop_t>::inplace<double>(fop_double);
    bindops::binop_definition<op::divides, fop_t>::external<all_types_t>(fop_double);

    using fop_cmplx_t = decltype(fop_cmplx_double);
    bindops::binop_definition<op::plus, fop_cmplx_t>::inplace<double, std::complex<double>>(fop_cmplx_double);
    bindops::binop_definition<op::plus, fop_cmplx_t>::external<all_types_t>(fop_cmplx_double);
    bindops::binop_definition<op::minus, fop_cmplx_t>::inplace<double, std::complex<double>>(fop_cmplx_double);
    bindops::binop_definition<op::minus, fop_cmplx_t>::external<all_types_t>(fop_cmplx_double);
    bindops::binop_definition<op::times, fop_cmplx_t>::inplace<double, std::complex<double>>(fop_cmplx_double);
    bindops::binop_definition<op::times, fop_cmplx_t>::external<all_types_t>(fop_cmplx_double);
    bindops::binop_definition<op::divides, fop_cmplx_t>::inplace<double, std::complex<double>>(fop_cmplx_double);
    bindops::binop_definition<op::divides, fop_cmplx_t>::external<all_types_t>(fop_cmplx_double);

    using fop_pr_t = decltype(fop_pr_double);
    bindops::binop_definition<op::plus, fop_pr_t>::inplace<double, pr_t>(fop_pr_double);
    bindops::binop_definition<op::plus, fop_pr_t>::external<all_types_t>(fop_pr_double);
    bindops::binop_definition<op::minus, fop_pr_t>::inplace<double, pr_t>(fop_pr_double);
    bindops::binop_definition<op::minus, fop_pr_t>::external<all_types_t>(fop_pr_double);
    bindops::binop_definition<op::times, fop_pr_t>::inplace<double, pr_t>(fop_pr_double);
    bindops::binop_definition<op::times, fop_pr_t>::external<all_types_t>(fop_pr_double);
    bindops::binop_definition<op::divides, fop_pr_t>::inplace<double, pr_t>(fop_pr_double);
    bindops::binop_definition<op::divides, fop_pr_t>::external<all_types_t>(fop_pr_double);

    using fop_pr_cmplx_t = decltype(fop_pr_cmplx_double);
    bindops::binop_definition<op::plus, fop_pr_cmplx_t>::inplace<all_types_t>(fop_pr_cmplx_double);
    bindops::binop_definition<op::plus, fop_pr_cmplx_t>::external<all_types_t>(fop_pr_cmplx_double);
    bindops::binop_definition<op::minus, fop_pr_cmplx_t>::inplace<all_types_t>(fop_pr_cmplx_double);
    bindops::binop_definition<op::minus, fop_pr_cmplx_t>::external<all_types_t>(fop_pr_cmplx_double);
    bindops::binop_definition<op::times, fop_pr_cmplx_t>::inplace<all_types_t>(fop_pr_cmplx_double);
    bindops::binop_definition<op::times, fop_pr_cmplx_t>::external<all_types_t>(fop_pr_cmplx_double);
    bindops::binop_definition<op::divides, fop_pr_cmplx_t>::inplace<all_types_t>(fop_pr_cmplx_double);
    bindops::binop_definition<op::divides, fop_pr_cmplx_t>::external<all_types_t>(fop_pr_cmplx_double);

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

    static_assert(std::is_same_v<decltype(qop_double), bindops::py_qop_t<double>>);
    static_assert(std::is_same_v<decltype(qop_cmplx_double), bindops::py_qop_t<std::complex<double>>>);

    // ---------------------------------

    using QubitOperatorD = decltype(qop_double)::type;
    using QubitOperatorCD = decltype(qop_cmplx_double)::type;
    using QubitOperatorPRD = decltype(qop_pr_double)::type;
    using QubitOperatorPRCD = decltype(qop_pr_cmplx_double)::type;

    qop_double.def("cast", bindops::cast<QubitOperatorD, QubitOperatorCD, QubitOperatorPRD, QubitOperatorPRCD>,
                   "Supported types: QubitOperatorD, QubitOperatorCD, QubitOperatorPRD, QubitOperatorPRCD");
    qop_cmplx_double.def("cast", bindops::cast<QubitOperatorCD, QubitOperatorPRCD>,
                         "Supported types: QubitOperatorCD, QubitOperatorPRCD");

    qop_pr_double.def("cast", bindops::cast<QubitOperatorD, QubitOperatorCD, QubitOperatorPRD, QubitOperatorPRCD>,
                      "Supported types: QubitOperatorD, QubitOperatorCD, QubitOperatorPRD, QubitOperatorPRCD");
    qop_pr_cmplx_double.def("cast", bindops::cast<QubitOperatorCD, QubitOperatorPRCD>,
                            "Supported types: QubitOperatorCD, QubitOperatorPRCD");

    // ---------------------------------

    using qop_t = decltype(qop_double);
    bindops::binop_definition<op::plus, qop_t>::inplace<double>(qop_double);
    bindops::binop_definition<op::plus, qop_t>::external<all_types_t>(qop_double);
    bindops::binop_definition<op::minus, qop_t>::inplace<double>(qop_double);
    bindops::binop_definition<op::minus, qop_t>::external<all_types_t>(qop_double);
    bindops::binop_definition<op::times, qop_t>::inplace<double>(qop_double);
    bindops::binop_definition<op::times, qop_t>::external<all_types_t>(qop_double);
    bindops::binop_definition<op::divides, qop_t>::inplace<double>(qop_double);
    bindops::binop_definition<op::divides, qop_t>::external<all_types_t>(qop_double);

    using qop_cmplx_t = decltype(qop_cmplx_double);
    bindops::binop_definition<op::plus, qop_cmplx_t>::inplace<double, std::complex<double>>(qop_cmplx_double);
    bindops::binop_definition<op::plus, qop_cmplx_t>::external<all_types_t>(qop_cmplx_double);
    bindops::binop_definition<op::minus, qop_cmplx_t>::inplace<double, std::complex<double>>(qop_cmplx_double);
    bindops::binop_definition<op::minus, qop_cmplx_t>::external<all_types_t>(qop_cmplx_double);
    bindops::binop_definition<op::times, qop_cmplx_t>::inplace<double, std::complex<double>>(qop_cmplx_double);
    bindops::binop_definition<op::times, qop_cmplx_t>::external<all_types_t>(qop_cmplx_double);
    bindops::binop_definition<op::divides, qop_cmplx_t>::inplace<double, std::complex<double>>(qop_cmplx_double);
    bindops::binop_definition<op::divides, qop_cmplx_t>::external<all_types_t>(qop_cmplx_double);

    using qop_pr_t = decltype(qop_pr_double);
    bindops::binop_definition<op::plus, qop_pr_t>::inplace<double, pr_t>(qop_pr_double);
    bindops::binop_definition<op::plus, qop_pr_t>::external<all_types_t>(qop_pr_double);
    bindops::binop_definition<op::minus, qop_pr_t>::inplace<double, pr_t>(qop_pr_double);
    bindops::binop_definition<op::minus, qop_pr_t>::external<all_types_t>(qop_pr_double);
    bindops::binop_definition<op::times, qop_pr_t>::inplace<double, pr_t>(qop_pr_double);
    bindops::binop_definition<op::times, qop_pr_t>::external<all_types_t>(qop_pr_double);
    bindops::binop_definition<op::divides, qop_pr_t>::inplace<double, pr_t>(qop_pr_double);
    bindops::binop_definition<op::divides, qop_pr_t>::external<all_types_t>(qop_pr_double);

    using qop_pr_cmplx_t = decltype(qop_pr_cmplx_double);
    bindops::binop_definition<op::plus, qop_pr_cmplx_t>::inplace<all_types_t>(qop_pr_cmplx_double);
    bindops::binop_definition<op::plus, qop_pr_cmplx_t>::external<all_types_t>(qop_pr_cmplx_double);
    bindops::binop_definition<op::minus, qop_pr_cmplx_t>::inplace<all_types_t>(qop_pr_cmplx_double);
    bindops::binop_definition<op::minus, qop_pr_cmplx_t>::external<all_types_t>(qop_pr_cmplx_double);
    bindops::binop_definition<op::times, qop_pr_cmplx_t>::inplace<all_types_t>(qop_pr_cmplx_double);
    bindops::binop_definition<op::times, qop_pr_cmplx_t>::external<all_types_t>(qop_pr_cmplx_double);
    bindops::binop_definition<op::divides, qop_pr_cmplx_t>::inplace<all_types_t>(qop_pr_cmplx_double);
    bindops::binop_definition<op::divides, qop_pr_cmplx_t>::external<all_types_t>(qop_pr_cmplx_double);

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
    module.def("parity", &transform::parity, "ops"_a, "n_qubits"_a = -1);
    module.def("reverse_jordan_wigner", &transform::reverse_jordan_wigner);
    module.def("jordan_wigner", &transform::jordan_wigner);
}

void mindquantum::python::init_ops(pybind11::module& module) {
    init_tweedledum_ops(module);
    init_mindquantum_ops(module);

    py::module trans = module.def_submodule("transform", "MindQuantum-C++ operators transform");
    init_transform(trans);
}
