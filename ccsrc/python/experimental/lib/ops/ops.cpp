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
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "config/constexpr_type_name.hpp"

#include "math/pr/parameter_resolver.hpp"

#include "experimental/ops/gates.hpp"
#include "experimental/ops/parametric/angle_gates.hpp"

#include "python/bindings.hpp"

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

void init_tweedledum_ops(pybind11::module& module) {  // NOLINT(runtime/references)
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

void init_mindquantum_ops(pybind11::module& module) {  // NOLINT(runtime/references)
    using namespace pybind11::literals;                // NOLINT(build/namespaces_literals)

    py::class_<ops::SqrtSwap>(module, "SqrtSwap").def(py::init<>()).def("__str__", &::to_string<ops::SqrtSwap>);

    py::class_<ops::Entangle>(module, "Entangle")
        .def(py::init<const uint32_t>())
        .def("__str__", &::to_string<ops::Entangle>);
    py::class_<ops::Ph>(module, "Ph").def(py::init<const double>());
    py::class_<ops::QFT>(module, "QFT").def(py::init<const uint32_t>()).def("__str__", &::to_string<ops::QFT>);

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

void mindquantum::python::init_ops(pybind11::module& module) {
    init_tweedledum_ops(module);
    init_mindquantum_ops(module);
}
