//   Copyright 2023 <Huawei Technologies Co., Ltd>
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

#include <pybind11/attr.h>
#include <pybind11/buffer_info.h>
#include <pybind11/cast.h>
#include <pybind11/complex.h>
#include <pybind11/detail/common.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>

#include "math/operators/fermion_operator_view.hpp"
#include "math/operators/qubit_operator_view.hpp"
#include "math/operators/transform/jordan_wigner.hpp"
#include "math/operators/transform/transform.hpp"
#include "math/pr/parameter_resolver.hpp"
#include "math/tensor/ops/memory_operator.hpp"
#include "math/tensor/tensor.hpp"
#include "math/tensor/traits.hpp"

namespace py = pybind11;
using namespace pybind11::literals;  // NOLINT(build/namespaces_literals)
#define BIND_TENSOR_OPS(op, ope)                                                                                       \
    def(py::self op py::self)                                                                                          \
        .def(py::self op float())                                                                                      \
        .def(py::self op double())                                                                                     \
        .def(py::self op std::complex<float>())                                                                        \
        .def(py::self op std::complex<double>())                                                                       \
        .def(py::self ope float())                                                                                     \
        .def(py::self ope double())                                                                                    \
        .def(py::self ope std::complex<float>())                                                                       \
        .def(py::self ope std::complex<double>())

#define BIND_TENSOR_OPS_REV(op)                                                                                        \
    def(float() op py::self)                                                                                           \
        .def(double() op py::self)                                                                                     \
        .def(std::complex<float>() op py::self)                                                                        \
        .def(std::complex<double>() op py::self)

void BindTensor(py::module &module) {  // NOLINT(runtime/references)
    py::class_<tensor::Tensor, std::shared_ptr<tensor::Tensor>>(module, "Tensor", py::buffer_protocol())
        .def(py::init<>())
        .def("__str__", &tensor::ops::to_string, "simplify"_a = false)
        .def("__repr__", &tensor::ops::to_string, "simplify"_a = false)
        .def_readonly("dtype", &tensor::Tensor::dtype)
        .def_readonly("size", &tensor::Tensor::dim)
        .BIND_TENSOR_OPS(+, +=)
        .BIND_TENSOR_OPS(-, -=)
        .BIND_TENSOR_OPS_REV(-)
        .BIND_TENSOR_OPS(*, *=)
        .BIND_TENSOR_OPS(/, /=)
        .BIND_TENSOR_OPS_REV(/)
        .def_buffer([](tensor::Tensor &t) -> py::buffer_info {
            auto format = py::format_descriptor<float>::format();
            switch (t.dtype) {
                case tensor::TDtype::Float32:
                    format = py::format_descriptor<tensor::to_device_t<tensor::TDtype::Float32>>::format();
                    break;
                case tensor::TDtype::Float64:
                    format = py::format_descriptor<tensor::to_device_t<tensor::TDtype::Float64>>::format();
                    break;
                case tensor::TDtype::Complex64:
                    format = py::format_descriptor<tensor::to_device_t<tensor::TDtype::Complex64>>::format();
                    break;
                case tensor::TDtype::Complex128:
                    format = py::format_descriptor<tensor::to_device_t<tensor::TDtype::Complex128>>::format();
                    break;
            }
            // clang-format off
            return py::buffer_info(
                t.data,
                tensor::bit_size(t.dtype),
                format,
                1,
                {t.dim,},
                {tensor::bit_size(t.dtype)}
            );
            // clang-format on
        });

    module.def("ones", &tensor::ops::ones, "len"_a, "dtype"_a = tensor::TDtype::Float64,
               "device"_a = tensor::TDevice::CPU);
    module.def("zeros", &tensor::ops::zeros, "len"_a, "dtype"_a = tensor::TDtype::Float64,
               "device"_a = tensor::TDevice::CPU);
}
#undef BIND_TENSOR_OPS
#undef BIND_TENSOR_OPS_REV

void BindPR(py::module &module) {  // NOLINT(runtime/references)
    namespace pr = parameter;
    using pr_t = pr::ParameterResolver;
    py::class_<pr_t, std::shared_ptr<pr_t>>(module, "ParameterResolver")
        .def(py::init<const std::string &, tensor::TDtype>(), "key"_a, "dtype"_a = tensor::TDtype::Float64)
        .def(py::init<const tensor::Tensor &>(), "const_value"_a)
        .def("__str__", &pr_t::ToString)
        .def("__len__", &pr_t::Size)
        .def_property_readonly("dtype", &pr_t::GetDtype)
        .def("astype",
             [](const pr_t &pr, tensor::TDtype dtype) {
                 auto out = pr;
                 out.CastTo(dtype);
                 return out;
             })
        .def("__contains__", &pr_t::Contains)
        .def("__copy__",
             [](const pr_t &pr) {
                 auto out = pr;
                 return out;
             })
        .def("pop", &pr_t::Pop)
        .def("real", &pr_t::Real)
        .def("imag", &pr_t::Imag)
        .def("no_grad", &pr_t::NoGrad)
        .def("is_const", &pr_t::IsConst)
        .def("keep_imag", &pr_t::KeepImag)
        .def("keep_real", &pr_t::KeepReal)
        .def("as_ansatz", &pr_t::AsAnsatz)
        .def("conjugate", &pr_t::Conjugate)
        .def("as_encoder", &pr_t::AsEncoder)
        .def("ansatz_part", &pr_t::AnsatzPart)
        .def("params_name", &pr_t::ParamsName)
        .def("combination", &pr_t::Combination)
        .def("no_grad_part", &pr_t::NoGradPart)
        .def("encoder_part", &pr_t::EncoderPart)
        .def("is_hermitian", &pr_t::IsHermitian)
        .def("requires_grad", &pr_t::RequiresGrad)
        .def("is_anti_hermitian", &pr_t::IsAntiHermitian)
        .def("requires_grad_part", &pr_t::RequiresGradPart);
}

// -----------------------------------------------------------------------------

void BindQubitOperator(py::module &module) {
    namespace pr = parameter;
    using pr_t = pr::ParameterResolver;
    using qop_t = operators::qubit::QubitOperator;
    using fop_t = operators::fermion::FermionOperator;
    py::class_<qop_t, std::shared_ptr<qop_t>>(module, "QubitOperator")
        .def(py::init<const std::string &, const pr_t &>(), "pauli_string"_a, "coeff"_a = pr_t(tensor::ops::ones(1)))
        .def(py::self += tensor::Tensor())
        .def(py::self += py::self)
        .def(py::self + tensor::Tensor())
        .def(py::self + py::self)
        .def(py::self *= py::self)
        .def("size", &qop_t::size)
        .def("__repr__", [](const qop_t &op) { return op.ToString(); });
    py::class_<fop_t, std::shared_ptr<fop_t>>(module, "FermionOperator")
        .def(py::init<const std::string &, const pr_t &>(), "fermion_string"_a, "coeff"_a = pr_t(tensor::ops::ones(1)))
        .def(py::self += tensor::Tensor())
        .def(py::self += py::self)
        .def(py::self + tensor::Tensor())
        .def(py::self + py::self)
        .def(py::self *= py::self)
        .def("size", &fop_t::size)
        .def("__repr__", [](const fop_t &op) { return op.ToString(); });
}

void BindTransform(py::module &module) {
    module.def("jordan_wigner", &operators::transform::jordan_wigner, "ops"_a);
}

PYBIND11_MODULE(_math, m) {
    m.doc() = "MindQuantum Math module.";
    auto dtype_id = py::enum_<tensor::TDtype>(m, "dtype")
                        .value("Complex64", tensor::TDtype::Complex64)
                        .value("Complex128", tensor::TDtype::Complex128)
                        .value("Float32", tensor::TDtype::Float32)
                        .value("Float64", tensor::TDtype::Float64);
    dtype_id.attr("__repr__") = pybind11::cpp_function(
        [](const tensor::TDtype &dtype) -> pybind11::str { return "mindquantum." + tensor::to_string(dtype); },
        pybind11::name("name"), pybind11::is_method(dtype_id));
    dtype_id.attr("__str__") = pybind11::cpp_function(
        [](const tensor::TDtype &dtype) -> pybind11::str { return "mindquantum." + tensor::to_string(dtype); },
        pybind11::name("name"), pybind11::is_method(dtype_id));
    auto device_id
        = py::enum_<tensor::TDevice>(m, "device").value("CPU", tensor::TDevice::CPU).value("GPU", tensor::TDevice::GPU);
    device_id.attr("__repr__") = pybind11::cpp_function(
        [](const tensor::TDevice &device) -> pybind11::str { return "mindquantum." + tensor::to_string(device); },
        pybind11::name("name"), pybind11::is_method(device_id));
    device_id.attr("__str__") = pybind11::cpp_function(
        [](const tensor::TDevice &device) -> pybind11::str { return "mindquantum." + tensor::to_string(device); },
        pybind11::name("name"), pybind11::is_method(device_id));

    py::module tensor_module = m.def_submodule("tensor", "MindQuantum Tensor module.");
    BindTensor(tensor_module);

    py::module pr_module = m.def_submodule("pr", "MindQuantum ParameterResolver module.");
    BindPR(pr_module);

    py::module ops_module = m.def_submodule("ops", "MindQuantum Operators module.");
    BindQubitOperator(ops_module);
    BindTransform(ops_module);
}
