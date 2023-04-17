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
#include <vector>

#include <pybind11/attr.h>
#include <pybind11/buffer_info.h>
#include <pybind11/cast.h>
#include <pybind11/complex.h>
#include <pybind11/detail/common.h>
#include <pybind11/functional.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "math/operators/fermion_operator_view.hpp"
#include "math/operators/qubit_operator_view.hpp"
#include "math/operators/transform.hpp"
#include "math/pr/parameter_resolver.hpp"
#include "math/tensor/matrix.hpp"
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
        .def(py::init<float, tensor::TDtype>(), "data"_a, "dtype"_a = tensor::TDtype::Float32)
        .def(py::init<double, tensor::TDtype>(), "data"_a, "dtype"_a = tensor::TDtype::Float64)
        .def(py::init<const std::complex<float> &, tensor::TDtype>(), "data"_a, "dtype"_a = tensor::TDtype::Complex64)
        .def(py::init<const std::complex<double> &, tensor::TDtype>(), "data"_a, "dtype"_a = tensor::TDtype::Complex128)
        .def(py::init<const std::vector<float> &, tensor::TDtype>(), "data"_a, "dtype"_a = tensor::TDtype::Float64)
        .def(py::init<const std::vector<double> &, tensor::TDtype>(), "data"_a, "dtype"_a = tensor::TDtype::Float32)
        .def(py::init<const std::vector<std::complex<float>> &, tensor::TDtype>(), "data"_a,
             "dtype"_a = tensor::TDtype::Complex64)
        .def(py::init<const std::vector<std::complex<double>> &, tensor::TDtype>(), "data"_a,
             "dtype"_a = tensor::TDtype::Complex128)
        .def("astype", &tensor::Tensor::astype, "dtype"_a)
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
                {t.dim, },
                {tensor::bit_size(t.dtype)});
            // clang-format on
        });

    module.def("ones", &tensor::ops::ones, "len"_a, "dtype"_a = tensor::TDtype::Float64,
               "device"_a = tensor::TDevice::CPU);
    module.def("zeros", &tensor::ops::zeros, "len"_a, "dtype"_a = tensor::TDtype::Float64,
               "device"_a = tensor::TDevice::CPU);
    py::class_<tensor::Matrix, tensor::Tensor, std::shared_ptr<tensor::Matrix>>(module, "Matrix", py::buffer_protocol())
        .def(py::init<>())
        .def(py::init<const std::vector<std::vector<std::complex<double>>> &, tensor::TDevice>(), "m"_a,
             "device"_a = tensor::TDevice::CPU);
}
#undef BIND_TENSOR_OPS
#undef BIND_TENSOR_OPS_REV

void BindPR(py::module &module) {  // NOLINT(runtime/references)
    namespace pr = parameter;
    using pr_t = pr::ParameterResolver;
    py::class_<pr_t, std::shared_ptr<pr_t>>(module, "ParameterResolver")
        .def(py::init<>())
        .def(py::init<const std::string &, const tensor::Tensor &, tensor::TDtype>(), "key"_a,
             "const_value"_a = tensor::ops::zeros(1), "dtype"_a = tensor::TDtype::Float64)
        .def(py::init<const std::map<std::string, tensor::Tensor> &, const tensor::Tensor &, tensor::TDtype>(),
             "data"_a, "const_value"_a = tensor::ops::zeros(1), "dtype"_a = tensor::TDtype::Float64)
        .def(py::init<const tensor::Tensor &>(), "const_value"_a)
        .def(py::init<const pr_t &>(), "other"_a)
        .def(py::self + py::self)
        .def(py::self + py::self)
        .def(py::self += py::self)
        .def(py::self - py::self)
        .def(py::self -= py::self)
        .def(py::self * py::self)
        .def(py::self *= py::self)
        .def(py::self / py::self)
        .def(py::self /= py::self)
        .def("__copy__",
             [](const pr_t &pr) {
                 auto out = pr;
                 return out;
             })
        .def("__contains__", &pr_t::Contains)
        .def("__len__", &pr_t::Size)
        .def("__str__", &pr_t::ToString)
        .def("ansatz_part", &pr_t::AnsatzPart)
        .def("astype",
             [](const pr_t &pr, tensor::TDtype dtype) {
                 auto out = pr;
                 out.CastTo(dtype);
                 return out;
             })
        .def("as_ansatz", &pr_t::AsAnsatz)
        .def("as_encoder", &pr_t::AsEncoder)
        .def("combination", &pr_t::Combination)
        .def("conjugate", &pr_t::Conjugate)
        .def("dtype", &pr_t::GetDtype)
        .def("encoder_part", &pr_t::EncoderPart)
        .def("get_const", &pr_t::GetConstValue)
        .def("get_encoder_parameters", [](const pr_t &a) { return a.encoder_parameters_; })
        .def("get_grad_parameters", [](const pr_t &a) { return a.no_grad_parameters_; })
        .def("get_item", &pr_t::GetItem)
        .def("is_hermitian", &pr_t::IsHermitian)
        .def("is_not_zero", &pr_t::IsNotZero)
        .def("imag", &pr_t::Imag)
        .def("is_const", &pr_t::IsConst)
        .def("is_anti_hermitian", &pr_t::IsAntiHermitian)
        .def("keep_imag", &pr_t::KeepImag)
        .def("keep_real", &pr_t::KeepReal)
        .def("no_grad", &pr_t::NoGrad)
        .def("no_grad_part", &pr_t::NoGradPart)
        .def("params_data", &pr_t::ParaData)
        .def("params_name", &pr_t::ParamsName)
        .def("params_value", &pr_t::ParaValue)
        .def("pop", &pr_t::Pop)
        .def("requires_grad", &pr_t::RequiresGrad)
        .def("requires_grad_part", &pr_t::RequiresGradPart)
        .def("real", &pr_t::Real)
        .def("set_const", &pr_t::SetConstValue)
        .def("set_item", &pr_t::SetItem<tensor::Tensor>)
        .def("subs", &pr_t::subs)
        .def("update", &pr_t::Update);
}

// -----------------------------------------------------------------------------

void BindQubitOperator(py::module &module) {  // NOLINT(runtime/references)
    namespace pr = parameter;
    using pr_t = pr::ParameterResolver;
    using qop_t = operators::qubit::QubitOperator;
    using fop_t = operators::fermion::FermionOperator;
    auto p_term_value = py::enum_<operators::qubit::TermValue>(module, "p_term_value")
                            .value("I", operators::qubit::TermValue::I)
                            .value("X", operators::qubit::TermValue::X)
                            .value("Y", operators::qubit::TermValue::Y)
                            .value("Z", operators::qubit::TermValue::Z);
    p_term_value.attr("__repr__") = pybind11::cpp_function(
        [](const operators::qubit::TermValue &dtype) -> pybind11::str { return operators::qubit::to_string(dtype); },
        pybind11::name("name"), pybind11::is_method(p_term_value));
    p_term_value.attr("__str__") = pybind11::cpp_function(
        [](const operators::qubit::TermValue &dtype) -> pybind11::str { return operators::qubit::to_string(dtype); },
        pybind11::name("name"), pybind11::is_method(p_term_value));

    auto f_term_value = py::enum_<operators::fermion::TermValue>(module, "f_term_value")
                            .value("I", operators::fermion::TermValue::I)
                            .value("a", operators::fermion::TermValue::A)
                            .value("adg", operators::fermion::TermValue::Ad);
    f_term_value.attr("__repr__") = pybind11::cpp_function(
        [](const operators::fermion::TermValue &dtype) -> pybind11::str {
            return operators::fermion::to_string(dtype);
        },
        pybind11::name("name"), pybind11::is_method(f_term_value));
    f_term_value.attr("__str__") = pybind11::cpp_function(
        [](const operators::fermion::TermValue &dtype) -> pybind11::str {
            return operators::fermion::to_string(dtype);
        },
        pybind11::name("name"), pybind11::is_method(f_term_value));

    // -----------------------------------------------------------------------------

    py::class_<qop_t, std::shared_ptr<qop_t>>(module, "QubitOperator")
        .def(py::init<>())
        .def(py::init<const qop_t &>(), "other"_a)
        .def(py::init<const qop_t::term_t &, const pr_t &>(), "term"_a, "coeff"_a = pr_t(tensor::ops::ones(1)))
        .def(py::init<const qop_t::terms_t &, const pr_t &>(), "terms"_a, "coeff"_a = pr_t(tensor::ops::ones(1)))
        .def(py::init<const qop_t::py_term_t &, const pr_t &>(), "term"_a, "coeff"_a = pr_t(tensor::ops::ones(1)))
        .def(py::init<const qop_t::py_terms_t &, const pr_t &>(), "terms"_a, "coeff"_a = pr_t(tensor::ops::ones(1)))
        .def(py::init<const std::string &, const pr_t &>(), "pauli_string"_a, "coeff"_a = pr_t(tensor::ops::ones(1)))
        .def(py::self += py::self)
        .def(py::self + py::self)
        .def(py::self *= py::self)
        .def(py::self * py::self)
        .def("__copy__", [](const qop_t &a) { return a; })
        .def("astype",
             [](const qop_t &a, tensor::TDtype dtype) {
                 auto out = a;
                 out.CastTo(dtype);
                 return out;
             })
        .def("count_qubits", &qop_t::count_qubits)
        .def("dtype", &qop_t::GetDtype)
        .def("get_terms", &qop_t::get_terms)
        .def("get_coeff", &qop_t::get_coeff)
        .def("hermitian_conjugated", &qop_t::hermitian_conjugated)
        .def("imag", &qop_t::imag)
        .def("is_singlet", &qop_t::is_singlet)
        .def("parameterized", &qop_t::parameterized)
        .def("real", &qop_t::real)
        .def("set_coeff", &qop_t::set_coeff)
        .def("split", &qop_t::split)
        .def("singlet_coeff", &qop_t::singlet_coeff)
        .def("singlet", &qop_t::singlet)
        .def("size", &qop_t::size)
        .def("subs", &qop_t::subs)
        .def("__repr__", [](const qop_t &op) { return op.ToString(); });

    // -----------------------------------------------------------------------------

    py::class_<fop_t, std::shared_ptr<fop_t>>(module, "FermionOperator")
        .def(py::init<>())
        .def(py::init<const fop_t &>(), "other"_a)
        .def(py::init<const fop_t::py_dict_t &>(), "py_terms"_a)
        .def(py::init<const fop_t::term_t &, const pr_t &>(), "term"_a, "coeff"_a = pr_t(tensor::ops::ones(1)))
        .def(py::init<const fop_t::terms_t &, const pr_t &>(), "terms"_a, "coeff"_a = pr_t(tensor::ops::ones(1)))
        .def(py::init<const fop_t::py_term_t &, const pr_t &>(), "term"_a, "coeff"_a = pr_t(tensor::ops::ones(1)))
        .def(py::init<const fop_t::py_terms_t &, const pr_t &>(), "terms"_a, "coeff"_a = pr_t(tensor::ops::ones(1)))
        .def(py::init<const std::string &, const pr_t &>(), "fermion_string"_a, "coeff"_a = pr_t(tensor::ops::ones(1)))
        .def(py::self += py::self)
        .def(py::self + py::self)
        .def(py::self *= py::self)
        .def(py::self * py::self)
        .def("__copy__", [](const fop_t &a) { return a; })
        .def("astype",
             [](const fop_t &a, tensor::TDtype dtype) {
                 auto out = a;
                 out.CastTo(dtype);
                 return out;
             })
        .def("count_qubits", &fop_t::count_qubits)
        .def("dtype", &fop_t::GetDtype)
        .def("get_terms", &fop_t::get_terms)
        .def("get_coeff", &fop_t::get_coeff)
        .def("hermitian_conjugated", &fop_t::hermitian_conjugated)
        .def("imag", &fop_t::imag)
        .def("is_singlet", &fop_t::is_singlet)
        .def("normal_ordered", &fop_t::normal_ordered)
        .def("parameterized", &fop_t::parameterized)
        .def("real", &fop_t::real)
        .def("set_coeff", &fop_t::set_coeff)
        .def("split", &fop_t::split)
        .def("singlet_coeff", &fop_t::singlet_coeff)
        .def("singlet", &fop_t::singlet)
        .def("size", &fop_t::size)
        .def("subs", &fop_t::subs)
        .def("__repr__", [](const fop_t &op) { return op.ToString(); });
}

void BindTransform(py::module &module) {  // NOLINT(runtime/references)
    module.def("jordan_wigner", &operators::transform::jordan_wigner, "ops"_a);
    module.def("reverse_jordan_wigner", &operators::transform::reverse_jordan_wigner, "ops"_a, "n_qubits"_a = -1);
    module.def("parity", &operators::transform::parity, "ops"_a, "n_qubits"_a = -1);
    module.def("bravyi_kitaev", &operators::transform::bravyi_kitaev, "ops"_a, "n_qubits"_a = -1);
    module.def("ternary_tree", &operators::transform::ternary_tree, "ops"_a, "n_qubits"_a);
    module.def("bravyi_kitaev_superfast", &operators::transform::bravyi_kitaev_superfast, "ops"_a);
}

PYBIND11_MODULE(_math, m) {
    m.doc() = "MindQuantum Math module.";
    auto dtype_id = py::enum_<tensor::TDtype>(m, "dtype")
                        .value("complex64", tensor::TDtype::Complex64)
                        .value("complex128", tensor::TDtype::Complex128)
                        .value("float32", tensor::TDtype::Float32)
                        .value("float64", tensor::TDtype::Float64);
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
