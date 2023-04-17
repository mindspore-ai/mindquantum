/**
 * Copyright 2021 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "python/device/binding.hpp"

#include <complex>
#include <memory>

#include <fmt/format.h>
#include <pybind11/attr.h>
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
#include "config/format/std_complex.hpp"
#include "config/type_traits.hpp"

#include "core/mq_base_types.hpp"
#include "core/sparse/algo.hpp"
#include "core/sparse/csrhdmatrix.hpp"
#include "core/sparse/paulimat.hpp"
#include "math/pr/parameter_resolver.hpp"
#include "math/tensor/matrix.hpp"
#include "ops/basic_gate.hpp"
#include "ops/gate_id.hpp"
#include "ops/gates.hpp"
#include "ops/hamiltonian.hpp"

#include "python/core/sparse/csrhdmatrix.hpp"
#include "python/details/create_from_container_class.hpp"
#include "python/details/define_binary_operator_helpers.hpp"
#include "python/ops/basic_gate.hpp"
#include "python/ops/build_env.hpp"

namespace py = pybind11;

// using mindquantum::sparse::Csr_Plus_Csr;
// using mindquantum::sparse::GetPauliMat;
// using mindquantum::sparse::PauliMat;
// using mindquantum::sparse::PauliMatToCsrHdMatrix;
// using mindquantum::sparse::SparseHamiltonian;
// using mindquantum::sparse::TransposeCsrHdMatrix;
using namespace pybind11::literals;  // NOLINT(build/namespaces_literals)

namespace mindquantum::python {
void init_logging(pybind11::module &module);  // NOLINT(runtime/references)NOLINT
}  // namespace mindquantum::python

void BindTypeIndependentGate(py::module &module) {  // NOLINT(runtime/references)
    using mindquantum::Index;
    using mindquantum::VT;
    py::class_<mindquantum::MeasureGate, mindquantum::BasicGate, std::shared_ptr<mindquantum::MeasureGate>>(
        module, "MeasureGate")
        .def(py::init<std::string, const VT<Index> &>(), "name"_a, "obj_qubits"_a);
    py::class_<mindquantum::IGate, mindquantum::BasicGate, std::shared_ptr<mindquantum::IGate>>(module, "IGate")
        .def(py::init<const VT<Index> &, const VT<Index> &>(), "obj_qubits"_a, "ctrl_qubits"_a = VT<Index>());
    py::class_<mindquantum::XGate, mindquantum::BasicGate, std::shared_ptr<mindquantum::XGate>>(module, "XGate")
        .def(py::init<const VT<Index> &, const VT<Index> &>(), "obj_qubits"_a, "ctrl_qubits"_a = VT<Index>());
    py::class_<mindquantum::YGate, mindquantum::BasicGate, std::shared_ptr<mindquantum::YGate>>(module, "YGate")
        .def(py::init<const VT<Index> &, const VT<Index> &>(), "obj_qubits"_a, "ctrl_qubits"_a = VT<Index>());
    py::class_<mindquantum::ZGate, mindquantum::BasicGate, std::shared_ptr<mindquantum::ZGate>>(module, "ZGate")
        .def(py::init<const VT<Index> &, const VT<Index> &>(), "obj_qubits"_a, "ctrl_qubits"_a = VT<Index>());
    py::class_<mindquantum::HGate, mindquantum::BasicGate, std::shared_ptr<mindquantum::HGate>>(module, "HGate")
        .def(py::init<const VT<Index> &, const VT<Index> &>(), "obj_qubits"_a, "ctrl_qubits"_a = VT<Index>());
    py::class_<mindquantum::ISWAPGate, mindquantum::BasicGate, std::shared_ptr<mindquantum::ISWAPGate>>(module,
                                                                                                        "ISWAPGate")
        .def(py::init<bool, const VT<Index> &, const VT<Index> &>(), "daggered"_a, "obj_qubits"_a,
             "ctrl_qubits"_a = VT<Index>());
    py::class_<mindquantum::SWAPGate, mindquantum::BasicGate, std::shared_ptr<mindquantum::SWAPGate>>(module,
                                                                                                      "SWAPGate")
        .def(py::init<const VT<Index> &, const VT<Index> &>(), "obj_qubits"_a, "ctrl_qubits"_a = VT<Index>());
    py::class_<mindquantum::SGate, mindquantum::BasicGate, std::shared_ptr<mindquantum::SGate>>(module, "SGate")
        .def(py::init<const VT<Index> &, const VT<Index> &>(), "obj_qubits"_a, "ctrl_qubits"_a = VT<Index>());
    py::class_<mindquantum::SdagGate, mindquantum::BasicGate, std::shared_ptr<mindquantum::SdagGate>>(module,
                                                                                                      "SdagGate")
        .def(py::init<const VT<Index> &, const VT<Index> &>(), "obj_qubits"_a, "ctrl_qubits"_a = VT<Index>());
    py::class_<mindquantum::TGate, mindquantum::BasicGate, std::shared_ptr<mindquantum::TGate>>(module, "TGate")
        .def(py::init<const VT<Index> &, const VT<Index> &>(), "obj_qubits"_a, "ctrl_qubits"_a = VT<Index>());
    py::class_<mindquantum::TdagGate, mindquantum::BasicGate, std::shared_ptr<mindquantum::TdagGate>>(module,
                                                                                                      "TdagGate")
        .def(py::init<const VT<Index> &, const VT<Index> &>(), "obj_qubits"_a, "ctrl_qubits"_a = VT<Index>());
    py::class_<mindquantum::PauliChannel, mindquantum::BasicGate, std::shared_ptr<mindquantum::PauliChannel>>(
        module, "PauliChannel")
        .def(py::init<double, double, double, const VT<Index> &, const VT<Index> &>(), "px"_a, "py"_a, "pz"_a,
             "obj_qubits"_a, "ctrl_qubits"_a = VT<Index>());
    py::class_<mindquantum::AmplitudeDampingChannel, mindquantum::BasicGate,
               std::shared_ptr<mindquantum::AmplitudeDampingChannel>>(module, "AmplitudeDampingChannel")
        .def(py::init<bool, double, const VT<Index> &, const VT<Index> &>(), "daggered"_a, "damping_coeff"_a,
             "obj_qubits"_a, "ctrl_qubits"_a = VT<Index>());
    py::class_<mindquantum::PhaseDampingChannel, mindquantum::BasicGate,
               std::shared_ptr<mindquantum::PhaseDampingChannel>>(module, "PhaseDampingChannel")
        .def(py::init<double, const VT<Index> &, const VT<Index> &>(), "damping_coeff"_a, "obj_qubits"_a,
             "ctrl_qubits"_a = VT<Index>());
}

void BindTypeDependentGate(py::module &module) {  // NOLINT(runtime/references)
    using mindquantum::CT;
    using mindquantum::Index;
    using mindquantum::VT;
    using mindquantum::VVT;
    using parameter::ParameterResolver;
    py::class_<mindquantum::RXGate, mindquantum::BasicGate, std::shared_ptr<mindquantum::RXGate>>(module, "RXGate")
        .def(py::init<const ParameterResolver &, const VT<Index> &, const VT<Index> &>(), "pr"_a, "obj_qubits"_a,
             "ctrl_qubits"_a = VT<Index>());
    py::class_<mindquantum::RYGate, mindquantum::BasicGate, std::shared_ptr<mindquantum::RYGate>>(module, "RYGate")
        .def(py::init<const ParameterResolver &, const VT<Index> &, const VT<Index> &>(), "pr"_a, "obj_qubits"_a,
             "ctrl_qubits"_a = VT<Index>());
    py::class_<mindquantum::RZGate, mindquantum::BasicGate, std::shared_ptr<mindquantum::RZGate>>(module, "RZGate")
        .def(py::init<const ParameterResolver &, const VT<Index> &, const VT<Index> &>(), "pr"_a, "obj_qubits"_a,
             "ctrl_qubits"_a = VT<Index>());
    py::class_<mindquantum::RxxGate, mindquantum::BasicGate, std::shared_ptr<mindquantum::RxxGate>>(module, "RxxGate")
        .def(py::init<const ParameterResolver &, const VT<Index> &, const VT<Index> &>(), "pr"_a, "obj_qubits"_a,
             "ctrl_qubits"_a = VT<Index>());
    py::class_<mindquantum::RyyGate, mindquantum::BasicGate, std::shared_ptr<mindquantum::RyyGate>>(module, "RyyGate")
        .def(py::init<const ParameterResolver &, const VT<Index> &, const VT<Index> &>(), "pr"_a, "obj_qubits"_a,
             "ctrl_qubits"_a = VT<Index>());
    py::class_<mindquantum::RzzGate, mindquantum::BasicGate, std::shared_ptr<mindquantum::RzzGate>>(module, "RzzGate")
        .def(py::init<const ParameterResolver &, const VT<Index> &, const VT<Index> &>(), "pr"_a, "obj_qubits"_a,
             "ctrl_qubits"_a = VT<Index>());
    py::class_<mindquantum::RxyGate, mindquantum::BasicGate, std::shared_ptr<mindquantum::RxyGate>>(module, "RxyGate")
        .def(py::init<const ParameterResolver &, const VT<Index> &, const VT<Index> &>(), "pr"_a, "obj_qubits"_a,
             "ctrl_qubits"_a = VT<Index>());
    py::class_<mindquantum::RxzGate, mindquantum::BasicGate, std::shared_ptr<mindquantum::RxzGate>>(module, "RxzGate")
        .def(py::init<const ParameterResolver &, const VT<Index> &, const VT<Index> &>(), "pr"_a, "obj_qubits"_a,
             "ctrl_qubits"_a = VT<Index>());
    py::class_<mindquantum::RyzGate, mindquantum::BasicGate, std::shared_ptr<mindquantum::RyzGate>>(module, "RyzGate")
        .def(py::init<const ParameterResolver &, const VT<Index> &, const VT<Index> &>(), "pr"_a, "obj_qubits"_a,
             "ctrl_qubits"_a = VT<Index>());
    py::class_<mindquantum::GPGate, mindquantum::BasicGate, std::shared_ptr<mindquantum::GPGate>>(module, "GPGate")
        .def(py::init<const ParameterResolver &, const VT<Index> &, const VT<Index> &>(), "pr"_a, "obj_qubits"_a,
             "ctrl_qubits"_a = VT<Index>());
    py::class_<mindquantum::PSGate, mindquantum::BasicGate, std::shared_ptr<mindquantum::PSGate>>(module, "PSGate")
        .def(py::init<const ParameterResolver &, const VT<Index> &, const VT<Index> &>(), "pr"_a, "obj_qubits"_a,
             "ctrl_qubits"_a = VT<Index>());
    py::class_<mindquantum::U3, mindquantum::BasicGate, std::shared_ptr<mindquantum::U3>>(module, "u3")
        .def(py::init<const ParameterResolver &, const ParameterResolver &, const ParameterResolver &,
                      const VT<Index> &, const VT<Index> &>(),
             "theta"_a, "phi"_a, "lambda"_a, "obj_qubits"_a, "ctrl_qubits"_a = VT<Index>());
    py::class_<mindquantum::FSim, mindquantum::BasicGate, std::shared_ptr<mindquantum::FSim>>(module, "fsim")
        .def(py::init<const ParameterResolver &, const ParameterResolver &, const VT<Index> &, const VT<Index> &>(),
             "theta"_a, "phi"_a, "obj_qubits"_a, "ctrl_qubits"_a = VT<Index>());
    py::class_<mindquantum::KrausChannel, mindquantum::BasicGate, std::shared_ptr<mindquantum::KrausChannel>>(
        module, "KrausChannel")
        .def(py::init<const VT<VVT<CT<double>>> &, const VT<Index> &, const VT<Index> &>(), "kraus_operator_set"_a,
             "obj_qubits"_a, "ctrl_qubits"_a = VT<Index>());
    py::class_<mindquantum::CustomGate, mindquantum::BasicGate, std::shared_ptr<mindquantum::CustomGate>>(module,
                                                                                                          "CustomGate")
        .def(py::init<std::string, uint64_t, uint64_t, int, const ParameterResolver, const VT<Index> &,
                      const VT<Index> &>(),
             "name"_a, "m_addr"_a, "dm_addr"_a, "dim"_a, "pr"_a, "obj_qubits"_a, "ctrl_qubits"_a = VT<Index>())
        .def(py::init<std::string, const tensor::Matrix &, const VT<Index> &, const VT<Index> &>(), "name"_a, "mat"_a,
             "obj_qubits"_a, "ctrl_qubits"_a);
}
template <typename T>
auto BindOther(py::module &module) {
    using namespace pybind11::literals;  // NOLINT(build/namespaces_literals)
    using mindquantum::CT;
    using mindquantum::Hamiltonian;
    using mindquantum::Index;
    using mindquantum::PauliTerm;
    using mindquantum::VS;
    using mindquantum::VT;
    using mindquantum::VVT;
    using mindquantum::python::CsrHdMatrix;
    using parameter::ParameterResolver;
    // matrix

    // parameter resolver
    using mindquantum::sparse::Csr_Plus_Csr;
    using mindquantum::sparse::GetPauliMat;
    using mindquantum::sparse::PauliMat;
    using mindquantum::sparse::PauliMatToCsrHdMatrix;
    using mindquantum::sparse::SparseHamiltonian;
    using mindquantum::sparse::TransposeCsrHdMatrix;
    // pauli mat
    py::class_<PauliMat<T>, std::shared_ptr<PauliMat<T>>>(module, "pauli_mat")
        .def(py::init<>())
        .def(py::init<const PauliTerm<T> &, Index>())
        .def_readonly("n_qubits", &PauliMat<T>::n_qubits_)
        .def_readonly("dim", &PauliMat<T>::dim_)
        .def_readwrite("coeff", &PauliMat<T>::p_)
        .def("PrintInfo", &PauliMat<T>::PrintInfo);

    module.def("get_pauli_mat", &GetPauliMat<T>);

    // // csr_hd_matrix
    py::class_<CsrHdMatrix<T>, std::shared_ptr<CsrHdMatrix<T>>>(module, "csr_hd_matrix")
        .def(py::init<>())
        .def(py::init<Index, Index, py::array_t<Index>, py::array_t<Index>, py::array_t<CT<T>>>())
        .def("PrintInfo", &CsrHdMatrix<T>::PrintInfo);
    module.def("csr_plus_csr", &Csr_Plus_Csr<T>);
    module.def("transpose_csr_hd_matrix", &TransposeCsrHdMatrix<T>);
    module.def("pauli_mat_to_csr_hd_matrix", &PauliMatToCsrHdMatrix<T>);

    // hamiltonian
    py::class_<Hamiltonian<T>, std::shared_ptr<Hamiltonian<T>>>(module, "hamiltonian")
        .def(py::init<>())
        .def(py::init<const VT<PauliTerm<T>> &>())
        .def(py::init<const VT<PauliTerm<T>> &, Index>())
        .def(py::init<std::shared_ptr<CsrHdMatrix<T>>, Index>())
        .def_readwrite("how_to", &Hamiltonian<T>::how_to_)
        .def_readwrite("n_qubits", &Hamiltonian<T>::n_qubits_)
        .def_readwrite("ham", &Hamiltonian<T>::ham_)
        .def_readwrite("ham_sparse_main", &Hamiltonian<T>::ham_sparse_main_)
        .def_readwrite("ham_sparse_second", &Hamiltonian<T>::ham_sparse_second_);
    module.def("sparse_hamiltonian", &SparseHamiltonian<T>);
}

// Interface with python
PYBIND11_MODULE(mqbackend, m) {
    m.doc() = "MindQuantum C++ plugin";

    py::module logging = m.def_submodule("logging", "MindQuantum-C++ logging module");
    mindquantum::python::init_logging(logging);

    auto gate_id = py::enum_<mindquantum::GateID>(m, "GateID")
                       .value("I", mindquantum::GateID::I)
                       .value("X", mindquantum::GateID::X)
                       .value("Y", mindquantum::GateID::Y)
                       .value("Z", mindquantum::GateID::Z)
                       .value("RX", mindquantum::GateID::RX)
                       .value("RY", mindquantum::GateID::RY)
                       .value("RZ", mindquantum::GateID::RZ)
                       .value("Rxx", mindquantum::GateID::Rxx)
                       .value("Ryy", mindquantum::GateID::Ryy)
                       .value("Rzz", mindquantum::GateID::Rzz)
                       .value("H", mindquantum::GateID::H)
                       .value("SWAP", mindquantum::GateID::SWAP)
                       .value("ISWAP", mindquantum::GateID::ISWAP)
                       .value("T", mindquantum::GateID::T)
                       .value("Tdag", mindquantum::GateID::T)
                       .value("S", mindquantum::GateID::Sdag)
                       .value("Sdag", mindquantum::GateID::Sdag)
                       .value("CNOT", mindquantum::GateID::CNOT)
                       .value("CZ", mindquantum::GateID::CZ)
                       .value("GP", mindquantum::GateID::GP)
                       .value("PS", mindquantum::GateID::PS)
                       .value("U3", mindquantum::GateID::U3)
                       .value("FSim", mindquantum::GateID::FSim)
                       .value("M", mindquantum::GateID::M)
                       .value("PL", mindquantum::GateID::PL)
                       .value("AD", mindquantum::GateID::AD)
                       .value("PD", mindquantum::GateID::PD)
                       .value("KRAUS", mindquantum::GateID::KRAUS)
                       .value("CUSTOM", mindquantum::GateID::CUSTOM);
    gate_id.attr("__repr__") = pybind11::cpp_function(
        [](const mindquantum::GateID &id) -> pybind11::str { return fmt::format("GateID.{}", id); },
        pybind11::name("name"), pybind11::is_method(gate_id));
    gate_id.attr("__str__") = pybind11::cpp_function(
        [](const mindquantum::GateID &id) -> pybind11::str { return fmt::format("{}", id); }, pybind11::name("name"),
        pybind11::is_method(gate_id));

    m.attr("EQ_TOLERANCE") = py::float_(1.e-8);

    py::module gate = m.def_submodule("gate", "MindQuantum-C++ gate");
    py::class_<mindquantum::BasicGate, std::shared_ptr<mindquantum::BasicGate>>(gate, "BasicGate").def(py::init<>());
    BindTypeIndependentGate(gate);
    BindTypeDependentGate(gate);

    py::module mqbackend_double = m.def_submodule("double", "MindQuantum-C++ double backend");
    BindOther<double>(mqbackend_double);
    py::module mqbackend_float = m.def_submodule("float", "MindQuantum-C++ double backend");
    BindOther<float>(mqbackend_float);

    py::module c = m.def_submodule("c", "pybind11 c++ env");
    mindquantum::BindPybind11Env(c);

    py::module device = m.def_submodule("device", "Quantum device module");
    BindTopology(device);
    BindQubitMapping(device);
}
