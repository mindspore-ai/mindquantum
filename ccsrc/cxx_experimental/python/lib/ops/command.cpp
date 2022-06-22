//   Copyright 2020 <Huawei Technologies Co., Ltd>
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

#include "python/ops/command.hpp"

#include <complex>
#include <iostream>
#include <map>
#include <optional>
#include <utility>
#include <vector>

#include <pybind11/complex.h>
#include <pybind11/iostream.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <tweedledum/Operators/Ising.h>
#include <tweedledum/Operators/Standard.h>

#include "ops/gates.hpp"
#include "ops/meta/dagger.hpp"
#include "python/details/python_api.hpp"

namespace py = pybind11;
namespace td = tweedledum;

// ==============================================================================

using PSG = mindquantum::details::PythonScopeGuard;

// ==============================================================================

namespace mindquantum::details {
td::Operator load_gate(PyObject* src, std::size_t n_targets, std::size_t n_controls, bool is_dagger) {
    using ComplexTermsDict = std::map<std::vector<std::pair<unsigned, char>>, std::complex<double>>;

    pybind11::scoped_ostream_redirect output;
    auto ptype = Py_TYPE(src);
    std::string_view gate_type(ptype->tp_name);

    const auto has_parameter = PyObject_HasAttrString(src, "angle");

    if (gate_type == "DaggeredGate") {
        if (auto attr = PSG(PyObject_GetAttrString(src, "_gate")); attr) {
            return load_gate(attr, n_targets, n_controls, true);
        }
    } else if (has_parameter) {
        if (auto attr = PSG(PyObject_GetAttrString(src, "angle")); attr) {
            using caster_t = py::detail::make_caster<double>;
            if (caster_t caster; caster.load(py::handle(attr), true)) {
                assert(!is_dagger);  // Should be taken care of in Python!
                double angle = caster;
                if (gate_type == "Rx" || gate_type == "RxNum") {
                    return td::Op::Rx(angle);
                } else if (gate_type == "Ry" || gate_type == "RyNum") {
                    return td::Op::Ry(angle);
                } else if (gate_type == "Rz" || gate_type == "RzNum") {
                    return td::Op::Rz(angle);
                } else if (gate_type == "R" || gate_type == "RNum") {
                    return td::Op::P(angle);
                } else if (gate_type == "Ph" || gate_type == "PhNum") {
                    return ops::Ph(angle);
                } else if (gate_type == "Rxx" || gate_type == "RxxNum") {
                    return td::Op::Rxx(angle / 2);
                } else if (gate_type == "Ryy" || gate_type == "RyyNum") {
                    return td::Op::Ryy(angle / 2);
                } else if (gate_type == "Rzz" || gate_type == "RzzNum") {
                    return td::Op::Rzz(angle);
                }
            }
        }
    } else if (gate_type == "QubitOperator") {
        if (auto attr = PSG(PyObject_GetAttrString(src, "terms")); attr) {
            using caster_t = py::detail::make_caster<ComplexTermsDict>;
            if (caster_t caster; caster.load(py::handle(attr), true)) {
                ComplexTermsDict terms = caster;
                if (caster) {
                    if (is_dagger) {
                        return ops::DaggerOperation(ops::QubitOperator(n_targets, terms));
                    } else {
                        return ops::QubitOperator(n_targets, terms);
                    }
                }
            }
        }
    } else if (gate_type == "TimeEvolution") {
        auto ham_attr = PSG(PyObject_GetAttrString(src, "hamiltonian"));
        auto terms_attr = PSG(PyObject_GetAttrString(ham_attr, "terms"));
        auto time_attr = PSG(PyObject_GetAttrString(src, "time"));
        if (terms_attr && time_attr) {
            py::detail::make_caster<ComplexTermsDict> terms_caster;
            py::detail::make_caster<double> time_caster;
            if (terms_caster.load(py::handle(terms_attr), true) && time_caster.load(py::handle(time_attr), true)) {
                ComplexTermsDict terms = terms_caster;
                double time = time_caster;
                if (is_dagger) {
                    time *= -1.;
                }
                return ops::TimeEvolution(ops::QubitOperator(n_targets, terms), time);
            } else {
                std::cerr << "Error: Couldn't load TimeEvolution attributes\n";
            }
        } else {
            std::cerr << "Error: Couldn't get TimeEvolution attributes\n";
        }
    } else if (gate_type == "MeasureGate") {
        assert(is_dagger == false);
        return ops::Measure();
    } else if (gate_type == "HGate") {
        return td::Op::H();
    } else if (gate_type == "XGate") {
        return td::Op::X();
    } else if (gate_type == "SqrtXGate") {
        if (is_dagger) {
            return td::Op::Sxdg();
        } else {
            return td::Op::Sx();
        }
    } else if (gate_type == "YGate") {
        return td::Op::Y();
    } else if (gate_type == "ZGate") {
        return td::Op::Z();
    } else if (gate_type == "SGate") {
        if (is_dagger) {
            return td::Op::Sdg();
        } else {
            return td::Op::S();
        }
    } else if (gate_type == "TGate") {
        if (is_dagger) {
            return td::Op::Tdg();
        } else {
            return td::Op::T();
        }
    } else if (gate_type == "SwapGate") {
        return td::Op::Swap();
    } else if (gate_type == "SqrtSwapGate") {
        if (is_dagger) {
            return ops::SqrtSwap().adjoint();
        } else {
            return ops::SqrtSwap();
        }
    } else if (gate_type == "EntangleGate") {
        if (is_dagger) {
            return ops::Entangle(n_targets).adjoint();
        } else {
            return ops::Entangle(n_targets);
        }
    } else if (gate_type == "QFTGate") {
        if (is_dagger) {
            return ops::QFT(n_targets).adjoint();
        } else {
            return ops::QFT(n_targets);
        }
    } else if (gate_type == "AllocateQubitGate") {
        if (is_dagger) {
            return ops::Deallocate();
        } else {
            return ops::Allocate();
        }
    } else if (gate_type == "DeallocateQubitGate") {
        if (is_dagger) {
            return ops::Allocate();
        } else {
            return ops::Deallocate();
        }
    }

    return ops::Invalid();
}
}  // namespace mindquantum::details

// ==============================================================================

std::optional<mindquantum::qubit_id_t> load_qubit(PyObject* src) {
    using qubit_id_t = mindquantum::qubit_id_t;
    if (auto attr = PSG(PyObject_GetAttrString(src, "id")); attr) {
        if (auto qubit_id_raw = PSG(PyNumber_Long(attr)); qubit_id_raw) {
            qubit_id_t qubit_id = PyLong_AsLong(qubit_id_raw);
            return qubit_id;
        }
    }
    return {};
}

std::optional<mindquantum::qureg_t> load_qureg(PyObject* src) {
    if (!PyList_Check(src)) {
        return {};
    }

    mindquantum::qureg_t qureg;

    const auto size = PyList_Size(src);
    for (auto i(0UL); i < size; ++i) {
        auto qubit_id = load_qubit(PyList_GetItem(src, i));
        if (!qubit_id) {
            return {};
        }
        qureg.emplace_back(qubit_id.value());
    }
    return qureg;
}

// ==============================================================================

bool mindquantum::details::load_command(pybind11::handle src, python::Command& cmd) {
    using str_caster_t = py::detail::make_caster<std::string>;

    qureg_t qubits;
    if (auto attr = PSG(PyObject_GetAttrString(src.ptr(), "qubits")); attr) {
        if (!PyTuple_Check(attr)) {
            return false;
        }

        const auto size = PyTuple_Size(attr);
        for (auto i(0UL); i < size; ++i) {
            if (auto qureg = load_qureg(PyTuple_GetItem(attr, i)); qureg) {
                for (auto qubit : qureg.value()) {
                    qubits.emplace_back(qubit);
                }
            } else {
                return false;
            }
        }
    } else {
        return false;
    }

    qureg_t control_qubits;
    if (auto attr = PSG(PyObject_GetAttrString(src.ptr(), "control_qubits")); attr) {
        if (auto qureg = load_qureg(attr); qureg) {
            control_qubits = qureg.value();
        } else {
            return false;
        }
    } else {
        return false;
    }

    if (auto* attr = PyObject_GetAttrString(src.ptr(), "gate"); attr) {
        auto gate = mindquantum::details::load_gate(attr, std::size(qubits), std::size(control_qubits));
        if (gate.kind() == ops::Invalid::kind()) {
            std::cerr << "Unsupported gate type!" << std::endl;
            return false;
        }
        cmd.set_gate(gate);
        cmd.set_qubits(qubits);
        cmd.set_control_qubits(control_qubits);
        return true;
    }
    return false;
}
