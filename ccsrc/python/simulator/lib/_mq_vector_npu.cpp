/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <algorithm>
#include <memory>

#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "simulator/vector/ascend/program.h"
#include "simulator/vector/ascend/runtime.h"

PYBIND11_MODULE(_mq_vector_npu, module) {
    namespace py = pybind11;
    using namespace pybind11::literals;        // NOLINT
    using namespace mindquantum::sim::ascend;  // NOLINT
    module.doc() = "Ascend 910B complex64 tiled statevector executor.";
    py::class_<Program>(module, "Program");
    py::class_<AdjointProgram>(module, "AdjointProgram")
        .def_property_readonly("tile_count", [](const AdjointProgram& program) { return program.tiles.size(); })
        .def_property_readonly("cube_tile_count", [](const AdjointProgram& program) {
            return std::count_if(program.tiles.begin(), program.tiles.end(), [](const auto& tile) {
                return std::holds_alternative<std::shared_ptr<const DenseAdjointProgram>>(tile);
            });
        });
    py::class_<ForwardPlan>(module, "ForwardPlan")
        .def(py::init<const std::vector<std::shared_ptr<mindquantum::BasicGate>>&, mindquantum::qbit_t, unsigned,
                      unsigned, unsigned>(),
             "circuit"_a, "n_qubits"_a, "tile_qubits"_a = 12, "fusion_qubits"_a = 6, "cube_min_gates"_a = 4,
             py::call_guard<py::gil_scoped_release>())
        .def("bind", &ForwardPlan::Bind, "pr"_a, py::call_guard<py::gil_scoped_release>());
    py::class_<AdjointPlan>(module, "AdjointPlan")
        .def(py::init<const std::vector<std::shared_ptr<mindquantum::BasicGate>>&, const mindquantum::MST<size_t>&,
                      mindquantum::qbit_t, unsigned, unsigned, unsigned>(),
             "circuit"_a, "parameter_map"_a, "n_qubits"_a, "tile_qubits"_a = 11, "fusion_qubits"_a = 8,
             "cube_min_gates"_a = 4, py::call_guard<py::gil_scoped_release>())
        .def("bind", &AdjointPlan::Bind, "pr"_a, py::call_guard<py::gil_scoped_release>());
    module.def("compile", &Compile, "circuit"_a, "parameters"_a, "n_qubits"_a, "tile_qubits"_a = 12,
               "fusion_qubits"_a = 6, "cube_min_gates"_a = 4, py::call_guard<py::gil_scoped_release>());
    module.def("compile_derivative", &CompileDerivative, "gate"_a, "pr"_a, "n_qubits"_a, "tile_qubits"_a = 12,
               py::call_guard<py::gil_scoped_release>());
    py::class_<Executable, std::shared_ptr<Executable>>(module, "Executable")
        .def_property_readonly("tile_count",
                               py::cpp_function(&Executable::TileCount, py::call_guard<py::gil_scoped_release>()))
        .def_property_readonly("cube_tile_count",
                               py::cpp_function(&Executable::CubeTileCount, py::call_guard<py::gil_scoped_release>()))
        .def_property_readonly("instruction_count", py::cpp_function(&Executable::InstructionCount,
                                                                     py::call_guard<py::gil_scoped_release>()))
        .def_property_readonly("metadata_bytes",
                               py::cpp_function(&Executable::MetadataBytes, py::call_guard<py::gil_scoped_release>()));
    py::class_<Observable, std::shared_ptr<Observable>>(module, "Observable");
    py::class_<State>(module, "State")
        .def(py::init<mindquantum::qbit_t, int>(), "n_qubits"_a, "device_id"_a = 0,
             py::call_guard<py::gil_scoped_release>())
        .def("prepare", &State::Prepare, "program"_a, py::call_guard<py::gil_scoped_release>())
        .def("apply", &State::Apply, "executable"_a, "repetitions"_a = 1, py::call_guard<py::gil_scoped_release>())
        .def("reset", &State::Reset, py::call_guard<py::gil_scoped_release>())
        .def("copy", &State::Copy, py::call_guard<py::gil_scoped_release>())
        .def("copy_from", &State::CopyFrom, "other"_a, py::call_guard<py::gil_scoped_release>())
        .def("prepare_observable", &State::PrepareObservable, "hamiltonian"_a, py::call_guard<py::gil_scoped_release>())
        .def("expectation", &State::Expectation, "observable"_a, "bra"_a = nullptr,
             py::call_guard<py::gil_scoped_release>())
        .def("apply_hamiltonian", &State::ApplyHamiltonian, "observable"_a, py::call_guard<py::gil_scoped_release>())
        .def("dot", &State::Dot, "ket"_a, py::call_guard<py::gil_scoped_release>())
        .def("norm", &State::Norm, "mask"_a = 0, "value"_a = 0, py::call_guard<py::gil_scoped_release>())
        .def("reduced_density", &State::ReducedDensity, "kept_qubits"_a, py::call_guard<py::gil_scoped_release>())
        .def("scale", &State::Scale, "factor"_a, py::call_guard<py::gil_scoped_release>())
        .def("measure", &State::Measure, "qubit"_a, "draw"_a, "reset_to"_a = -1,
             py::call_guard<py::gil_scoped_release>())
        .def("sample", &State::Sample, "shots"_a, "seed"_a, py::call_guard<py::gil_scoped_release>())
        .def("set_qs",
             [](State& state, py::array_t<std::complex<float>, py::array::c_style> values) {
                 if (values.ndim() != 1) {
                     throw py::value_error("State must be one-dimensional.");
                 }
                 const auto* data = values.data();
                 const auto size = values.size();
                 py::gil_scoped_release release;
                 state.Set(data, size);
             })
        .def("get_qs",
             [](const State& state) {
                 std::vector<std::complex<float>> values;
                 {
                     py::gil_scoped_release release;
                     values = state.Get();
                 }
                 auto buffer = std::make_unique<std::vector<std::complex<float>>>(std::move(values));
                 py::capsule owner(buffer.get(), [](void* pointer) {
                     delete static_cast<std::vector<std::complex<float>>*>(pointer);  // NOLINT
                 });
                 const auto size = static_cast<py::ssize_t>(buffer->size());
                 auto* data = buffer->data();
                 buffer.release();  // ownership transfers to the NumPy array's capsule
                 return py::array_t<std::complex<float>>({size}, {sizeof(std::complex<float>)}, data, owner);
             })
        .def_property_readonly("soc_name", py::cpp_function(&State::SocName, py::call_guard<py::gil_scoped_release>()))
        .def_property_readonly("vector_cores",
                               py::cpp_function(&State::VectorCores, py::call_guard<py::gil_scoped_release>()));
    py::class_<GradientPlan>(module, "GradientPlan")
        .def("evaluate", &GradientPlan::Evaluate, "right_state"_a, "left_state"_a, "parameters"_a,
             py::call_guard<py::gil_scoped_release>());
    module.def(
        "make_gradient",
        [](const State& owner, const std::vector<mindquantum::Hamiltonian<float>>& hams,
           const std::vector<mindquantum::Hamiltonian<float>>& hams_dagger,
           const std::vector<std::shared_ptr<mindquantum::BasicGate>>& right_gates,
           const std::vector<std::shared_ptr<mindquantum::BasicGate>>& left_gates,
           const std::vector<std::string>& parameter_names, bool hermitian, unsigned tile_qubits,
           unsigned fusion_qubits, unsigned cube_min_gates) {
            return std::make_unique<GradientPlan>(owner, hams, hams_dagger, right_gates, left_gates, parameter_names,
                                                  hermitian, tile_qubits, fusion_qubits, cube_min_gates);
        },
        "owner"_a, "hams"_a, "hams_dagger"_a, "right_gates"_a, "left_gates"_a, "parameter_names"_a, "hermitian"_a,
        "tile_qubits"_a = 12, "fusion_qubits"_a = 6, "cube_min_gates"_a = 4, py::call_guard<py::gil_scoped_release>());
}
